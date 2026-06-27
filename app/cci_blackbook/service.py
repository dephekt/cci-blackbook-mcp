from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np

from .chunking import PageText, chunk_pages
from .embeddings import EmbeddingProvider, HashEmbeddingProvider, build_embedding_provider
from .settings import Settings, load_settings
from .store import BlackBookIndex, SearchHit


class IndexUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FusedHit:
    hit: SearchHit
    score: float
    sources: tuple[str, ...]


class BlackBookService:
    def __init__(self, settings: Settings | None = None, provider: EmbeddingProvider | None = None):
        self.settings = settings or load_settings()
        self.index = BlackBookIndex(self.settings.sqlite_path)
        self._provider = provider
        self._lock = threading.Lock()

    def status(self) -> dict:
        index_status = self.index.status()
        source_status = _source_status(self.settings.source_pdf)
        provider = self._provider or self._new_provider()
        return {
            "service": "cci-blackbook-mcp",
            "source": source_status,
            "index": index_status,
            "embedding": provider.status(),
            "paths": {
                "source_pdf": str(self.settings.source_pdf),
                "index_dir": str(self.settings.index_dir),
                "cache_dir": str(self.settings.cache_dir),
            },
        }

    def ensure_index(self, *, force: bool = False) -> dict:
        with self._lock:
            source_metadata = _source_metadata(self.settings.source_pdf)
            index_status = self.index.status()
            if not force and _index_current(index_status, source_metadata, self.settings):
                return {"rebuilt": False, "status": index_status}

            if not self.settings.source_pdf.exists():
                raise IndexUnavailable(f"source PDF missing: {self.settings.source_pdf}")

            pages = _read_pdf_pages(self.settings.source_pdf)
            return self.rebuild_from_pages(
                pages,
                source_metadata=source_metadata,
                force_provider_refresh=True,
            )

    def rebuild_from_pages(
        self,
        pages: list[PageText],
        *,
        source_metadata: dict | None = None,
        force_provider_refresh: bool = False,
    ) -> dict:
        chunks = chunk_pages(
            pages,
            chunk_chars=self.settings.chunk_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        if not chunks:
            raise IndexUnavailable("source PDF had no extractable text chunks")

        if force_provider_refresh:
            self._provider = None
        provider = self._provider or self._new_provider()
        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = _embed_all(provider, texts, batch_size=self.settings.embedding_batch_size)
        except Exception as exc:
            provider = HashEmbeddingProvider(
                detail=f"{provider.status().get('backend')} failed with {type(exc).__name__}"
            )
            self._provider = provider
            embeddings = _embed_all(provider, texts, batch_size=self.settings.embedding_batch_size)
        else:
            self._provider = provider

        metadata = {
            "source": source_metadata or {"kind": "in-memory"},
            "embedding": provider.status(),
            "chunking": {
                "chunk_chars": self.settings.chunk_chars,
                "chunk_overlap_chars": self.settings.chunk_overlap_chars,
            },
            "built_at": int(time()),
        }
        self.index.rebuild(chunks, embeddings, metadata)
        return {"rebuilt": True, "chunk_count": len(chunks), "status": self.index.status()}

    def search(self, query: str, *, limit: int = 10, mode: str = "hybrid") -> dict:
        limit = _clamp(limit, 1, 20)
        mode = mode.lower()
        if mode not in {"hybrid", "fts", "vector"}:
            mode = "hybrid"

        try:
            self.ensure_index()
        except IndexUnavailable as exc:
            return {
                "query": query,
                "mode": mode,
                "results": [],
                "abstain": True,
                "confidence_notes": [str(exc)],
            }

        fetch_limit = max(limit * 4, 20)
        fts_hits: list[SearchHit] = []
        vector_hits: list[SearchHit] = []
        vector_error: str | None = None
        provider = self._provider or self._new_provider()

        if mode in {"fts", "hybrid"}:
            fts_hits = self.index.search_fts(query, limit=fetch_limit)
        if mode in {"vector", "hybrid"}:
            try:
                query_vector = provider.embed_query(query)
                vector_hits = [
                    hit
                    for hit in self.index.search_vector(query_vector, limit=fetch_limit)
                    if hit.score >= self.settings.min_vector_score
                ]
            except Exception as exc:
                vector_error = f"vector search unavailable: {type(exc).__name__}"

        if mode == "fts":
            fused = [FusedHit(hit=hit, score=hit.score, sources=("fts",)) for hit in fts_hits[:limit]]
        elif mode == "vector":
            fused = [
                FusedHit(hit=hit, score=hit.score, sources=("vector",))
                for hit in vector_hits[:limit]
            ]
        else:
            fused = _fuse_hits([("fts", fts_hits), ("vector", vector_hits)], limit=limit)

        confidence_notes = _confidence_notes(fused, mode, provider.status())
        if vector_error:
            confidence_notes.append(vector_error)

        return {
            "query": query,
            "mode": mode,
            "results": [_format_hit(item, query) for item in fused],
            "abstain": not fused,
            "confidence_notes": confidence_notes,
        }

    def ask(
        self,
        question: str,
        *,
        crop_context: str | None = None,
        facility_context: str | None = None,
        max_citations: int = 6,
    ) -> dict:
        parts = [question]
        if crop_context:
            parts.append(f"crop context: {crop_context}")
        if facility_context:
            parts.append(f"facility context: {facility_context}")
        search_query = "\n".join(parts)
        search_result = self.search(search_query, limit=_clamp(max_citations, 1, 10), mode="hybrid")
        return {
            "question": question,
            "crop_context": crop_context,
            "facility_context": facility_context,
            "abstain": search_result["abstain"],
            "answer_instruction": (
                "Compose the answer from these cited excerpts only. "
                "If the excerpts do not answer the question, say the Black Book evidence is insufficient."
            ),
            "evidence": search_result["results"],
            "confidence_notes": search_result["confidence_notes"],
        }

    def read_citation(self, chunk_id: str) -> dict:
        try:
            self.ensure_index()
        except IndexUnavailable as exc:
            return {"chunk_id": chunk_id, "found": False, "error": str(exc)}
        hit = self.index.read_chunk(chunk_id)
        if hit is None:
            return {"chunk_id": chunk_id, "found": False}
        return {
            "chunk_id": hit.chunk_id,
            "found": True,
            "page": hit.page,
            "citation": _citation(hit),
            "text": _bounded_text(hit.text, 2500),
            "bounded": len(hit.text) > 2500,
        }

    def _new_provider(self) -> EmbeddingProvider:
        return build_embedding_provider(
            backend=self.settings.embedding_backend,
            model_name=self.settings.embedding_model,
            cache_dir=self.settings.cache_dir,
            openvino_device=self.settings.openvino_device,
            render_device=self.settings.render_device,
        )


def _embed_all(provider: EmbeddingProvider, texts: list[str], *, batch_size: int = 64) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for offset in range(0, len(texts), batch_size):
        vectors.extend(provider.embed_passages(texts[offset : offset + batch_size]))
    return vectors


def _read_pdf_pages(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append(PageText(page=page_number, text=page.extract_text() or ""))
    return pages


def _source_status(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {"exists": True, "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_metadata(path: Path) -> dict:
    status = _source_status(path)
    if not status["exists"]:
        return status
    return {
        "path": status["path"],
        "size": status["size"],
        "mtime_ns": status["mtime_ns"],
    }


def _index_current(index_status: dict, source_metadata: dict, settings: Settings) -> bool:
    if not index_status.get("ready"):
        return False
    metadata = index_status.get("metadata", {})
    return (
        metadata.get("source") == source_metadata
        and metadata.get("embedding", {}).get("model") == settings.embedding_model
        and metadata.get("chunking", {}).get("chunk_chars") == settings.chunk_chars
        and metadata.get("chunking", {}).get("chunk_overlap_chars") == settings.chunk_overlap_chars
    )


def _fuse_hits(named_hit_lists: list[tuple[str, list[SearchHit]]], *, limit: int) -> list[FusedHit]:
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    sources: dict[str, set[str]] = {}

    for source, hit_list in named_hit_lists:
        for rank, hit in enumerate(hit_list, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (60.0 + rank)
            hits.setdefault(hit.chunk_id, hit)
            sources.setdefault(hit.chunk_id, set()).add(source)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        FusedHit(hit=hits[chunk_id], score=score, sources=tuple(sorted(sources[chunk_id])))
        for chunk_id, score in ordered[:limit]
    ]


def _format_hit(item: FusedHit, query: str) -> dict:
    hit = item.hit
    return {
        "chunk_id": hit.chunk_id,
        "page": hit.page,
        "citation": _citation(hit),
        "retrieval_score": round(item.score, 6),
        "sources": list(item.sources),
        "excerpt": _excerpt(hit.text, query),
    }


def _citation(hit: SearchHit) -> str:
    return f"CCI Black Book page {hit.page}, chunk {hit.chunk_id}"


def _excerpt(text: str, query: str, *, max_chars: int = 700) -> str:
    terms = [term.lower() for term in query.split() if len(term) > 2]
    lower = text.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) != -1]
    if positions:
        center = min(positions)
        start = max(0, center - max_chars // 3)
    else:
        start = 0
    end = min(len(text), start + max_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(text):
        snippet += " ..."
    return snippet


def _bounded_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 4].rstrip() + " ..."


def _confidence_notes(results: list[FusedHit], mode: str, provider_status: dict) -> list[str]:
    notes = [
        f"retrieval mode: {mode}",
        "excerpts are bounded; call blackbook_read_citation for one full bounded chunk",
    ]
    if not results:
        notes.append("no matching Black Book chunks were found")
    if provider_status.get("backend") == "hash":
        notes.append("vector ranking is degraded because the semantic embedding backend fell back")
    if provider_status.get("accelerator") != "openvino":
        notes.append("OpenVINO GPU acceleration is not active for this query")
    return notes


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
