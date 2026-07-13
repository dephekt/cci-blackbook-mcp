from __future__ import annotations

import logging
import re
import statistics
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np

from .chunking import PageText, chunk_pages, group_chunks_into_documents
from .embeddings import DenseEmbeddingProvider, VoyageUnavailable, build_dense_provider
from .pagefilter import PageDecision, classify_page, summarize
from .render import ImageUnit, RenderedPage, page_image_tokens, render_pages
from .settings import Settings, load_settings, settings_fingerprint, voyage_configured
from .store import BlackBookIndex, PageRecord, SearchHit

log = logging.getLogger("cci_blackbook")


class IndexUnavailable(RuntimeError):
    """Source PDF missing or the index is not ready."""


class IngestFailed(RuntimeError):
    """A build could not complete (Voyage down, retention not confirmed, broken extraction).
    Raised loudly so a partial/degraded index is never silently produced."""


@dataclass(frozen=True)
class FusedHit:
    hit: SearchHit
    score: float
    sources: tuple[str, ...]


class BlackBookService:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: DenseEmbeddingProvider | None = None,
        *,
        page_source: Callable[[Path], Iterable[RenderedPage]] | None = None,
    ):
        self.settings = settings or load_settings()
        self.index = BlackBookIndex(self.settings.sqlite_path)
        self._provider = provider
        self._page_source = page_source or self._default_page_source
        self._lock = threading.Lock()

    def _default_page_source(self, path: Path) -> Iterable[RenderedPage]:
        return render_pages(
            path,
            dpi=self.settings.render_dpi,
            max_pixels=self.settings.render_max_pixels,
            ink_luma_threshold=self.settings.ink_luma_threshold,
            color_sat_threshold=self.settings.color_sat_threshold,
        )

    def status(self) -> dict:
        index_status = self.index.status()
        source_status = _source_status(self.settings.source_pdf)
        provider = self._provider or build_dense_provider(self.settings)
        return {
            "service": "cci-blackbook-mcp",
            "source": source_status,
            "index": index_status,
            "embedding": provider.status(),
            "voyage_configured": voyage_configured(),
            "paths": {
                "source_pdf": str(self.settings.source_pdf),
                "index_dir": str(self.settings.index_dir),
                "cache_dir": str(self.settings.cache_dir),
            },
        }

    # ------------------------------------------------------------------ ingest

    def ensure_index(self, *, force: bool = False) -> dict:
        """Build or refresh the index. Called ONLY by the ingest CLI — never on the
        query path (which must never trigger a paid rebuild)."""
        with self._lock:
            source_metadata = _source_metadata(self.settings.source_pdf)
            index_status = self.index.status()
            if not force and _index_current(index_status, source_metadata, self.settings):
                return {"rebuilt": False, "status": index_status}
            if not self.settings.source_pdf.exists():
                raise IndexUnavailable(f"source PDF missing: {self.settings.source_pdf}")
            self._guard_retention()
            try:
                return self._rebuild_from_source(source_metadata)
            except VoyageUnavailable as exc:
                raise IngestFailed(f"embedding provider failed during ingest: {exc}") from exc

    def _guard_retention(self) -> None:
        if self.settings.embedding_backend == "voyage" and not self.settings.voyage_retention_confirmed:
            raise IngestFailed(
                "refusing to send the private Black Book to Voyage: confirm the zero-retention "
                "opt-out (or accept retention), then set CCI_VOYAGE_RETENTION_CONFIRMED=true"
            )

    def _rebuild_from_source(self, source_metadata: dict) -> dict:
        provider = self._provider or build_dense_provider(self.settings)
        s = self.settings
        fk, fd = s.force_keep_pages, s.force_drop_pages

        page_texts: list[PageText] = []
        page_records: list[PageRecord] = []
        decisions: list[PageDecision] = []
        image_vectors: dict[int, np.ndarray] = {}
        batch: list[ImageUnit] = []
        batch_tokens = 0

        def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            for unit, vec in zip(batch, provider.embed_image_units(batch), strict=True):
                image_vectors[unit.page] = vec
            batch, batch_tokens = [], 0  # page images released here → one batch resident

        for rp in self._page_source(s.source_pdf):
            decision = classify_page(
                rp,
                blank_min_chars=s.blank_min_chars,
                blank_max_ink=s.blank_max_ink,
                blank_max_color=s.blank_max_color,
                force_keep=fk,
                force_drop=fd,
                disabled=s.blank_filter_disable,
            )
            decisions.append(decision)
            page_records.append(_page_record(rp, decision))
            page_texts.append(PageText(rp.page, rp.ocr_text))  # chunk EVERY page's OCR
            if decision.kept:  # every kept page -> exactly one image unit (even at 0 OCR chars)
                cost = page_image_tokens(
                    rp.width, rp.height, len(rp.ocr_text),
                    pixels_per_token=s.mm_pixels_per_token, chars_per_token=s.chars_per_token,
                )
                if batch and (len(batch) >= s.mm_max_inputs or batch_tokens + cost > s.mm_token_budget):
                    flush()
                batch.append(ImageUnit(rp.page, rp.ocr_text, rp.image))
                batch_tokens += cost
        flush()

        self._check_extraction_tripwire(decisions)
        log.info("blackbook page filter: %s", summarize(decisions))

        chunks = chunk_pages(page_texts, chunk_chars=s.chunk_chars, overlap_chars=s.chunk_overlap_chars)
        if not chunks and not image_vectors:
            raise IngestFailed("source produced no indexable text chunks or page images")

        groups = group_chunks_into_documents(
            chunks, token_budget=s.doc_token_budget,
            chars_per_token=s.chars_per_token, max_chunk_tokens=s.max_chunk_tokens,
        )
        documents = [[chunks[i].text for i in group] for group in groups]
        doc_vecs = provider.embed_text_documents(documents)
        chunk_vectors = _scatter(groups, doc_vecs, n=len(chunks))

        metadata = {
            "source": source_metadata,
            "fingerprint": settings_fingerprint(s),
            "text_embedding": {
                **provider.status(), "space": "text",
                "observed_dim": int(chunk_vectors[0].shape[0]) if chunk_vectors else None,
            },
            "image_embedding": {
                **provider.status(), "space": "image",
                "observed_dim": int(next(iter(image_vectors.values())).shape[0]) if image_vectors else None,
            },
            "chunking": {"chunk_chars": s.chunk_chars, "chunk_overlap_chars": s.chunk_overlap_chars},
            "grouping": {"documents": len(documents), "doc_token_budget": s.doc_token_budget},
            "filter": summarize(decisions),
            "built_at": int(time()),
        }
        self.index.rebuild(
            chunks=chunks, chunk_vectors=chunk_vectors,
            page_records=page_records, page_vectors=image_vectors, metadata=metadata,
        )
        self._provider = provider
        return {
            "rebuilt": True,
            "chunk_count": len(chunks),
            "image_unit_count": len(image_vectors),
            "pages_dropped": sum(1 for d in decisions if not d.kept),
            "status": self.index.status(),
        }

    def _check_extraction_tripwire(self, decisions: list[PageDecision]) -> None:
        """Guard the fitz-based OCR extraction against a silent break (e.g. the
        HiddenHorzOCR layer not being read). Measured over TEXT-BEARING pages only,
        since ~30% of this scanned book is legitimately near-empty (figures, blank
        "Notes:" templates) and would otherwise drag an all-pages median under the
        threshold and false-abort a healthy ingest."""
        threshold = self.settings.min_expected_median_chars
        if threshold <= 0 or not decisions:
            return
        text_pages = [d.char_count for d in decisions if d.char_count > 0]
        if not text_pages:
            raise IngestFailed(
                "text extraction produced no text on any page; the scanned OCR layer "
                "may not be readable — aborting before a broken index"
            )
        median_chars = statistics.median(text_pages)
        if median_chars < threshold:
            raise IngestFailed(
                f"text extraction looks sparse (median {median_chars:.0f} chars over "
                f"text-bearing pages < {threshold}); OCR layer may be partially unreadable — aborting"
            )

    # ---------------------------------------------------------------- retrieve

    def search(self, query: str, *, limit: int = 10, mode: str = "hybrid") -> dict:
        limit = _clamp(limit, 1, 20)
        mode = mode.lower()
        if mode not in {"hybrid", "vector", "fts", "text", "image"}:
            mode = "hybrid"

        status = self.index.status()  # NEVER ensure_index here (no paid rebuild on a query)
        if not status.get("ready"):
            return {
                "query": query, "mode": mode, "results": [], "abstain": True,
                "confidence_notes": ["index not built; run cci-blackbook-ingest"],
            }

        provider = self._provider or build_dense_provider(self.settings)
        fetch = max(limit * 4, 20)
        notes_extra: list[str] = []

        fts_hits = self.index.search_fts(query, limit=fetch) if mode in {"fts", "hybrid"} else []

        text_hits: list[SearchHit] = []
        if mode in {"text", "vector", "hybrid"}:
            try:
                qvec = provider.embed_text_query(query)
                raw = self.index.search_vector(qvec, limit=fetch)
                text_hits = [h for h in raw if h.score >= self.settings.min_vector_score]
                notes_extra.append(f"text-dense: {len(raw)} fetched, {len(text_hits)} >= gate")
            except Exception as exc:
                notes_extra.append(f"text-dense unavailable: {type(exc).__name__}")

        image_hits: list[SearchHit] = []
        if mode in {"image", "vector", "hybrid"}:
            try:
                qvec = provider.embed_image_query(query)
                raw = self.index.search_page_vector(qvec, limit=fetch)
                image_hits = [h for h in raw if h.score >= self.settings.min_image_score]
                notes_extra.append(f"image-dense: {len(raw)} fetched, {len(image_hits)} >= gate")
            except Exception as exc:
                notes_extra.append(f"image-dense unavailable: {type(exc).__name__}")

        named = _select_lists(mode, fts_hits, text_hits, image_hits)
        weights = {
            "fts": self.settings.rrf_weight_fts,
            "text_dense": self.settings.rrf_weight_text,
            "image_dense": self.settings.rrf_weight_image,
        }
        fused = _fuse_hits(
            named, limit=limit, k=self.settings.rrf_k, weights=weights,
            max_units_per_page=self.settings.max_units_per_page,
        )
        notes = _confidence_notes(fused, mode, provider.status()) + notes_extra
        return {
            "query": query, "mode": mode,
            "results": [_format_hit(item, query) for item in fused],
            "abstain": not fused, "confidence_notes": notes,
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
                "Compose the answer from these cited excerpts only. Some citations are scanned "
                "figure/page images (unit_type='image') whose OCR text may be sparse — call "
                "blackbook_read_citation to inspect one. If the excerpts do not answer the "
                "question, say the Black Book evidence is insufficient."
            ),
            "evidence": search_result["results"],
            "confidence_notes": search_result["confidence_notes"],
        }

    def read_citation(self, chunk_id: str) -> dict:
        status = self.index.status()  # read directly; never ensure_index
        if not status.get("ready"):
            return {"chunk_id": chunk_id, "found": False, "error": "index not built; run cci-blackbook-ingest"}

        page = parse_image_unit_id(chunk_id)
        if page is not None:
            hit = self.index.read_page(page)
            if hit is None:
                return {"chunk_id": chunk_id, "found": False}
            result = {
                "chunk_id": chunk_id, "unit_id": chunk_id, "found": True, "page": hit.page,
                "unit_type": "image", "citation": _citation(hit),
                "text": _bounded_text(hit.text, 2500), "bounded": len(hit.text) > 2500,
                "note": "match came from the page image; text shown is page OCR (may be sparse)",
            }
            thumb = self._maybe_thumbnail(hit.page)
            if thumb:
                result["thumbnail_png_base64"] = thumb
            return result

        hit = self.index.read_chunk(chunk_id)
        if hit is None:
            return {"chunk_id": chunk_id, "found": False}
        return {
            "chunk_id": hit.chunk_id, "unit_id": hit.chunk_id, "found": True, "page": hit.page,
            "unit_type": "text", "citation": _citation(hit),
            "text": _bounded_text(hit.text, 2500), "bounded": len(hit.text) > 2500,
        }

    def _maybe_thumbnail(self, page: int) -> str | None:
        if not self.settings.citation_thumbnails or not self.settings.source_pdf.exists():
            return None
        try:
            import base64
            import io

            import fitz
            from PIL import Image

            doc = fitz.open(str(self.settings.source_pdf))
            try:
                pg = doc[page - 1]
                zoom = 1024.0 / max(pg.rect.width, pg.rect.height, 1.0)
                pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
            finally:
                doc.close()
        except Exception:
            return None


# ---------------------------------------------------------------------- helpers


def _page_record(rp: RenderedPage, decision: PageDecision) -> PageRecord:
    return PageRecord(
        page=rp.page,
        ocr_text=rp.ocr_text,
        char_count=rp.char_count,
        ink_coverage=rp.ink_coverage,
        color_fraction=rp.color_fraction,
        width=rp.width,
        height=rp.height,
        kept=decision.kept,
        reason=decision.reason,
    )


def _scatter(groups: list[list[int]], doc_vecs: list[list[np.ndarray]], *, n: int) -> list[np.ndarray]:
    out: list[np.ndarray | None] = [None] * n
    for group, vecs in zip(groups, doc_vecs, strict=True):
        for idx, vec in zip(group, vecs, strict=True):
            out[idx] = vec
    if any(v is None for v in out):
        raise IngestFailed("internal: not every chunk received a text vector")
    return out  # type: ignore[return-value]


def parse_image_unit_id(uid: str) -> int | None:
    match = re.match(r"^p(\d+)-img$", uid)
    return int(match.group(1)) if match else None


def _source_status(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {"exists": True, "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_metadata(path: Path) -> dict:
    status = _source_status(path)
    if not status["exists"]:
        return status
    return {"path": status["path"], "size": status["size"], "mtime_ns": status["mtime_ns"]}


def _index_current(index_status: dict, source_metadata: dict, settings: Settings) -> bool:
    if not index_status.get("ready"):
        return False
    metadata = index_status.get("metadata", {})
    return (
        metadata.get("source") == source_metadata
        and metadata.get("fingerprint") == settings_fingerprint(settings)
    )


def _select_lists(
    mode: str,
    fts_hits: list[SearchHit],
    text_hits: list[SearchHit],
    image_hits: list[SearchHit],
) -> list[tuple[str, list[SearchHit]]]:
    if mode == "fts":
        return [("fts", fts_hits)]
    if mode == "text":
        return [("text_dense", text_hits)]
    if mode == "image":
        return [("image_dense", image_hits)]
    if mode == "vector":
        return [("text_dense", text_hits), ("image_dense", image_hits)]
    return [("fts", fts_hits), ("text_dense", text_hits), ("image_dense", image_hits)]  # hybrid


def _fuse_hits(
    named_hit_lists: list[tuple[str, list[SearchHit]]],
    *,
    limit: int,
    k: int = 60,
    weights: dict[str, float] | None = None,
    max_units_per_page: int = 0,
) -> list[FusedHit]:
    weights = weights or {}
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    sources: dict[str, set[str]] = {}

    for source, hit_list in named_hit_lists:
        weight = weights.get(source, 1.0)
        for rank, hit in enumerate(hit_list, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            hits.setdefault(hit.chunk_id, hit)
            sources.setdefault(hit.chunk_id, set()).add(source)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    out: list[FusedHit] = []
    per_page: dict[int, int] = {}
    for uid, score in ordered:
        hit = hits[uid]
        if max_units_per_page and per_page.get(hit.page, 0) >= max_units_per_page:
            continue
        per_page[hit.page] = per_page.get(hit.page, 0) + 1
        out.append(FusedHit(hit=hit, score=score, sources=tuple(sorted(sources[uid]))))
        if len(out) >= limit:
            break
    return out


def _format_hit(item: FusedHit, query: str) -> dict:
    hit = item.hit
    return {
        "chunk_id": hit.chunk_id,
        "unit_id": hit.chunk_id,
        "unit_type": hit.unit_type,
        "page": hit.page,
        "citation": _citation(hit),
        "retrieval_score": round(item.score, 6),
        "sources": list(item.sources),
        "excerpt": _excerpt_for(hit, query),
    }


def _citation(hit: SearchHit) -> str:
    if hit.unit_type == "image":
        return f"CCI Black Book page {hit.page} (scanned figure/page image)"
    return f"CCI Black Book page {hit.page}, chunk {hit.chunk_id}"


def _excerpt_for(hit: SearchHit, query: str) -> str:
    if hit.unit_type == "image" and len((hit.text or "").strip()) < 40:
        return "[scanned figure/page image — little/no OCR text; call blackbook_read_citation]"
    return _excerpt(hit.text, query)


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
        "excerpts are bounded; call blackbook_read_citation for one full bounded unit",
    ]
    if not results:
        notes.append("no matching Black Book units were found")
    if provider_status.get("backend") == "stub":
        notes.append("dense ranking uses the offline stub provider (not Voyage)")
    return notes


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
