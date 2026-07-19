from __future__ import annotations

import hashlib
import logging
import os
import statistics
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import time

import numpy as np

from .chunking import Chunk, PageText, chunk_pages, group_chunks_into_documents
from .embeddings import DenseEmbeddingProvider, VoyageUnavailable, build_dense_provider
from .pagefilter import PageDecision, classify_page, summarize
from .render import ImageUnit, RenderedPage, page_image_tokens, render_pages
from .settings import (
    SCHEMA_VERSION,
    Settings,
    image_fingerprint,
    load_settings,
    settings_fingerprint,
    text_fingerprint,
    voyage_configured,
)
from .sources import SourceMeta, discover_sources, parse_unit_id
from .store import BlackBookIndex, PageRecord, PreparedSource, SearchHit

log = logging.getLogger("cci_blackbook")


class IndexUnavailable(RuntimeError):
    """Source directory missing/empty or the index is not ready."""


class IngestFailed(RuntimeError):
    """A build could not complete (Voyage down, retention not confirmed, broken extraction).
    Raised loudly so a partial/degraded index is never silently produced."""


@dataclass(frozen=True)
class FusedHit:
    hit: SearchHit
    score: float
    sources: tuple[str, ...]


@dataclass
class _SourceIngest:
    chunks: list[Chunk]
    chunk_vectors: list[np.ndarray]
    page_records: list[PageRecord]
    page_vectors: dict[tuple[str, int], np.ndarray]
    documents: int = 0


@dataclass(frozen=True)
class _SourceIdentity:
    source: SourceMeta
    content_sha256: str
    text_fingerprint: str
    image_fingerprint: str


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
        d = self.settings.source_dir
        specs = discover_sources(d)
        index_status = self.index.status()
        provider = self._provider or build_dense_provider(self.settings)
        return {
            "service": "cci-blackbook-mcp",
            "source_dir": {
                "path": str(d),
                "exists": d.exists(),
                "is_dir": d.is_dir(),
                "pdf_count": len(specs),
                "discovered": [
                    {"source_id": s.id, "title": s.title, "path": str(s.path)} for s in specs
                ],
            },
            "sources": index_status.get("sources", []),  # what is actually indexed, with counts
            "index": index_status,
            "embedding": provider.status(),
            "voyage_configured": voyage_configured(),
            "paths": {
                "source_dir": str(d),
                "index_dir": str(self.settings.index_dir),
                "cache_dir": str(self.settings.cache_dir),
            },
        }

    # ------------------------------------------------------------------ ingest

    def ensure_index(self, *, force: bool = False) -> dict:
        """Build or refresh the index. Called ONLY by the ingest CLI — never on the
        query path (which must never trigger a paid rebuild)."""
        with self._lock:
            d = self.settings.source_dir
            if not d.exists():
                raise IndexUnavailable(f"source directory missing: {d}")
            if not d.is_dir():
                raise IndexUnavailable(f"source path is not a directory: {d}")
            specs = discover_sources(d)
            if not specs:
                raise IndexUnavailable(f"no source PDFs found in {d}")

            version = self.index.schema_version()
            if version is not None and version != SCHEMA_VERSION:
                if not force:
                    raise IngestFailed(_legacy_rebuild_message(version))
                return self._force_rebuild_legacy(specs, version)

            try:
                identities = [_source_identity(spec, self.settings) for spec in specs]
            except OSError as exc:
                raise IngestFailed(
                    f"failed to hash source PDF before ingest: {type(exc).__name__}: {exc}"
                ) from exc

            manifests = self.index.source_manifests() if version == SCHEMA_VERSION else []
            changes = _classify_sources(identities, manifests, force=force)
            affected = set(changes["added"]) | set(changes["modified"])
            if not affected and not changes["removed"]:
                return self._ingest_result(
                    changes=changes,
                    prepared=[],
                    rebuilt=False,
                    forced=force,
                )

            provider: DenseEmbeddingProvider | None = None
            prepared: list[PreparedSource] = []
            total_documents = 0
            if affected:
                self._guard_retention()
                provider = self._provider or build_dense_provider(self.settings)
                prepared, total_documents = self._prepare_sources(
                    provider, identities, affected
                )

            retained_artifacts = sum(
                manifest["chunk_count"] + manifest["image_unit_count"]
                for manifest in manifests
                if manifest["source_id"] in changes["unchanged"]
            )
            prepared_artifacts = sum(
                len(source.chunks) + len(source.page_vectors) for source in prepared
            )
            if retained_artifacts + prepared_artifacts == 0:
                raise IngestFailed("corpus produced no indexable text chunks or page images")

            metadata = self._metadata(total_documents)
            try:
                self.index.replace_sources(
                    remove_source_ids=set(changes["removed"]) | set(changes["modified"]),
                    prepared_sources=prepared,
                    discovered_sources=specs,
                    metadata=metadata,
                    expected_text_dim=self.settings.voyage_output_dim,
                    expected_image_dim=self.settings.voyage_output_dim,
                    initialize=version is None,
                )
            except Exception as exc:
                raise IngestFailed(
                    f"atomic index update failed: {type(exc).__name__}: {exc}"
                ) from exc
            if provider is not None:
                self._provider = provider
            return self._ingest_result(
                changes=changes,
                prepared=prepared,
                rebuilt=True,
                forced=force,
            )

    def _guard_retention(self) -> None:
        if self.settings.embedding_backend == "voyage" and not self.settings.voyage_retention_confirmed:
            raise IngestFailed(
                "refusing to send the source PDFs to Voyage: confirm the zero-retention "
                "opt-out (or accept retention), then set CCI_VOYAGE_RETENTION_CONFIRMED=true"
            )

    def _prepare_sources(
        self,
        provider: DenseEmbeddingProvider,
        identities: list[_SourceIdentity],
        affected: set[str],
    ) -> tuple[list[PreparedSource], int]:
        prepared: list[PreparedSource] = []
        total_docs = 0
        indexed_at = int(time())
        for identity in identities:
            spec = identity.source
            if spec.id not in affected:
                continue
            try:
                one = self._ingest_one_source(provider, spec)
            except (IngestFailed, VoyageUnavailable) as exc:
                raise IngestFailed(f"source {spec.id!r}: {exc}") from exc
            except Exception as exc:  # e.g. fitz.open on a corrupt PDF
                raise IngestFailed(f"source {spec.id!r}: {type(exc).__name__}: {exc}") from exc
            prepared.append(PreparedSource(
                source=spec,
                content_sha256=identity.content_sha256,
                text_fingerprint=identity.text_fingerprint,
                image_fingerprint=identity.image_fingerprint,
                indexed_at=indexed_at,
                chunks=one.chunks,
                chunk_vectors=one.chunk_vectors,
                page_records=one.page_records,
                page_vectors=one.page_vectors,
            ))
            total_docs += one.documents
        return prepared, total_docs

    def _metadata(self, documents_embedded: int) -> dict:
        if self.settings.embedding_backend == "stub":
            text_model = image_model = "stub"
            configured = True
        else:
            text_model = self.settings.voyage_text_model
            image_model = self.settings.voyage_image_model
            configured = voyage_configured()
        embedding = {
            "backend": self.settings.embedding_backend,
            "text_model": text_model,
            "image_model": image_model,
            "dim": self.settings.voyage_output_dim,
            "configured": configured,
        }
        return {
            "fingerprint": settings_fingerprint(self.settings),
            "text_embedding": {
                **embedding,
                "space": "text",
                "observed_dim": self.settings.voyage_output_dim,
            },
            "image_embedding": {
                **embedding,
                "space": "image",
                "observed_dim": self.settings.voyage_output_dim,
            },
            "chunking": {
                "chunk_chars": self.settings.chunk_chars,
                "chunk_overlap_chars": self.settings.chunk_overlap_chars,
            },
            "grouping": {
                "documents_embedded": documents_embedded,
                "doc_token_budget": self.settings.doc_token_budget,
                "chars_per_token": self.settings.chars_per_token,
                "max_chunk_tokens": self.settings.max_chunk_tokens,
            },
            "built_at": int(time()),
        }

    def _ingest_result(
        self,
        *,
        changes: dict[str, list[str]],
        prepared: list[PreparedSource],
        rebuilt: bool,
        forced: bool,
    ) -> dict:
        status = self.index.status()
        return {
            "rebuilt": rebuilt,
            "forced": forced,
            "added": changes["added"],
            "modified": changes["modified"],
            "removed": changes["removed"],
            "unchanged": changes["unchanged"],
            "artifact_spaces_rebuilt": {
                source_id: ["text", "image"]
                for source_id in sorted(changes["added"] + changes["modified"])
            },
            "sources_embedded": len(prepared),
            "chunks_embedded": sum(len(source.chunks) for source in prepared),
            "pages_embedded": sum(len(source.page_vectors) for source in prepared),
            "source_count": status.get("source_count", 0),
            "chunk_count": status.get("chunk_count", 0),
            "image_unit_count": status.get("image_unit_count", 0),
            "pages_dropped": int(status["metadata"]["filter"]["dropped"]),
            "sources": status.get("sources", []),
            "status": status,
        }

    def _force_rebuild_legacy(self, specs: list[SourceMeta], version: int) -> dict:
        """Build and validate v4 separately before atomically replacing a legacy index."""
        self._guard_retention()
        provider = self._provider or build_dense_provider(self.settings)
        try:
            identities = [_source_identity(spec, self.settings) for spec in specs]
        except OSError as exc:
            raise IngestFailed(
                f"failed to hash source PDF before forced rebuild: {type(exc).__name__}: {exc}"
            ) from exc
        affected = {spec.id for spec in specs}
        prepared, total_documents = self._prepare_sources(
            provider, identities, affected
        )
        if not any(source.chunks or source.page_vectors for source in prepared):
            raise IngestFailed("corpus produced no indexable text chunks or page images")

        self.settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{self.settings.sqlite_path.name}.v4-",
            suffix=".tmp",
            dir=self.settings.sqlite_path.parent,
            delete=False,
        )
        temporary_path = Path(handle.name)
        handle.close()
        temporary_index = BlackBookIndex(temporary_path, journal_mode="DELETE")
        expected_counts = {
            "sources": len(prepared),
            "chunks": sum(len(source.chunks) for source in prepared),
            "pages": sum(len(source.page_records) for source in prepared),
            "image_units": sum(len(source.page_vectors) for source in prepared),
        }
        try:
            temporary_index.replace_sources(
                remove_source_ids=set(),
                prepared_sources=prepared,
                discovered_sources=specs,
                metadata=self._metadata(total_documents),
                expected_text_dim=self.settings.voyage_output_dim,
                expected_image_dim=self.settings.voyage_output_dim,
                initialize=True,
            )
            temporary_index.validate_database(expected_counts)
            os.replace(temporary_path, self.settings.sqlite_path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise IngestFailed(
                f"forced schema-v4 temporary build failed; legacy schema v{version} was "
                f"left unchanged: {type(exc).__name__}: {exc}"
            ) from exc

        self._provider = provider
        changes = {
            "added": [],
            "modified": sorted(affected),
            "removed": [],
            "unchanged": [],
        }
        log.info("replaced legacy schema v%s with validated schema v%s", version, SCHEMA_VERSION)
        return self._ingest_result(
            changes=changes,
            prepared=prepared,
            rebuilt=True,
            forced=True,
        )

    def _ingest_one_source(self, provider: DenseEmbeddingProvider, spec: SourceMeta) -> _SourceIngest:
        s = self.settings
        sid = spec.id
        fk = s.force_keep_pages.for_source(sid)
        fd = s.force_drop_pages.for_source(sid)

        page_texts: list[PageText] = []
        page_records: list[PageRecord] = []
        decisions: list[PageDecision] = []
        page_vectors: dict[tuple[str, int], np.ndarray] = {}
        batch: list[ImageUnit] = []
        batch_tokens = 0

        def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            for unit, vec in zip(batch, provider.embed_image_units(batch), strict=True):
                page_vectors[(sid, unit.page)] = vec  # (source_id, page) key from enclosing scope
            batch, batch_tokens = [], 0  # page images released here → one batch resident

        for rp in self._page_source(spec.path):
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
            page_records.append(_page_record(sid, rp, decision))
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

        self._check_extraction_tripwire(decisions, sid)
        log.info("blackbook page filter [%s]: %s", sid, summarize(decisions))

        chunks = chunk_pages(sid, page_texts, chunk_chars=s.chunk_chars, overlap_chars=s.chunk_overlap_chars)
        if chunks:  # per-source grouping: a book's chunks only see their own siblings
            groups = group_chunks_into_documents(
                chunks, token_budget=s.doc_token_budget,
                chars_per_token=s.chars_per_token, max_chunk_tokens=s.max_chunk_tokens,
            )
            documents = [[chunks[i].text for i in g] for g in groups]
            chunk_vectors = _scatter(groups, provider.embed_text_documents(documents), n=len(chunks))
            ndocs = len(groups)
        else:
            chunk_vectors, ndocs = [], 0

        return _SourceIngest(chunks, chunk_vectors, page_records, page_vectors, ndocs)

    def _check_extraction_tripwire(self, decisions: list[PageDecision], source_id: str) -> None:
        """Opt-in guard against silently broken text extraction, measured over TEXT-BEARING
        pages only (a source can legitimately have many near-empty figure pages). Threshold
        is per-source so a value tuned for a scanned book doesn't abort a sparse native one."""
        threshold = self.settings.min_expected_median_chars.for_source(source_id)
        if threshold <= 0 or not decisions:
            return
        text_pages = [d.char_count for d in decisions if d.char_count > 0]
        if not text_pages:
            raise IngestFailed(
                f"[{source_id}] text extraction produced no text on any page; the text layer "
                "may be unreadable (image-only source?) — aborting before a broken index"
            )
        median_chars = statistics.median(text_pages)
        if median_chars < threshold:
            raise IngestFailed(
                f"[{source_id}] text extraction looks sparse (median {median_chars:.0f} chars "
                f"over text-bearing pages < {threshold}); text layer may be unreadable — aborting"
            )

    # ---------------------------------------------------------------- retrieve

    def search(
        self, query: str, *, limit: int = 10, mode: str = "hybrid",
        sources: list[str] | str | None = None,
    ) -> dict:
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

        known = {r["source_id"] for r in status.get("sources", [])}
        resolved, source_notes = _resolve_sources(sources, known)
        if resolved is not None and not resolved:
            return {
                "query": query, "mode": mode, "results": [], "abstain": True, "source_filter": sources,
                "confidence_notes": [f"no requested source is indexed; known: {sorted(known)}", *source_notes],
            }

        provider = self._provider or build_dense_provider(self.settings)
        n = len(resolved) if resolved else status.get("source_count", 1)
        fetch = max(limit * 4, 20) * max(1, min(n, 5))  # widen the window with #books (cross-book recall)
        notes_extra: list[str] = []

        fts_hits = (
            self.index.search_fts(query, limit=fetch, source_ids=resolved)
            if mode in {"fts", "hybrid"} else []
        )

        text_hits: list[SearchHit] = []
        if mode in {"text", "vector", "hybrid"}:
            try:
                qvec = provider.embed_text_query(query)
                raw = self.index.search_vector(qvec, limit=fetch, source_ids=resolved)
                text_hits = [h for h in raw if h.score >= self.settings.min_vector_score]
                notes_extra.append(f"text-dense: {len(raw)} fetched, {len(text_hits)} >= gate")
            except Exception as exc:
                notes_extra.append(f"text-dense unavailable: {type(exc).__name__}")

        image_hits: list[SearchHit] = []
        if mode in {"image", "vector", "hybrid"}:
            try:
                qvec = provider.embed_image_query(query)
                raw = self.index.search_page_vector(qvec, limit=fetch, source_ids=resolved)
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
        notes = _confidence_notes(fused, mode, provider.status()) + notes_extra + source_notes
        return {
            "query": query, "mode": mode,
            "source_filter": (sorted(resolved) if resolved is not None else "all"),
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
        sources: list[str] | str | None = None,
    ) -> dict:
        parts = [question]
        if crop_context:
            parts.append(f"crop context: {crop_context}")
        if facility_context:
            parts.append(f"facility context: {facility_context}")
        search_query = "\n".join(parts)
        search_result = self.search(
            search_query, limit=_clamp(max_citations, 1, 10), mode="hybrid", sources=sources
        )
        return {
            "question": question,
            "crop_context": crop_context,
            "facility_context": facility_context,
            "abstain": search_result["abstain"],
            "answer_instruction": (
                "Compose the answer from these cited excerpts only. Each citation names its "
                "source (source_title / source_id). Some citations are page images "
                "(unit_type='image') whose extracted text may be sparse — call "
                "blackbook_read_citation to inspect one. If the excerpts do not answer the "
                "question, say the evidence is insufficient."
            ),
            "evidence": search_result["results"],
            "confidence_notes": search_result["confidence_notes"],
        }

    def read_citation(self, chunk_id: str) -> dict:
        status = self.index.status()  # read directly; never ensure_index
        if not status.get("ready"):
            return {"chunk_id": chunk_id, "found": False, "error": "index not built; run cci-blackbook-ingest"}

        uid = self._canonicalize_unit_id(chunk_id, status)
        parsed = parse_unit_id(uid)
        if parsed is None:
            return {"chunk_id": chunk_id, "found": False, "error": "unrecognized unit id"}
        source_id, page, kind = parsed

        if kind == "image":
            hit = self.index.read_page(source_id, page)
            if hit is None:
                return {"chunk_id": chunk_id, "found": False}
            result = {
                "chunk_id": uid, "unit_id": uid, "found": True, "page": hit.page,
                "source_id": hit.source_id, "source_title": hit.source_title, "unit_type": "image",
                "citation": _citation(hit), "text": _bounded_text(hit.text, 2500),
                "bounded": len(hit.text) > 2500,
                "note": "match came from the page image; text shown is the page's extracted text (may be sparse)",
            }
            thumb = self._maybe_thumbnail(source_id, hit.page)
            if thumb:
                result["thumbnail_png_base64"] = thumb
            return result

        hit = self.index.read_chunk(uid)
        if hit is None:
            return {"chunk_id": chunk_id, "found": False}
        return {
            "chunk_id": hit.chunk_id, "unit_id": hit.chunk_id, "found": True, "page": hit.page,
            "source_id": hit.source_id, "source_title": hit.source_title, "unit_type": "text",
            "citation": _citation(hit), "text": _bounded_text(hit.text, 2500),
            "bounded": len(hit.text) > 2500,
        }

    def _canonicalize_unit_id(self, uid: str, status: dict) -> str:
        """Back-compat for cached bare ids (p0042-img). Only resolvable — and only ever
        guessed — when exactly one source is indexed; a multi-source corpus always uses
        namespaced ids returned by search."""
        if ":" in uid:
            return uid
        src = status.get("sources", [])
        return f"{src[0]['source_id']}:{uid}" if len(src) == 1 else uid

    def _maybe_thumbnail(self, source_id: str, page: int) -> str | None:
        if not self.settings.citation_thumbnails:
            return None
        row = self.index.get_source(source_id)
        if not row or not Path(row["path"]).exists():
            return None
        try:
            import base64
            import io

            import fitz
            from PIL import Image

            doc = fitz.open(row["path"])
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


def _page_record(source_id: str, rp: RenderedPage, decision: PageDecision) -> PageRecord:
    return PageRecord(
        source_id=source_id,
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


def _resolve_sources(
    sources: list[str] | str | None, known: set[str]
) -> tuple[list[str] | None, list[str]]:
    """Return (resolved_ids, notes). None = all sources (no filter). [] = a filter was
    requested but no requested id is indexed (caller abstains)."""
    if sources is None:
        return None, []
    req = [sources] if isinstance(sources, str) else list(sources)
    req = [x.strip().lower() for x in req if isinstance(x, str) and x.strip()]
    if not req:
        return None, []
    valid = [r for r in req if r in known]
    unknown = [r for r in req if r not in known]
    notes = [f"ignored unknown sources: {sorted(set(unknown))}"] if unknown else []
    if valid:
        notes.insert(0, f"scoped to sources: {sorted(set(valid))}")
    return valid, notes


def _source_identity(source: SourceMeta, settings: Settings) -> _SourceIdentity:
    digest = hashlib.sha256()
    with source.path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return _SourceIdentity(
        source=source,
        content_sha256=digest.hexdigest(),
        text_fingerprint=text_fingerprint(settings),
        image_fingerprint=image_fingerprint(settings, source.id),
    )


def _classify_sources(
    identities: list[_SourceIdentity],
    manifests: list[dict],
    *,
    force: bool,
) -> dict[str, list[str]]:
    discovered = {identity.source.id: identity for identity in identities}
    indexed = {manifest["source_id"]: manifest for manifest in manifests}
    removed = sorted(set(indexed) - set(discovered))
    if force:
        return {
            "added": [],
            "modified": sorted(discovered),
            "removed": removed,
            "unchanged": [],
        }

    added = sorted(set(discovered) - set(indexed))
    modified: list[str] = []
    unchanged: list[str] = []
    for source_id in sorted(set(discovered) & set(indexed)):
        identity = discovered[source_id]
        manifest = indexed[source_id]
        current = (
            identity.content_sha256,
            identity.text_fingerprint,
            identity.image_fingerprint,
        )
        stored = (
            manifest["content_sha256"],
            manifest["text_fingerprint"],
            manifest["image_fingerprint"],
        )
        (unchanged if current == stored else modified).append(source_id)
    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "unchanged": unchanged,
    }


def _legacy_rebuild_message(version: int) -> str:
    return (
        f"index schema {version} is unavailable under schema v{SCHEMA_VERSION}. Stop the MCP "
        "first, then run cci-blackbook-ingest --force once. This regenerates all embeddings "
        "and may consume Voyage allowance or incur charges"
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
    per_page: dict[tuple[str, int], int] = {}  # cap is per (source, page), not per bare page number
    for uid, score in ordered:
        hit = hits[uid]
        key = (hit.source_id, hit.page)
        if max_units_per_page and per_page.get(key, 0) >= max_units_per_page:
            continue
        per_page[key] = per_page.get(key, 0) + 1
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
        "source_id": hit.source_id,
        "source_title": hit.source_title,
        "page": hit.page,
        "citation": _citation(hit),
        "retrieval_score": round(item.score, 6),
        "sources": list(item.sources),  # RANKER origins (fts/text_dense/image_dense)
        "excerpt": _excerpt_for(hit, query),
    }


def _citation(hit: SearchHit) -> str:
    title = hit.source_title or hit.source_id
    if hit.unit_type == "image":
        return f"{title}, p.{hit.page} (page image)"
    return f"{title}, p.{hit.page}"


def _excerpt_for(hit: SearchHit, query: str) -> str:
    if hit.unit_type == "image" and len((hit.text or "").strip()) < 40:
        return "[page image — little/no extracted text; call blackbook_read_citation]"
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
        notes.append("no matching units were found")
    if provider_status.get("backend") == "stub":
        notes.append("dense ranking uses the offline stub provider (not Voyage)")
    return notes


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))
