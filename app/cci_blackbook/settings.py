from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Bumping this invalidates every existing index (it is part of the fingerprint) AND is
# written to the DB header (PRAGMA user_version), so a physically stale-shaped index is
# detected before any source-aware query runs. Forces a clean rebuild on a breaking change.
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ScopedPages:
    """A page-set config that can apply to all sources (default) plus per-source additions.
    Env grammar: whitespace tokens, each `[<source_id>:]<ranges>` (a bare token = default)."""

    default: frozenset[int]
    per_source: tuple[tuple[str, frozenset[int]], ...]  # sorted; source-specific additions

    def for_source(self, sid: str) -> frozenset[int]:
        return self.default | dict(self.per_source).get(sid, frozenset())  # UNION

    def fingerprint(self) -> list:
        return [sorted(self.default), [[k, sorted(v)] for k, v in self.per_source]]

    @classmethod
    def all(cls, pages) -> ScopedPages:
        return cls(frozenset(pages), ())


@dataclass(frozen=True)
class ScopedInt:
    """An int config with a default plus per-source overrides."""

    default: int
    per_source: tuple[tuple[str, int], ...]  # sorted

    def for_source(self, sid: str) -> int:
        return dict(self.per_source).get(sid, self.default)  # OVERRIDE

    def fingerprint(self) -> list:
        return [self.default, [list(x) for x in self.per_source]]

    @classmethod
    def all(cls, value: int) -> ScopedInt:
        return cls(value, ())


@dataclass(frozen=True)
class Settings:
    source_dir: Path
    index_dir: Path
    cache_dir: Path
    sqlite_path: Path
    embedding_backend: str
    # --- chunking ---
    chunk_chars: int
    chunk_overlap_chars: int
    # --- Voyage provider ---
    voyage_text_model: str
    voyage_image_model: str
    voyage_output_dim: int
    voyage_output_dtype: str
    voyage_timeout: float
    voyage_max_retries: int
    voyage_retention_confirmed: bool
    # --- context-4 document grouping ---
    doc_token_budget: int
    chars_per_token: float
    max_chunk_tokens: int
    # --- multimodal batching ---
    mm_token_budget: int
    mm_max_inputs: int
    mm_pixels_per_token: int
    # --- rendering + blank filter ---
    render_dpi: int
    render_max_pixels: int
    blank_min_chars: int
    blank_max_ink: float
    blank_max_color: float
    ink_luma_threshold: int
    color_sat_threshold: int
    blank_filter_disable: bool
    force_keep_pages: ScopedPages
    force_drop_pages: ScopedPages
    # --- retrieval ---
    min_vector_score: float
    min_image_score: float
    rrf_k: int
    rrf_weight_fts: float
    rrf_weight_text: float
    rrf_weight_image: float
    max_units_per_page: int
    # --- ingest safety ---
    min_expected_median_chars: ScopedInt
    # --- optional ---
    citation_thumbnails: bool
    # --- server ---
    host: str
    port: int
    log_level: str


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_scoped(raw: str) -> tuple[list[str], dict[str, list[str]]]:
    """Split whitespace tokens into (default_values, {source_id: values}). A token with a
    `<sid>:` prefix is scoped to that source; a bare token is a default applied to all."""
    default: list[str] = []
    per_source: dict[str, list[str]] = {}
    for token in (raw or "").split():
        if ":" in token:
            sid, val = token.split(":", 1)
            per_source.setdefault(sid, []).append(val)
        else:
            default.append(token)
    return default, per_source


def _expand_pages(values: list[str]) -> frozenset[int]:
    """Expand ["1-4,7", "17"] etc. (comma or whitespace separated ranges) into a page set."""
    pages: set[int] = set()
    for val in values:
        for token in val.replace(",", " ").split():
            if "-" in token:
                lo_s, hi_s = token.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    lo, hi = hi, lo
                pages.update(range(lo, hi + 1))
            else:
                pages.add(int(token))
    return frozenset(pages)


def _scoped_pages_from_env(name: str) -> ScopedPages:
    default_vals, per_source_vals = _parse_scoped(os.environ.get(name, "") or "")
    default = _expand_pages(default_vals)
    per_source = tuple(sorted((sid, _expand_pages(vals)) for sid, vals in per_source_vals.items()))
    return ScopedPages(default, per_source)


def _scoped_int_from_env(name: str) -> ScopedInt:
    default_vals, per_source_vals = _parse_scoped(os.environ.get(name, "") or "")
    default = int(default_vals[-1]) if default_vals else 0
    per_source = tuple(sorted((sid, int(vals[-1])) for sid, vals in per_source_vals.items()))
    return ScopedInt(default, per_source)


def load_settings() -> Settings:
    index_dir = Path(os.environ.get("CCI_INDEX_DIR", "/data/index"))
    cache_dir = Path(os.environ.get("CCI_CACHE_DIR", "/data/cache"))
    sqlite_path = Path(os.environ.get("CCI_SQLITE_PATH", index_dir / "blackbook.sqlite3"))

    return Settings(
        source_dir=Path(os.environ.get("CCI_SOURCE_DIR", "/data/source")),
        index_dir=index_dir,
        cache_dir=cache_dir,
        sqlite_path=sqlite_path,
        embedding_backend=os.environ.get("CCI_EMBEDDING_BACKEND", "voyage").lower(),
        chunk_chars=_int_from_env("CCI_CHUNK_CHARS", 1800),
        chunk_overlap_chars=_int_from_env("CCI_CHUNK_OVERLAP_CHARS", 250),
        voyage_text_model=os.environ.get("CCI_VOYAGE_TEXT_MODEL", "voyage-context-4"),
        voyage_image_model=os.environ.get("CCI_VOYAGE_IMAGE_MODEL", "voyage-multimodal-3.5"),
        voyage_output_dim=_int_from_env("CCI_VOYAGE_OUTPUT_DIM", 1024),
        voyage_output_dtype=os.environ.get("CCI_VOYAGE_OUTPUT_DTYPE", "float"),
        voyage_timeout=_float_from_env("CCI_VOYAGE_TIMEOUT", 60.0),
        voyage_max_retries=_int_from_env("CCI_VOYAGE_MAX_RETRIES", 4),
        voyage_retention_confirmed=_bool_from_env("CCI_VOYAGE_RETENTION_CONFIRMED", False),
        # voyage-context-4 rejects a single example (document) whose chunks sum to
        # > 32000 tokens (manual-chunking mode has no truncation); keep well under it.
        # Our token estimate over-counts vs the real tokenizer, so this is conservative.
        doc_token_budget=_int_from_env("CCI_DOC_TOKEN_BUDGET", 28000),
        chars_per_token=_float_from_env("CCI_CHARS_PER_TOKEN", 3.0),
        max_chunk_tokens=_int_from_env("CCI_MAX_CHUNK_TOKENS", 32000),
        mm_token_budget=_int_from_env("CCI_MM_TOKEN_BUDGET", 200000),
        mm_max_inputs=_int_from_env("CCI_MM_MAX_INPUTS", 400),
        mm_pixels_per_token=_int_from_env("CCI_MM_PIXELS_PER_TOKEN", 560),
        render_dpi=_int_from_env("CCI_RENDER_DPI", 100),
        render_max_pixels=_int_from_env("CCI_RENDER_MAX_PIXELS", 12000000),
        blank_min_chars=_int_from_env("CCI_BLANK_MIN_CHARS", 100),
        blank_max_ink=_float_from_env("CCI_BLANK_MAX_INK", 0.02),
        blank_max_color=_float_from_env("CCI_BLANK_MAX_COLOR", 0.005),
        ink_luma_threshold=_int_from_env("CCI_INK_LUMA_THRESHOLD", 240),
        color_sat_threshold=_int_from_env("CCI_COLOR_SAT_THRESHOLD", 30),
        blank_filter_disable=_bool_from_env("CCI_BLANK_FILTER_DISABLE", False),
        # Optional operator overrides. In a multi-source corpus a bare `1-4` applies to
        # every book; scope it to one with `<source_id>:1-4` (books share low page numbers).
        force_keep_pages=_scoped_pages_from_env("CCI_FORCE_KEEP_PAGES"),
        force_drop_pages=_scoped_pages_from_env("CCI_FORCE_DROP_PAGES"),
        min_vector_score=_float_from_env("CCI_MIN_VECTOR_SCORE", 0.20),
        min_image_score=_float_from_env("CCI_MIN_IMAGE_SCORE", 0.0),
        rrf_k=_int_from_env("CCI_RRF_K", 60),
        rrf_weight_fts=_float_from_env("CCI_RRF_WEIGHT_FTS", 1.0),
        rrf_weight_text=_float_from_env("CCI_RRF_WEIGHT_TEXT", 1.0),
        rrf_weight_image=_float_from_env("CCI_RRF_WEIGHT_IMAGE", 2.0),
        max_units_per_page=_int_from_env("CCI_MAX_UNITS_PER_PAGE", 2),
        # Opt-in text-extraction tripwire (median chars over text-bearing pages). Off by
        # default so the tool works on any PDF, incl. small/visual docs; raise it (e.g.
        # 200, or `<source_id>:200`) for a large scanned book to catch a broken text layer.
        min_expected_median_chars=_scoped_int_from_env("CCI_MIN_EXPECTED_MEDIAN_CHARS"),
        citation_thumbnails=_bool_from_env("CCI_CITATION_THUMBNAILS", False),
        host=os.environ.get("CCI_MCP_HOST", "0.0.0.0"),
        port=_int_from_env("CCI_MCP_PORT", 8000),
        log_level=os.environ.get("CCI_LOG_LEVEL", "info"),
    )


def voyage_configured() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


def settings_fingerprint(s: Settings) -> dict:
    """Identity of the pipeline that produced an index. Any change here invalidates a
    stored index via _index_current, forcing a clean rebuild. Combined with the corpus
    identity (the `sources` table) and the DB `user_version`, this is what makes an
    out-of-date index rebuild cleanly."""
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": s.embedding_backend,
        "text_model": s.voyage_text_model,
        "image_model": s.voyage_image_model,
        "output_dim": s.voyage_output_dim,
        "output_dtype": s.voyage_output_dtype,
        "chunk_chars": s.chunk_chars,
        "chunk_overlap_chars": s.chunk_overlap_chars,
        "doc_token_budget": s.doc_token_budget,
        "chars_per_token": s.chars_per_token,
        "render_dpi": s.render_dpi,
        "render_max_pixels": s.render_max_pixels,
        "blank": [
            s.blank_min_chars,
            s.blank_max_ink,
            s.blank_max_color,
            s.ink_luma_threshold,
            s.color_sat_threshold,
            s.blank_filter_disable,
        ],
        "force_keep": s.force_keep_pages.fingerprint(),
        "force_drop": s.force_drop_pages.fingerprint(),
    }
