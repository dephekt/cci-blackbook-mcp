from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Bumping this invalidates every existing index (it is part of the fingerprint),
# forcing a clean rebuild after a breaking schema/pipeline change.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Settings:
    source_pdf: Path
    index_dir: Path
    cache_dir: Path
    sqlite_path: Path
    embedding_backend: str
    # --- legacy FastEmbed/OpenVINO fields (still parsed for the installed but now
    # unused local provider + compose; removed in the deferred GPU-stack follow-up) ---
    embedding_model: str
    openvino_device: str
    render_device: Path
    embedding_batch_size: int
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
    force_keep_pages: frozenset[int]
    force_drop_pages: frozenset[int]
    # --- retrieval ---
    min_vector_score: float
    min_image_score: float
    rrf_k: int
    rrf_weight_fts: float
    rrf_weight_text: float
    rrf_weight_image: float
    max_units_per_page: int
    # --- ingest safety ---
    min_expected_median_chars: int
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


def _int_set_from_env(name: str, default: str = "") -> frozenset[int]:
    """Parse "1-4,17,200-205" (whitespace/comma separated ranges) into a set of ints."""
    raw = os.environ.get(name, default) or ""
    pages: set[int] = set()
    for token in raw.replace(",", " ").split():
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            pages.update(range(lo, hi + 1))
        else:
            pages.add(int(token))
    return frozenset(pages)


def load_settings() -> Settings:
    index_dir = Path(os.environ.get("CCI_INDEX_DIR", "/data/index"))
    cache_dir = Path(os.environ.get("CCI_CACHE_DIR", "/data/cache"))
    sqlite_path = Path(os.environ.get("CCI_SQLITE_PATH", index_dir / "blackbook.sqlite3"))

    return Settings(
        source_pdf=Path(os.environ.get("CCI_SOURCE_PDF", "/data/source/CCI Black Book.pdf")),
        index_dir=index_dir,
        cache_dir=cache_dir,
        sqlite_path=sqlite_path,
        embedding_backend=os.environ.get("CCI_EMBEDDING_BACKEND", "voyage").lower(),
        embedding_model=os.environ.get("CCI_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        openvino_device=os.environ.get("CCI_OPENVINO_DEVICE", "GPU"),
        render_device=Path(os.environ.get("CCI_RENDER_DEVICE", "/dev/dri/renderD129")),
        embedding_batch_size=_int_from_env("CCI_EMBEDDING_BATCH_SIZE", 8),
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
        force_keep_pages=_int_set_from_env("CCI_FORCE_KEEP_PAGES", ""),
        force_drop_pages=_int_set_from_env("CCI_FORCE_DROP_PAGES", ""),
        min_vector_score=_float_from_env("CCI_MIN_VECTOR_SCORE", 0.20),
        min_image_score=_float_from_env("CCI_MIN_IMAGE_SCORE", 0.0),
        rrf_k=_int_from_env("CCI_RRF_K", 60),
        rrf_weight_fts=_float_from_env("CCI_RRF_WEIGHT_FTS", 1.0),
        rrf_weight_text=_float_from_env("CCI_RRF_WEIGHT_TEXT", 1.0),
        rrf_weight_image=_float_from_env("CCI_RRF_WEIGHT_IMAGE", 2.0),
        max_units_per_page=_int_from_env("CCI_MAX_UNITS_PER_PAGE", 2),
        min_expected_median_chars=_int_from_env("CCI_MIN_EXPECTED_MEDIAN_CHARS", 200),
        citation_thumbnails=_bool_from_env("CCI_CITATION_THUMBNAILS", False),
        host=os.environ.get("CCI_MCP_HOST", "0.0.0.0"),
        port=_int_from_env("CCI_MCP_PORT", 8000),
        log_level=os.environ.get("CCI_LOG_LEVEL", "info"),
    )


def voyage_configured() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY"))


def settings_fingerprint(s: Settings) -> dict:
    """Identity of the pipeline that produced an index. Any change here invalidates
    a stored index via _index_current, forcing a clean rebuild. A legacy bge-384
    index has no fingerprint at all, and a `stub` index carries backend="stub",
    so both mismatch a voyage/1024 config and rebuild cleanly."""
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
        "force_keep": sorted(s.force_keep_pages),
        "force_drop": sorted(s.force_drop_pages),
    }
