from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    source_pdf: Path
    index_dir: Path
    cache_dir: Path
    sqlite_path: Path
    embedding_backend: str
    embedding_model: str
    openvino_device: str
    render_device: Path
    chunk_chars: int
    chunk_overlap_chars: int
    min_vector_score: float
    embedding_batch_size: int
    host: str
    port: int
    log_level: str


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def load_settings() -> Settings:
    index_dir = Path(os.environ.get("CCI_INDEX_DIR", "/data/index"))
    cache_dir = Path(os.environ.get("CCI_CACHE_DIR", "/data/cache"))
    sqlite_path = Path(os.environ.get("CCI_SQLITE_PATH", index_dir / "blackbook.sqlite3"))

    return Settings(
        source_pdf=Path(os.environ.get("CCI_SOURCE_PDF", "/data/source/CCI Black Book.pdf")),
        index_dir=index_dir,
        cache_dir=cache_dir,
        sqlite_path=sqlite_path,
        embedding_backend=os.environ.get("CCI_EMBEDDING_BACKEND", "auto").lower(),
        embedding_model=os.environ.get("CCI_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        openvino_device=os.environ.get("CCI_OPENVINO_DEVICE", "GPU"),
        render_device=Path(os.environ.get("CCI_RENDER_DEVICE", "/dev/dri/renderD129")),
        chunk_chars=_int_from_env("CCI_CHUNK_CHARS", 1800),
        chunk_overlap_chars=_int_from_env("CCI_CHUNK_OVERLAP_CHARS", 250),
        min_vector_score=float(os.environ.get("CCI_MIN_VECTOR_SCORE", "0.20")),
        embedding_batch_size=_int_from_env("CCI_EMBEDDING_BATCH_SIZE", 8),
        host=os.environ.get("CCI_MCP_HOST", "0.0.0.0"),
        port=_int_from_env("CCI_MCP_PORT", 8000),
        log_level=os.environ.get("CCI_LOG_LEVEL", "info"),
    )
