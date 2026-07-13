from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str          # unit id: "p0042-c001" (text) or "p0042-img" (image)
    page: int
    chunk_index: int       # real index for text units; 0 for image units
    text: str
    score: float
    source: str            # "fts" | "text_dense" | "image_dense" | "citation"
    unit_type: str         # "text" | "image"


@dataclass(frozen=True)
class PageRecord:
    page: int
    ocr_text: str
    char_count: int
    ink_coverage: float
    color_fraction: float
    width: int
    height: int
    kept: bool
    reason: str


class BlackBookIndex:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path

    def connect(self) -> sqlite3.Connection:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            self._create_schema(conn)

    def rebuild(
        self,
        *,
        chunks: list[Chunk],
        chunk_vectors: list[np.ndarray],
        page_records: list[PageRecord],
        page_vectors: dict[int, np.ndarray],
        metadata: dict,
    ) -> None:
        """Atomic swap: everything is validated, then the whole index is replaced in
        one transaction. A caller that raises before this point leaves the prior
        index fully intact (fail-loud ingest)."""
        if len(chunks) != len(chunk_vectors):
            raise ValueError("chunks and chunk_vectors length mismatch")
        kept_pages = {pr.page for pr in page_records if pr.kept}
        stray = set(page_vectors) - kept_pages
        if stray:
            raise ValueError(f"page_vectors reference non-kept pages: {sorted(stray)}")

        with self.connect() as conn:
            self._create_schema(conn)
            for table in ("meta", "chunks", "chunks_fts", "embeddings", "page_embeddings", "pages"):
                conn.execute(f"DELETE FROM {table}")

            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, page, chunk_index, text, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (c.chunk_id, c.page, c.chunk_index, c.text, c.char_start, c.char_end)
                    for c in chunks
                ],
            )
            conn.executemany(
                "INSERT INTO chunks_fts(chunk_id, page, text) VALUES (?, ?, ?)",
                [(c.chunk_id, c.page, c.text) for c in chunks],
            )
            conn.executemany(
                "INSERT INTO embeddings(chunk_id, dim, vector) VALUES (?, ?, ?)",
                [
                    (c.chunk_id, int(v.shape[0]), _serialize_vector(v))
                    for c, v in zip(chunks, chunk_vectors, strict=True)
                ],
            )
            # pages must be inserted before page_embeddings (FK, PRAGMA foreign_keys=ON).
            conn.executemany(
                """
                INSERT INTO pages(page, ocr_text, char_count, ink_coverage, color_fraction,
                                  width, height, kept, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        pr.page, pr.ocr_text, pr.char_count, pr.ink_coverage, pr.color_fraction,
                        pr.width, pr.height, int(pr.kept), pr.reason,
                    )
                    for pr in page_records
                ],
            )
            conn.executemany(
                "INSERT INTO page_embeddings(page, dim, vector) VALUES (?, ?, ?)",
                [
                    (page, int(vec.shape[0]), _serialize_vector(vec))
                    for page, vec in sorted(page_vectors.items())
                ],
            )
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in metadata.items()],
            )

    def status(self) -> dict:
        if not self.sqlite_path.exists():
            return {"ready": False, "reason": "index database missing"}
        with self.connect() as conn:
            self._create_schema(conn)
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedding_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            page_count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            image_unit_count = conn.execute("SELECT COUNT(*) FROM page_embeddings").fetchone()[0]
            indexed_image_count = conn.execute(
                "SELECT COUNT(*) FROM pages WHERE kept = 1"
            ).fetchone()[0]
            metadata = {
                row["key"]: json.loads(row["value"])
                for row in conn.execute("SELECT key, value FROM meta").fetchall()
            }
        return {
            "ready": chunk_count > 0 or image_unit_count > 0,
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "page_count": page_count,
            "image_unit_count": image_unit_count,
            "indexed_image_count": indexed_image_count,
            "metadata": metadata,
            "sqlite_path": str(self.sqlite_path),
        }

    def read_chunk(self, chunk_id: str) -> SearchHit | None:
        with self.connect() as conn:
            self._create_schema(conn)
            row = conn.execute(
                """
                SELECT chunk_id, page, chunk_index, text
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return SearchHit(
            chunk_id=row["chunk_id"],
            page=int(row["page"]),
            chunk_index=int(row["chunk_index"]),
            text=row["text"],
            score=1.0,
            source="citation",
            unit_type="text",
        )

    def read_page(self, page: int) -> SearchHit | None:
        with self.connect() as conn:
            self._create_schema(conn)
            row = conn.execute(
                """
                SELECT p.page, p.ocr_text
                FROM pages p
                JOIN page_embeddings pe ON pe.page = p.page
                WHERE p.page = ?
                """,
                (page,),
            ).fetchone()
        if row is None:
            return None
        return SearchHit(
            chunk_id=f"p{int(row['page']):04d}-img",
            page=int(row["page"]),
            chunk_index=0,
            text=row["ocr_text"],
            score=1.0,
            source="citation",
            unit_type="image",
        )

    def search_fts(self, query: str, *, limit: int) -> list[SearchHit]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        with self.connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.page,
                    c.chunk_index,
                    c.text,
                    bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [
            SearchHit(
                chunk_id=row["chunk_id"],
                page=int(row["page"]),
                chunk_index=int(row["chunk_index"]),
                text=row["text"],
                score=float(-row["rank"]),
                source="fts",
                unit_type="text",
            )
            for row in rows
        ]

    def search_vector(self, query_vector: np.ndarray, *, limit: int) -> list[SearchHit]:
        query_vector = _normalize(query_vector)
        with self.connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.page, c.chunk_index, c.text, e.vector
                FROM embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                """
            ).fetchall()

        scored: list[SearchHit] = []
        for row in rows:
            vector = _deserialize_vector(row["vector"])
            scored.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    page=int(row["page"]),
                    chunk_index=int(row["chunk_index"]),
                    text=row["text"],
                    score=float(np.dot(query_vector, vector)),
                    source="text_dense",
                    unit_type="text",
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def search_page_vector(self, query_vector: np.ndarray, *, limit: int) -> list[SearchHit]:
        query_vector = _normalize(query_vector)
        with self.connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(
                """
                SELECT p.page, p.ocr_text, pe.vector
                FROM page_embeddings pe
                JOIN pages p ON p.page = pe.page
                """
            ).fetchall()

        scored: list[SearchHit] = []
        for row in rows:
            vector = _deserialize_vector(row["vector"])
            scored.append(
                SearchHit(
                    chunk_id=f"p{int(row['page']):04d}-img",
                    page=int(row["page"]),
                    chunk_index=0,
                    text=row["ocr_text"],
                    score=float(np.dot(query_vector, vector)),
                    source="image_dense",
                    unit_type="image",
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                page UNINDEXED,
                text,
                tokenize = 'porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pages (
                page INTEGER PRIMARY KEY,
                ocr_text TEXT NOT NULL DEFAULT '',
                char_count INTEGER NOT NULL,
                ink_coverage REAL NOT NULL,
                color_fraction REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                kept INTEGER NOT NULL,
                reason TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS page_embeddings (
                page INTEGER PRIMARY KEY REFERENCES pages(page) ON DELETE CASCADE,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            );
            """
        )


def _serialize_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return _normalize(np.frombuffer(blob, dtype=np.float32))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _fts_query(query: str) -> str:
    terms = [term for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query) if len(term) > 1]
    if not terms:
        return ""
    return " OR ".join(_quote_fts_term(term) for term in terms[:16])


def _quote_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'
