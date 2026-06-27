from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    page: int
    chunk_index: int
    text: str
    score: float
    source: str


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

    def rebuild(self, chunks: list[Chunk], embeddings: list[np.ndarray], metadata: dict) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        with self.connect() as conn:
            self._create_schema(conn)
            conn.execute("DELETE FROM meta")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM embeddings")

            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, page, chunk_index, text, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.page,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.char_start,
                        chunk.char_end,
                    )
                    for chunk in chunks
                ],
            )
            conn.executemany(
                "INSERT INTO chunks_fts(chunk_id, page, text) VALUES (?, ?, ?)",
                [(chunk.chunk_id, chunk.page, chunk.text) for chunk in chunks],
            )
            conn.executemany(
                "INSERT INTO embeddings(chunk_id, dim, vector) VALUES (?, ?, ?)",
                [
                    (chunk.chunk_id, int(vector.shape[0]), _serialize_vector(vector))
                    for chunk, vector in zip(chunks, embeddings, strict=True)
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
            metadata = {
                row["key"]: json.loads(row["value"])
                for row in conn.execute("SELECT key, value FROM meta").fetchall()
            }
        return {
            "ready": chunk_count > 0,
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
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
                    source="vector",
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


def merge_unique_hits(hit_lists: Iterable[list[SearchHit]]) -> dict[str, SearchHit]:
    merged: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for hit in hits:
            if hit.chunk_id not in merged:
                merged[hit.chunk_id] = hit
    return merged
