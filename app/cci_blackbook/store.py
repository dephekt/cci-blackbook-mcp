from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .settings import SCHEMA_VERSION as _DB_SCHEMA
from .sources import SourceMeta, build_image_unit_id


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str          # namespaced: "aroya-guide-to-drying:p0012-c000" | "...:p0012-img"
    source_id: str
    source_title: str
    page: int
    chunk_index: int       # real index for text units; 0 for image units
    text: str
    score: float
    source: str            # RANKER origin: "fts" | "text_dense" | "image_dense" | "citation"
    unit_type: str         # "text" | "image"


@dataclass(frozen=True)
class PageRecord:
    source_id: str
    page: int
    ocr_text: str
    char_count: int
    ink_coverage: float
    color_fraction: float
    width: int
    height: int
    kept: bool
    reason: str


_CREATE_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY, title TEXT NOT NULL, path TEXT NOT NULL,
        size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
        page_count INTEGER NOT NULL, kept_page_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL, image_unit_count INTEGER NOT NULL,
        ordinal INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
        page INTEGER NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
        char_start INTEGER NOT NULL, char_end INTEGER NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_id UNINDEXED, page UNINDEXED, text, tokenize = 'porter unicode61')""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        dim INTEGER NOT NULL, vector BLOB NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS pages (
        source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
        page INTEGER NOT NULL, ocr_text TEXT NOT NULL DEFAULT '',
        char_count INTEGER NOT NULL, ink_coverage REAL NOT NULL, color_fraction REAL NOT NULL,
        width INTEGER NOT NULL, height INTEGER NOT NULL, kept INTEGER NOT NULL, reason TEXT NOT NULL,
        PRIMARY KEY (source_id, page))""",
    """CREATE TABLE IF NOT EXISTS page_embeddings (
        source_id TEXT NOT NULL, page INTEGER NOT NULL,
        dim INTEGER NOT NULL, vector BLOB NOT NULL,
        PRIMARY KEY (source_id, page),
        FOREIGN KEY (source_id, page) REFERENCES pages(source_id, page) ON DELETE CASCADE)""",
]
_DROP_ORDER = [  # child → parent, so FK-on drops never dangle
    "page_embeddings", "pages", "embeddings", "chunks_fts", "chunks", "sources", "meta",
]

_SOURCE_COLS = (
    "source_id, title, path, size, mtime_ns, page_count, kept_page_count, "
    "chunk_count, image_unit_count, ordinal"
)


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
        sources: list[SourceMeta],
        chunks: list[Chunk],
        chunk_vectors: list[np.ndarray],
        page_records: list[PageRecord],
        page_vectors: dict[tuple[str, int], np.ndarray],
        metadata: dict,
    ) -> None:
        """Atomic swap across the whole corpus. Everything is validated, then the index is
        dropped and rebuilt in ONE explicit transaction (DROP+CREATE converges an
        old-shaped index; per-statement execute + explicit BEGIN keeps it atomic — a
        failure rolls back to the prior index). Sets PRAGMA user_version so a physically
        stale index is detected on the next status()."""
        if len(chunks) != len(chunk_vectors):
            raise ValueError("chunks and chunk_vectors length mismatch")
        source_ids = {s.id for s in sources}
        kept = {(pr.source_id, pr.page) for pr in page_records if pr.kept}
        stray = set(page_vectors) - kept
        if stray:
            raise ValueError(f"page_vectors reference non-kept (source,page): {sorted(stray)}")
        unknown = ({c.source_id for c in chunks} | {pr.source_id for pr in page_records}) - source_ids
        if unknown:
            raise ValueError(f"chunks/pages reference unknown sources: {sorted(unknown)}")

        pc = Counter(pr.source_id for pr in page_records)
        kpc = Counter(pr.source_id for pr in page_records if pr.kept)
        cc = Counter(c.source_id for c in chunks)
        ic = Counter(sid for (sid, _p) in page_vectors)

        with self.connect() as conn:
            conn.execute("BEGIN")  # DDL does not open an implicit txn; required for atomicity
            self._drop_schema(conn)  # per-statement (NOT executescript, which implicit-commits)
            self._create_schema(conn)
            conn.executemany(
                f"INSERT INTO sources ({_SOURCE_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (s.id, s.title, str(s.path), s.size, s.mtime_ns,
                     pc[s.id], kpc[s.id], cc[s.id], ic[s.id], s.ordinal)
                    for s in sources
                ],
            )
            conn.executemany(
                "INSERT INTO chunks(chunk_id,source_id,page,chunk_index,text,char_start,char_end) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    (c.chunk_id, c.source_id, c.page, c.chunk_index, c.text, c.char_start, c.char_end)
                    for c in chunks
                ],
            )
            conn.executemany(
                "INSERT INTO chunks_fts(chunk_id,page,text) VALUES (?,?,?)",
                [(c.chunk_id, c.page, c.text) for c in chunks],
            )
            conn.executemany(
                "INSERT INTO embeddings(chunk_id,dim,vector) VALUES (?,?,?)",
                [
                    (c.chunk_id, int(v.shape[0]), _serialize_vector(v))
                    for c, v in zip(chunks, chunk_vectors, strict=True)
                ],
            )
            conn.executemany(
                "INSERT INTO pages(source_id,page,ocr_text,char_count,ink_coverage,color_fraction,"
                "width,height,kept,reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (pr.source_id, pr.page, pr.ocr_text, pr.char_count, pr.ink_coverage,
                     pr.color_fraction, pr.width, pr.height, int(pr.kept), pr.reason)
                    for pr in page_records
                ],
            )
            conn.executemany(
                "INSERT INTO page_embeddings(source_id,page,dim,vector) VALUES (?,?,?,?)",
                [
                    (sid, page, int(vec.shape[0]), _serialize_vector(vec))
                    for (sid, page), vec in sorted(page_vectors.items())
                ],
            )
            conn.executemany(
                "INSERT INTO meta(key,value) VALUES (?,?)",
                [(key, json.dumps(value)) for key, value in metadata.items()],
            )
            conn.execute(f"PRAGMA user_version = {int(_DB_SCHEMA)}")  # transactional; rolls back on failure

    def status(self) -> dict:
        if not self.sqlite_path.exists():
            return {"ready": False, "reason": "index database missing"}
        with self.connect() as conn:
            # Guard FIRST on the header int (no table dependency) — never run _create_schema
            # or source-aware SQL against an old-shaped index (e.g. its CREATE INDEX on
            # chunks(source_id) would fail on a pre-v3 chunks table). Only rebuild() sets
            # user_version=SCHEMA_VERSION, and it creates the current-shape tables, so a
            # match guarantees the queries below are safe.
            if conn.execute("PRAGMA user_version").fetchone()[0] != int(_DB_SCHEMA):
                return {
                    "ready": False,
                    "reason": "index not built or schema out of date; run cci-blackbook-ingest",
                    "sqlite_path": str(self.sqlite_path),
                }
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedding_count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            page_count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            image_unit_count = conn.execute("SELECT COUNT(*) FROM page_embeddings").fetchone()[0]
            indexed_image_count = conn.execute("SELECT COUNT(*) FROM pages WHERE kept = 1").fetchone()[0]
            sources = [
                dict(r)
                for r in conn.execute(f"SELECT {_SOURCE_COLS} FROM sources ORDER BY ordinal")
            ]
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
            "source_count": len(sources),
            "sources": sources,
            "metadata": metadata,
            "sqlite_path": str(self.sqlite_path),
        }

    def list_sources(self) -> list[dict]:
        with self.connect() as conn:
            self._create_schema(conn)
            return [
                dict(r)
                for r in conn.execute(f"SELECT {_SOURCE_COLS} FROM sources ORDER BY ordinal")
            ]

    def get_source(self, source_id: str) -> dict | None:
        with self.connect() as conn:
            self._create_schema(conn)
            row = conn.execute(
                f"SELECT {_SOURCE_COLS} FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        return dict(row) if row else None

    def read_chunk(self, chunk_id: str) -> SearchHit | None:
        # INVARIANT (shared by all read/search methods below): callers must gate on
        # status()["ready"] first. These run source-aware SQL and call _create_schema,
        # which would raise on a pre-v3 (old-shaped) index; status()'s user_version guard
        # already returns not-ready for any such index, so we never reach here on one.
        with self.connect() as conn:
            self._create_schema(conn)
            row = conn.execute(
                """
                SELECT c.chunk_id, c.source_id, s.title AS source_title, c.page, c.chunk_index, c.text
                FROM chunks c
                JOIN sources s ON s.source_id = c.source_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return SearchHit(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            source_title=row["source_title"],
            page=int(row["page"]),
            chunk_index=int(row["chunk_index"]),
            text=row["text"],
            score=1.0,
            source="citation",
            unit_type="text",
        )

    def read_page(self, source_id: str, page: int) -> SearchHit | None:
        with self.connect() as conn:
            self._create_schema(conn)
            row = conn.execute(
                """
                SELECT p.source_id, s.title AS source_title, p.page, p.ocr_text
                FROM pages p
                JOIN page_embeddings pe ON pe.source_id = p.source_id AND pe.page = p.page
                JOIN sources s ON s.source_id = p.source_id
                WHERE p.source_id = ? AND p.page = ?
                """,
                (source_id, page),
            ).fetchone()
        if row is None:
            return None
        return SearchHit(
            chunk_id=build_image_unit_id(source_id, int(row["page"])),
            source_id=row["source_id"],
            source_title=row["source_title"],
            page=int(row["page"]),
            chunk_index=0,
            text=row["ocr_text"],
            score=1.0,
            source="citation",
            unit_type="image",
        )

    def search_fts(
        self, query: str, *, limit: int, source_ids: list[str] | None = None
    ) -> list[SearchHit]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        frag, fp = _source_in_clause("c.source_id", source_ids)
        where = "WHERE chunks_fts MATCH ?" + (f" AND {frag}" if frag else "")
        sql = f"""
            SELECT c.chunk_id, c.source_id, s.title AS source_title,
                   c.page, c.chunk_index, c.text, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN sources s ON s.source_id = c.source_id
            {where}
            ORDER BY rank
            LIMIT ?
        """
        with self.connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(sql, (fts_query, *fp, limit)).fetchall()
        return [
            SearchHit(
                chunk_id=row["chunk_id"],
                source_id=row["source_id"],
                source_title=row["source_title"],
                page=int(row["page"]),
                chunk_index=int(row["chunk_index"]),
                text=row["text"],
                score=float(-row["rank"]),
                source="fts",
                unit_type="text",
            )
            for row in rows
        ]

    def search_vector(
        self, query_vector: np.ndarray, *, limit: int, source_ids: list[str] | None = None
    ) -> list[SearchHit]:
        query_vector = _normalize(query_vector)
        frag, fp = _source_in_clause("c.source_id", source_ids)
        where = f"WHERE {frag}" if frag else ""
        sql = f"""
            SELECT c.chunk_id, c.source_id, s.title AS source_title,
                   c.page, c.chunk_index, c.text, e.vector
            FROM embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            JOIN sources s ON s.source_id = c.source_id
            {where}
        """
        with self.connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(sql, fp).fetchall()

        scored: list[SearchHit] = []
        for row in rows:
            vector = _deserialize_vector(row["vector"])
            scored.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    source_id=row["source_id"],
                    source_title=row["source_title"],
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

    def search_page_vector(
        self, query_vector: np.ndarray, *, limit: int, source_ids: list[str] | None = None
    ) -> list[SearchHit]:
        query_vector = _normalize(query_vector)
        frag, fp = _source_in_clause("pe.source_id", source_ids)
        where = f"WHERE {frag}" if frag else ""
        sql = f"""
            SELECT p.source_id, s.title AS source_title, p.page, p.ocr_text, pe.vector
            FROM page_embeddings pe
            JOIN pages p ON p.source_id = pe.source_id AND p.page = pe.page
            JOIN sources s ON s.source_id = pe.source_id
            {where}
        """
        with self.connect() as conn:
            self._create_schema(conn)
            rows = conn.execute(sql, fp).fetchall()

        scored: list[SearchHit] = []
        for row in rows:
            vector = _deserialize_vector(row["vector"])
            scored.append(
                SearchHit(
                    chunk_id=build_image_unit_id(row["source_id"], int(row["page"])),
                    source_id=row["source_id"],
                    source_title=row["source_title"],
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
        for statement in _CREATE_STATEMENTS:
            conn.execute(statement)

    def _drop_schema(self, conn: sqlite3.Connection) -> None:
        for table in _DROP_ORDER:
            conn.execute(f"DROP TABLE IF EXISTS {table}")


def _source_in_clause(column: str, source_ids: list[str] | None) -> tuple[str, list[str]]:
    if not source_ids:
        return "", []
    return f"{column} IN ({','.join('?' for _ in source_ids)})", list(source_ids)


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
