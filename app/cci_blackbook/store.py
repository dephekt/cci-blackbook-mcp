from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class PreparedSource:
    source: SourceMeta
    content_sha256: str
    text_fingerprint: str
    image_fingerprint: str
    indexed_at: int
    chunks: list[Chunk]
    chunk_vectors: list[np.ndarray]
    page_records: list[PageRecord]
    page_vectors: dict[tuple[str, int], np.ndarray]


_CREATE_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY, title TEXT NOT NULL, path TEXT NOT NULL,
        size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
        page_count INTEGER NOT NULL, kept_page_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL, image_unit_count INTEGER NOT NULL,
        ordinal INTEGER NOT NULL, content_sha256 TEXT NOT NULL,
        text_fingerprint TEXT NOT NULL, image_fingerprint TEXT NOT NULL,
        indexed_at INTEGER NOT NULL)""",
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
_SOURCE_COLS = (
    "source_id, title, path, size, mtime_ns, page_count, kept_page_count, "
    "chunk_count, image_unit_count, ordinal"
)
_SOURCE_MANIFEST_COLS = (
    f"{_SOURCE_COLS}, content_sha256, text_fingerprint, image_fingerprint, indexed_at"
)


class BlackBookIndex:
    def __init__(self, sqlite_path: Path, *, journal_mode: str = "WAL"):
        self.sqlite_path = sqlite_path
        self.journal_mode = journal_mode

    def connect(self) -> sqlite3.Connection:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA journal_mode = {self.journal_mode}")
        return conn

    def _read_connect(self) -> sqlite3.Connection:
        uri = f"{self.sqlite_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def schema_version(self) -> int | None:
        if not self.sqlite_path.exists():
            return None
        try:
            with self._read_connect() as conn:
                return int(conn.execute("PRAGMA user_version").fetchone()[0])
        except sqlite3.DatabaseError:
            return -1

    def initialize(self) -> None:
        version = self.schema_version()
        if version is not None:
            if version != int(_DB_SCHEMA):
                raise ValueError(
                    f"refusing to initialize legacy or unknown schema {version}"
                )
            return
        with self.connect() as conn:
            conn.execute("BEGIN")
            self._create_schema(conn)
            conn.execute(f"PRAGMA user_version = {int(_DB_SCHEMA)}")

    def replace_sources(
        self,
        *,
        remove_source_ids: set[str],
        prepared_sources: list[PreparedSource],
        discovered_sources: list[SourceMeta],
        metadata: dict,
        expected_text_dim: int,
        expected_image_dim: int,
        initialize: bool = False,
    ) -> None:
        """Atomically delete and replace complete sources after all remote work is done."""
        self._validate_replacement(
            remove_source_ids=remove_source_ids,
            prepared_sources=prepared_sources,
            discovered_sources=discovered_sources,
            expected_text_dim=expected_text_dim,
            expected_image_dim=expected_image_dim,
        )

        with self.connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if initialize:
                tables = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('table','view')"
                ).fetchone()[0]
                if version != 0 or tables:
                    raise ValueError("refusing to initialize a non-empty database")
            elif version != int(_DB_SCHEMA):
                raise ValueError(
                    f"source replacement requires schema {_DB_SCHEMA}, found schema {version}"
                )

            conn.execute("BEGIN")
            if initialize:
                self._create_schema(conn)
                conn.execute(f"PRAGMA user_version = {int(_DB_SCHEMA)}")

            for source_id in sorted(remove_source_ids):
                conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE source_id = ?)",
                    (source_id,),
                )
            conn.executemany(
                "DELETE FROM sources WHERE source_id = ?",
                [(source_id,) for source_id in sorted(remove_source_ids)],
            )

            for prepared in prepared_sources:
                self._insert_prepared(conn, prepared)

            actual_ids = {
                row[0] for row in conn.execute("SELECT source_id FROM sources").fetchall()
            }
            discovered_ids = {source.id for source in discovered_sources}
            if actual_ids != discovered_ids:
                raise ValueError(
                    "post-replacement sources differ from discovery: "
                    f"indexed={sorted(actual_ids)}, discovered={sorted(discovered_ids)}"
                )
            conn.executemany(
                "UPDATE sources SET title=?, path=?, size=?, mtime_ns=?, ordinal=? "
                "WHERE source_id=?",
                [
                    (source.title, str(source.path), source.size, source.mtime_ns,
                     source.ordinal, source.id)
                    for source in discovered_sources
                ],
            )

            refreshed_metadata = self._aggregate_metadata(conn, metadata)
            conn.execute("DELETE FROM meta")
            conn.executemany(
                "INSERT INTO meta(key,value) VALUES (?,?)",
                [(key, json.dumps(value)) for key, value in refreshed_metadata.items()],
            )
            self._validate_transaction(conn)

    def _validate_replacement(
        self,
        *,
        remove_source_ids: set[str],
        prepared_sources: list[PreparedSource],
        discovered_sources: list[SourceMeta],
        expected_text_dim: int,
        expected_image_dim: int,
    ) -> None:
        discovered_ids = [source.id for source in discovered_sources]
        if len(discovered_ids) != len(set(discovered_ids)):
            raise ValueError("discovered source IDs are not unique")
        prepared_ids = [prepared.source.id for prepared in prepared_sources]
        if len(prepared_ids) != len(set(prepared_ids)):
            raise ValueError("prepared source IDs are not unique")
        if set(prepared_ids) - set(discovered_ids):
            raise ValueError("prepared sources are not present in discovery")
        for prepared in prepared_sources:
            source_id = prepared.source.id
            if len(prepared.content_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in prepared.content_sha256
            ):
                raise ValueError(f"invalid SHA-256 for source {source_id!r}")
            if not prepared.text_fingerprint or not prepared.image_fingerprint:
                raise ValueError(f"missing pipeline fingerprint for source {source_id!r}")
            if len(prepared.chunks) != len(prepared.chunk_vectors):
                raise ValueError(f"chunk/vector count mismatch for source {source_id!r}")
            if any(
                chunk.source_id != source_id or not chunk.chunk_id.startswith(f"{source_id}:")
                for chunk in prepared.chunks
            ):
                raise ValueError(f"chunk ownership mismatch for source {source_id!r}")
            if any(record.source_id != source_id for record in prepared.page_records):
                raise ValueError(f"page ownership mismatch for source {source_id!r}")
            kept = {
                (record.source_id, record.page)
                for record in prepared.page_records
                if record.kept
            }
            if set(prepared.page_vectors) != kept:
                raise ValueError(
                    f"kept-page/image-vector mismatch for source {source_id!r}"
                )
            _validate_vectors(
                prepared.chunk_vectors, expected_text_dim, f"text source {source_id!r}"
            )
            _validate_vectors(
                list(prepared.page_vectors.values()), expected_image_dim,
                f"image source {source_id!r}",
            )

    def _insert_prepared(self, conn: sqlite3.Connection, prepared: PreparedSource) -> None:
        source = prepared.source
        conn.execute(
            f"INSERT INTO sources ({_SOURCE_MANIFEST_COLS}) VALUES ({','.join('?' for _ in range(14))})",
            (
                source.id, source.title, str(source.path), source.size, source.mtime_ns,
                len(prepared.page_records),
                sum(1 for record in prepared.page_records if record.kept),
                len(prepared.chunks), len(prepared.page_vectors), source.ordinal,
                prepared.content_sha256, prepared.text_fingerprint,
                prepared.image_fingerprint, prepared.indexed_at,
            ),
        )
        conn.executemany(
            "INSERT INTO chunks(chunk_id,source_id,page,chunk_index,text,char_start,char_end) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (chunk.chunk_id, chunk.source_id, chunk.page, chunk.chunk_index,
                 chunk.text, chunk.char_start, chunk.char_end)
                for chunk in prepared.chunks
            ],
        )
        conn.executemany(
            "INSERT INTO chunks_fts(chunk_id,page,text) VALUES (?,?,?)",
            [(chunk.chunk_id, chunk.page, chunk.text) for chunk in prepared.chunks],
        )
        conn.executemany(
            "INSERT INTO embeddings(chunk_id,dim,vector) VALUES (?,?,?)",
            [
                (chunk.chunk_id, int(vector.shape[0]), _serialize_vector(vector))
                for chunk, vector in zip(
                    prepared.chunks, prepared.chunk_vectors, strict=True
                )
            ],
        )
        conn.executemany(
            "INSERT INTO pages(source_id,page,ocr_text,char_count,ink_coverage,color_fraction,"
            "width,height,kept,reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (record.source_id, record.page, record.ocr_text, record.char_count,
                 record.ink_coverage, record.color_fraction, record.width, record.height,
                 int(record.kept), record.reason)
                for record in prepared.page_records
            ],
        )
        conn.executemany(
            "INSERT INTO page_embeddings(source_id,page,dim,vector) VALUES (?,?,?,?)",
            [
                (source_id, page, int(vector.shape[0]), _serialize_vector(vector))
                for (source_id, page), vector in sorted(prepared.page_vectors.items())
            ],
        )

    def _aggregate_metadata(self, conn: sqlite3.Connection, metadata: dict) -> dict:
        result: dict[str, Any] = dict(metadata)
        counts = {
            "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "pages": conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
            "image_units": conn.execute("SELECT COUNT(*) FROM page_embeddings").fetchone()[0],
        }
        reasons = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT reason, COUNT(*) FROM pages GROUP BY reason ORDER BY reason"
            ).fetchall()
        }
        dropped_pages = [
            row[0]
            for row in conn.execute(
                "SELECT page FROM pages WHERE kept=0 ORDER BY source_id, page LIMIT 200"
            ).fetchall()
        ]
        result["counts"] = counts
        result["filter"] = {
            "pages": counts["pages"],
            "kept": counts["image_units"],
            "dropped": counts["pages"] - counts["image_units"],
            "by_reason": reasons,
            "dropped_pages": dropped_pages,
        }
        return result

    def _validate_transaction(self, conn: sqlite3.Connection) -> None:
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise sqlite3.IntegrityError(f"foreign key violations: {foreign_keys}")
        checks = {
            "orphaned FTS rows": """
                SELECT COUNT(*) FROM chunks_fts f
                LEFT JOIN chunks c ON c.chunk_id=f.chunk_id WHERE c.chunk_id IS NULL
            """,
            "chunks missing FTS rows": """
                SELECT COUNT(*) FROM chunks c
                LEFT JOIN chunks_fts f ON f.chunk_id=c.chunk_id WHERE f.chunk_id IS NULL
            """,
            "kept pages missing image vectors": """
                SELECT COUNT(*) FROM pages p LEFT JOIN page_embeddings pe
                ON pe.source_id=p.source_id AND pe.page=p.page
                WHERE p.kept=1 AND pe.source_id IS NULL
            """,
            "non-kept pages with image vectors": """
                SELECT COUNT(*) FROM page_embeddings pe JOIN pages p
                ON p.source_id=pe.source_id AND p.page=pe.page WHERE p.kept=0
            """,
        }
        for label, sql in checks.items():
            if conn.execute(sql).fetchone()[0]:
                raise sqlite3.IntegrityError(label)

    def status(self) -> dict:
        if not self.sqlite_path.exists():
            return {"ready": False, "reason": "index database missing"}
        try:
            with self._read_connect() as conn:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version != int(_DB_SCHEMA):
                    return _schema_unavailable(version, self.sqlite_path)
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
        except sqlite3.DatabaseError as exc:
            return {
                "ready": False,
                "reason": f"index schema is invalid or unreadable: {type(exc).__name__}",
                "sqlite_path": str(self.sqlite_path),
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
        with self._read_connect() as conn:
            return [
                dict(r)
                for r in conn.execute(f"SELECT {_SOURCE_COLS} FROM sources ORDER BY ordinal")
            ]

    def source_manifests(self) -> list[dict]:
        with self._read_connect() as conn:
            if conn.execute("PRAGMA user_version").fetchone()[0] != int(_DB_SCHEMA):
                raise ValueError("source manifests require the current schema")
            return [
                dict(row)
                for row in conn.execute(
                    f"SELECT {_SOURCE_MANIFEST_COLS} FROM sources ORDER BY ordinal"
                )
            ]

    def validate_database(self, expected_counts: dict[str, int]) -> None:
        with self._read_connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version != int(_DB_SCHEMA):
                raise ValueError(f"expected schema {_DB_SCHEMA}, found {version}")
            actual_counts = {
                "sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "pages": conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
                "image_units": conn.execute(
                    "SELECT COUNT(*) FROM page_embeddings"
                ).fetchone()[0],
            }
            if actual_counts != expected_counts:
                raise ValueError(
                    f"temporary index counts differ: expected={expected_counts}, "
                    f"actual={actual_counts}"
                )
            self._validate_transaction(conn)
            integrity = conn.execute("PRAGMA integrity_check").fetchall()
            if [row[0] for row in integrity] != ["ok"]:
                raise sqlite3.DatabaseError(f"integrity_check failed: {integrity}")

    def get_source(self, source_id: str) -> dict | None:
        with self._read_connect() as conn:
            row = conn.execute(
                f"SELECT {_SOURCE_COLS} FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        return dict(row) if row else None

    def read_chunk(self, chunk_id: str) -> SearchHit | None:
        # INVARIANT (shared by all read/search methods below): callers must gate on
        # status()["ready"] first. These run source-aware SQL and call _create_schema,
        # which would raise on a legacy (old-shaped) index; status()'s user_version guard
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


def _source_in_clause(column: str, source_ids: list[str] | None) -> tuple[str, list[str]]:
    if not source_ids:
        return "", []
    return f"{column} IN ({','.join('?' for _ in source_ids)})", list(source_ids)


def _validate_vectors(vectors: list[np.ndarray], expected_dim: int, label: str) -> None:
    for vector in vectors:
        array = np.asarray(vector)
        if array.ndim != 1 or int(array.shape[0]) != expected_dim:
            raise ValueError(
                f"{label} vector has shape {array.shape}, expected ({expected_dim},)"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label} vector contains non-finite values")


def _schema_unavailable(version: int, sqlite_path: Path) -> dict:
    return {
        "ready": False,
        "reason": (
            f"index schema {version} is unavailable; stop the MCP, then run "
            "cci-blackbook-ingest --force to rebuild schema v4. All embeddings will be "
            "regenerated and may consume Voyage allowance or incur charges"
        ),
        "sqlite_path": str(sqlite_path),
        "schema_version": version,
    }


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
