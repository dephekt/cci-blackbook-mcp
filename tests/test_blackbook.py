from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cci_blackbook.auth import is_authorized
from cci_blackbook.chunking import (
    Chunk,
    chunk_page,
    estimate_tokens,
    group_chunks_into_documents,
)
from cci_blackbook.embeddings import (
    StubDenseProvider,
    VoyageUnavailable,
    _unpack_multimodal,
    _unpack_one_document,
    build_dense_provider,
    mm_element,
    pack_multimodal_batches,
)
from cci_blackbook.pagefilter import classify_page
from cci_blackbook.render import ImageUnit, RenderedPage, page_image_tokens, visual_signals
from cci_blackbook.service import (
    BlackBookService,
    IndexUnavailable,
    IngestFailed,
    _fuse_hits,
    _index_current,
    _resolve_sources,
)
from cci_blackbook.settings import (
    SCHEMA_VERSION,
    ScopedInt,
    ScopedPages,
    Settings,
    _scoped_int_from_env,
    _scoped_pages_from_env,
    settings_fingerprint,
)
from cci_blackbook.sources import (
    SourceMeta,
    build_image_unit_id,
    build_text_unit_id,
    corpus_identity,
    discover_sources,
    parse_unit_id,
    slugify,
    titleize,
)
from cci_blackbook.store import BlackBookIndex, PageRecord, SearchHit
from PIL import Image


def make_settings(root: Path, **over) -> Settings:
    defaults = dict(
        source_dir=root / "source",
        index_dir=root / "index",
        cache_dir=root / "cache",
        sqlite_path=root / "index" / "blackbook.sqlite3",
        embedding_backend="stub",
        chunk_chars=300,
        chunk_overlap_chars=50,
        voyage_text_model="voyage-context-4",
        voyage_image_model="voyage-multimodal-3.5",
        voyage_output_dim=64,
        voyage_output_dtype="float",
        voyage_timeout=5.0,
        voyage_max_retries=0,
        voyage_retention_confirmed=True,
        doc_token_budget=28000,
        chars_per_token=3.0,
        max_chunk_tokens=32000,
        mm_token_budget=200000,
        mm_max_inputs=400,
        mm_pixels_per_token=560,
        render_dpi=100,
        render_max_pixels=12000000,
        blank_min_chars=100,
        blank_max_ink=0.02,
        blank_max_color=0.005,
        ink_luma_threshold=240,
        color_sat_threshold=30,
        blank_filter_disable=False,
        force_keep_pages=ScopedPages.all(()),
        force_drop_pages=ScopedPages.all(()),
        min_vector_score=0.0,
        min_image_score=0.0,
        rrf_k=60,
        rrf_weight_fts=1.0,
        rrf_weight_text=1.0,
        rrf_weight_image=2.0,
        max_units_per_page=2,
        min_expected_median_chars=ScopedInt(0, ()),
        citation_thumbnails=False,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
    defaults.update(over)
    return Settings(**defaults)


def write_pdf(directory: Path, name: str = "CCI Black Book.pdf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text("x")
    return p


def img(w: int = 60, h: int = 80, fill=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (w, h), fill)


def rp(page: int, text: str, *, ink: float = 0.0, color: float = 0.0, image=None) -> RenderedPage:
    image = image if image is not None else img()
    return RenderedPage(
        page=page,
        ocr_text=text,
        char_count=len(text),
        width=image.width,
        height=image.height,
        ink_coverage=ink,
        color_fraction=color,
        image=image,
    )


def hit(uid, page, source, unit_type="text", sid="src", title="Src") -> SearchHit:
    return SearchHit(uid, sid, title, page, 0, "x", 0.0, source, unit_type)


def _corpus_identity(source_dir: Path) -> list[dict]:
    return corpus_identity(
        (s.id, str(s.path), s.size, s.mtime_ns) for s in discover_sources(source_dir)
    )


def dispatch_source(pages_by_name: dict[str, list[RenderedPage]]):
    """A page_source that yields a different page list per PDF filename."""
    def source(path: Path):
        return iter(pages_by_name[path.name])
    return source


class ChunkingTest(unittest.TestCase):
    def test_chunk_page_namespaces_and_advances(self):
        text = "alpha beta gamma. " * 80
        chunks = chunk_page("cci-black-book", 7, text, chunk_chars=120, overlap_chars=30)
        self.assertGreater(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "cci-black-book:p0007-c000")
        self.assertTrue(all(c.page == 7 for c in chunks))
        self.assertTrue(all(c.source_id == "cci-black-book" for c in chunks))
        starts = [c.char_start for c in chunks]
        self.assertEqual(starts, sorted(set(starts)))


class AuthTest(unittest.TestCase):
    def test_bearer_auth_uses_exact_token(self):
        self.assertTrue(is_authorized("Bearer secret-token", "secret-token"))
        self.assertFalse(is_authorized("Bearer wrong", "secret-token"))
        self.assertFalse(is_authorized("Basic secret-token", "secret-token"))
        self.assertFalse(is_authorized("", "secret-token"))


class SourcesTest(unittest.TestCase):
    def test_slug_and_title(self):
        self.assertEqual(slugify("CCI Black Book"), "cci-black-book")
        self.assertEqual(titleize("CCI Black Book"), "CCI Black Book")  # acronym/case preserved
        self.assertEqual(slugify("aroya_guide_to_drying"), "aroya-guide-to-drying")
        self.assertEqual(titleize("aroya_guide_to_drying"), "Aroya Guide To Drying")
        self.assertRegex(slugify("!!!weird***name!!!"), r"^[a-z0-9][a-z0-9-]*$")
        self.assertNotIn(":", slugify("a:b:c"))
        self.assertEqual(slugify("***"), "source")
        self.assertEqual(titleize("   "), "Source")

    def test_unit_id_roundtrip(self):
        self.assertEqual(
            parse_unit_id(build_text_unit_id("cci-black-book", 42, 1)), ("cci-black-book", 42, "text")
        )
        self.assertEqual(
            parse_unit_id(build_image_unit_id("cci-black-book", 42)), ("cci-black-book", 42, "image")
        )
        self.assertIsNone(parse_unit_id("p0042-img"))  # bare (no source) not parseable
        self.assertIsNone(parse_unit_id("garbage"))
        self.assertIsNone(parse_unit_id("cci-black-book:pxx-img"))

    def test_disambiguation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            sd = Path(d)
            (sd / "CCI Black Book.pdf").write_text("a")
            (sd / "CCI-Black-Book.pdf").write_text("b")  # slugs collide
            got1 = [s.id for s in discover_sources(sd)]
            got2 = [s.id for s in discover_sources(sd)]
            self.assertEqual(got1, ["cci-black-book", "cci-black-book-2"])  # name-sorted
            self.assertEqual(got1, got2)


class GroupingTest(unittest.TestCase):
    def _chunks(self, spec, sid="src"):
        out = []
        for page, count in spec:
            for i in range(count):
                out.append(Chunk(build_text_unit_id(sid, page, i), sid, page, i, "word " * 30, 0, 1))
        return out

    def test_page_aligned_within_budget_every_chunk_once(self):
        chunks = self._chunks([(1, 2), (2, 3), (3, 1)])
        per_chunk = estimate_tokens("word " * 30, 3.0)
        budget = per_chunk * 4
        groups = group_chunks_into_documents(
            chunks, token_budget=budget, chars_per_token=3.0, max_chunk_tokens=32000
        )
        flat = [i for g in groups for i in g]
        self.assertEqual(sorted(flat), list(range(len(chunks))))
        page_to_docs = {}
        for di, g in enumerate(groups):
            for i in g:
                page_to_docs.setdefault(chunks[i].page, set()).add(di)
        self.assertTrue(all(len(v) == 1 for v in page_to_docs.values()))
        self.assertEqual(len(groups), 2)  # budget MUST split
        for g in groups:
            doc_tokens = sum(estimate_tokens(chunks[i].text, 3.0) for i in g)
            self.assertLessEqual(doc_tokens, budget)

    def test_documents_never_mix_sources(self):
        # Two books whose chunks together fit under budget must STILL be split by source,
        # so one book is never contextualized with another.
        chunks = self._chunks([(1, 1)], sid="book-a") + self._chunks([(1, 1)], sid="book-b")
        big = estimate_tokens("word " * 30, 3.0) * 100  # far above the total
        groups = group_chunks_into_documents(
            chunks, token_budget=big, chars_per_token=3.0, max_chunk_tokens=32000
        )
        self.assertEqual(len(groups), 2)  # split by source despite ample budget for both
        for g in groups:
            self.assertEqual(len({chunks[i].source_id for i in g}), 1)

    def test_over_budget_chunk_raises(self):
        chunks = self._chunks([(1, 1)])
        with self.assertRaises(ValueError):
            group_chunks_into_documents(
                chunks, token_budget=100000, chars_per_token=3.0, max_chunk_tokens=1
            )

    def test_empty(self):
        self.assertEqual(
            group_chunks_into_documents([], token_budget=100, chars_per_token=3.0, max_chunk_tokens=10),
            [],
        )


class BatchingTest(unittest.TestCase):
    def test_pack_respects_caps_and_never_splits(self):
        units = [ImageUnit(i, "text", img(100, 100)) for i in range(1, 8)]
        batches = list(
            pack_multimodal_batches(
                units, token_budget=10_000, max_inputs=3, pixels_per_token=560, chars_per_token=3.0
            )
        )
        seen = [u.page for b in batches for u in b]
        self.assertEqual(seen, [u.page for u in units])
        for b in batches:
            self.assertLessEqual(len(b), 3)

    def test_token_budget_forces_smaller_batches(self):
        units = [ImageUnit(i, "", img(1000, 1000)) for i in range(1, 6)]
        one_cost = page_image_tokens(1000, 1000, 0, pixels_per_token=560, chars_per_token=3.0)
        batches = list(
            pack_multimodal_batches(
                units, token_budget=one_cost, max_inputs=999, pixels_per_token=560, chars_per_token=3.0
            )
        )
        self.assertTrue(all(len(b) == 1 for b in batches))
        self.assertEqual(len(batches), 5)

    def test_page_image_tokens_monotonic(self):
        small = page_image_tokens(100, 100, 0, pixels_per_token=560, chars_per_token=3.0)
        big = page_image_tokens(1000, 1000, 0, pixels_per_token=560, chars_per_token=3.0)
        self.assertGreater(big, small)
        self.assertGreater(
            page_image_tokens(100, 100, 300, pixels_per_token=560, chars_per_token=3.0), small
        )


class VisualSignalTest(unittest.TestCase):
    def test_visual_signals(self):
        ink, color = visual_signals(img(200, 200), 240, 30)
        self.assertAlmostEqual(ink, 0.0, places=2)
        self.assertAlmostEqual(color, 0.0, places=2)
        ink_b, _ = visual_signals(Image.new("RGB", (200, 200), (0, 0, 0)), 240, 30)
        self.assertGreater(ink_b, 0.9)
        _, color_r = visual_signals(Image.new("RGB", (200, 200), (255, 0, 0)), 240, 30)
        self.assertGreater(color_r, 0.9)

    def test_classify_truth_table(self):
        common = dict(
            blank_min_chars=100, blank_max_ink=0.02, blank_max_color=0.005,
            force_keep=frozenset(), force_drop=frozenset(), disabled=False,
        )
        self.assertEqual(classify_page(rp(1, "x" * 200, ink=0.0), **common).reason, "kept:content")
        figure = classify_page(rp(2, "", ink=0.30), **common)
        self.assertEqual(figure.reason, "kept:visual")
        self.assertTrue(figure.kept)
        self.assertEqual(classify_page(rp(3, "", ink=0.0, color=0.10), **common).reason, "kept:visual")
        blank = classify_page(rp(4, "", ink=0.001, color=0.0), **common)
        self.assertEqual(blank.reason, "dropped:blank")
        self.assertFalse(blank.kept)
        forced_keep = classify_page(rp(5, "", ink=0.0), **{**common, "force_keep": frozenset({5})})
        self.assertEqual(forced_keep.reason, "kept:forced")
        forced_drop = classify_page(rp(6, "x" * 500, ink=0.5), **{**common, "force_drop": frozenset({6})})
        self.assertEqual(forced_drop.reason, "dropped:forced")
        self.assertTrue(classify_page(rp(7, "", ink=0.0), **{**common, "disabled": True}).kept)

    def test_boundary_at_thresholds(self):
        common = dict(
            blank_min_chars=100, blank_max_ink=0.02, blank_max_color=0.005,
            force_keep=frozenset(), force_drop=frozenset(), disabled=False,
        )
        self.assertTrue(classify_page(rp(1, "x" * 100, ink=0.0), **common).kept)
        self.assertEqual(classify_page(rp(2, "", ink=0.02), **common).reason, "kept:visual")


class UnpackTest(unittest.TestCase):
    def test_unpack_context(self):
        resp = SimpleNamespace(results=[SimpleNamespace(embeddings=[[3.0, 4.0], [1.0, 0.0]])])
        vecs = _unpack_one_document(resp, 2)
        self.assertEqual(len(vecs), 2)
        self.assertAlmostEqual(float(np.linalg.norm(vecs[0])), 1.0, places=5)
        self.assertAlmostEqual(float(vecs[0][0]), 0.6, places=5)

    def test_unpack_multimodal(self):
        resp = SimpleNamespace(embeddings=[[0.0, 2.0]])
        vecs = _unpack_multimodal(resp, 1)
        self.assertAlmostEqual(float(vecs[0][1]), 1.0, places=5)

    def test_wrong_count_raises(self):
        resp = SimpleNamespace(results=[SimpleNamespace(embeddings=[[1.0, 0.0]])])
        with self.assertRaises(VoyageUnavailable):
            _unpack_one_document(resp, 3)
        with self.assertRaises(VoyageUnavailable):
            _unpack_multimodal(SimpleNamespace(embeddings=[[1.0], [2.0]]), 1)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = BlackBookIndex(Path(self.tmp.name) / "bb.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def _rebuild(self):
        sources = [SourceMeta("src", "Src", Path("/x/Src.pdf"), 1, 1, 0)]
        chunks = [
            Chunk("src:p0001-c000", "src", 1, 0, "vapor pressure deficit transpiration", 0, 1),
            Chunk("src:p0002-c000", "src", 2, 0, "reverse osmosis water ppm", 0, 1),
        ]
        cvecs = [np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)]
        pages = [
            PageRecord("src", 1, "vpd text", 8, 0.3, 0.0, 60, 80, True, "kept:content"),
            PageRecord("src", 2, "", 0, 0.3, 0.0, 60, 80, True, "kept:visual"),
            PageRecord("src", 3, "", 0, 0.001, 0.0, 60, 80, False, "dropped:blank"),
        ]
        pvecs = {("src", 1): np.array([1.0, 0.0], np.float32), ("src", 2): np.array([0.0, 1.0], np.float32)}
        self.index.rebuild(
            sources=sources, chunks=chunks, chunk_vectors=cvecs,
            page_records=pages, page_vectors=pvecs, metadata={"k": 1},
        )

    def test_rebuild_rolls_back_to_prior_index_on_insert_failure(self):
        # The headline atomicity guarantee: a failure INSIDE the rebuild txn (after
        # BEGIN + DROP + CREATE) must roll back to the prior good index, not leave it
        # dropped/half-built. Guards against a regression to executescript / dropping BEGIN.
        self._rebuild()
        self.assertEqual(self.index.status()["chunk_count"], 2)

        # Two chunks share one chunk_id -> PK violation on the chunks INSERT, which runs
        # after BEGIN/DROP/CREATE. Everything else is valid (passes pre-txn validation).
        dup = [
            Chunk("src:p0001-c000", "src", 1, 0, "a", 0, 1),
            Chunk("src:p0001-c000", "src", 1, 1, "b", 0, 1),
        ]
        with self.assertRaises(sqlite3.IntegrityError):
            self.index.rebuild(
                sources=[SourceMeta("src", "Src", Path("/x/Src.pdf"), 1, 1, 0)],
                chunks=dup,
                chunk_vectors=[np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)],
                page_records=[PageRecord("src", 1, "a", 1, 0.3, 0.0, 10, 10, True, "kept:content")],
                page_vectors={("src", 1): np.array([1.0, 0.0], np.float32)},
                metadata={"k": 2},
            )

        st = self.index.status()
        self.assertTrue(st["ready"])  # prior index restored, not dropped/empty
        self.assertEqual(st["chunk_count"], 2)
        self.assertEqual(st["source_count"], 1)
        with self.index.connect() as conn:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_dual_space_roundtrip(self):
        self._rebuild()
        st = self.index.status()
        self.assertTrue(st["ready"])
        self.assertEqual(st["chunk_count"], 2)
        self.assertEqual(st["image_unit_count"], 2)
        self.assertEqual(st["source_count"], 1)

        tv = self.index.search_vector(np.array([1.0, 0.0], np.float32), limit=5)
        self.assertEqual(tv[0].chunk_id, "src:p0001-c000")
        self.assertEqual(tv[0].source, "text_dense")
        self.assertEqual(tv[0].source_id, "src")
        self.assertEqual(tv[0].source_title, "Src")

        iv = self.index.search_page_vector(np.array([0.0, 1.0], np.float32), limit=5)
        self.assertEqual(iv[0].chunk_id, "src:p0002-img")
        self.assertEqual(iv[0].source, "image_dense")
        self.assertEqual(iv[0].unit_type, "image")

        self.assertEqual(self.index.read_chunk("src:p0001-c000").unit_type, "text")
        self.assertEqual(self.index.read_page("src", 2).unit_type, "image")
        self.assertIsNone(self.index.read_page("src", 3))  # dropped page has no image unit

    def test_legacy_embeddings_table_reused(self):
        self._rebuild()
        with self.index.connect() as conn:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")]
        self.assertEqual(cols, ["chunk_id", "dim", "vector"])

    def test_pages_composite_pk(self):
        self._rebuild()
        with self.index.connect() as conn:
            page_pk = {r["name"] for r in conn.execute("PRAGMA table_info(pages)") if r["pk"] > 0}
            pe_pk = {r["name"] for r in conn.execute("PRAGMA table_info(page_embeddings)") if r["pk"] > 0}
        self.assertEqual(page_pk, {"source_id", "page"})
        self.assertEqual(pe_pk, {"source_id", "page"})


class MigrationTest(unittest.TestCase):
    def test_rebuild_over_old_v2_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            pdf = write_pdf(sd, "CCI Black Book.pdf")
            sqlite_path = root / "index" / "blackbook.sqlite3"
            sqlite_path.parent.mkdir(parents=True)
            # Seed an OLD (v2) index: legacy page-as-PK pages, chunks WITHOUT source_id,
            # user_version left 0. The DROP+CREATE rebuild must converge it.
            conn = sqlite3.connect(sqlite_path)
            conn.executescript(
                """
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, page INTEGER, chunk_index INTEGER,
                    text TEXT, char_start INTEGER, char_end INTEGER);
                CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, dim INTEGER, vector BLOB);
                CREATE TABLE pages (page INTEGER PRIMARY KEY, ocr_text TEXT, char_count INTEGER,
                    ink_coverage REAL, color_fraction REAL, width INTEGER, height INTEGER,
                    kept INTEGER, reason TEXT);
                CREATE TABLE page_embeddings (page INTEGER PRIMARY KEY REFERENCES pages(page),
                    dim INTEGER, vector BLOB);
                """
            )
            conn.execute("INSERT INTO chunks VALUES ('p0001-c000',1,0,'old text',0,8)")
            conn.execute(
                "INSERT INTO embeddings VALUES ('p0001-c000',384,?)",
                (np.zeros(384, np.float32).tobytes(),),
            )
            conn.execute("INSERT INTO pages VALUES (1,'old',3,0.3,0.0,60,80,1,'kept:content')")
            conn.execute(
                "INSERT INTO page_embeddings VALUES (1,384,?)", (np.zeros(384, np.float32).tobytes(),)
            )
            conn.execute("INSERT INTO meta VALUES ('source', '{\"stale\": true}')")
            conn.commit()
            conn.close()

            settings = make_settings(root, sqlite_path=sqlite_path)

            def source(_p):
                yield rp(1, "vapor pressure deficit " * 20, ink=0.3, image=img())

            svc = BlackBookService(settings, page_source=source)
            self.assertTrue(svc.ensure_index(force=False)["rebuilt"])  # stale schema => rebuild

            with svc.index.connect() as c:
                cols = {r["name"] for r in c.execute("PRAGMA table_info(pages)")}
                self.assertIn("source_id", cols)
                self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 3)
                dim = c.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()[0]
                chunk_id = c.execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()[0]
            self.assertEqual(dim, settings.voyage_output_dim)
            self.assertTrue(chunk_id.startswith("cci-black-book:"))
            st = svc.index.status()
            self.assertEqual({s["source_id"] for s in st["sources"]}, {"cci-black-book"})
            self.assertIn("fingerprint", st["metadata"])
            self.assertTrue(_index_current(st, _corpus_identity(sd), settings))
            self.assertTrue(pdf.exists())


class FuseTest(unittest.TestCase):
    def test_image_weight_lifts_figure(self):
        named = [
            ("fts", [hit("s:p1-c000", 1, "fts"), hit("s:p2-c000", 2, "fts"), hit("s:p3-c000", 3, "fts")]),
            ("text_dense", [hit("s:p2-c000", 2, "text_dense"), hit("s:p1-c000", 1, "text_dense")]),
            ("image_dense", [hit("s:p9-img", 9, "image_dense", "image")]),
        ]
        weighted = _fuse_hits(named, limit=5, k=60, weights={"fts": 1, "text_dense": 1, "image_dense": 2})
        self.assertEqual(weighted[0].hit.chunk_id, "s:p9-img")
        flat = _fuse_hits(named, limit=5, k=60, weights={"fts": 1, "text_dense": 1, "image_dense": 1})
        self.assertNotIn("s:p9-img", {f.hit.chunk_id for f in flat[:2]})

    def test_per_page_cap(self):
        named = [
            ("fts", [hit("s:p5-c000", 5, "fts"), hit("s:p5-c001", 5, "fts")]),
            ("image_dense", [hit("s:p5-img", 5, "image_dense", "image")]),
        ]
        capped = _fuse_hits(named, limit=10, k=60, weights={}, max_units_per_page=1)
        self.assertEqual(sum(1 for f in capped if f.hit.page == 5), 1)

    def test_per_page_cap_is_per_source(self):
        # same page NUMBER but different books must both survive a per-page cap of 1.
        named = [("fts", [hit("a:p5-c000", 5, "fts", sid="a"), hit("b:p5-c000", 5, "fts", sid="b")])]
        capped = _fuse_hits(named, limit=10, k=60, weights={}, max_units_per_page=1)
        self.assertEqual(len(capped), 2)


class MultiSourceCollisionTest(unittest.TestCase):
    def test_two_books_same_page_numbers_coexist(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            source = dispatch_source({
                "CCI Black Book.pdf": [
                    rp(1, "vapor pressure deficit transpiration canopy " * 4, ink=0.4, image=img(fill=(10, 20, 30))),
                    rp(2, "", ink=0.3, image=img(60, 80, fill=(200, 50, 50))),  # distinct color
                ],
                "aroya_guide_to_drying.pdf": [
                    rp(1, "drying curing terpene retention humidity " * 4, ink=0.4, image=img(fill=(30, 20, 10))),
                    rp(2, "", ink=0.3, image=img(60, 80, fill=(50, 50, 200))),  # distinct color
                ],
            })
            svc = BlackBookService(make_settings(root), page_source=source)
            result = svc.ensure_index(force=True)  # must NOT raise on the page-number collision
            self.assertEqual(result["source_count"], 2)

            with svc.index.connect() as c:
                ids = {r[0] for r in c.execute("SELECT chunk_id FROM chunks")}
                self.assertIn("cci-black-book:p0001-c000", ids)
                self.assertIn("aroya-guide-to-drying:p0001-c000", ids)
                self.assertEqual(c.execute("SELECT COUNT(*) FROM pages").fetchone()[0], 4)
                self.assertEqual(c.execute("SELECT COUNT(*) FROM page_embeddings").fetchone()[0], 4)

            by_id = {s["source_id"]: s for s in svc.index.status()["sources"]}
            self.assertEqual(set(by_id), {"cci-black-book", "aroya-guide-to-drying"})
            self.assertEqual(by_id["cci-black-book"]["title"], "CCI Black Book")
            self.assertEqual(by_id["aroya-guide-to-drying"]["title"], "Aroya Guide To Drying")
            self.assertTrue(all(s["chunk_count"] > 0 and s["image_unit_count"] > 0 for s in by_id.values()))

            cite = svc.read_citation("aroya-guide-to-drying:p0001-c000")
            self.assertEqual(cite["source_title"], "Aroya Guide To Drying")
            self.assertIn("Aroya Guide To Drying", cite["citation"])

            cci2 = svc.index.read_page("cci-black-book", 2)
            aro2 = svc.index.read_page("aroya-guide-to-drying", 2)
            self.assertEqual(cci2.source_title, "CCI Black Book")
            self.assertEqual(aro2.source_title, "Aroya Guide To Drying")


class SourceFilterTest(unittest.TestCase):
    def _svc(self, root):
        sd = root / "source"
        write_pdf(sd, "CCI Black Book.pdf")
        write_pdf(sd, "aroya_guide_to_drying.pdf")
        source = dispatch_source({
            "CCI Black Book.pdf": [rp(1, "alpha vapor pressure deficit " * 6, ink=0.4, image=img(fill=(10, 20, 30)))],
            "aroya_guide_to_drying.pdf": [rp(1, "bravo drying curing terpene " * 6, ink=0.4, image=img(fill=(30, 20, 10)))],
        })
        # Score gates off so a scoped DENSE search always returns the source's units
        # (a filter bug then shows as wrong-source or empty, not a masked pass).
        svc = BlackBookService(
            make_settings(root, min_vector_score=-2.0, min_image_score=-2.0), page_source=source
        )
        svc.ensure_index(force=True)
        return svc

    def test_filter_scopes_without_leaks(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            # fts/hybrid match aroya's text -> non-empty AND only aroya
            for mode in ("fts", "hybrid"):
                res = svc.search("drying curing terpene", mode=mode, sources=["aroya-guide-to-drying"])
                self.assertTrue(res["results"], mode)
                self.assertTrue(all(r["source_id"] == "aroya-guide-to-drying" for r in res["results"]), mode)
            # dense-only modes: the SQL WHERE must return the scoped book's units
            # (non-empty), and scoping the other way must return the OTHER book.
            for mode in ("text", "image"):
                aro = svc.search("drying curing", mode=mode, sources=["aroya-guide-to-drying"])
                self.assertTrue(aro["results"], mode)
                self.assertTrue(all(r["source_id"] == "aroya-guide-to-drying" for r in aro["results"]), mode)
                cci = svc.search("alpha vapor", mode=mode, sources=["cci-black-book"])
                self.assertTrue(cci["results"], mode)
                self.assertTrue(all(r["source_id"] == "cci-black-book" for r in cci["results"]), mode)

    def test_filter_string_and_case(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            res = svc.search("alpha vapor pressure", mode="fts", sources="cci-black-book")  # bare string
            self.assertTrue(res["results"])
            self.assertTrue(all(r["source_id"] == "cci-black-book" for r in res["results"]))
            res = svc.search("alpha vapor pressure", mode="fts", sources=["CCI-BLACK-BOOK"])  # uppercase
            self.assertTrue(res["results"])
            self.assertTrue(all(r["source_id"] == "cci-black-book" for r in res["results"]))

    def test_unknown_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            res = svc.search("alpha", mode="hybrid", sources=["nope"])  # all-unknown -> abstain
            self.assertTrue(res["abstain"])
            self.assertTrue(any("no requested source" in n for n in res["confidence_notes"]))
            res = svc.search("alpha vapor", mode="fts", sources=["cci-black-book", "nope"])  # partial
            self.assertTrue(res["results"])
            self.assertTrue(all(r["source_id"] == "cci-black-book" for r in res["results"]))
            self.assertTrue(any("ignored unknown" in n for n in res["confidence_notes"]))

    def test_none_searches_all(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            res = svc.search("alpha bravo vapor drying", mode="fts", sources=None)
            self.assertEqual(
                {r["source_id"] for r in res["results"]}, {"cci-black-book", "aroya-guide-to-drying"}
            )


class _RecordingStub(StubDenseProvider):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.doc_batches: list[list[list[str]]] = []

    def embed_text_documents(self, documents):
        self.doc_batches.append([list(doc) for doc in documents])
        return super().embed_text_documents(documents)


class PerSourceContextualizationTest(unittest.TestCase):
    def test_documents_never_mix_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            source = dispatch_source({
                "CCI Black Book.pdf": [rp(1, "MARKERCCI vapor pressure deficit " * 6, ink=0.4, image=img())],
                "aroya_guide_to_drying.pdf": [rp(1, "MARKERAROYA drying curing terpene " * 6, ink=0.4, image=img())],
            })
            prov = _RecordingStub(dim=64)
            svc = BlackBookService(make_settings(root), provider=prov, page_source=source)
            svc.ensure_index(force=True)
            self.assertEqual(len(prov.doc_batches), 2)  # exactly one call per non-empty source
            for batch in prov.doc_batches:
                for doc in batch:
                    joined = " ".join(doc)
                    self.assertTrue(("MARKERCCI" in joined) ^ ("MARKERAROYA" in joined))  # exactly one book


class SchemaGuardTest(unittest.TestCase):
    def test_stale_schema_version_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "vapor pressure deficit " * 10, ink=0.4, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            self.assertTrue(svc.index.status()["ready"])

            conn = svc.index.connect()  # simulate a pre-v3 index
            try:
                conn.execute("PRAGMA user_version = 2")
                conn.commit()
            finally:
                conn.close()

            st = svc.index.status()
            self.assertFalse(st["ready"])
            self.assertIn("schema out of date", st["reason"])
            self.assertTrue(svc.search("vapor", mode="hybrid")["abstain"])
            self.assertFalse(svc.read_citation("cci-black-book:p0001-c000")["found"])


class ScopedConfigTest(unittest.TestCase):
    def test_parse_scoped_pages(self):
        import os

        os.environ["CCI_TEST_PAGES"] = "cci-black-book:1-4 aroya-guide-to-drying:5 7"
        try:
            sp = _scoped_pages_from_env("CCI_TEST_PAGES")
        finally:
            del os.environ["CCI_TEST_PAGES"]
        self.assertEqual(sp.for_source("cci-black-book"), frozenset({1, 2, 3, 4, 7}))  # scoped + default
        self.assertEqual(sp.for_source("aroya-guide-to-drying"), frozenset({5, 7}))
        self.assertEqual(sp.for_source("other"), frozenset({7}))

    def test_parse_scoped_int(self):
        import os

        os.environ["CCI_TEST_INT"] = "cci-black-book:200"
        try:
            si = _scoped_int_from_env("CCI_TEST_INT")
        finally:
            del os.environ["CCI_TEST_INT"]
        self.assertEqual(si.for_source("cci-black-book"), 200)
        self.assertEqual(si.for_source("other"), 0)

    def test_force_drop_is_source_scoped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            source = dispatch_source({
                "CCI Black Book.pdf": [rp(p, f"cci page {p} content here " * 8, ink=0.4, image=img()) for p in (1, 2, 3, 4)],
                "aroya_guide_to_drying.pdf": [rp(p, f"aroya page {p} content here " * 8, ink=0.4, image=img()) for p in (1, 2, 3, 4)],
            })
            settings = make_settings(
                root,
                force_drop_pages=ScopedPages(frozenset(), (("cci-black-book", frozenset({1, 2, 3, 4})),)),
            )
            svc = BlackBookService(settings, page_source=source)
            svc.ensure_index(force=True)
            with svc.index.connect() as c:
                cci_kept = c.execute(
                    "SELECT COUNT(*) FROM pages WHERE source_id='cci-black-book' AND kept=1"
                ).fetchone()[0]
                aro_kept = c.execute(
                    "SELECT COUNT(*) FROM pages WHERE source_id='aroya-guide-to-drying' AND kept=1"
                ).fetchone()[0]
            self.assertEqual(cci_kept, 0)  # dropped in cci only
            self.assertEqual(aro_kept, 4)  # aroya's pages 1-4 untouched

    def test_tripwire_is_source_scoped(self):
        # cci is HEALTHY and carries the 200 override; aroya is SPARSE but on the default 0.
        # A GLOBAL 200 threshold would abort on aroya -> ingest succeeding proves the 200 is
        # scoped to cci and aroya's own default (0 = disabled) governs aroya.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            source = dispatch_source({
                "CCI Black Book.pdf": [rp(1, "x" * 300, ink=0.4, image=img())],  # healthy (>=200)
                "aroya_guide_to_drying.pdf": [rp(1, "ab", ink=0.4, image=img())],  # sparse (median 2)
            })
            settings = make_settings(
                root, min_expected_median_chars=ScopedInt(0, (("cci-black-book", 200),))
            )
            svc = BlackBookService(settings, page_source=source)
            self.assertTrue(svc.ensure_index(force=True)["rebuilt"])  # aroya not aborted by cci's 200

    def test_tripwire_override_fires_on_its_source(self):
        # The scoped override still aborts when ITS source is the sparse one.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")
            source = dispatch_source({"CCI Black Book.pdf": [rp(1, "ab", ink=0.4, image=img())]})
            settings = make_settings(
                root, min_expected_median_chars=ScopedInt(0, (("cci-black-book", 200),))
            )
            svc = BlackBookService(settings, page_source=source)
            with self.assertRaises(IngestFailed) as ctx:
                svc.ensure_index(force=True)
            self.assertIn("cci-black-book", str(ctx.exception))


class ServiceTrapTest(unittest.TestCase):
    def _service(self, root):
        write_pdf(root / "source", "CCI Black Book.pdf")

        def source(_p):
            yield rp(1, "vapor pressure deficit controls transpiration " * 4, ink=0.4, image=img())
            yield rp(2, "", ink=0.30, image=img(61, 81))  # text-less figure -> kept:visual
            yield rp(3, "", ink=0.001, color=0.0, image=img())  # blank -> dropped

        return BlackBookService(make_settings(root), page_source=source)

    def test_textless_figure_indexed_and_retrievable_blank_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            svc = self._service(root)
            result = svc.ensure_index(force=True)
            self.assertEqual(result["image_unit_count"], 2)
            self.assertEqual(result["pages_dropped"], 1)

            with svc.index.connect() as c:
                page2_chunks = c.execute("SELECT COUNT(*) FROM chunks WHERE page=2").fetchone()[0]
                reason2 = c.execute("SELECT reason FROM pages WHERE page=2").fetchone()[0]
                reason3 = c.execute("SELECT reason FROM pages WHERE page=3").fetchone()[0]
                page3_img = c.execute("SELECT COUNT(*) FROM page_embeddings WHERE page=3").fetchone()[0]
            self.assertEqual(page2_chunks, 0)
            self.assertEqual(reason2, "kept:visual")
            self.assertEqual(reason3, "dropped:blank")
            self.assertEqual(page3_img, 0)

            provider = StubDenseProvider(dim=svc.settings.voyage_output_dim)
            v2 = provider.embed_image_units([ImageUnit(2, "", img(61, 81))])[0]
            iv = svc.index.search_page_vector(v2, limit=5)
            self.assertEqual(iv[0].page, 2)

            # bare cached id resolves via the single-source shim
            self.assertTrue(svc.read_citation("p0002-img")["found"])
            self.assertTrue(svc.read_citation("cci-black-book:p0002-img")["found"])

    def test_mm_element_never_sends_empty_string(self):
        i = img()
        self.assertEqual(mm_element(ImageUnit(2, "", i)), [i])
        self.assertEqual(len(mm_element(ImageUnit(2, "  ", i))), 1)
        self.assertEqual(mm_element(ImageUnit(1, "caption", i)), ["caption", i])

    def test_image_query_path_surfaces_captioned_figure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "reverse osmosis water treatment ppm " * 10, ink=0.4, image=img())
                yield rp(2, "seedling clone morphology comparison", ink=0.3, image=img(61, 81))

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            res = svc.search("seedling morphology comparison", mode="image")
            self.assertFalse(res["abstain"])
            self.assertEqual(res["results"][0]["unit_type"], "image")
            self.assertEqual(res["results"][0]["page"], 2)
            self.assertEqual(res["results"][0]["source_id"], "cci-black-book")


class GateTest(unittest.TestCase):
    def test_image_gate_filters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "some text here " * 20, ink=0.4, image=img())

            svc = BlackBookService(make_settings(root, min_image_score=0.99), page_source=source)
            svc.ensure_index(force=True)
            self.assertTrue(svc.search("unrelated zzzz query", mode="image")["abstain"])

            svc2 = BlackBookService(make_settings(root, min_image_score=0.0), page_source=source)
            svc2.ensure_index(force=True)
            self.assertFalse(svc2.search("some text", mode="image")["abstain"])


class FailLoudTest(unittest.TestCase):
    def _source(self):
        def source(_p):
            yield rp(1, "vapor pressure deficit " * 20, ink=0.4, image=img())

        return source

    def test_text_embed_failure_leaves_prior_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")
            good = BlackBookService(make_settings(root), page_source=self._source())
            good.ensure_index(force=True)
            old_chunks = good.index.status()["chunk_count"]

            failing = StubDenseProvider(dim=64, fail_on=("embed_text_documents", 1))
            svc = BlackBookService(make_settings(root), provider=failing, page_source=self._source())
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)
            self.assertTrue(good.index.status()["ready"])
            self.assertEqual(good.index.status()["chunk_count"], old_chunks)

    def test_image_embed_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")
            failing = StubDenseProvider(dim=64, fail_on=("embed_image_units", 1))
            svc = BlackBookService(make_settings(root), provider=failing, page_source=self._source())
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)
            self.assertFalse(svc.index.status()["ready"])

    def test_extraction_tripwire_no_text_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "", ink=0.3, image=img())
                yield rp(2, "", ink=0.3, image=img())

            svc = BlackBookService(
                make_settings(root, min_expected_median_chars=ScopedInt.all(200)), page_source=source
            )
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)

    def test_extraction_tripwire_sparse_median(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "abc", ink=0.3, image=img())
                yield rp(2, "de", ink=0.3, image=img())

            svc = BlackBookService(
                make_settings(root, min_expected_median_chars=ScopedInt.all(200)), page_source=source
            )
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)

    def test_extraction_tripwire_ignores_empty_figure_pages(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "x" * 400, ink=0.4, image=img())
                yield rp(2, "", ink=0.3, image=img())
                yield rp(3, "", ink=0.3, image=img())

            svc = BlackBookService(
                make_settings(root, min_expected_median_chars=ScopedInt.all(200)), page_source=source
            )
            self.assertTrue(svc.ensure_index(force=True)["rebuilt"])


class FailLoudCorpusTest(unittest.TestCase):
    def test_missing_source_dir(self):
        with tempfile.TemporaryDirectory() as d:
            svc = BlackBookService(make_settings(Path(d)))  # root/source does not exist
            with self.assertRaises(IndexUnavailable):
                svc.ensure_index(force=True)

    def test_empty_source_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "source").mkdir()
            svc = BlackBookService(make_settings(root))
            with self.assertRaises(IndexUnavailable):
                svc.ensure_index(force=True)

    def test_second_source_failure_leaves_prior_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")

            def good_source(_p):
                return iter([rp(1, "vapor pressure " * 10, ink=0.4, image=img())])

            good = BlackBookService(make_settings(root), page_source=good_source)
            good.ensure_index(force=True)
            old_count = good.index.status()["source_count"]

            write_pdf(sd, "aroya_guide_to_drying.pdf")  # a second source that will fail

            def bad_source(path):
                if path.name == "aroya_guide_to_drying.pdf":
                    raise RuntimeError("corrupt pdf")
                return iter([rp(1, "vapor pressure " * 10, ink=0.4, image=img())])

            svc = BlackBookService(make_settings(root), page_source=bad_source)
            with self.assertRaises(IngestFailed) as ctx:
                svc.ensure_index(force=True)
            self.assertIn("aroya-guide-to-drying", str(ctx.exception))

            st = good.index.status()  # prior good index intact (abort before rebuild())
            self.assertTrue(st["ready"])
            self.assertEqual(st["source_count"], old_count)


class RetentionTest(unittest.TestCase):
    def test_voyage_backend_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "text " * 40, ink=0.3, image=img())

            settings = make_settings(root, embedding_backend="voyage", voyage_retention_confirmed=False)
            boom = StubDenseProvider(dim=64, fail_on=("embed_text_documents", 1))
            svc = BlackBookService(settings, provider=boom, page_source=source)
            with self.assertRaises(IngestFailed) as ctx:
                svc.ensure_index(force=True)
            self.assertIn("retention", str(ctx.exception).lower())

    def test_stub_backend_proceeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "text " * 40, ink=0.3, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            self.assertTrue(svc.ensure_index(force=True)["rebuilt"])


class _QueryFailProvider:
    text_dim = image_dim = 64

    def embed_text_documents(self, documents):
        return [[np.zeros(64, np.float32) for _ in doc] for doc in documents]

    def embed_text_query(self, text):
        raise VoyageUnavailable("boom text query")

    def embed_image_units(self, units):
        return [np.zeros(64, np.float32) for _ in units]

    def embed_image_query(self, text):
        raise VoyageUnavailable("boom image query")

    def status(self):
        return {"backend": "stub", "dim": 64, "configured": True}


class DegradationTest(unittest.TestCase):
    def test_never_built_index_abstains_without_dense_call(self):
        with tempfile.TemporaryDirectory() as d:
            res = BlackBookService(make_settings(Path(d))).search("anything", mode="hybrid")
            self.assertTrue(res["abstain"])
            self.assertEqual(res["results"], [])

    def test_query_embed_failure_degrades_to_fts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "vapor pressure deficit transpiration canopy " * 4, ink=0.4, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            svc._provider = _QueryFailProvider()
            res = svc.search("transpiration deficit", mode="hybrid")
            self.assertFalse(res["abstain"])
            self.assertTrue(any("unavailable" in n for n in res["confidence_notes"]))


class FingerprintTest(unittest.TestCase):
    def _ready_status(self, settings, source_dir):
        specs = discover_sources(source_dir)
        return {
            "ready": True,
            "sources": [
                {"source_id": s.id, "path": str(s.path), "size": s.size, "mtime_ns": s.mtime_ns}
                for s in specs
            ],
            "metadata": {"fingerprint": settings_fingerprint(settings)},
        }

    def test_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            base = make_settings(root)
            status = self._ready_status(base, sd)
            disc = _corpus_identity(sd)
            self.assertTrue(_index_current(status, disc, base))

            for change in (
                dict(voyage_text_model="voyage-context-3"),
                dict(voyage_output_dim=256),
                dict(render_dpi=150),
                dict(chunk_chars=1000),
                dict(blank_min_chars=50),
                dict(force_keep_pages=ScopedPages.all({7})),
                dict(embedding_backend="voyage"),
            ):
                self.assertFalse(
                    _index_current(status, disc, replace(base, **change)),
                    f"expected invalidation for {change}",
                )

            # corpus change: add a second PDF -> discovered differs from stored
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            self.assertFalse(_index_current(status, _corpus_identity(sd), base))


class ResolveSourcesTest(unittest.TestCase):
    def test_resolve(self):
        known = {"cci-black-book", "aroya-guide-to-drying"}
        self.assertEqual(_resolve_sources(None, known), (None, []))
        self.assertEqual(_resolve_sources([], known), (None, []))
        r, _ = _resolve_sources("cci-black-book", known)
        self.assertEqual(r, ["cci-black-book"])
        r, _ = _resolve_sources(["CCI-BLACK-BOOK"], known)  # lowercased
        self.assertEqual(r, ["cci-black-book"])
        r, notes = _resolve_sources(["nope"], known)  # all unknown -> [] (caller abstains)
        self.assertEqual(r, [])
        r, notes = _resolve_sources(["cci-black-book", "nope"], known)
        self.assertEqual(r, ["cci-black-book"])
        self.assertTrue(any("ignored unknown" in n for n in notes))


class CitationStatusTest(unittest.TestCase):
    def test_image_citation_and_status_no_leak(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_pdf(root / "source", "CCI Black Book.pdf")

            def source(_p):
                yield rp(1, "vpd text here " * 20, ink=0.4, image=img())
                yield rp(2, "", ink=0.3, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)

            cite = svc.read_citation("cci-black-book:p0002-img")
            self.assertTrue(cite["found"])
            self.assertEqual(cite["unit_type"], "image")
            self.assertEqual(cite["source_id"], "cci-black-book")
            self.assertIn("CCI Black Book", cite["citation"])
            self.assertIn("page image", cite["citation"])
            self.assertNotIn("thumbnail_png_base64", cite)

            st = svc.status()
            self.assertIn("voyage_configured", st)
            self.assertIn("filter", st["index"]["metadata"])
            self.assertEqual(st["embedding"]["backend"], "stub")
            self.assertGreaterEqual(st["source_dir"]["pdf_count"], 1)
            self.assertEqual({s["source_id"] for s in st["sources"]}, {"cci-black-book"})
            self.assertNotIn("VOYAGE_API_KEY", json.dumps(st))


class StatusSourcesTest(unittest.TestCase):
    def test_status_lists_sources_with_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            source = dispatch_source({
                "CCI Black Book.pdf": [rp(1, "vapor pressure deficit " * 8, ink=0.4, image=img())],
                "aroya_guide_to_drying.pdf": [rp(1, "drying curing terpene " * 8, ink=0.4, image=img())],
            })
            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            st = svc.status()
            self.assertEqual(st["source_dir"]["pdf_count"], 2)
            by_id = {s["source_id"]: s for s in st["sources"]}
            self.assertEqual(set(by_id), {"cci-black-book", "aroya-guide-to-drying"})
            for s in st["sources"]:
                self.assertGreater(s["page_count"], 0)
                self.assertGreater(s["chunk_count"], 0)
                self.assertGreater(s["image_unit_count"], 0)
            self.assertNotIn("VOYAGE_API_KEY", json.dumps(st))


class OfflineGuaranteeTest(unittest.TestCase):
    def test_no_voyage_import_offline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sd = root / "source"
            write_pdf(sd, "CCI Black Book.pdf")
            write_pdf(sd, "aroya_guide_to_drying.pdf")
            source = dispatch_source({
                "CCI Black Book.pdf": [rp(1, "text here " * 30, ink=0.4, image=img())],
                "aroya_guide_to_drying.pdf": [rp(1, "drying text " * 30, ink=0.4, image=img())],
            })
            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            svc.search("text", mode="hybrid")
            self.assertNotIn("voyageai", sys.modules)

    def test_build_dense_provider_rejects_unknown_backend(self):
        with tempfile.TemporaryDirectory() as d:
            settings = make_settings(Path(d), embedding_backend="hash")
            with self.assertRaises(ValueError):
                build_dense_provider(settings)


class SettingsDefaultsTest(unittest.TestCase):
    def test_default_doc_budget_fits_context4_example_limit(self):
        import os

        from cci_blackbook.settings import load_settings

        old = os.environ.pop("CCI_DOC_TOKEN_BUDGET", None)
        try:
            self.assertLess(load_settings().doc_token_budget, 32000)
        finally:
            if old is not None:
                os.environ["CCI_DOC_TOKEN_BUDGET"] = old


if __name__ == "__main__":
    unittest.main()
