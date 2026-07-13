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
    IngestFailed,
    _fuse_hits,
    _index_current,
    parse_image_unit_id,
)
from cci_blackbook.settings import Settings, settings_fingerprint
from cci_blackbook.store import BlackBookIndex, PageRecord, SearchHit
from PIL import Image


def make_settings(root: Path, **over) -> Settings:
    defaults = dict(
        source_pdf=root / "CCI Black Book.pdf",
        index_dir=root / "index",
        cache_dir=root / "cache",
        sqlite_path=root / "index" / "blackbook.sqlite3",
        embedding_backend="stub",
        embedding_model="stub",
        openvino_device="GPU",
        render_device=root / "renderD129",
        embedding_batch_size=4,
        chunk_chars=300,
        chunk_overlap_chars=50,
        voyage_text_model="voyage-context-4",
        voyage_image_model="voyage-multimodal-3.5",
        voyage_output_dim=64,
        voyage_output_dtype="float",
        voyage_timeout=5.0,
        voyage_max_retries=0,
        voyage_retention_confirmed=True,
        doc_token_budget=100000,
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
        force_keep_pages=frozenset(),
        force_drop_pages=frozenset(),
        min_vector_score=0.0,
        min_image_score=0.0,
        rrf_k=60,
        rrf_weight_fts=1.0,
        rrf_weight_text=1.0,
        rrf_weight_image=2.0,
        max_units_per_page=2,
        min_expected_median_chars=0,
        citation_thumbnails=False,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
    defaults.update(over)
    return Settings(**defaults)


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


def hit(uid: str, page: int, source: str, unit_type: str = "text") -> SearchHit:
    return SearchHit(uid, page, 0, "x", 0.0, source, unit_type)


def _src_meta(pdf: Path) -> dict:
    stat = pdf.stat()
    return {"path": str(pdf), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class ChunkingTest(unittest.TestCase):
    def test_chunk_page_preserves_page_and_advances(self):
        text = "alpha beta gamma. " * 80
        chunks = chunk_page(7, text, chunk_chars=120, overlap_chars=30)
        self.assertGreater(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "p0007-c000")
        self.assertTrue(all(c.page == 7 for c in chunks))
        starts = [c.char_start for c in chunks]
        self.assertEqual(starts, sorted(set(starts)))


class AuthTest(unittest.TestCase):
    def test_bearer_auth_uses_exact_token(self):
        self.assertTrue(is_authorized("Bearer secret-token", "secret-token"))
        self.assertFalse(is_authorized("Bearer wrong", "secret-token"))
        self.assertFalse(is_authorized("Basic secret-token", "secret-token"))
        self.assertFalse(is_authorized("", "secret-token"))


class GroupingTest(unittest.TestCase):
    def _chunks(self, spec):
        out = []
        for page, count in spec:
            for i in range(count):
                out.append(Chunk(f"p{page:04d}-c{i:03d}", page, i, "word " * 30, 0, 1))
        return out

    def test_page_aligned_within_budget_every_chunk_once(self):
        chunks = self._chunks([(1, 2), (2, 3), (3, 1)])
        per_chunk = estimate_tokens("word " * 30, 3.0)
        budget = per_chunk * 4
        groups = group_chunks_into_documents(
            chunks, token_budget=budget, chars_per_token=3.0, max_chunk_tokens=32000
        )
        flat = [i for g in groups for i in g]
        self.assertEqual(sorted(flat), list(range(len(chunks))))  # every chunk exactly once
        page_to_docs = {}
        for di, g in enumerate(groups):
            for i in g:
                page_to_docs.setdefault(chunks[i].page, set()).add(di)
        self.assertTrue(all(len(v) == 1 for v in page_to_docs.values()))  # no page split across docs
        # the budget MUST split: pages are 100/150/50 tok, budget 200 -> [[p1],[p2,p3]].
        # (guards the 120K-token/document Voyage ceiling; a no-op grouper would fail here.)
        self.assertEqual(len(groups), 2)
        for g in groups:
            doc_tokens = sum(estimate_tokens(chunks[i].text, 3.0) for i in g)
            self.assertLessEqual(doc_tokens, budget)

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
        self.assertEqual(seen, [u.page for u in units])  # order + completeness
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
        self.assertTrue(classify_page(rp(1, "x" * 100, ink=0.0), **common).kept)  # >= min chars
        self.assertEqual(classify_page(rp(2, "", ink=0.02), **common).reason, "kept:visual")  # >= max ink


class UnpackTest(unittest.TestCase):
    def test_unpack_context(self):
        resp = SimpleNamespace(results=[SimpleNamespace(embeddings=[[3.0, 4.0], [1.0, 0.0]])])
        vecs = _unpack_one_document(resp, 2)
        self.assertEqual(len(vecs), 2)
        self.assertAlmostEqual(float(np.linalg.norm(vecs[0])), 1.0, places=5)
        self.assertAlmostEqual(float(vecs[0][0]), 0.6, places=5)  # (3,4) -> (.6,.8), order kept

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
        chunks = [
            Chunk("p0001-c000", 1, 0, "vapor pressure deficit transpiration", 0, 1),
            Chunk("p0002-c000", 2, 0, "reverse osmosis water ppm", 0, 1),
        ]
        cvecs = [np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32)]
        pages = [
            PageRecord(1, "vpd text", 8, 0.3, 0.0, 60, 80, True, "kept:content"),
            PageRecord(2, "", 0, 0.3, 0.0, 60, 80, True, "kept:visual"),
            PageRecord(3, "", 0, 0.001, 0.0, 60, 80, False, "dropped:blank"),
        ]
        pvecs = {1: np.array([1.0, 0.0], np.float32), 2: np.array([0.0, 1.0], np.float32)}
        self.index.rebuild(
            chunks=chunks, chunk_vectors=cvecs, page_records=pages, page_vectors=pvecs, metadata={"k": 1}
        )

    def test_dual_space_roundtrip(self):
        self._rebuild()
        st = self.index.status()
        self.assertTrue(st["ready"])
        self.assertEqual(st["chunk_count"], 2)
        self.assertEqual(st["image_unit_count"], 2)  # pages 1 and 2 (kept), not 3

        tv = self.index.search_vector(np.array([1.0, 0.0], np.float32), limit=5)
        self.assertEqual(tv[0].chunk_id, "p0001-c000")
        self.assertEqual(tv[0].source, "text_dense")

        iv = self.index.search_page_vector(np.array([0.0, 1.0], np.float32), limit=5)
        self.assertEqual(iv[0].chunk_id, "p0002-img")
        self.assertEqual(iv[0].source, "image_dense")
        self.assertEqual(iv[0].unit_type, "image")

        self.assertEqual(self.index.read_chunk("p0001-c000").unit_type, "text")
        self.assertEqual(self.index.read_page(2).unit_type, "image")
        self.assertIsNone(self.index.read_page(3))  # dropped page has no image unit

    def test_legacy_embeddings_table_reused(self):
        self._rebuild()
        with self.index.connect() as conn:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")]
        self.assertEqual(cols, ["chunk_id", "dim", "vector"])  # unchanged legacy schema


class MigrationTest(unittest.TestCase):
    def test_rebuild_over_old_384_index(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / "CCI Black Book.pdf"
            pdf.write_text("x")
            sqlite_path = root / "index" / "blackbook.sqlite3"
            sqlite_path.parent.mkdir(parents=True)
            conn = sqlite3.connect(sqlite_path)
            conn.executescript(
                """
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, page INTEGER, chunk_index INTEGER,
                    text TEXT, char_start INTEGER, char_end INTEGER);
                CREATE TABLE embeddings (chunk_id TEXT PRIMARY KEY, dim INTEGER, vector BLOB);
                """
            )
            conn.execute("INSERT INTO chunks VALUES ('p0001-c000',1,0,'old text',0,8)")
            conn.execute(
                "INSERT INTO embeddings VALUES ('p0001-c000',384,?)",
                (np.zeros(384, np.float32).tobytes(),),
            )
            conn.execute("INSERT INTO meta VALUES ('source', '{\"stale\": true}')")
            conn.commit()
            conn.close()

            settings = make_settings(root, sqlite_path=sqlite_path)

            def source(_p):
                yield rp(1, "vapor pressure deficit " * 20, ink=0.3, image=img())

            svc = BlackBookService(settings, page_source=source)
            self.assertTrue(svc.ensure_index(force=False)["rebuilt"])  # missing fingerprint => rebuild

            with svc.index.connect() as c:
                self.assertTrue(list(c.execute("PRAGMA table_info(pages)")))
                self.assertTrue(list(c.execute("PRAGMA table_info(page_embeddings)")))
                dim = c.execute("SELECT dim FROM embeddings LIMIT 1").fetchone()[0]
            self.assertEqual(dim, settings.voyage_output_dim)  # old 384 rows replaced
            self.assertIn("fingerprint", svc.index.status()["metadata"])
            self.assertTrue(_index_current(svc.index.status(), _src_meta(pdf), settings))


class FuseTest(unittest.TestCase):
    def test_image_weight_lifts_figure(self):
        named = [
            ("fts", [hit("p1-c000", 1, "fts"), hit("p2-c000", 2, "fts"), hit("p3-c000", 3, "fts")]),
            ("text_dense", [hit("p2-c000", 2, "text_dense"), hit("p1-c000", 1, "text_dense")]),
            ("image_dense", [hit("p9-img", 9, "image_dense", "image")]),
        ]
        weighted = _fuse_hits(named, limit=5, k=60, weights={"fts": 1, "text_dense": 1, "image_dense": 2})
        self.assertEqual(weighted[0].hit.chunk_id, "p9-img")  # weight 2 -> top

        flat = _fuse_hits(named, limit=5, k=60, weights={"fts": 1, "text_dense": 1, "image_dense": 1})
        self.assertNotIn("p9-img", {f.hit.chunk_id for f in flat[:2]})  # weight 1 regresses it

    def test_per_page_cap(self):
        named = [
            ("fts", [hit("p5-c000", 5, "fts"), hit("p5-c001", 5, "fts")]),
            ("image_dense", [hit("p5-img", 5, "image_dense", "image")]),
        ]
        capped = _fuse_hits(named, limit=10, k=60, weights={}, max_units_per_page=1)
        self.assertEqual(sum(1 for f in capped if f.hit.page == 5), 1)


class ServiceTrapTest(unittest.TestCase):
    def _service(self, root):
        def source(_p):
            yield rp(1, "vapor pressure deficit controls transpiration " * 4, ink=0.4, image=img())
            yield rp(2, "", ink=0.30, image=img(61, 81))  # text-less figure -> kept:visual
            yield rp(3, "", ink=0.001, color=0.0, image=img())  # blank -> dropped

        return BlackBookService(make_settings(root), page_source=source)

    def test_textless_figure_indexed_and_retrievable_blank_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")
            svc = self._service(root)
            result = svc.ensure_index(force=True)
            self.assertEqual(result["image_unit_count"], 2)  # pages 1 and 2
            self.assertEqual(result["pages_dropped"], 1)  # page 3

            with svc.index.connect() as c:
                page2_chunks = c.execute("SELECT COUNT(*) FROM chunks WHERE page=2").fetchone()[0]
                reason2 = c.execute("SELECT reason FROM pages WHERE page=2").fetchone()[0]
                reason3 = c.execute("SELECT reason FROM pages WHERE page=3").fetchone()[0]
                page3_img = c.execute("SELECT COUNT(*) FROM page_embeddings WHERE page=3").fetchone()[0]
            self.assertEqual(page2_chunks, 0)  # text-less page yields ZERO chunks
            self.assertEqual(reason2, "kept:visual")
            self.assertEqual(reason3, "dropped:blank")
            self.assertEqual(page3_img, 0)  # blank page not embedded

            provider = StubDenseProvider(dim=svc.settings.voyage_output_dim)
            v2 = provider.embed_image_units([ImageUnit(2, "", img(61, 81))])[0]
            iv = svc.index.search_page_vector(v2, limit=5)
            self.assertEqual(iv[0].page, 2)  # stored + searchable in the image space

            self.assertTrue(svc.read_citation("p0002-img")["found"])

    def test_mm_element_never_sends_empty_string(self):
        i = img()
        self.assertEqual(mm_element(ImageUnit(2, "", i)), [i])  # [image] only, no ""
        self.assertEqual(len(mm_element(ImageUnit(2, "  ", i))), 1)  # whitespace -> [image]
        self.assertEqual(mm_element(ImageUnit(1, "caption", i)), ["caption", i])

    def test_image_query_path_surfaces_captioned_figure(self):
        # Exercises the REAL query path (embed_image_query -> search_page_vector -> fuse),
        # not a circular self-vector round-trip: a caption-bearing figure must outrank an
        # unrelated page for a query that shares its caption tokens.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "reverse osmosis water treatment ppm " * 10, ink=0.4, image=img())
                yield rp(2, "seedling clone morphology comparison", ink=0.3, image=img(61, 81))

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            res = svc.search("seedling morphology comparison", mode="image")
            self.assertFalse(res["abstain"])
            self.assertEqual(res["results"][0]["unit_type"], "image")
            self.assertEqual(res["results"][0]["page"], 2)


class GateTest(unittest.TestCase):
    def test_image_gate_filters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

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
            (root / "CCI Black Book.pdf").write_text("x")
            good = BlackBookService(make_settings(root), page_source=self._source())
            good.ensure_index(force=True)
            old_chunks = good.index.status()["chunk_count"]

            failing = StubDenseProvider(dim=64, fail_on=("embed_text_documents", 1))
            svc = BlackBookService(make_settings(root), provider=failing, page_source=self._source())
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)
            self.assertTrue(good.index.status()["ready"])  # prior index intact
            self.assertEqual(good.index.status()["chunk_count"], old_chunks)

    def test_image_embed_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")
            failing = StubDenseProvider(dim=64, fail_on=("embed_image_units", 1))
            svc = BlackBookService(make_settings(root), provider=failing, page_source=self._source())
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)
            self.assertFalse(svc.index.status()["ready"])

    def test_extraction_tripwire_no_text_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "", ink=0.3, image=img())
                yield rp(2, "", ink=0.3, image=img())

            svc = BlackBookService(make_settings(root, min_expected_median_chars=200), page_source=source)
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)  # OCR read nothing anywhere

    def test_extraction_tripwire_sparse_median(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "abc", ink=0.3, image=img())  # text-bearing but tiny
                yield rp(2, "de", ink=0.3, image=img())

            svc = BlackBookService(make_settings(root, min_expected_median_chars=200), page_source=source)
            with self.assertRaises(IngestFailed):
                svc.ensure_index(force=True)

    def test_extraction_tripwire_ignores_empty_figure_pages(self):
        # The regression the fix targets: many empty figure pages must NOT drag an
        # all-pages median under the threshold and false-abort a healthy book.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "x" * 400, ink=0.4, image=img())  # dense text page
                yield rp(2, "", ink=0.3, image=img())  # empty figure (excluded from median)
                yield rp(3, "", ink=0.3, image=img())  # empty figure

            svc = BlackBookService(make_settings(root, min_expected_median_chars=200), page_source=source)
            self.assertTrue(svc.ensure_index(force=True)["rebuilt"])  # median over text pages = 400


class RetentionTest(unittest.TestCase):
    def test_voyage_backend_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "text " * 40, ink=0.3, image=img())

            settings = make_settings(root, embedding_backend="voyage", voyage_retention_confirmed=False)
            boom = StubDenseProvider(dim=64, fail_on=("embed_text_documents", 1))  # would explode if reached
            svc = BlackBookService(settings, provider=boom, page_source=source)
            with self.assertRaises(IngestFailed) as ctx:
                svc.ensure_index(force=True)
            self.assertIn("retention", str(ctx.exception).lower())

    def test_stub_backend_proceeds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

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
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")
            res = BlackBookService(make_settings(root)).search("anything", mode="hybrid")
            self.assertTrue(res["abstain"])
            self.assertEqual(res["results"], [])

    def test_query_embed_failure_degrades_to_fts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "vapor pressure deficit transpiration canopy " * 4, ink=0.4, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            svc._provider = _QueryFailProvider()
            res = svc.search("transpiration deficit", mode="hybrid")
            self.assertFalse(res["abstain"])  # FTS still answers
            self.assertTrue(any("unavailable" in n for n in res["confidence_notes"]))


class FingerprintTest(unittest.TestCase):
    def _ready_status(self, settings, pdf):
        return {
            "ready": True,
            "metadata": {"source": _src_meta(pdf), "fingerprint": settings_fingerprint(settings)},
        }

    def test_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / "CCI Black Book.pdf"
            pdf.write_text("x")
            base = make_settings(root)
            status = self._ready_status(base, pdf)
            self.assertTrue(_index_current(status, _src_meta(pdf), base))

            for change in (
                dict(voyage_text_model="voyage-context-3"),
                dict(voyage_output_dim=256),
                dict(render_dpi=150),
                dict(chunk_chars=1000),
                dict(blank_min_chars=50),
                dict(force_keep_pages=frozenset({7})),
                dict(embedding_backend="voyage"),
            ):
                self.assertFalse(
                    _index_current(status, _src_meta(pdf), replace(base, **change)),
                    f"expected invalidation for {change}",
                )

            self.assertFalse(_index_current(status, {"path": "other", "size": 9, "mtime_ns": 9}, base))


class CitationStatusTest(unittest.TestCase):
    def test_image_citation_and_status_no_leak(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "vpd text here " * 20, ink=0.4, image=img())
                yield rp(2, "", ink=0.3, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)

            cite = svc.read_citation("p0002-img")
            self.assertTrue(cite["found"])
            self.assertEqual(cite["unit_type"], "image")
            self.assertIn("figure/page image", cite["citation"])
            self.assertNotIn("thumbnail_png_base64", cite)  # thumbnails disabled

            st = svc.status()
            self.assertIn("voyage_configured", st)
            self.assertIn("filter", st["index"]["metadata"])
            self.assertEqual(st["embedding"]["backend"], "stub")
            self.assertNotIn("VOYAGE_API_KEY", json.dumps(st))

    def test_parse_image_unit_id(self):
        self.assertEqual(parse_image_unit_id("p0042-img"), 42)
        self.assertEqual(parse_image_unit_id("p12-img"), 12)
        self.assertIsNone(parse_image_unit_id("p0042-c001"))
        self.assertIsNone(parse_image_unit_id("garbage"))


class OfflineGuaranteeTest(unittest.TestCase):
    def test_no_voyage_import_offline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CCI Black Book.pdf").write_text("x")

            def source(_p):
                yield rp(1, "text here " * 30, ink=0.4, image=img())

            svc = BlackBookService(make_settings(root), page_source=source)
            svc.ensure_index(force=True)
            svc.search("text", mode="hybrid")
            self.assertNotIn("voyageai", sys.modules)

    def test_build_dense_provider_rejects_unknown_backend(self):
        with tempfile.TemporaryDirectory() as d:
            settings = make_settings(Path(d), embedding_backend="hash")
            with self.assertRaises(ValueError):
                build_dense_provider(settings)


if __name__ == "__main__":
    unittest.main()
