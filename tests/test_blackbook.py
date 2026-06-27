from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cci_blackbook.auth import is_authorized
from cci_blackbook.chunking import PageText, chunk_page
from cci_blackbook.embeddings import HashEmbeddingProvider
from cci_blackbook.service import BlackBookService
from cci_blackbook.settings import Settings


class ChunkingTest(unittest.TestCase):
    def test_chunk_page_preserves_page_and_advances(self):
        text = "alpha beta gamma. " * 80

        chunks = chunk_page(7, text, chunk_chars=120, overlap_chars=30)

        self.assertGreater(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "p0007-c000")
        self.assertTrue(all(chunk.page == 7 for chunk in chunks))
        starts = [chunk.char_start for chunk in chunks]
        self.assertEqual(starts, sorted(set(starts)))


class AuthTest(unittest.TestCase):
    def test_bearer_auth_uses_exact_token(self):
        self.assertTrue(is_authorized("Bearer secret-token", "secret-token"))
        self.assertFalse(is_authorized("Bearer wrong", "secret-token"))
        self.assertFalse(is_authorized("Basic secret-token", "secret-token"))
        self.assertFalse(is_authorized("", "secret-token"))


class BlackBookServiceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.source_pdf = root / "CCI Black Book.pdf"
        self.source_pdf.write_text("test source", encoding="utf-8")
        self.settings = Settings(
            source_pdf=self.source_pdf,
            index_dir=root / "index",
            cache_dir=root / "cache",
            sqlite_path=root / "index" / "blackbook.sqlite3",
            embedding_backend="hash",
            embedding_model="hashing-vectorizer",
            openvino_device="GPU",
            render_device=root / "renderD129",
            chunk_chars=300,
            chunk_overlap_chars=50,
            min_vector_score=0.15,
            embedding_batch_size=4,
            host="127.0.0.1",
            port=8000,
            log_level="info",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_search_returns_bounded_cited_evidence(self):
        service = self._indexed_service()

        result = service.search("tomato vapor pressure deficit", limit=3, mode="hybrid")

        self.assertFalse(result["abstain"])
        self.assertEqual(result["results"][0]["page"], 12)
        self.assertIn("CCI Black Book page 12", result["results"][0]["citation"])
        self.assertLessEqual(len(result["results"][0]["excerpt"]), 705)

    def test_read_citation_is_bounded(self):
        service = self._indexed_service(long_text=True)

        result = service.read_citation("p0012-c000")

        self.assertTrue(result["found"])
        self.assertTrue(result["bounded"])
        self.assertLessEqual(len(result["text"]), 2500)

    def test_search_abstains_when_index_source_is_missing(self):
        service = BlackBookService(self.settings, HashEmbeddingProvider())
        self.source_pdf.unlink()

        result = service.search("tomato", limit=3)

        self.assertTrue(result["abstain"])
        self.assertIn("source PDF missing", result["confidence_notes"][0])

    def _indexed_service(self, *, long_text: bool = False) -> BlackBookService:
        settings = (
            replace(self.settings, chunk_chars=4000, chunk_overlap_chars=100)
            if long_text
            else self.settings
        )
        service = BlackBookService(settings, HashEmbeddingProvider())
        stat = self.source_pdf.stat()
        source_metadata = {
            "path": str(self.source_pdf),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        text = (
            "Tomato crops should be managed with vapor pressure deficit in mind. "
            "Leaf temperature, humidity, and transpiration all affect irrigation decisions."
        )
        if long_text:
            text = text + " Nutrient and canopy note." * 300
        service.rebuild_from_pages(
            [
                PageText(page=12, text=text),
                PageText(
                    page=40,
                    text="Lighting schedules and photoperiod notes for flowering crops.",
                ),
            ],
            source_metadata=source_metadata,
        )
        return service


if __name__ == "__main__":
    unittest.main()
