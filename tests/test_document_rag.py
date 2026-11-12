import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_rag import (
    DIRECT_MAX_BYTES,
    choose_document_plan,
    chunk_text,
    chunk_document_sections,
    extract_document_sections,
    DocumentSection,
)


class DocumentStrategyTests(unittest.TestCase):
    def test_small_document_starts_direct_and_switches_to_rag_after_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "document_registry.json"
            document = root / "ventas.txt"
            document.write_text("Venta V-001: 100", encoding="utf-8")

            with patch("document_rag.REGISTRY_PATH", registry), patch("document_rag.LOG_DIR", root):
                first = choose_document_plan(document, user_id="user-1")
                second = choose_document_plan(document, user_id="user-1")
                third = choose_document_plan(document, user_id="user-1")

            self.assertEqual(first.mode, "multimodal_direct")
            self.assertEqual(second.mode, "multimodal_direct")
            self.assertEqual(third.mode, "document_rag")
            self.assertEqual(third.previous_queries, 2)

    def test_large_extractable_document_uses_rag_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "large.txt"
            document.write_text("ventas " * (DIRECT_MAX_BYTES // 6), encoding="utf-8")
            with patch("document_rag.REGISTRY_PATH", Path(directory) / "registry.json"), patch(
                "document_rag.LOG_DIR", Path(directory)
            ):
                plan = choose_document_plan(document, user_id="user-1")
            self.assertEqual(plan.mode, "document_rag")

    def test_multimodal_only_file_stays_direct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "captura.png"
            document.write_bytes(b"fake image")
            with patch("document_rag.REGISTRY_PATH", Path(directory) / "registry.json"), patch(
                "document_rag.LOG_DIR", Path(directory)
            ):
                plan = choose_document_plan(document, user_id="user-1")
            self.assertEqual(plan.mode, "multimodal_direct")

    def test_chunk_text_keeps_overlap(self) -> None:
        chunks = chunk_text("abcdefghijklmnop", size=10, overlap=3)
        self.assertEqual(chunks, ["abcdefghij", "hijklmnop"])

    def test_structured_chunks_keep_document_locator(self) -> None:
        chunks = chunk_document_sections(
            [DocumentSection(text="venta V-003 por 2392", locator="Hoja Ventas · filas 1-2")],
            doc_id="abc123",
        )

        self.assertEqual(chunks[0].id, "document_chunk:abc123:0")
        self.assertEqual(chunks[0].locator, "Hoja Ventas · filas 1-2")

    def test_csv_extraction_returns_row_locator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "ventas.csv"
            document.write_text("id,total\nV-001,100\nV-002,200\n", encoding="utf-8")
            sections = extract_document_sections(document)

        self.assertEqual(len(sections), 1)
        self.assertIn("V-002", sections[0].text)
        self.assertIn("filas 1-3", sections[0].locator)
if __name__ == "__main__":
    unittest.main()
