import unittest

from rag import (
    BM25Index,
    RetrievalDocument,
    hybrid_search,
    reciprocal_rank_fusion,
    tokenize_text,
)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            RetrievalDocument("memory-1", "Python y memoria episódica"),
            RetrievalDocument("memory-2", "Clima actual en Lima"),
            RetrievalDocument("memory-3", "Gemini-2.5-flash y RAG híbrido"),
        ]

    def test_tokenizer_normalizes_spanish_and_keeps_technical_tokens(self) -> None:
        tokens = tokenize_text("¿Qué hace Gemini-2.5-flash con IA?")

        self.assertIn("que", tokens)
        self.assertIn("gemini-2.5-flash", tokens)
        self.assertIn("ia", tokens)

    def test_bm25_returns_stable_document_ids(self) -> None:
        index = BM25Index(self.documents)

        results = index.search("memoria Python")

        self.assertEqual(results[0], "memory-1")

    def test_rrf_combines_rankings_by_id(self) -> None:
        results = reciprocal_rank_fusion(
            vector_rankings=["memory-1", "memory-2"],
            bm25_rankings=["memory-3", "memory-1"],
        )

        self.assertEqual(results[0][0], "memory-1")

    def test_hybrid_search_returns_documents_not_duplicate_text_keys(self) -> None:
        index = BM25Index(self.documents)

        results = hybrid_search(
            query_text="Python",
            keywords=[],
            corpus_docs=self.documents,
            vector_results=[self.documents[0], self.documents[1]],
            top_k=2,
            bm25_index=index,
        )

        self.assertEqual(results[0].id, "memory-1")
        self.assertEqual(len({document.id for document in results}), len(results))


if __name__ == "__main__":
    unittest.main()
