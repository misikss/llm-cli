import unittest

from benchmarks.faithfulness_benchmark import evaluate_annotations, render_markdown


class FaithfulnessBenchmarkTests(unittest.TestCase):
    def test_evaluates_claim_support_and_citations(self) -> None:
        metrics = evaluate_annotations([
            {
                "id": "q01",
                "answer": "respuesta",
                "gold_source_ids": ["memory-001"],
                "claims": [
                    {
                        "text": "claim supported",
                        "supported": True,
                        "source_ids": ["memory-001"],
                    },
                    {
                        "text": "claim unsupported",
                        "supported": False,
                        "source_ids": ["memory-999"],
                    },
                ],
            },
        ])

        self.assertEqual(metrics["claim_support_rate"], 0.5)
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertIn("no se invoca ningún LLM", render_markdown(metrics))


if __name__ == "__main__":
    unittest.main()
