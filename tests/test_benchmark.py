import unittest

from benchmarks.retrieval_benchmark import (
    load_dataset,
    render_markdown,
    retrieval_metrics,
    run_benchmark,
)


class BenchmarkTests(unittest.TestCase):
    def test_dataset_is_valid_and_offline_benchmark_runs(self) -> None:
        result = run_benchmark(load_dataset(), live=False, limit=2)

        self.assertEqual(result["queries"], 2)
        self.assertIn("BM25", result["summary"])
        self.assertNotIn("Híbrido RRF", result["summary"])

    def test_metrics_find_first_relevant_document(self) -> None:
        metrics = retrieval_metrics(
            ["irrelevant", "memory-001", "other"],
            ["memory-001"],
        )

        self.assertEqual(metrics["hit_at_1"], 0.0)
        self.assertEqual(metrics["hit_at_3"], 1.0)
        self.assertEqual(metrics["mrr_at_3"], 0.5)

    def test_offline_report_does_not_claim_live_methods(self) -> None:
        report = render_markdown(run_benchmark(load_dataset(), live=False, limit=1))

        self.assertIn("| BM25 |", report)
        self.assertIn("requieren `--live`", report)
        self.assertNotIn("Híbrido RRF |", report)


if __name__ == "__main__":
    unittest.main()
