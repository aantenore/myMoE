from __future__ import annotations

import unittest

from local_moe.config import load_config
from local_moe.orchestrator import LocalMoE


class OrchestratorTests(unittest.TestCase):
    def test_preserves_correlation_id(self) -> None:
        moe = LocalMoE(load_config("tests/fixtures/moe.synthetic.json"))
        response = moe.generate("Write Python code", correlation_id="case-1")
        self.assertEqual(response.correlation_id, "case-1")
        self.assertEqual(response.results[0].correlation_id, "case-1")

    def test_parallel_compare_calls_top_k(self) -> None:
        raw = load_config("tests/fixtures/moe.synthetic.json")
        config = type(raw)(
            routing=type(raw.routing)(
                top_k=2,
                fallback_order=(),
                aggregation="compare",
            ),
            experts=raw.experts,
            rules=raw.rules,
        )
        moe = LocalMoE(config)
        response = moe.generate("Design Python architecture", correlation_id="case-2")
        self.assertEqual(len(response.results), 2)
        self.assertIsNotNone(response.disagreement)
        self.assertIn("Deterministic disagreement report", response.content)
        self.assertEqual(len(response.disagreement.pairwise_overlaps), 1)
        self.assertIn(response.disagreement.status, {"agreement_likely", "review_recommended"})


if __name__ == "__main__":
    unittest.main()
