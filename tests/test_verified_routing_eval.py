from __future__ import annotations

from itertools import combinations
from pathlib import Path
import unittest

from experiments.eval_verified_routing import (
    _load_fixture,
    evaluate_cases,
    expand_cases,
)
from local_moe.verified_routing_contracts import VerifiedRoutingError


FIXTURE = "tests/fixtures/verified-routing-eval.json"


class VerifiedRoutingEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _load_fixture(Path(FIXTURE))
        self.cases = expand_cases(self.fixture)

    def test_expands_exactly_sixty_four_pairwise_covered_cases(self) -> None:
        dimensions = ("capability", "difficulty", "language", "context")

        self.assertEqual(len(self.cases), 64)
        self.assertEqual(len({case["id"] for case in self.cases}), 64)
        for left, right in combinations(dimensions, 2):
            with self.subTest(left=left, right=right):
                pairs = {(case[left], case[right]) for case in self.cases}
                self.assertEqual(len(pairs), 16)

    def test_shadow_fixture_validates_metric_formulas_without_empirical_claims(self) -> None:
        report = evaluate_cases(self.cases, calibration_bins=10)
        local = report["strategies"]["local_only"]["overall"]
        baseline = report["strategies"]["current_baseline"]["overall"]
        shadow = report["strategies"]["verified_shadow"]["overall"]

        self.assertEqual(local["false_local"], 1.0)
        self.assertEqual(baseline["premium_calls"], 32)
        self.assertEqual(shadow["premium_calls"], 16)
        self.assertEqual(shadow["verified_success"], 0.75)
        self.assertEqual(shadow["escalation_precision"], 1.0)
        self.assertEqual(shadow["escalation_recall"], 1.0)
        self.assertLess(shadow["brier_score"], baseline["brier_score"])

    def test_rejects_a_non_covering_case_count(self) -> None:
        with self.assertRaisesRegex(VerifiedRoutingError, "64 cases"):
            evaluate_cases(self.cases[:-1], calibration_bins=10)


if __name__ == "__main__":
    unittest.main()
