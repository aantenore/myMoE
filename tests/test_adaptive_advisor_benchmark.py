from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.benchmark_adaptive_cell_advisor import run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveCellAdvisorBenchmarkTests(unittest.TestCase):
    def test_contract_report_is_deterministic_and_machine_readable(self) -> None:
        first = run_benchmark()
        second = run_benchmark()

        self.assertEqual(first, second)
        self.assertTrue(first["contract_checks_passed"])
        self.assertEqual(
            json.loads(json.dumps(first, allow_nan=False, sort_keys=True)),
            first,
        )

    def test_checked_in_ci_artifact_matches_the_canonical_report(self) -> None:
        expected = (
            json.dumps(
                run_benchmark(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        self.assertEqual(
            (ROOT / "outputs" / "adaptive-cell-advisor-contract.json").read_text(
                encoding="utf-8"
            ),
            expected,
        )

    def test_profiles_can_select_different_eligible_cells(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        self.assertEqual(
            scenarios["efficiency_profile"]["selected_cell_id"],
            "fast-small-cell",
        )
        self.assertEqual(
            scenarios["quality_profile"]["selected_cell_id"],
            "high-quality-cell",
        )
        self.assertEqual(scenarios["efficiency_profile"]["status"], "recommended")
        self.assertEqual(scenarios["quality_profile"]["status"], "recommended")

    def test_staleness_and_resource_pressure_fail_closed(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        self.assertEqual(scenarios["stale_snapshot"]["status"], "abstained")
        self.assertIn(
            "snapshot_stale",
            scenarios["stale_snapshot"]["reason_codes"],
        )
        self.assertEqual(scenarios["resource_pressure"]["status"], "abstained")
        self.assertTrue(
            all(
                "host_memory_headroom_insufficient" in reasons
                for reasons in scenarios["resource_pressure"][
                    "candidate_rejections"
                ].values()
            )
        )

    def test_intent_family_never_collapses_exact_lineage_or_reuses_a_response(
        self,
    ) -> None:
        report = run_benchmark()
        lineage = report["scenarios"]["paraphrase_lineage"]

        self.assertNotEqual(
            lineage["first_exact_request_fingerprint"],
            lineage["second_exact_request_fingerprint"],
        )
        self.assertNotEqual(
            lineage["first_advice_sha256"],
            lineage["second_advice_sha256"],
        )
        self.assertFalse(lineage["response_cache_lookup_performed"])
        self.assertFalse(lineage["response_reuse_authorized"])
        self.assertIn("does_not_measure_model_quality", report["limits"])
        self.assertIn("does_not_measure_real_latency_or_memory", report["limits"])


if __name__ == "__main__":
    unittest.main()
