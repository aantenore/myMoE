from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.benchmark_cell_execution_gate import run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class CellExecutionGateBenchmarkTests(unittest.TestCase):
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
            (ROOT / "outputs" / "cell-execution-gate-contract.json").read_text(
                encoding="utf-8"
            ),
            expected,
        )

    def test_exact_lineage_and_resource_drift_fail_closed(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        self.assertEqual(
            scenarios["exact_task_drift"]["reason_codes"],
            ["task_fingerprint_mismatch"],
        )
        self.assertEqual(
            scenarios["catalog_drift"]["reason_codes"],
            ["catalog_drift"],
        )
        self.assertIn(
            "fresh_admission_blocked",
            scenarios["fresh_resource_pressure"]["reason_codes"],
        )

    def test_every_scenario_remains_dry_run_and_non_authorizing(self) -> None:
        report = run_benchmark()

        for scenario in report["scenarios"].values():
            self.assertFalse(scenario["applied"])
            self.assertFalse(scenario["authorizes_execution"])
            self.assertFalse(scenario["network_used"])
            self.assertEqual(scenario["model_invocations"], 0)
        self.assertIn("does_not_measure_model_quality", report["limits"])
        self.assertIn("does_not_authorize_or_apply_execution", report["limits"])


if __name__ == "__main__":
    unittest.main()
