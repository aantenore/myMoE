from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.benchmark_speculative_cell_qualifier import (
    PRIVATE_OUTPUT_MARKER,
    render_report,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class SpeculativeCellBenchmarkTests(unittest.TestCase):
    def test_report_is_deterministic_and_matches_checked_artifact(self) -> None:
        first = run_benchmark()
        second = run_benchmark()
        rendered = render_report(first)

        self.assertEqual(first, second)
        self.assertTrue(first["contract_checks_passed"])
        self.assertEqual(first["pass_count"], first["check_count"])
        self.assertEqual(json.loads(rendered), first)
        self.assertEqual(
            (ROOT / "outputs" / "speculative-cell-qualifier-contract.json").read_text(
                encoding="utf-8"
            ),
            rendered,
        )

    def test_artifact_is_payload_free_and_non_authorizing(self) -> None:
        report = run_benchmark()
        rendered = render_report(report)
        qualified = report["scenarios"]["qualified_cell"]

        self.assertNotIn(PRIVATE_OUTPUT_MARKER, rendered)
        self.assertEqual(qualified["decision"], "qualified")
        self.assertFalse(qualified["activation_authorized"])
        self.assertIn(
            "does_not_establish_live_speedup_or_memory_savings",
            report["limits"],
        )


if __name__ == "__main__":
    unittest.main()
