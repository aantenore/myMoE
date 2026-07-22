from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.benchmark_bound_cell_run import (
    FAILED_RESPONSE_BODY,
    RESPONSE_BODY,
    TASK_BODY,
    render_report,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class BoundCellRunBenchmarkTests(unittest.TestCase):
    def test_report_is_deterministic_and_byte_identical(self) -> None:
        first = run_benchmark()
        second = run_benchmark()

        self.assertEqual(first, second)
        self.assertEqual(render_report(first), render_report(second))
        self.assertTrue(first["contract_checks_passed"])
        self.assertEqual(
            json.loads(json.dumps(first, allow_nan=False, sort_keys=True)),
            first,
        )

    def test_checked_in_artifact_matches_the_canonical_report(self) -> None:
        self.assertEqual(
            (ROOT / "outputs" / "bound-cell-run-contract.json").read_text(
                encoding="utf-8"
            ),
            render_report(run_benchmark()),
        )

    def test_state_transitions_and_request_counts_fail_closed(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        self.assertEqual(scenarios["completed"]["receipt"]["status"], "completed")
        self.assertEqual(
            scenarios["completed"]["envelope"]["contract"],
            "BoundCellRunEnvelopeV2",
        )
        self.assertEqual(
            scenarios["completed"]["envelope"]["run_receipt"],
            scenarios["completed"]["receipt"],
        )
        blocked = scenarios["precondition_blocked"]
        self.assertEqual(blocked["receipt"]["status"], "blocked")
        self.assertEqual(blocked["transport_counters"]["probe_requests"], 0)
        self.assertEqual(blocked["transport_counters"]["post_requests"], 0)
        failed = scenarios["transport_failure"]
        self.assertEqual(failed["receipt"]["status"], "failed")
        self.assertEqual(failed["receipt"]["invocation_attempts"], 1)
        self.assertEqual(failed["transport_counters"]["retries"], 0)
        self.assertEqual(
            scenarios["post_binding_drift"]["receipt"]["status"],
            "invalidated",
        )
        self.assertEqual(
            scenarios["post_model_identity_drift"]["receipt"]["status"],
            "invalidated",
        )

    def test_public_artifact_contains_no_bodies_or_process_authority(self) -> None:
        report = run_benchmark()
        rendered = render_report(report)

        for marker in (TASK_BODY, RESPONSE_BODY, FAILED_RESPONSE_BODY):
            self.assertNotIn(marker, rendered)
        self.assertNotIn("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", rendered)
        for scenario in report["scenarios"].values():
            receipt = scenario["receipt"]
            envelope = scenario["envelope"]
            self.assertTrue(envelope["cooperative_only"])
            self.assertFalse(envelope["os_memory_reserved"])
            self.assertFalse(envelope["runtime_managed"])
            self.assertEqual(envelope["run_receipt_sha256"], receipt["digest"])
            self.assertFalse(receipt["process_mutations"])
            self.assertEqual(receipt["lifecycle_operations"], 0)
            self.assertFalse(receipt["endpoint_process_identity_verified"])
            self.assertFalse(receipt["semantic_outcome_verified"])
            self.assertFalse(receipt["authorizes_future_execution"])
            self.assertFalse(receipt["remote_egress"])
            self.assertEqual(receipt["retries"], 0)
            self.assertEqual(receipt["tools_invoked"], 0)
            self.assertLessEqual(receipt["invocation_attempts"], 1)

        completed = report["scenarios"]["completed"]["receipt"]
        self.assertIsNotNone(completed["pre_binding_request_sha256"])
        self.assertEqual(
            completed["pre_binding_request_sha256"],
            completed["post_binding_request_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
