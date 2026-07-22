from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.benchmark_cooperative_resource_lease import (
    ANSWER_MARKER,
    DEFAULT_ARTIFACT,
    SENSITIVE_ARTIFACT_KEYS,
    TASK_MARKER,
    render_report,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def _artifact_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested for item in value.values() for nested in _artifact_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _artifact_keys(item)}
    return set()


class CooperativeResourceLeaseBenchmarkTests(unittest.TestCase):
    def test_report_is_deterministic_and_machine_readable(self) -> None:
        first = run_benchmark()
        second = run_benchmark()

        self.assertEqual(first, second)
        self.assertEqual(render_report(first), render_report(second))
        self.assertTrue(first["contract_checks_passed"])
        self.assertEqual(first["pass_count"], first["check_count"])
        self.assertEqual(first["contract"], "cooperative_resource_lease")
        self.assertEqual(
            json.loads(json.dumps(first, allow_nan=False, sort_keys=True)),
            first,
        )

    def test_checked_in_artifact_matches_canonical_bytes(self) -> None:
        self.assertEqual(
            DEFAULT_ARTIFACT.read_bytes(),
            render_report(run_benchmark()).encode("utf-8"),
        )

    def test_capacity_release_and_sticky_unknown_contracts(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        system = scenarios["shared_system_capacity"]
        self.assertEqual(
            system["admission_statuses"],
            ["acquired", "acquired", "denied"],
        )
        self.assertEqual(system["denied_path"]["simulated_invocations"], 0)
        self.assertEqual(
            system["reserve_accounting"]["applied_pool_reserve_bytes"],
            300,
        )
        self.assertEqual(system["replacement_status"], "acquired")

        unknown = scenarios["armed_attempted_unknown"]
        self.assertTrue(unknown["delivery_armed"])
        self.assertEqual(unknown["ambiguous_release_status"], "unknown_blocking")
        self.assertEqual(
            unknown["subsequent_admission_status"],
            "unknown_blocking",
        )
        self.assertFalse(unknown["subsequent_handle_issued"])

    def test_unified_and_discrete_fixture_pool_contracts(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        unified = scenarios["unified_pool"]
        self.assertEqual(unified["claim_pools"], ["system", "unified"])
        self.assertEqual(unified["armed"], [True, True])
        self.assertEqual(unified["release_statuses"], ["released", "released"])

        discrete = scenarios["discrete_pool_contract_fixture"]
        self.assertEqual(discrete["fixture_scope"], "contract_only")
        self.assertEqual(
            discrete["third_reason_codes"],
            ["accelerator_capacity_insufficient"],
        )
        self.assertEqual(
            discrete["accelerator_accounting"],
            {
                "active_accelerator_claim_bytes": 1_200,
                "applied_accelerator_reserve_bytes": 100,
                "available_accelerator_bytes": 1_300,
                "requested_accelerator_claim_bytes": 600,
            },
        )

    def test_artifact_is_privacy_safe_and_states_authority_limits(self) -> None:
        report = run_benchmark()
        rendered = render_report(report)
        scenario_keys = _artifact_keys(report["scenarios"])

        self.assertNotIn(TASK_MARKER, rendered)
        self.assertNotIn(ANSWER_MARKER, rendered)
        self.assertTrue(SENSITIVE_ARTIFACT_KEYS.isdisjoint(scenario_keys))
        self.assertIn("cooperative_participants_only", report["limits"])
        self.assertIn(
            "does_not_reserve_ram_or_vram_at_operating_system_level",
            report["limits"],
        )
        self.assertIn(
            "does_not_start_load_unload_stop_or_evict_models",
            report["limits"],
        )

    def test_check_mode_is_byte_exact_and_never_rewrites_mismatch(self) -> None:
        checked = subprocess.run(
            [
                sys.executable,
                "experiments/benchmark_cooperative_resource_lease.py",
                "--check",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)

        with tempfile.TemporaryDirectory() as temp:
            stale = Path(temp) / "stale.json"
            original = b'{"stale":true}\n'
            stale.write_bytes(original)
            mismatch = subprocess.run(
                [
                    sys.executable,
                    "experiments/benchmark_cooperative_resource_lease.py",
                    "--check",
                    "--out",
                    str(stale),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("artifact is out of date", mismatch.stderr)
            self.assertEqual(stale.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
