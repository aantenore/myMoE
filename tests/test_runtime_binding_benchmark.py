from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.benchmark_runtime_binding import _Fixture, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class RuntimeBindingBenchmarkTests(unittest.TestCase):
    def test_fixture_json_is_platform_neutral_binary_utf8(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                Path,
                "write_text",
                side_effect=AssertionError(
                    "fixture JSON must bypass newline translation"
                ),
            ),
        ):
            fixture = _Fixture(Path(temporary))
            self.assertEqual(
                fixture.request_path.read_bytes(),
                json.dumps(fixture.request, indent=2, sort_keys=True).encode("utf-8"),
            )
            self.assertEqual(
                fixture.config_path.read_bytes(),
                json.dumps(fixture.config, indent=2, sort_keys=True).encode("utf-8"),
            )
            for path in (
                fixture.request_path,
                fixture.config_path,
                fixture.catalog_path,
            ):
                value = path.read_bytes()
                self.assertNotIn(b"\r\n", value)
                json.loads(value)

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
            (ROOT / "outputs" / "runtime-binding-contract.json").read_text(
                encoding="utf-8"
            ),
            expected,
        )

    def test_unpinned_verified_and_drift_scenarios_fail_closed(self) -> None:
        scenarios = run_benchmark()["scenarios"]

        self.assertEqual(scenarios["first_run_unpinned"]["status"], "abstained")
        self.assertEqual(
            set(scenarios["first_run_unpinned"]["reason_codes"]),
            {
                "harness_identity_unknown",
                "model_identity_unknown",
                "runtime_identity_unknown",
                "tool_contract_identity_unknown",
            },
        )
        self.assertTrue(
            all(
                len(value) == 64
                for value in scenarios["first_run_unpinned"][
                    "observed_identities"
                ].values()
            )
        )
        self.assertEqual(
            scenarios["verified_after_expected_pinning"]["status"],
            "verified",
        )
        self.assertEqual(
            scenarios["model_content_drift"]["reason_codes"],
            ["model_identity_mismatch"],
        )
        self.assertEqual(
            scenarios["runtime_content_drift"]["reason_codes"],
            ["runtime_identity_mismatch"],
        )

    def test_reordering_and_clock_do_not_ambiguate_the_binding(self) -> None:
        scenarios = run_benchmark()["scenarios"]
        reorder = scenarios["expert_reorder"]
        clock = scenarios["fresh_receipt"]

        self.assertTrue(reorder["selected_binding_preserved"])
        self.assertEqual(
            reorder["before"]["launch_plan_sha256"],
            reorder["after"]["launch_plan_sha256"],
        )
        self.assertEqual(
            reorder["before"]["expert_config_sha256"],
            reorder["after"]["expert_config_sha256"],
        )
        self.assertTrue(clock["manifest_clock_stable"])
        self.assertTrue(clock["receipt_changed"])
        self.assertNotEqual(clock["first_captured_at"], clock["second_captured_at"])

    def test_every_inspection_is_static_and_hash_reads_are_bounded(self) -> None:
        report = run_benchmark()
        fixture = report["fixture"]

        self.assertTrue(
            report["criteria"]["every_inspection_is_static_and_non_authorizing"]
        )
        self.assertTrue(
            report["criteria"][
                "every_receipt_reports_zero_network_process_and_model_use"
            ]
        )
        self.assertTrue(
            report["criteria"]["artifact_hashing_uses_bounded_streaming_reads"]
        )
        self.assertGreater(
            fixture["model_artifact_bytes"], fixture["stream_read_bound_bytes"]
        )
        self.assertLessEqual(
            fixture["observed_max_read_request_bytes"],
            fixture["stream_read_bound_bytes"],
        )
        self.assertGreater(fixture["observed_model_read_request_count"], 1)
        self.assertLessEqual(
            fixture["observed_model_max_read_request_bytes"],
            fixture["stream_read_bound_bytes"],
        )
        self.assertTrue(fixture["whole_file_path_reads_blocked"])
        self.assertEqual(
            fixture["guarded_side_effect_surfaces"],
            [
                "model_server_start",
                "network_socket",
                "process_spawn",
                "url_fetch",
            ],
        )
        self.assertIn("does_not_measure_model_quality", report["limits"])
        self.assertIn("does_not_benchmark_hash_performance", report["limits"])
        self.assertIn(
            "does_not_authorize_or_apply_execution",
            report["limits"],
        )


if __name__ == "__main__":
    unittest.main()
