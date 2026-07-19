from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.freeze_verified_routing_plan import main as freeze_plan
from experiments.qualify_verified_routing import main as qualify
from local_moe.paired_execution import paired_execution_harness_sha256
from local_moe.route_policy import load_route_policy
from local_moe.route_scorecard import build_route_scorecard
from local_moe.route_signals import MetadataTaskSignalProvider
from local_moe.verified_routing_contracts import VerifiedRoutingError
from tests.test_route_promotion import (
    ATTESTATION_POLICY_SHA256,
    CONFIG_SHA256,
    EXECUTION_HARNESS_SHA256,
    EXECUTOR_HARNESS_SHA256,
    FIXTURES,
    PRICING,
    RUNNER_SOURCE_SHA256,
    _SyntheticPairedVerifier,
    _synthetic_verify_record,
    _cases,
    _gate_policy,
    _holdout_records,
    _training_records,
)


class RoutePromotionCliTests(unittest.TestCase):
    def test_freezes_and_qualifies_paired_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=400)

            with redirect_stdout(io.StringIO()):
                freeze_exit = self._freeze(paths)
                self._write_holdout(paths, candidate_latency_ms=400)
                qualify_exit = self._qualify(paths)

            self.assertEqual(freeze_exit, 0)
            self.assertEqual(qualify_exit, 0)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "eligible")
            self.assertEqual(manifest["target_mode"], "canary")
            self.assertFalse(manifest["applied"])
            self.assertEqual(plan["pricing_contract"], PRICING.payload())
            self.assertEqual(plan["pricing_sha256"], PRICING.pricing_sha256)

    def test_regression_writes_report_but_not_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=1_200)

            with redirect_stdout(io.StringIO()):
                self._freeze(paths)
                self._write_holdout(paths, candidate_latency_ms=1_200)
                exit_code = self._qualify(paths)

            self.assertEqual(exit_code, 3)
            self.assertTrue(paths["report"].exists())
            self.assertFalse(paths["manifest"].exists())
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ineligible")

    def test_existing_manifest_path_is_rejected_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=1_200)
            with redirect_stdout(io.StringIO()):
                self._freeze(paths)
            paths["manifest"].write_text('{"stale":true}\n', encoding="utf-8")

            with self.assertRaisesRegex(VerifiedRoutingError, "already exists"):
                self._qualify(paths)

            self.assertFalse(paths["report"].exists())

    def test_cases_file_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=400)
            case = _cases(1)[0].payload()
            encoded = json.dumps(case)
            paths["cases"].write_text(
                "["
                + encoded[:-1]
                + ',"task_fingerprint":"'
                + str(case["task_fingerprint"])
                + '"}]',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(VerifiedRoutingError, "Duplicate JSON key"):
                self._freeze(paths)

    def test_freeze_rejects_a_signal_provider_the_runner_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=400)
            cases = json.loads(paths["cases"].read_text(encoding="utf-8"))
            for case in cases:
                case["signal_provider_config_sha256"] = "f" * 64
            paths["cases"].write_text(json.dumps(cases), encoding="utf-8")

            with self.assertRaisesRegex(
                VerifiedRoutingError,
                "executable MetadataTaskSignalProvider",
            ):
                self._freeze(paths)

    def _write_inputs(
        self, root: Path, *, candidate_latency_ms: int
    ) -> dict[str, Path]:
        route_policy = load_route_policy(
            FIXTURES / "verified-routing-policy.json"
        )
        signal_provider_config_sha256 = MetadataTaskSignalProvider().config_sha256
        execution_harness_sha256 = paired_execution_harness_sha256(
            executor_harness_sha256=EXECUTOR_HARNESS_SHA256,
            signal_provider_config_sha256=signal_provider_config_sha256,
        )
        training = _training_records(
            signal_provider_config_sha256=signal_provider_config_sha256,
            execution_harness_sha256=execution_harness_sha256,
        )
        scorecard = build_route_scorecard(
            training,
            minimum_evidence_strength="independent",
            minimum_confidence=0.7,
            generated_at="2026-07-19T00:00:00+00:00",
            ttl_seconds=86_400,
        )
        paths = {
            name: root / filename
            for name, filename in {
                "cases": "cases.json",
                "gate": "gate.json",
                "scorecard": "scorecard.json",
                "training": "training.json",
                "holdout": "holdout.json",
                "plan": "plan.json",
                "report": "report.json",
                "manifest": "manifest.json",
                "pricing": "pricing.json",
                "bridge": "bridge.json",
                "attestation": "attestation.json",
                "exchange": "exchange",
                "cas": "cas",
            }.items()
        }
        paths["cases"].write_text(
            json.dumps(
                [
                    case.payload()
                    for case in _cases(
                        20,
                        signal_provider_config_sha256=(
                            signal_provider_config_sha256
                        ),
                    )
                ]
            ),
            encoding="utf-8",
        )
        paths["gate"].write_text(
            json.dumps(_gate_policy().payload()), encoding="utf-8"
        )
        paths["scorecard"].write_text(
            json.dumps(scorecard.payload()), encoding="utf-8"
        )
        paths["training"].write_text(
            json.dumps({"records": [record.payload() for record in training]}),
            encoding="utf-8",
        )
        paths["pricing"].write_text(
            json.dumps(PRICING.payload()), encoding="utf-8"
        )
        self.assertEqual(route_policy.mode, "shadow")
        return paths

    def _write_holdout(
        self,
        paths: dict[str, Path],
        *,
        candidate_latency_ms: int,
    ) -> None:
        plan_sha256 = str(
            json.loads(paths["plan"].read_text(encoding="utf-8"))[
                "plan_sha256"
            ]
        )
        signal_provider_config_sha256 = MetadataTaskSignalProvider().config_sha256
        execution_harness_sha256 = paired_execution_harness_sha256(
            executor_harness_sha256=EXECUTOR_HARNESS_SHA256,
            signal_provider_config_sha256=signal_provider_config_sha256,
        )
        paths["holdout"].write_text(
            json.dumps(
                {
                    "records": [
                        record.payload()
                        for record in _holdout_records(
                            20,
                            plan_sha256=plan_sha256,
                            candidate_latency_ms=candidate_latency_ms,
                            signal_provider_config_sha256=(
                                signal_provider_config_sha256
                            ),
                            execution_harness_sha256=(
                                execution_harness_sha256
                            ),
                        )
                    ]
                }
            ),
            encoding="utf-8",
        )

    def _freeze_args(self, paths: dict[str, Path]) -> list[str]:
        return [
            "--cases",
            str(paths["cases"]),
            "--route-policy",
            str(FIXTURES / "verified-routing-policy.json"),
            "--scorecard",
            str(paths["scorecard"]),
            "--gate-policy",
            str(paths["gate"]),
            "--created-at",
            "2026-07-19T01:00:00+00:00",
            "--canary-basis-points",
            "100",
            "--manifest-ttl-seconds",
            "3600",
            "--assignment-salt-sha256",
            "4" * 64,
            "--assistant-bridge-config",
            str(paths["bridge"]),
            "--attestation-config",
            str(paths["attestation"]),
            "--attestation-exchange-dir",
            str(paths["exchange"]),
            "--pricing-contract",
            str(paths["pricing"]),
            "--out",
            str(paths["plan"]),
        ]

    def _qualify_args(self, paths: dict[str, Path]) -> list[str]:
        return [
            "--plan",
            str(paths["plan"]),
            "--gate-policy",
            str(paths["gate"]),
            "--route-policy",
            str(FIXTURES / "verified-routing-policy.json"),
            "--scorecard",
            str(paths["scorecard"]),
            "--training-records",
            str(paths["training"]),
            "--holdout-records",
            str(paths["holdout"]),
            "--assistant-bridge-config",
            str(paths["bridge"]),
            "--attestation-config",
            str(paths["attestation"]),
            "--evidence-cas",
            str(paths["cas"]),
            "--evaluated-at",
            "2026-07-19T02:00:00+00:00",
            "--report",
            str(paths["report"]),
            "--manifest",
            str(paths["manifest"]),
        ]

    def _freeze(self, paths: dict[str, Path]) -> int:
        with patch(
            "experiments.freeze_verified_routing_plan._inspect_execution_harness",
            return_value=(
                CONFIG_SHA256,
                ATTESTATION_POLICY_SHA256,
                EXECUTOR_HARNESS_SHA256,
                RUNNER_SOURCE_SHA256,
            ),
        ):
            return freeze_plan(self._freeze_args(paths))

    def _qualify(self, paths: dict[str, Path]) -> int:
        with patch(
            "experiments.qualify_verified_routing._load_paired_verifier",
            return_value=_SyntheticPairedVerifier(),
        ), patch(
            "local_moe.route_promotion._verify_concrete_paired_record",
            _synthetic_verify_record,
        ):
            return qualify(self._qualify_args(paths))


if __name__ == "__main__":
    unittest.main()
