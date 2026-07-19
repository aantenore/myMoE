from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.freeze_verified_routing_plan import main as freeze_plan
from experiments.qualify_verified_routing import main as qualify
from local_moe.route_policy import load_route_policy
from local_moe.route_scorecard import build_route_scorecard
from local_moe.verified_routing_contracts import VerifiedRoutingError
from tests.test_route_promotion import (
    FIXTURES,
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
                freeze_exit = freeze_plan(self._freeze_args(paths))
                qualify_exit = qualify(self._qualify_args(paths))

            self.assertEqual(freeze_exit, 0)
            self.assertEqual(qualify_exit, 0)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "eligible")
            self.assertEqual(manifest["target_mode"], "canary")
            self.assertFalse(manifest["applied"])

    def test_regression_writes_report_but_not_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=1_200)

            with redirect_stdout(io.StringIO()):
                freeze_plan(self._freeze_args(paths))
                exit_code = qualify(self._qualify_args(paths))

            self.assertEqual(exit_code, 3)
            self.assertTrue(paths["report"].exists())
            self.assertFalse(paths["manifest"].exists())
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ineligible")

    def test_existing_manifest_path_is_rejected_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(Path(tmp), candidate_latency_ms=1_200)
            with redirect_stdout(io.StringIO()):
                freeze_plan(self._freeze_args(paths))
            paths["manifest"].write_text('{"stale":true}\n', encoding="utf-8")

            with self.assertRaisesRegex(VerifiedRoutingError, "already exists"):
                qualify(self._qualify_args(paths))

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
                freeze_plan(self._freeze_args(paths))

    def _write_inputs(
        self, root: Path, *, candidate_latency_ms: int
    ) -> dict[str, Path]:
        route_policy = load_route_policy(
            FIXTURES / "verified-routing-policy.json"
        )
        training = _training_records()
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
            }.items()
        }
        paths["cases"].write_text(
            json.dumps([case.payload() for case in _cases(20)]),
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
        paths["holdout"].write_text(
            json.dumps(
                {
                    "records": [
                        record.payload()
                        for record in _holdout_records(
                            20, candidate_latency_ms=candidate_latency_ms
                        )
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(route_policy.mode, "shadow")
        return paths

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
            "--evaluated-at",
            "2026-07-19T02:00:00+00:00",
            "--report",
            str(paths["report"]),
            "--manifest",
            str(paths["manifest"]),
        ]


if __name__ == "__main__":
    unittest.main()
