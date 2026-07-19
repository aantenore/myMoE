from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.derive_route_signals import main as derive_signals
from experiments.recommend_verified_route import main as recommend_route
from local_moe.route_signals import MetadataTaskSignalProvider, TaskSignals
from local_moe.verified_routing_contracts import sha256_json


class VerifiedRoutingCliTests(unittest.TestCase):
    def test_derives_content_free_signals_file_from_receipt(self) -> None:
        fingerprint = hashlib.sha256(b"task").hexdigest()
        receipt = {
            "contract": "RouteDecisionReceipt",
            "task": {
                "task_fingerprint": fingerprint,
                "objective_chars": 400,
                "constraint_count": 1,
                "capability_demand": {
                    "required": ["analysis"],
                    "tools": [],
                    "risk_class": "read_only",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "receipt.json"
            destination = Path(tmp) / "signals.json"
            source.write_text(json.dumps(receipt), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                derive_signals(
                    ["--receipt", str(source), "--out", str(destination)]
                )

            payload = json.loads(destination.read_text(encoding="utf-8"))

        signals = TaskSignals.from_payload(payload)
        self.assertEqual(signals.request_fingerprint, fingerprint)
        self.assertEqual(signals.capabilities, ("analysis",))
        self.assertNotIn("objective", payload)

    def test_signal_cli_rejects_unattested_context_override(self) -> None:
        receipt = {
            "contract": "RouteDecisionReceipt",
            "task": {
                "task_fingerprint": hashlib.sha256(b"task").hexdigest(),
                "objective_chars": 400,
                "constraint_count": 1,
                "capability_demand": {
                    "required": ["analysis"],
                    "tools": [],
                    "risk_class": "read_only",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "receipt.json"
            destination = Path(tmp) / "signals.json"
            source.write_text(json.dumps(receipt), encoding="utf-8")

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    derive_signals(
                        [
                            "--receipt",
                            str(source),
                            "--context-tokens",
                            "5000",
                            "--out",
                            str(destination),
                        ]
                    )
            self.assertFalse(destination.exists())

    def test_writes_a_non_applying_shadow_recommendation(self) -> None:
        fingerprint = hashlib.sha256(b"shadow-task").hexdigest()
        local_runtime = {
            "provider_id": "local-a",
            "model": "local/model-a",
            "execution_scope": "device_only",
        }
        local_runtime["runtime_sha256"] = sha256_json(local_runtime)
        premium_runtime = {
            "provider_id": "premium-a",
            "model": "premium/model-a",
            "execution_scope": "paid_remote",
        }
        premium_runtime["runtime_sha256"] = sha256_json(premium_runtime)
        receipt = {
            "schema_version": "2.0",
            "contract": "RouteDecisionReceipt",
            "receipt_id": "route-cli-test",
            "config_sha256": "1" * 64,
            "local_gaps": [],
            "local_provider": "local-a",
            "local_runtime": local_runtime,
            "premium_call_budget": 1,
            "premium_gaps": [],
            "premium_provider": "premium-a",
            "premium_runtime": premium_runtime,
            "rationale_codes": ["profile_selected"],
            "expected_flow": ["local", "verify"],
            "remote_allowed": True,
            "route": "local_then_verify",
            "task": {
                "task_id": "task-cli-test",
                "objective_sha256": hashlib.sha256(b"objective").hexdigest(),
                "profile": "balanced",
                "task_fingerprint": fingerprint,
                "objective_chars": 400,
                "capability_demand": {
                    "required": ["analysis"],
                    "tools": [],
                    "risk_class": "read_only",
                },
                "constraint_count": 2,
                "no_change_expected": False,
                "required_verifier_ids": ["tests"],
                "allow_remote": True,
                "allow_remote_workspace": False,
                "max_premium_calls": 1,
            },
            "workspace": {"fingerprint": "4" * 64},
        }
        signals = MetadataTaskSignalProvider().signals_from_metadata(
            receipt["task"],  # type: ignore[arg-type]
        )
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "receipt.json"
            signals_path = Path(tmp) / "signals.json"
            destination = Path(tmp) / "decision.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            signals_path.write_text(json.dumps(signals.payload()), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                recommend_route(
                    [
                        "--receipt",
                        str(receipt_path),
                        "--signals",
                        str(signals_path),
                        "--scorecard",
                        "tests/fixtures/verified-routing-scorecard.json",
                        "--policy",
                        "tests/fixtures/verified-routing-policy.json",
                        "--now",
                        "2026-07-20T00:00:00+00:00",
                        "--out",
                        str(destination),
                    ]
                )
            decision = json.loads(destination.read_text(encoding="utf-8"))

        self.assertFalse(decision["applied"])
        self.assertEqual(decision["mode"], "shadow")
        self.assertEqual(decision["baseline_route"], "local_then_verify")
        self.assertEqual(decision["route_receipt_id"], "route-cli-test")
        self.assertIn(
            decision["recommended_route"],
            {"local", "local_then_verify", "premium"},
        )


if __name__ == "__main__":
    unittest.main()
