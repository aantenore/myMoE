from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from experiments.derive_route_signals import main as derive_signals
from experiments.recommend_verified_route import main as recommend_route
from local_moe.route_signals import TaskSignals


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

    def test_writes_a_non_applying_shadow_recommendation(self) -> None:
        fingerprint = hashlib.sha256(b"shadow-task").hexdigest()
        receipt = {
            "config_sha256": "1" * 64,
            "local_gaps": [],
            "premium_call_budget": 1,
            "premium_gaps": [],
            "remote_allowed": True,
            "route": "local_then_verify",
            "task": {
                "profile": "balanced",
                "task_fingerprint": fingerprint,
            },
        }
        signals = TaskSignals(
            request_fingerprint=fingerprint,
            capabilities=("analysis",),
            difficulty="medium",
            confidence=0.9,
            abstained=False,
            source="test-signals",
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
        self.assertIn(
            decision["recommended_route"],
            {"local", "local_then_verify", "premium"},
        )


if __name__ == "__main__":
    unittest.main()
