from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import local_moe.assistant_bridge as assistant_bridge
import local_moe.verified_routing_contracts as routing_contracts
from local_moe.assistant_bridge import (
    build_assistant_task,
    load_assistant_bridge_config,
    plan_assistant_route,
)
from local_moe.assistant_bridge_attestation import ed25519_public_key_sha256
from local_moe.assistant_bridge_integrity import (
    canonical_json_bytes,
    canonical_sha256,
)
from local_moe.route_canary import (
    AUTHORIZATION_PAYLOAD_TYPE,
    CanaryRouteDecision,
    VerifiedRoutingCanaryManifest,
    canary_assignment_bucket,
)
from local_moe.route_outcomes import runtime_plan_sha256
from local_moe.route_policy import load_route_policy, recommend_shadow_route
from local_moe.route_scorecard import route_scorecard_from_payload
from local_moe.route_signals import signals_from_route_receipt
from local_moe.verified_routing_contracts import sha256_json


ROOT = Path(__file__).resolve().parents[1]
FIXED_NOW = "2026-07-20T01:00:00+00:00"
ROUTE_RANK = {"local": 0, "local_then_verify": 1, "premium": 2}


class RouteCanaryBridgeIntegrationTests(unittest.TestCase):
    def test_signed_exact_cell_changes_route_and_failures_retain_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            configs = root / "configs"
            workspace = root / "workspace"
            artifacts.mkdir()
            configs.mkdir()
            workspace.mkdir()
            (workspace / "tracked.txt").write_text("stable\n", encoding="utf-8")

            private_key = Ed25519PrivateKey.generate()
            public_key_path = artifacts / "operator-public.pem"
            public_key_path.write_bytes(
                private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            scorecard_path = artifacts / "scorecard.json"
            manifest_path = artifacts / "manifest.json"
            authorization_path = artifacts / "authorization.dsse.json"
            runtime_path = configs / "verified-routing-runtime.json"
            assignment_env = "MYMOE_TEST_ROUTE_CANARY_SECRET"
            runtime_raw = {
                "schema_version": "1.0",
                "mode": "canary",
                "route_policy_path": str(
                    ROOT / "tests" / "fixtures" / "verified-routing-policy.json"
                ),
                "scorecard_path": str(scorecard_path),
                "manifest_path": str(manifest_path),
                "authorization_path": str(authorization_path),
                "operator_key_id": "integration-operator",
                "operator_public_key_path": str(public_key_path),
                "operator_public_key_sha256": ed25519_public_key_sha256(
                    private_key.public_key()
                ),
                "assignment_secret_env": assignment_env,
                "chronology_path": str(artifacts / "chronology.json"),
            }
            runtime_path.write_text(
                json.dumps(runtime_raw, sort_keys=True),
                encoding="utf-8",
            )

            bridge_raw = json.loads(
                (ROOT / "configs" / "assistant-bridge.json").read_text(
                    encoding="utf-8"
                )
            )
            executable = str(Path(sys.executable).resolve(strict=True))
            bridge_raw["providers"]["local"]["executable"] = executable
            bridge_raw["providers"]["premium"]["executable"] = executable
            bridge_raw["state"]["ledger_path"] = str(artifacts / "ledger.json")
            bridge_raw["workspace"]["transaction_state_dir"] = str(
                artifacts / "transactions"
            )
            bridge_raw["verified_routing"] = {
                "enabled": True,
                "config_path": str(runtime_path),
            }
            bridge_path = configs / "assistant-bridge.json"
            bridge_path.write_text(
                json.dumps(bridge_raw, sort_keys=True),
                encoding="utf-8",
            )
            config = load_assistant_bridge_config(bridge_path)

            task = build_assistant_task(
                "Analyze the bounded integration contract.",
                profile="quality",
                required_capabilities=("analysis",),
                allow_remote=True,
            )
            baseline = plan_assistant_route(task, config, workspace=workspace)
            self.assertEqual(baseline.route, "premium")
            self.assertIsNone(baseline.route_canary)

            signals = signals_from_route_receipt(baseline)
            runtime_sha256 = runtime_plan_sha256(baseline)
            scorecard_raw = json.loads(
                (
                    ROOT
                    / "tests"
                    / "fixtures"
                    / "verified-routing-scorecard.json"
                ).read_text(encoding="utf-8")
            )
            metrics = {
                "local": (0.99, 100.0, 400.0, 0.0, 0.0, 0.0),
                "local_then_verify": (0.95, 900.0, 800.0, 0.01, 0.25, 500.0),
                "premium": (0.90, 1800.0, 1000.0, 0.05, 1.0, 3000.0),
            }
            for entry in scorecard_raw["entries"]:
                route = str(entry["route"])
                success, latency, tokens, cost, premium_calls, egress = metrics[route]
                entry.update(
                    {
                        "config_sha256": config.source_sha256,
                        "signal_provider_config_sha256": (
                            signals.provider_config_sha256
                        ),
                        "runtime_plan_sha256": runtime_sha256,
                        "capabilities": list(signals.capabilities),
                        "difficulty": signals.difficulty,
                        "verified_samples": 50,
                        "success_rate": success,
                        "p95_latency_ms": latency,
                        "mean_tokens": tokens,
                        "cost_sample_count": 50,
                        "mean_cost_usd": cost,
                        "mean_premium_calls": premium_calls,
                        "mean_egress_chars": egress,
                    }
                )
            scorecard_raw.update(
                {
                    "generated_at": "2026-07-19T00:00:00+00:00",
                    "expires_at": "2026-07-21T00:00:00+00:00",
                    "source_digest": "a" * 64,
                }
            )
            scorecard_raw["digest"] = sha256_json(
                {
                    key: value
                    for key, value in scorecard_raw.items()
                    if key != "digest"
                }
            )
            scorecard = route_scorecard_from_payload(
                scorecard_raw,
                now=FIXED_NOW,
            )
            scorecard_path.write_text(
                json.dumps(scorecard.payload(), sort_keys=True),
                encoding="utf-8",
            )

            policy = load_route_policy(
                ROOT / "tests" / "fixtures" / "verified-routing-policy.json"
            )
            shadow = recommend_shadow_route(
                baseline,
                signals,
                scorecard,
                policy,
                profile="quality",
                now=FIXED_NOW,
            )
            self.assertEqual(shadow.recommended_route, "local")
            self.assertFalse(shadow.abstained)

            assignment_secret = _secret_in_first_canary_cohort(
                task.task_fingerprint
            )
            manifest_content = {
                "schema_version": "1.0",
                "contract": "VerifiedRoutingCanaryManifest",
                "current_mode": "shadow",
                "target_mode": "canary",
                "authority": "structural_eligibility_only",
                "producer_authenticity": "not_attested",
                "applied": False,
                "not_before": "2026-07-20T00:00:00+00:00",
                "expires_at": "2026-07-20T12:00:00+00:00",
                "evidence_valid_until": "2026-07-20T12:00:00+00:00",
                "canary_basis_points": 500,
                "assignment_salt_sha256": hashlib.sha256(
                    assignment_secret.encode("utf-8")
                ).hexdigest(),
                "lineage": {
                    "plan_sha256": "1" * 64,
                    "report_sha256": "2" * 64,
                    "gate_policy_digest": "3" * 64,
                    "route_policy_digest": policy.digest,
                    "scorecard_digest": scorecard.digest,
                    "training_source_digest": scorecard.source_digest,
                    "evaluator_sha256": "4" * 64,
                },
                "enabled_cells": [
                    {
                        "profile": shadow.profile,
                        "capabilities": list(shadow.task_signals.capabilities),
                        "difficulty": shadow.task_signals.difficulty,
                        "baseline_route": shadow.baseline_route,
                        "candidate_route": shadow.recommended_route,
                        "config_sha256": config.source_sha256,
                        "signal_provider_config_sha256": (
                            shadow.task_signals.provider_config_sha256
                        ),
                        "runtime_plan_sha256": shadow.runtime_plan_sha256,
                        "paired_tasks": 50,
                        "candidate_success_rate": 0.99,
                        "candidate_success_ci_lower": 0.94,
                    }
                ],
                "invariants": {
                    "monotone_less_premium_only": True,
                    "privacy_budget_and_capability_guards_preserved": True,
                    "runtime_integration_required_before_application": True,
                    "trusted_signature_required_before_runtime_consumption": True,
                },
            }
            manifest_payload = {
                **manifest_content,
                "manifest_sha256": sha256_json(manifest_content),
            }
            manifest = VerifiedRoutingCanaryManifest.from_payload(manifest_payload)
            manifest_path.write_text(
                json.dumps(manifest.payload(), sort_keys=True),
                encoding="utf-8",
            )

            authorization_content = {
                "schema_version": "1.0",
                "contract": "VerifiedRoutingCanaryAuthorization",
                "activation_id": "integration-activation",
                "operator_key_id": "integration-operator",
                "manifest_sha256": manifest.manifest_sha256,
                "bridge_config_sha256": config.source_sha256,
                "route_policy_digest": policy.digest,
                "scorecard_digest": scorecard.digest,
                "issued_at": "2026-07-19T23:59:00+00:00",
                "not_before": "2026-07-20T00:00:00+00:00",
                "expires_at": "2026-07-20T12:00:00+00:00",
                "maximum_canary_basis_points": 500,
            }
            authorization_payload = canonical_json_bytes(
                {
                    **authorization_content,
                    "authorization_sha256": canonical_sha256(
                        authorization_content
                    ),
                }
            )
            valid_envelope = canonical_json_bytes(
                {
                    "payloadType": AUTHORIZATION_PAYLOAD_TYPE,
                    "payload": base64.b64encode(authorization_payload).decode(
                        "ascii"
                    ),
                    "signatures": [
                        {
                            "keyid": "integration-operator",
                            "sig": base64.b64encode(
                                private_key.sign(
                                    _dsse_pae(
                                        AUTHORIZATION_PAYLOAD_TYPE,
                                        authorization_payload,
                                    )
                                )
                            ).decode("ascii"),
                        }
                    ],
                }
            )
            authorization_path.write_bytes(valid_envelope)

            with (
                patch.dict(os.environ, {assignment_env: assignment_secret}),
                patch.object(
                    routing_contracts,
                    "now_utc",
                    return_value=FIXED_NOW,
                ),
            ):
                effective = assistant_bridge._apply_verified_route_canary(
                    baseline,
                    config,
                )

            self.assertEqual(
                effective.route,
                "local",
                msg=json.dumps(effective.payload(), sort_keys=True),
            )
            self.assertNotEqual(effective.receipt_id, baseline.receipt_id)
            self.assertIsNotNone(effective.route_canary)
            decision = CanaryRouteDecision.from_payload(effective.route_canary or {})
            self.assertTrue(decision.applied)
            self.assertEqual(decision.baseline_route, "premium")
            self.assertEqual(decision.effective_route, "local")
            self.assertEqual(decision.route_receipt_id, baseline.receipt_id)
            self.assertEqual(
                decision.assignment_bucket,
                canary_assignment_bucket(
                    assignment_secret.encode("utf-8"),
                    task.task_fingerprint,
                ),
            )
            self.assertLess(decision.assignment_bucket, 500)

            execution_binding = {
                "contract": "integration-execution-binding",
                "runtime_plan_sha256": runtime_sha256,
            }
            self.assertNotEqual(
                assistant_bridge._confirmation_binding_sha256(
                    baseline,
                    execution_binding,
                ),
                assistant_bridge._confirmation_binding_sha256(
                    effective,
                    execution_binding,
                ),
            )

            envelope = json.loads(valid_envelope)
            signature = bytearray(
                base64.b64decode(envelope["signatures"][0]["sig"])
            )
            signature[0] ^= 1
            envelope["signatures"][0]["sig"] = base64.b64encode(
                bytes(signature)
            ).decode("ascii")
            invalid_cases = [(FIXED_NOW, assignment_secret)]
            invalid_cases.extend(
                [
                    ("2026-07-20T13:00:00+00:00", assignment_secret),
                    (FIXED_NOW, "mismatch-secret-" + "x" * 40),
                ]
            )

            tampered_envelope = canonical_json_bytes(envelope)
            for index, (evaluated_at, secret) in enumerate(invalid_cases):
                authorization_path.write_bytes(
                    tampered_envelope if index == 0 else valid_envelope
                )
                with (
                    self.subTest(case=index),
                    patch.dict(os.environ, {assignment_env: secret}),
                    patch.object(
                        routing_contracts,
                        "now_utc",
                        return_value=evaluated_at,
                    ),
                ):
                    retained = assistant_bridge._apply_verified_route_canary(
                        baseline,
                        config,
                    )
                self.assertEqual(retained.route, baseline.route)
                self.assertIsNone(retained.route_canary)
                self.assertIn(
                    "verified_route_canary_authority_unavailable",
                    retained.rationale_codes,
                )
                self.assertLessEqual(
                    ROUTE_RANK[retained.route],
                    ROUTE_RANK[baseline.route],
                )


def _secret_in_first_canary_cohort(task_fingerprint: str) -> str:
    for attempt in range(10_000):
        candidate = f"integration-canary-secret-{attempt:04d}-" + "x" * 32
        if canary_assignment_bucket(candidate.encode("utf-8"), task_fingerprint) < 500:
            return candidate
    raise AssertionError("Could not find a deterministic canary cohort secret.")


def _dsse_pae(payload_type: str, payload: bytes) -> bytes:
    encoded_type = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )


if __name__ == "__main__":
    unittest.main()
