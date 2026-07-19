from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_moe.assistant_bridge_attestation import ed25519_public_key_sha256
from local_moe.assistant_bridge_integrity import canonical_json_bytes, canonical_sha256
from local_moe.route_canary import (
    AUTHORIZATION_PAYLOAD_TYPE,
    CanaryRouteDecision,
    RouteCanaryError,
    VerifiedRoutingCanaryManifest,
    canary_assignment_bucket,
    decide_route_canary,
    load_and_verify_canary_authorization,
    load_verified_routing_canary_manifest,
    load_verified_routing_runtime_config,
)
from local_moe.route_policy import load_route_policy, recommend_shadow_route
from local_moe.route_scorecard import load_route_scorecard
from local_moe.verified_routing_contracts import sha256_json
from tests.test_route_policy import (
    CONFIG_SHA256,
    FIXTURES,
    NOW,
    TASK_FINGERPRINT,
    _receipt,
    _signals,
)


APPLY_SECRET = b"canary-secret-039-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OUTSIDE_SECRET = b"0123456789abcdef0123456789abcdef"


class RouteCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_route_policy(FIXTURES / "verified-routing-policy.json")
        self.scorecard = load_route_scorecard(
            FIXTURES / "verified-routing-scorecard.json",
            now=NOW,
        )
        self.shadow = recommend_shadow_route(
            _receipt(route="premium", profile="economy"),
            _signals(),
            self.scorecard,
            self.policy,
            profile="economy",
            now=NOW,
        )
        self.assertEqual(self.shadow.recommended_route, "local")
        self.private_key = Ed25519PrivateKey.generate()

    def test_signed_exact_cell_inside_cohort_applies_less_premium_route(self) -> None:
        manifest = self._manifest(APPLY_SECRET)
        authorization = self._authorization(manifest)

        decision = decide_route_canary(
            self.shadow,
            manifest=manifest,
            authorization=authorization,
            bridge_config_sha256=CONFIG_SHA256,
            assignment_secret=APPLY_SECRET,
            evaluated_at=NOW,
        )

        self.assertEqual(canary_assignment_bucket(APPLY_SECRET, TASK_FINGERPRINT), 479)
        self.assertTrue(decision.applied)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.baseline_route, "premium")
        self.assertEqual(decision.effective_route, "local")
        self.assertEqual(decision.reason_codes, ("authorized_canary_applied",))
        self.assertEqual(CanaryRouteDecision.from_payload(decision.payload()), decision)
        self.assertNotIn("objective", json.dumps(decision.payload()))

    def test_outside_cohort_and_cell_mismatch_retain_baseline(self) -> None:
        outside_manifest = self._manifest(OUTSIDE_SECRET)
        outside = decide_route_canary(
            self.shadow,
            manifest=outside_manifest,
            authorization=self._authorization(outside_manifest),
            bridge_config_sha256=CONFIG_SHA256,
            assignment_secret=OUTSIDE_SECRET,
            evaluated_at=NOW,
        )
        self.assertEqual(
            canary_assignment_bucket(OUTSIDE_SECRET, TASK_FINGERPRINT),
            6744,
        )
        self.assertFalse(outside.applied)
        self.assertEqual(outside.effective_route, "premium")
        self.assertIn("outside_canary_cohort", outside.reason_codes)

        mismatch_payload = self._manifest_payload(APPLY_SECRET)
        mismatch_payload["enabled_cells"][0]["difficulty"] = "very_complex"
        mismatch_payload["manifest_sha256"] = sha256_json(
            {key: value for key, value in mismatch_payload.items() if key != "manifest_sha256"}
        )
        mismatch = VerifiedRoutingCanaryManifest.from_payload(mismatch_payload)
        retained = decide_route_canary(
            self.shadow,
            manifest=mismatch,
            authorization=self._authorization(mismatch),
            bridge_config_sha256=CONFIG_SHA256,
            assignment_secret=APPLY_SECRET,
            evaluated_at=NOW,
        )
        self.assertFalse(retained.applied)
        self.assertIn("cell_not_enabled", retained.reason_codes)

    def test_assignment_secret_and_lineage_mismatch_fail_closed(self) -> None:
        manifest = self._manifest(APPLY_SECRET)
        authorization = self._authorization(manifest)

        with self.assertRaisesRegex(RouteCanaryError, "secret"):
            decide_route_canary(
                self.shadow,
                manifest=manifest,
                authorization=authorization,
                bridge_config_sha256=CONFIG_SHA256,
                assignment_secret=OUTSIDE_SECRET,
                evaluated_at=NOW,
            )
        with self.assertRaisesRegex(RouteCanaryError, "lineage"):
            decide_route_canary(
                self.shadow,
                manifest=manifest,
                authorization=authorization,
                bridge_config_sha256="a" * 64,
                assignment_secret=APPLY_SECRET,
                evaluated_at=NOW,
            )

    def test_manifest_rejects_tamper_unknown_fields_and_route_widening(self) -> None:
        manifest = self._manifest_payload(APPLY_SECRET)
        tampered = json.loads(json.dumps(manifest))
        tampered["canary_basis_points"] = 499
        with self.assertRaisesRegex(RouteCanaryError, "digest"):
            VerifiedRoutingCanaryManifest.from_payload(tampered)

        unknown = json.loads(json.dumps(manifest))
        unknown["prompt"] = "must never appear"
        with self.assertRaisesRegex(RouteCanaryError, "Unknown"):
            VerifiedRoutingCanaryManifest.from_payload(unknown)

        widening = json.loads(json.dumps(manifest))
        widening["enabled_cells"][0]["baseline_route"] = "local"
        widening["enabled_cells"][0]["candidate_route"] = "premium"
        widening["manifest_sha256"] = sha256_json(
            {key: value for key, value in widening.items() if key != "manifest_sha256"}
        )
        with self.assertRaisesRegex(RouteCanaryError, "less premium"):
            VerifiedRoutingCanaryManifest.from_payload(widening)

    def test_authorization_verifies_dsse_key_time_and_every_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(APPLY_SECRET)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.payload()), encoding="utf-8")
            public_key_path = self._write_public_key(root, self.private_key)
            authorization_path = root / "authorization.json"
            authorization_path.write_bytes(self._authorization_envelope(manifest))
            runtime_path = self._write_runtime_config(
                root,
                public_key_path=public_key_path,
                manifest_path=manifest_path,
            )
            runtime_raw = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime = load_verified_routing_runtime_config(
                runtime_path,
                expected_source_sha256=sha256_json(runtime_raw),
            )

            verified = load_and_verify_canary_authorization(
                authorization_path,
                manifest=load_verified_routing_canary_manifest(manifest_path),
                runtime=runtime,
                bridge_config_sha256=CONFIG_SHA256,
                now=NOW,
            )
            self.assertEqual(verified.manifest_sha256, manifest.manifest_sha256)

            envelope = json.loads(authorization_path.read_text(encoding="utf-8"))
            signature = base64.b64decode(envelope["signatures"][0]["sig"])
            envelope["signatures"][0]["sig"] = base64.b64encode(
                bytes([signature[0] ^ 1]) + signature[1:]
            ).decode("ascii")
            authorization_path.write_bytes(canonical_json_bytes(envelope))
            with self.assertRaisesRegex(RouteCanaryError, "signature"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=manifest,
                    runtime=runtime,
                    bridge_config_sha256=CONFIG_SHA256,
                    now=NOW,
                )

    def test_expired_or_wrong_config_authorization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(APPLY_SECRET)
            public_key_path = self._write_public_key(root, self.private_key)
            authorization_path = root / "authorization.json"
            authorization_path.write_bytes(self._authorization_envelope(manifest))
            runtime_path = self._write_runtime_config(
                root,
                public_key_path=public_key_path,
                manifest_path=root / "manifest.json",
            )
            runtime_raw = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime = load_verified_routing_runtime_config(
                runtime_path,
                expected_source_sha256=sha256_json(runtime_raw),
            )

            with self.assertRaisesRegex(RouteCanaryError, "not currently active"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=manifest,
                    runtime=runtime,
                    bridge_config_sha256=CONFIG_SHA256,
                    now="2026-07-20T13:00:00+00:00",
                )
            with self.assertRaisesRegex(RouteCanaryError, "bindings"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=manifest,
                    runtime=runtime,
                    bridge_config_sha256="a" * 64,
                    now=NOW,
                )

    def test_durable_chronology_rejects_local_clock_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._manifest(APPLY_SECRET)
            public_key_path = self._write_public_key(root, self.private_key)
            authorization_path = root / "authorization.json"
            authorization_path.write_bytes(self._authorization_envelope(manifest))
            runtime_path = self._write_runtime_config(
                root,
                public_key_path=public_key_path,
                manifest_path=root / "manifest.json",
            )
            runtime_raw = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime = load_verified_routing_runtime_config(
                runtime_path,
                expected_source_sha256=sha256_json(runtime_raw),
            )
            load_and_verify_canary_authorization(
                authorization_path,
                manifest=manifest,
                runtime=runtime,
                bridge_config_sha256=CONFIG_SHA256,
                now="2026-07-20T01:00:00+00:00",
            )

            with self.assertRaisesRegex(RouteCanaryError, "clock rollback"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=manifest,
                    runtime=runtime,
                    bridge_config_sha256=CONFIG_SHA256,
                    now="2026-07-20T00:30:00+00:00",
                )

    def test_runtime_config_is_content_bound_strict_and_rejects_linked_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_key_path = self._write_public_key(root, self.private_key)
            runtime_path = self._write_runtime_config(
                root,
                public_key_path=public_key_path,
                manifest_path=root / "manifest.json",
            )
            raw = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime = load_verified_routing_runtime_config(
                runtime_path,
                expected_source_sha256=sha256_json(raw),
            )
            self.assertEqual(runtime.operator_key_id, "operator-test")

            with self.assertRaisesRegex(RouteCanaryError, "changed"):
                load_verified_routing_runtime_config(
                    runtime_path,
                    expected_source_sha256="a" * 64,
                )
            raw["unknown"] = True
            runtime_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RouteCanaryError, "Unknown"):
                load_verified_routing_runtime_config(
                    runtime_path,
                    expected_source_sha256=sha256_json(raw),
                )

    def test_decision_digest_detects_nested_tamper(self) -> None:
        manifest = self._manifest(APPLY_SECRET)
        decision = decide_route_canary(
            self.shadow,
            manifest=manifest,
            authorization=self._authorization(manifest),
            bridge_config_sha256=CONFIG_SHA256,
            assignment_secret=APPLY_SECRET,
            evaluated_at=NOW,
        )
        payload = decision.payload()
        payload["capabilities"] = ["analysis", "web"]
        with self.assertRaisesRegex(RouteCanaryError, "digest"):
            CanaryRouteDecision.from_payload(payload)

    def test_privacy_can_only_remove_remote_use_and_offline_never_adds_it(self) -> None:
        privacy_shadow = recommend_shadow_route(
            _receipt(
                route="local_then_verify",
                profile="privacy",
                remote_allowed=False,
            ),
            _signals(),
            self.scorecard,
            self.policy,
            profile="privacy",
            now=NOW,
        )
        privacy_manifest = self._manifest(APPLY_SECRET, shadow=privacy_shadow)
        privacy = decide_route_canary(
            privacy_shadow,
            manifest=privacy_manifest,
            authorization=self._authorization(privacy_manifest),
            bridge_config_sha256=CONFIG_SHA256,
            assignment_secret=APPLY_SECRET,
            evaluated_at=NOW,
        )
        self.assertTrue(privacy.applied)
        self.assertEqual(privacy.effective_route, "local")

        offline_shadow = recommend_shadow_route(
            _receipt(
                route="local",
                profile="offline",
                remote_allowed=False,
                premium_call_budget=0,
            ),
            _signals(),
            self.scorecard,
            self.policy,
            profile="offline",
            now=NOW,
        )
        retained = decide_route_canary(
            offline_shadow,
            manifest=privacy_manifest,
            authorization=self._authorization(privacy_manifest),
            bridge_config_sha256=CONFIG_SHA256,
            assignment_secret=APPLY_SECRET,
            evaluated_at=NOW,
        )
        self.assertFalse(retained.applied)
        self.assertEqual(retained.effective_route, "local")
        self.assertIn("not_monotone_less_premium", retained.reason_codes)

    def _manifest(
        self,
        secret: bytes,
        *,
        shadow=None,
    ) -> VerifiedRoutingCanaryManifest:
        return VerifiedRoutingCanaryManifest.from_payload(
            self._manifest_payload(secret, shadow=shadow)
        )

    def _manifest_payload(self, secret: bytes, *, shadow=None) -> dict[str, object]:
        selected = shadow or self.shadow
        content: dict[str, object] = {
            "schema_version": "1.0",
            "contract": "VerifiedRoutingCanaryManifest",
            "current_mode": "shadow",
            "target_mode": "canary",
            "authority": "structural_eligibility_only",
            "producer_authenticity": "not_attested",
            "applied": False,
            "not_before": "2026-07-19T00:00:00+00:00",
            "expires_at": "2026-07-20T12:00:00+00:00",
            "evidence_valid_until": "2026-07-20T12:00:00+00:00",
            "canary_basis_points": 500,
            "assignment_salt_sha256": hashlib.sha256(secret).hexdigest(),
            "lineage": {
                "plan_sha256": "1" * 64,
                "report_sha256": "2" * 64,
                "gate_policy_digest": "3" * 64,
                "route_policy_digest": selected.policy_digest,
                "scorecard_digest": selected.scorecard_digest,
                "training_source_digest": "4" * 64,
                "evaluator_sha256": "5" * 64,
            },
            "enabled_cells": [
                {
                    "profile": selected.profile,
                    "capabilities": list(selected.task_signals.capabilities),
                    "difficulty": selected.task_signals.difficulty,
                    "baseline_route": selected.baseline_route,
                    "candidate_route": selected.recommended_route,
                    "config_sha256": CONFIG_SHA256,
                    "signal_provider_config_sha256": (
                        selected.task_signals.provider_config_sha256
                    ),
                    "runtime_plan_sha256": selected.runtime_plan_sha256,
                    "paired_tasks": 50,
                    "candidate_success_rate": 0.96,
                    "candidate_success_ci_lower": 0.88,
                }
            ],
            "invariants": {
                "monotone_less_premium_only": True,
                "privacy_budget_and_capability_guards_preserved": True,
                "runtime_integration_required_before_application": True,
                "trusted_signature_required_before_runtime_consumption": True,
            },
        }
        return {**content, "manifest_sha256": sha256_json(content)}

    def _authorization(self, manifest: VerifiedRoutingCanaryManifest):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_key_path = self._write_public_key(root, self.private_key)
            envelope_path = root / "authorization.json"
            envelope_path.write_bytes(self._authorization_envelope(manifest))
            runtime_path = self._write_runtime_config(
                root,
                public_key_path=public_key_path,
                manifest_path=root / "manifest.json",
            )
            runtime_raw = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime = load_verified_routing_runtime_config(
                runtime_path,
                expected_source_sha256=sha256_json(runtime_raw),
            )
            return load_and_verify_canary_authorization(
                envelope_path,
                manifest=manifest,
                runtime=runtime,
                bridge_config_sha256=CONFIG_SHA256,
                now=NOW,
            )

    def _authorization_payload(
        self,
        manifest: VerifiedRoutingCanaryManifest,
    ) -> dict[str, object]:
        content: dict[str, object] = {
            "schema_version": "1.0",
            "contract": "VerifiedRoutingCanaryAuthorization",
            "activation_id": "activation-test",
            "operator_key_id": "operator-test",
            "manifest_sha256": manifest.manifest_sha256,
            "bridge_config_sha256": CONFIG_SHA256,
            "route_policy_digest": manifest.lineage["route_policy_digest"],
            "scorecard_digest": manifest.lineage["scorecard_digest"],
            "issued_at": "2026-07-19T23:59:00+00:00",
            "not_before": "2026-07-20T00:00:00+00:00",
            "expires_at": "2026-07-20T12:00:00+00:00",
            "maximum_canary_basis_points": 500,
        }
        return {**content, "authorization_sha256": canonical_sha256(content)}

    def _authorization_envelope(
        self,
        manifest: VerifiedRoutingCanaryManifest,
    ) -> bytes:
        payload = canonical_json_bytes(self._authorization_payload(manifest))
        signature = self.private_key.sign(_pae(AUTHORIZATION_PAYLOAD_TYPE, payload))
        return canonical_json_bytes(
            {
                "payloadType": AUTHORIZATION_PAYLOAD_TYPE,
                "payload": base64.b64encode(payload).decode("ascii"),
                "signatures": [
                    {
                        "keyid": "operator-test",
                        "sig": base64.b64encode(signature).decode("ascii"),
                    }
                ],
            }
        )

    def _write_public_key(
        self,
        root: Path,
        private_key: Ed25519PrivateKey,
    ) -> Path:
        path = root / "operator-public.pem"
        path.write_bytes(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return path

    def _write_runtime_config(
        self,
        root: Path,
        *,
        public_key_path: Path,
        manifest_path: Path,
    ) -> Path:
        path = root / "configs" / "verified-routing-runtime.json"
        path.parent.mkdir(exist_ok=True)
        raw = {
            "schema_version": "1.0",
            "mode": "canary",
            "route_policy_path": str(FIXTURES / "verified-routing-policy.json"),
            "scorecard_path": str(FIXTURES / "verified-routing-scorecard.json"),
            "manifest_path": str(manifest_path),
            "authorization_path": str(root / "authorization.json"),
            "operator_key_id": "operator-test",
            "operator_public_key_path": str(public_key_path),
            "operator_public_key_sha256": ed25519_public_key_sha256(
                self.private_key.public_key()
            ),
            "assignment_secret_env": "MYMOE_TEST_CANARY_SECRET",
            "chronology_path": str(root / "chronology.json"),
        }
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path


def _pae(payload_type: str, payload: bytes) -> bytes:
    encoded_type = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (
        len(encoded_type),
        encoded_type,
        len(payload),
        payload,
    )


if __name__ == "__main__":
    unittest.main()
