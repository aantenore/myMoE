from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import local_moe.assistant_bridge as assistant_bridge
from local_moe.assistant_bridge import VerifiedRoutingRuntimeReference
from local_moe.assistant_bridge_attestation import ed25519_public_key_sha256
from local_moe.assistant_bridge_integrity import canonical_json_bytes, canonical_sha256
from local_moe.route_canary import (
    AUTHORIZATION_PAYLOAD_TYPE,
    CanaryRouteDecision,
    RouteCanaryError,
    VerifiedRoutingCanaryAuthorization,
    VerifiedRoutingCanaryManifest,
    decide_route_canary,
    load_and_verify_canary_authorization,
    load_verified_routing_runtime_config,
    validate_canary_receipt_binding,
)
from local_moe.route_outcomes import runtime_plan_sha256
from local_moe.route_policy import load_route_policy, recommend_shadow_route
from local_moe.route_scorecard import load_route_scorecard
from local_moe.route_signals import signals_from_route_receipt
from local_moe.verified_routing_contracts import VerifiedRoutingError, sha256_json
from tests.test_route_policy import CONFIG_SHA256, FIXTURES, NOW, _receipt, _signals


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_SECRET = b"canary-security-secret-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


class RouteCanarySecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
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
        self.manifest = self._manifest()

    def test_only_fresh_opaque_verified_authority_can_drive_a_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, authorization_path = self._trusted_runtime(root)
            authorization_path.write_bytes(
                self._authorization_envelope(
                    activation_id="activation-current",
                    issued_at="2026-07-19T23:59:00+00:00",
                )
            )
            authority = load_and_verify_canary_authorization(
                authorization_path,
                manifest=self.manifest,
                runtime=runtime,
                bridge_config_sha256=CONFIG_SHA256,
                now=NOW,
            )

            raw_authorization = VerifiedRoutingCanaryAuthorization.from_payload(
                authority.authorization.payload()
            )
            with self.assertRaisesRegex(RouteCanaryError, "verified operator authority"):
                decide_route_canary(
                    self.shadow,
                    manifest=self.manifest,
                    authorization=raw_authorization,  # type: ignore[arg-type]
                    bridge_config_sha256=CONFIG_SHA256,
                    assignment_secret=ASSIGNMENT_SECRET,
                    evaluated_at=NOW,
                )

            decision = decide_route_canary(
                self.shadow,
                manifest=self.manifest,
                authorization=authority,
                bridge_config_sha256=CONFIG_SHA256,
                assignment_secret=ASSIGNMENT_SECRET,
                evaluated_at=NOW,
            )
            self.assertEqual(decision.authorization_sha256, authority.authorization_sha256)

            with self.assertRaisesRegex(RouteCanaryError, "verified evaluation time"):
                decide_route_canary(
                    self.shadow,
                    manifest=self.manifest,
                    authorization=authority,
                    bridge_config_sha256=CONFIG_SHA256,
                    assignment_secret=ASSIGNMENT_SECRET,
                    evaluated_at="2026-07-20T00:00:01+00:00",
                )

    def test_runtime_config_and_configured_public_key_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_path, _, _ = self._write_runtime_files(root)
            linked_runtime = root / "linked-runtime.json"
            try:
                linked_runtime.symlink_to(runtime_path)
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            with self.assertRaisesRegex(RouteCanaryError, "regular non-link"):
                load_verified_routing_runtime_config(linked_runtime)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_public_key = self._write_public_key(root / "operator-public.pem")
            linked_public_key = root / "linked-operator-public.pem"
            linked_public_key.symlink_to(real_public_key)
            runtime_path, authorization_path, _ = self._write_runtime_files(
                root,
                public_key_path=linked_public_key,
            )
            runtime = load_verified_routing_runtime_config(runtime_path)
            authorization_path.write_bytes(
                self._authorization_envelope(
                    activation_id="activation-linked-key",
                    issued_at="2026-07-19T23:59:00+00:00",
                )
            )

            with self.assertRaisesRegex(RouteCanaryError, "regular non-link"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=self.manifest,
                    runtime=runtime,
                    bridge_config_sha256=CONFIG_SHA256,
                    now=NOW,
                )

    def test_chronology_rejects_rollback_and_same_time_equivocation(self) -> None:
        """Exercise continuity while state exists; deletion resistance is not asserted."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, authorization_path = self._trusted_runtime(root)
            authorization_path.write_bytes(
                self._authorization_envelope(
                    activation_id="activation-newer",
                    issued_at="2026-07-19T23:59:30+00:00",
                )
            )
            load_and_verify_canary_authorization(
                authorization_path,
                manifest=self.manifest,
                runtime=runtime,
                bridge_config_sha256=CONFIG_SHA256,
                now="2026-07-20T00:30:00+00:00",
            )

            authorization_path.write_bytes(
                self._authorization_envelope(
                    activation_id="activation-older",
                    issued_at="2026-07-19T23:59:00+00:00",
                )
            )
            with self.assertRaisesRegex(RouteCanaryError, "activation rollback"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=self.manifest,
                    runtime=runtime,
                    bridge_config_sha256=CONFIG_SHA256,
                    now="2026-07-20T00:30:00+00:00",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, authorization_path = self._trusted_runtime(root)
            authorization_path.write_bytes(
                self._authorization_envelope(
                    activation_id="activation-first",
                    issued_at="2026-07-19T23:59:00+00:00",
                )
            )
            load_and_verify_canary_authorization(
                authorization_path,
                manifest=self.manifest,
                runtime=runtime,
                bridge_config_sha256=CONFIG_SHA256,
                now="2026-07-20T00:30:00+00:00",
            )

            authorization_path.write_bytes(
                self._authorization_envelope(
                    activation_id="activation-second",
                    issued_at="2026-07-19T23:59:00+00:00",
                )
            )
            with self.assertRaisesRegex(RouteCanaryError, "signed equivocation"):
                load_and_verify_canary_authorization(
                    authorization_path,
                    manifest=self.manifest,
                    runtime=runtime,
                    bridge_config_sha256=CONFIG_SHA256,
                    now="2026-07-20T00:30:00+00:00",
                )

    def test_transplanted_or_rewritten_canary_lineage_is_rejected(self) -> None:
        baseline, final, decision = self._bound_receipts()
        self.assertEqual(
            validate_canary_receipt_binding(decision, final.payload()),
            decision,
        )

        mutations = {
            "task": lambda payload: payload["task"].update(
                {"task_fingerprint": "f" * 64}
            ),
            "profile": lambda payload: payload["task"].update(
                {"profile": "economy"}
            ),
            "capabilities": lambda payload: payload["task"][
                "capability_demand"
            ].update({"required": ["analysis", "web"]}),
            "runtime": self._mutate_runtime,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = json.loads(json.dumps(final.payload()))
                mutate(payload)
                with self.assertRaisesRegex(
                    RouteCanaryError,
                    "task or runtime lineage",
                ):
                    validate_canary_receipt_binding(decision, payload)

        transplanted = json.loads(json.dumps(final.payload()))
        transplanted["task"]["task_fingerprint"] = "f" * 64
        with self.assertRaises(VerifiedRoutingError):
            signals_from_route_receipt(transplanted)

        for field, replacement in (
            ("route_receipt_id", "route-mutated-baseline"),
            ("route_receipt_sha256", "e" * 64),
        ):
            with self.subTest(field=field):
                payload = self._rewrite_embedded_decision(
                    final.payload(),
                    **{field: replacement},
                )
                with self.assertRaisesRegex(
                    RouteCanaryError,
                    "baseline receipt lineage",
                ):
                    validate_canary_receipt_binding(payload["route_canary"], payload)

        self.assertEqual(decision.route_receipt_id, baseline.receipt_id)

    def test_disabled_authority_returns_the_identical_baseline_without_loading(self) -> None:
        config = assistant_bridge.load_assistant_bridge_config(
            ROOT / "configs" / "assistant-bridge.json"
        )
        disabled = replace(
            config,
            verified_routing=VerifiedRoutingRuntimeReference(
                enabled=False,
                config_path="/missing/verified-routing-runtime.json",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "input.txt").write_text("bounded\n", encoding="utf-8")
            task = assistant_bridge.build_assistant_task(
                "Assess the bounded local contract.",
                profile="quality",
                required_capabilities=("analysis",),
                allow_remote=True,
            )
            baseline = assistant_bridge.plan_assistant_route(
                task,
                disabled,
                workspace=workspace,
            )

            with patch(
                "local_moe.route_canary.load_verified_routing_runtime_config",
                side_effect=AssertionError("disabled authority loaded artifacts"),
            ) as loader:
                effective = assistant_bridge._apply_verified_route_canary(
                    baseline,
                    disabled,
                )

            self.assertIs(effective, baseline)
            loader.assert_not_called()

    def _manifest(self) -> VerifiedRoutingCanaryManifest:
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
            "assignment_salt_sha256": hashlib.sha256(ASSIGNMENT_SECRET).hexdigest(),
            "lineage": {
                "plan_sha256": "1" * 64,
                "report_sha256": "2" * 64,
                "gate_policy_digest": "3" * 64,
                "route_policy_digest": self.shadow.policy_digest,
                "scorecard_digest": self.shadow.scorecard_digest,
                "training_source_digest": self.scorecard.source_digest,
                "evaluator_sha256": "5" * 64,
                "pricing_sha256": "6" * 64,
            },
            "enabled_cells": [
                {
                    "profile": self.shadow.profile,
                    "capabilities": list(self.shadow.task_signals.capabilities),
                    "difficulty": self.shadow.task_signals.difficulty,
                    "baseline_route": self.shadow.baseline_route,
                    "candidate_route": self.shadow.recommended_route,
                    "config_sha256": CONFIG_SHA256,
                    "signal_provider_config_sha256": (
                        self.shadow.task_signals.provider_config_sha256
                    ),
                    "runtime_plan_sha256": self.shadow.runtime_plan_sha256,
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
        return VerifiedRoutingCanaryManifest.from_payload(
            {**content, "manifest_sha256": sha256_json(content)}
        )

    def _trusted_runtime(self, root: Path):
        runtime_path, authorization_path, _ = self._write_runtime_files(root)
        return load_verified_routing_runtime_config(runtime_path), authorization_path

    def _write_runtime_files(
        self,
        root: Path,
        *,
        public_key_path: Path | None = None,
    ) -> tuple[Path, Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        public_key = public_key_path or self._write_public_key(
            root / "operator-public.pem"
        )
        authorization_path = root / "authorization.dsse.json"
        chronology_path = root / "chronology.json"
        runtime_path = root / "runtime.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "mode": "canary",
                    "route_policy_path": str(FIXTURES / "verified-routing-policy.json"),
                    "scorecard_path": str(FIXTURES / "verified-routing-scorecard.json"),
                    "manifest_path": str(root / "manifest.json"),
                    "authorization_path": str(authorization_path),
                    "operator_key_id": "operator-test",
                    "operator_public_key_path": str(public_key),
                    "operator_public_key_sha256": ed25519_public_key_sha256(
                        self.private_key.public_key()
                    ),
                    "assignment_secret_env": "MYMOE_TEST_CANARY_SECRET",
                    "chronology_path": str(chronology_path),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return runtime_path, authorization_path, chronology_path

    def _write_public_key(self, path: Path) -> Path:
        path.write_bytes(
            self.private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return path

    def _authorization_envelope(
        self,
        *,
        activation_id: str,
        issued_at: str,
    ) -> bytes:
        content: dict[str, object] = {
            "schema_version": "1.0",
            "contract": "VerifiedRoutingCanaryAuthorization",
            "activation_id": activation_id,
            "operator_key_id": "operator-test",
            "manifest_sha256": self.manifest.manifest_sha256,
            "bridge_config_sha256": CONFIG_SHA256,
            "route_policy_digest": self.manifest.lineage["route_policy_digest"],
            "scorecard_digest": self.manifest.lineage["scorecard_digest"],
            "issued_at": issued_at,
            "not_before": "2026-07-20T00:00:00+00:00",
            "expires_at": "2026-07-20T12:00:00+00:00",
            "maximum_canary_basis_points": 500,
        }
        payload = canonical_json_bytes(
            {**content, "authorization_sha256": canonical_sha256(content)}
        )
        signature = self.private_key.sign(
            _dsse_pae(AUTHORIZATION_PAYLOAD_TYPE, payload)
        )
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

    def _bound_receipts(self):
        config = assistant_bridge.load_assistant_bridge_config(
            ROOT / "configs" / "assistant-bridge.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "input.txt").write_text("bounded\n", encoding="utf-8")
            task = assistant_bridge.build_assistant_task(
                "Assess the bounded integration contract.",
                profile="quality",
                required_capabilities=("analysis",),
                allow_remote=True,
            )
            baseline = assistant_bridge.plan_assistant_route(
                task,
                config,
                workspace=workspace,
            )
        self.assertEqual(baseline.route, "premium")
        signals = signals_from_route_receipt(baseline)
        decision = CanaryRouteDecision(
            task_fingerprint=task.task_fingerprint,
            profile="quality",
            capabilities=signals.capabilities,
            difficulty=signals.difficulty,
            baseline_route="premium",
            effective_route="local",
            shadow_recommended_route="local",
            applied=True,
            abstained=False,
            reason_codes=("authorized_canary_applied",),
            route_receipt_id=baseline.receipt_id,
            route_receipt_sha256=canonical_sha256(baseline.payload()),
            runtime_plan_sha256=runtime_plan_sha256(baseline),
            signal_provider_config_sha256=signals.provider_config_sha256,
            shadow_decision_sha256="1" * 64,
            policy_digest="2" * 64,
            scorecard_digest="3" * 64,
            bridge_config_sha256=config.source_sha256,
            manifest_sha256="4" * 64,
            authorization_sha256="5" * 64,
            operator_key_id="operator-test",
            assignment_bucket=1,
            canary_basis_points=500,
        )
        final = assistant_bridge._receipt_with_canary_decision(
            baseline,
            decision.payload(),
        )
        return baseline, final, decision

    @staticmethod
    def _mutate_runtime(payload: dict[str, object]) -> None:
        runtime = payload["local_runtime"]
        runtime["model"] = "local/model-mutated"
        unsigned = dict(runtime)
        unsigned.pop("runtime_sha256")
        runtime["runtime_sha256"] = sha256_json(unsigned)

    @staticmethod
    def _rewrite_embedded_decision(
        receipt_payload: dict[str, object],
        **updates: object,
    ) -> dict[str, object]:
        payload = json.loads(json.dumps(receipt_payload))
        decision = payload["route_canary"]
        decision.update(updates)
        unsigned_decision = dict(decision)
        unsigned_decision.pop("decision_sha256")
        decision["decision_sha256"] = canonical_sha256(unsigned_decision)
        unsigned_receipt = dict(payload)
        unsigned_receipt.pop("receipt_id")
        payload["receipt_id"] = f"route-{canonical_sha256(unsigned_receipt)[:32]}"
        return payload


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
