from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.route_outcomes import (
    OutcomeStore,
    VerifiedOutcomeRecord,
    build_verified_outcome,
)
from local_moe.route_signals import TaskSignals
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    canonical_json,
    sha256_json,
)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64


class RouteOutcomeTests(unittest.TestCase):
    def test_builds_metadata_only_record_and_aggregates_metrics(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("check-a", passed=True)],
            commands=[_command(11, 5, 3), _command(17, 7, 4)],
            capsule=_capsule(321),
            premium_calls=1,
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            estimated_cost_usd=0.0125,
            created_at="2026-07-19T03:00:00+00:00",
        )
        rendered = canonical_json(record.payload())

        self.assertEqual(record.outcome, "passed")
        self.assertEqual(record.evidence_strength, "deterministic")
        self.assertEqual(record.failure_class, "none")
        self.assertEqual(record.latency_ms, 28)
        self.assertEqual(record.prompt_tokens, 12)
        self.assertEqual(record.completion_tokens, 7)
        self.assertEqual(record.premium_calls, 1)
        self.assertEqual(record.remote_payload_chars, 321)
        self.assertEqual(record.model, "local/model-a")
        self.assertEqual(record.provider_runtime_sha256, metadata["route_receipt"]["local_runtime"]["runtime_sha256"])
        self.assertNotIn("reasoning", rendered.lower())
        self.assertNotIn("private-result-body", rendered)
        self.assertNotIn("content", record.payload())

    def test_failed_evidence_wins_and_external_evidence_is_independent(self) -> None:
        metadata = _bridge_metadata(
            evidence=[
                _evidence("check-a", passed=True),
                _evidence("contract-failed", passed=False, kind="external"),
            ]
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "failed")
        self.assertEqual(record.failure_class, "contract-failed")
        self.assertEqual(record.evidence_strength, "independent")

    def test_missing_final_evidence_is_inconclusive(self) -> None:
        record = build_verified_outcome(
            _bridge_metadata(evidence=[], required_verifier_ids=[]),
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "inconclusive")
        self.assertEqual(record.evidence_strength, "implicit")
        self.assertEqual(record.failure_class, "verification_missing")

    def test_required_verifier_must_pass_in_final_phase(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("unrelated-check", passed=True)],
            prior_evidence=[_evidence("check-a", passed=True)],
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "failed")
        self.assertEqual(record.failure_class, "required_verifier_missing")

    def test_failed_bridge_cannot_be_rescued_by_unrelated_passing_evidence(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("unrelated-check", passed=True)],
            status="failed",
            code="premium-runtime-failed",
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "failed")
        self.assertEqual(record.failure_class, "premium-runtime-failed")

    def test_prior_failure_does_not_override_verified_final_recovery(self) -> None:
        metadata = _bridge_metadata(
            evidence=[_evidence("check-a", passed=True)],
            prior_evidence=[_evidence("local-check", passed=False)],
        )

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.outcome, "passed")
        self.assertEqual(record.failure_class, "none")

    def test_capability_comparison_is_order_independent(self) -> None:
        metadata = _bridge_metadata(evidence=[])
        metadata["route_receipt"]["task"]["capability_demand"]["required"] = [
            "tests",
            "code",
        ]

        record = build_verified_outcome(
            metadata,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )

        self.assertEqual(record.capabilities, ("code", "tests"))

    def test_store_is_idempotent_and_fails_closed_on_corruption(self) -> None:
        record = build_verified_outcome(
            _bridge_metadata(evidence=[_evidence("check-a", passed=True)]),
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            store = OutcomeStore(path)
            self.assertTrue(store.append(record))
            self.assertFalse(store.append(record))
            self.assertEqual(store.list_records(), (record,))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

            with path.open("a", encoding="utf-8") as handle:
                handle.write("{broken\n")
            with self.assertRaisesRegex(VerifiedRoutingError, "corrupt"):
                store.list_records()
            with self.assertRaises(VerifiedRoutingError):
                store.append(record)

    def test_strict_parsing_rejects_leaks_tampering_and_non_finite_cost(self) -> None:
        metadata = _bridge_metadata(evidence=[])
        metadata["raw_output"] = "private-result-body"
        with self.assertRaises(VerifiedRoutingError):
            build_verified_outcome(metadata, _signals())

        clean = _bridge_metadata(evidence=[])
        with self.assertRaisesRegex(VerifiedRoutingError, "finite"):
            build_verified_outcome(clean, _signals(), estimated_cost_usd=float("nan"))

        record = build_verified_outcome(
            clean,
            _signals(),
            created_at="2026-07-19T03:00:00+00:00",
        )
        tampered = record.payload()
        tampered["latency_ms"] = 999
        with self.assertRaisesRegex(VerifiedRoutingError, "record_id"):
            VerifiedOutcomeRecord.from_payload(tampered)

        serialized = canonical_json(record.payload()).replace(
            '"confidence":0.9', '"confidence":NaN'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            path.write_text(serialized + "\n", encoding="utf-8")
            with self.assertRaisesRegex(VerifiedRoutingError, "Non-finite"):
                OutcomeStore(path).list_records()


def _signals() -> TaskSignals:
    return TaskSignals(
        request_fingerprint=_DIGEST_A,
        capabilities=("tests", "code"),
        difficulty="complex",
        confidence=0.9,
        abstained=False,
        source="task-metadata-v1",
        objective_chars=1200,
        context_tokens=400,
        constraint_count=2,
        tool_count=2,
    )


def _bridge_metadata(
    *,
    evidence: list[dict[str, object]],
    prior_evidence: list[dict[str, object]] | None = None,
    commands: list[dict[str, object]] | None = None,
    capsule: dict[str, object] | None = None,
    premium_calls: int = 0,
    required_verifier_ids: list[str] | None = None,
    status: str = "completed",
    code: str = "completed",
) -> dict[str, object]:
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
    return {
        "schema_version": "2.0",
        "mode": "assistant_bridge",
        "status": status,
        "code": code,
        "route_receipt": {
            "schema_version": "2.0",
            "contract": "RouteDecisionReceipt",
            "receipt_id": "route-1234",
            "task": {
                "task_id": "task-a",
                "objective_sha256": _DIGEST_B,
                "task_fingerprint": _DIGEST_A,
                "objective_chars": 1200,
                "profile": "balanced",
                "capability_demand": {
                    "required": ["code", "tests"],
                    "tools": ["filesystem", "shell"],
                    "risk_class": "write_local",
                },
                "constraint_count": 2,
                "no_change_expected": False,
                "required_verifier_ids": (
                    ["check-a"]
                    if required_verifier_ids is None
                    else required_verifier_ids
                ),
                "allow_remote": True,
                "allow_remote_workspace": False,
                "max_premium_calls": 1,
            },
            "route": "local_then_verify",
            "local_provider": "local-a",
            "premium_provider": "premium-a",
            "local_gaps": [],
            "premium_gaps": [],
            "remote_allowed": True,
            "premium_call_budget": 1,
            "rationale_codes": ["profile_balanced"],
            "expected_flow": ["local", "verify"],
            "config_sha256": _DIGEST_C,
            "workspace": {"fingerprint": _DIGEST_D},
            "local_runtime": local_runtime,
            "premium_runtime": premium_runtime,
        },
        "verification": {"prior": prior_evidence or [], "final": evidence},
        "commands": commands or [],
        "capsule": capsule,
        "final_provider": "local-a",
        "premium_calls_used": premium_calls,
        "privacy": "metadata_only",
    }


def _evidence(
    code: str, *, passed: bool, kind: str = "command"
) -> dict[str, object]:
    return {
        "id": code,
        "verifier": "verifier-a",
        "kind": kind,
        "passed": passed,
        "code": code,
        "artifact_sha256": _DIGEST_B,
        "observed_chars": 0,
        "evidence_ref": None,
        "task_fingerprint": _DIGEST_A,
        "workspace_fingerprint": _DIGEST_D,
        "verifier_spec_sha256": _DIGEST_C,
    }


def _command(duration_ms: int, prompt_tokens: int, completion_tokens: int) -> dict[str, object]:
    return {
        "provider_id": "local-a",
        "status": "completed",
        "code": "completed",
        "returncode": 0,
        "duration_ms": duration_ms,
        "output_sha256": _DIGEST_A,
        "output_chars": 10,
        "stdout_sha256": _DIGEST_B,
        "stdout_bytes": 10,
        "stderr_sha256": None,
        "stderr_bytes": 0,
        "command_sha256": _DIGEST_C,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": None,
            "cost_status": "not_computed_without_pricing_contract",
        },
    }


def _capsule(characters: int) -> dict[str, object]:
    return {
        "capsule_id": "capsule-a",
        "sha256": _DIGEST_A,
        "characters": characters,
        "objective_sha256": _DIGEST_B,
        "constraint_count": 2,
        "verification_count": 1,
        "failure_codes": ["check-failed"],
        "diff_sha256": None,
        "redaction_count": 0,
        "residual_assured": True,
        "residual_detector": "detector-a",
        "truncated": False,
        "content_in_metadata": False,
    }


if __name__ == "__main__":
    unittest.main()
