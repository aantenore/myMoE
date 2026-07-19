from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from local_moe.route_outcomes import VerifiedOutcomeRecord
from local_moe.route_policy import load_route_policy
from local_moe.route_promotion import (
    ContentAddressedDocument,
    PromotionCase,
    PromotionGatePolicy,
    build_evidence_plan,
    evaluate_route_promotion,
    load_evidence_plan,
    load_promotion_gate_policy,
    write_content_addressed_json,
)
from local_moe.route_scorecard import build_route_scorecard
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    sha256_json,
)


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_SHA256 = "1" * 64
SIGNAL_PROVIDER_CONFIG_SHA256 = "2" * 64
RUNTIME_PLAN_SHA256 = "3" * 64


class RoutePromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route_policy = load_route_policy(
            FIXTURES / "verified-routing-policy.json"
        )
        self.gate_policy = _gate_policy()
        self.training = _training_records()
        self.scorecard = build_route_scorecard(
            self.training,
            minimum_evidence_strength="independent",
            minimum_confidence=0.7,
            generated_at="2026-07-19T00:00:00+00:00",
            ttl_seconds=86_400,
        )
        self.cases = _cases(20)
        self.plan = build_evidence_plan(
            self.cases,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            gate_policy=self.gate_policy,
            created_at="2026-07-19T01:00:00+00:00",
            canary_basis_points=100,
            manifest_ttl_seconds=3_600,
            assignment_salt_sha256="4" * 64,
        )

    def test_eligible_paired_holdout_emits_content_addressed_manifest(self) -> None:
        report, manifest = self._evaluate(_holdout_records(20))

        self.assertEqual(report.payload()["status"], "eligible")
        self.assertIsNotNone(manifest)
        assert manifest is not None
        manifest_payload = manifest.payload()
        self.assertFalse(manifest_payload["applied"])
        self.assertEqual(
            manifest_payload["authority"], "structural_eligibility_only"
        )
        self.assertEqual(manifest_payload["canary_basis_points"], 100)
        self.assertEqual(len(manifest_payload["enabled_cells"]), 1)
        self.assertEqual(
            manifest_payload["manifest_sha256"],
            sha256_json(manifest.content),
        )
        serialized = json.dumps([report.payload(), manifest_payload])
        for case in self.cases:
            self.assertNotIn(case.task_fingerprint, serialized)

    def test_training_overlap_makes_evidence_inconclusive(self) -> None:
        overlapping_fingerprint = self.training[0].task_fingerprint
        cases = list(self.cases)
        cases[0] = replace(
            cases[0],
            task_fingerprint=overlapping_fingerprint,
            normalized_item_sha256=_digest("overlap-item"),
        )
        cases.sort(key=lambda case: case.task_fingerprint)
        plan = build_evidence_plan(
            cases,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            gate_policy=self.gate_policy,
            created_at="2026-07-19T01:00:00+00:00",
            canary_basis_points=100,
            manifest_ttl_seconds=3_600,
            assignment_salt_sha256="4" * 64,
        )
        records = _holdout_records(20)
        original = self.cases[0].task_fingerprint
        records = [
            _with_task_fingerprint(record, overlapping_fingerprint)
            if record.task_fingerprint == original
            else record
            for record in records
        ]

        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=records,
            evaluated_at="2026-07-19T02:00:00+00:00",
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_holdout_disjoint", report.payload()["reason_codes"]
        )

    def test_missing_candidate_arm_counts_as_inconclusive(self) -> None:
        records = _holdout_records(20)
        records.pop()

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn("paired_arm_completeness", report.payload()["reason_codes"])

    def test_candidate_regression_is_ineligible(self) -> None:
        report, manifest = self._evaluate(
            _holdout_records(20, candidate_latency_ms=1_200)
        )

        self.assertEqual(report.payload()["status"], "ineligible")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_quality_and_efficiency", report.payload()["reason_codes"]
        )

    def test_abstained_arm_makes_evidence_inconclusive(self) -> None:
        records = _holdout_records(20)
        baseline_index = next(
            index
            for index, record in enumerate(records)
            if record.planned_route == "premium"
        )
        records[baseline_index] = _mutate_record(
            records[baseline_index], abstained=True
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        cell = report.payload()["cells"][0]
        self.assertEqual(cell["baseline_non_successes"], 1)
        self.assertIn("holdout_evidence_floor", report.payload()["reason_codes"])

    def test_training_must_predate_scorecard_and_plan(self) -> None:
        future_training = [
            _mutate_record(
                record, created_at="2026-07-19T01:30:00+00:00"
            )
            for record in self.training
        ]
        scorecard = build_route_scorecard(
            future_training,
            minimum_evidence_strength="independent",
            minimum_confidence=0.7,
            generated_at="2026-07-19T00:00:00+00:00",
            ttl_seconds=86_400,
        )
        plan = build_evidence_plan(
            self.cases,
            route_policy=self.route_policy,
            scorecard=scorecard,
            gate_policy=self.gate_policy,
            created_at="2026-07-19T01:00:00+00:00",
            canary_basis_points=100,
            manifest_ttl_seconds=3_600,
            assignment_salt_sha256="4" * 64,
        )

        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=scorecard,
            training_records=future_training,
            holdout_records=_holdout_records(20),
            evaluated_at="2026-07-19T02:00:00+00:00",
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_chronology_and_profile", report.payload()["reason_codes"]
        )

    def test_manifest_expiry_is_bounded_by_scorecard(self) -> None:
        report, manifest = evaluate_route_promotion(
            plan=self.plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=_holdout_records(20),
            evaluated_at="2026-07-19T23:59:50+00:00",
        )

        self.assertEqual(report.payload()["status"], "eligible")
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(
            manifest.payload()["expires_at"], "2026-07-20T00:00:00+00:00"
        )

    def test_expired_scorecard_never_emits_zero_validity_manifest(self) -> None:
        report, manifest = evaluate_route_promotion(
            plan=self.plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=_holdout_records(20),
            evaluated_at="2026-07-20T00:00:00+00:00",
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "plan_and_scorecard_freshness", report.payload()["reason_codes"]
        )

    def test_receipt_or_evidence_replay_is_inconclusive(self) -> None:
        records = _holdout_records(20)
        records[0] = _mutate_record(
            records[0],
            route_receipt_sha256=self.training[0].route_receipt_sha256,
            evidence_sha256=self.training[0].evidence_sha256,
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_holdout_disjoint", report.payload()["reason_codes"]
        )

    def test_replayed_holdout_execution_digests_are_inconclusive(self) -> None:
        records = _holdout_records(20)
        repeated_receipt_id = "receipt-replayed-baseline"
        repeated_receipt_digest = _digest("replayed-baseline-receipt")
        repeated_evidence_digest = _digest("replayed-baseline-evidence")
        records = [
            _mutate_record(
                record,
                route_receipt_id=repeated_receipt_id,
                route_receipt_sha256=repeated_receipt_digest,
                evidence_sha256=repeated_evidence_digest,
            )
            if record.planned_route == "premium"
            else record
            for record in records
        ]

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "holdout_execution_uniqueness", report.payload()["reason_codes"]
        )

    def test_pairwise_resource_increase_cannot_hide_in_means(self) -> None:
        records = _holdout_records(20)
        candidate_index = next(
            index
            for index, record in enumerate(records)
            if record.planned_route == "local"
        )
        records[candidate_index] = _mutate_record(
            records[candidate_index],
            premium_calls=2,
            remote_payload_chars=2_000,
            estimated_cost_usd=0.04,
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "ineligible")
        self.assertIsNone(manifest)
        cell = report.payload()["cells"][0]
        self.assertEqual(cell["pairwise_premium_increases"], 1)
        self.assertEqual(cell["pairwise_egress_increases"], 1)
        self.assertEqual(cell["pairwise_cost_increases"], 1)

    def test_real_scope_failure_code_is_blocking_by_default(self) -> None:
        records = _holdout_records(20)
        task_fingerprint = records[0].task_fingerprint
        records = [
            _mutate_record(
                record,
                outcome="failed",
                failure_class="candidate_scope_invalid",
            )
            if record.task_fingerprint == task_fingerprint
            else record
            for record in records
        ]

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "ineligible")
        self.assertIsNone(manifest)
        cell = report.payload()["cells"][0]
        self.assertEqual(cell["blocking_failure_count"], 2)
        self.assertIn("hard_invariant_failures", report.payload()["reason_codes"])

    def test_repeated_training_tasks_do_not_inflate_effective_sample(self) -> None:
        repeated_training: list[VerifiedOutcomeRecord] = []
        for index in range(2):
            repeated_training.extend(
                (
                    _record(
                        f"training-repeat-{index}",
                        task_name="one-training-task",
                        route="premium",
                        created_at="2026-07-18T23:00:00+00:00",
                        latency_ms=1_000,
                        premium_calls=1,
                        remote_payload_chars=1_000,
                        estimated_cost_usd=0.02,
                    ),
                    _record(
                        f"training-repeat-{index}",
                        task_name="one-training-task",
                        route="local",
                        created_at="2026-07-18T23:00:00+00:00",
                        latency_ms=400,
                        premium_calls=0,
                        remote_payload_chars=0,
                        estimated_cost_usd=0.0,
                    ),
                )
            )
        scorecard = build_route_scorecard(
            repeated_training,
            minimum_evidence_strength="independent",
            minimum_confidence=0.7,
            generated_at="2026-07-19T00:00:00+00:00",
            ttl_seconds=86_400,
        )
        plan = build_evidence_plan(
            self.cases,
            route_policy=self.route_policy,
            scorecard=scorecard,
            gate_policy=self.gate_policy,
            created_at="2026-07-19T01:00:00+00:00",
            canary_basis_points=100,
            manifest_ttl_seconds=3_600,
            assignment_salt_sha256="4" * 64,
        )

        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=scorecard,
            training_records=repeated_training,
            holdout_records=_holdout_records(20),
            evaluated_at="2026-07-19T02:00:00+00:00",
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_effective_sample_uniqueness",
            report.payload()["reason_codes"],
        )

    def test_scorecard_must_rebuild_from_exact_training_set(self) -> None:
        altered = list(self.training)
        altered.pop()

        with self.assertRaisesRegex(VerifiedRoutingError, "source digest"):
            evaluate_route_promotion(
                plan=self.plan,
                gate_policy=self.gate_policy,
                route_policy=self.route_policy,
                scorecard=self.scorecard,
                training_records=altered,
                holdout_records=_holdout_records(20),
                evaluated_at="2026-07-19T02:00:00+00:00",
            )

    def test_plan_loader_rejects_content_tampering(self) -> None:
        payload = self.plan.payload()
        payload["canary_basis_points"] = 200
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerifiedRoutingError, "plan digest"):
                load_evidence_plan(path)

    def test_gate_loader_rejects_duplicate_and_unknown_fields(self) -> None:
        payload = self.gate_policy.payload()
        with tempfile.TemporaryDirectory() as tmp:
            duplicate = Path(tmp) / "duplicate.json"
            encoded = json.dumps(payload)
            duplicate.write_text(
                encoded[:-1] + ',"minimum_paired_tasks":99}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(VerifiedRoutingError, "Duplicate JSON key"):
                load_promotion_gate_policy(duplicate)

            unknown = Path(tmp) / "unknown.json"
            payload["unexpected"] = True
            unknown.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
                load_promotion_gate_policy(unknown)

        with self.assertRaisesRegex(VerifiedRoutingError, "mandatory"):
            replace(self.gate_policy, blocking_failure_classes=())
        with self.assertRaisesRegex(VerifiedRoutingError, "strictly positive"):
            replace(self.gate_policy, minimum_relative_improvement=0.0)

    def test_content_addressed_writer_is_idempotent_and_no_clobber(self) -> None:
        first = ContentAddressedDocument(
            {"contract": "test", "value": 1}, "digest"
        )
        second = ContentAddressedDocument(
            {"contract": "test", "value": 2}, "digest"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "document.json"
            write_content_addressed_json(path, first)
            write_content_addressed_json(path, first)
            self.assertEqual(json.loads(path.read_text()), first.payload())
            with self.assertRaisesRegex(VerifiedRoutingError, "Refusing"):
                write_content_addressed_json(path, second)

    def _evaluate(
        self, records: list[VerifiedOutcomeRecord]
    ) -> tuple[ContentAddressedDocument, ContentAddressedDocument | None]:
        return evaluate_route_promotion(
            plan=self.plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=records,
            evaluated_at="2026-07-19T02:00:00+00:00",
        )


def _gate_policy() -> PromotionGatePolicy:
    return PromotionGatePolicy(
        minimum_paired_tasks=20,
        minimum_paired_tasks_per_cell=20,
        minimum_evidence_strength="independent",
        minimum_confidence=0.7,
        confidence_level=0.95,
        maximum_candidate_latency_ratio=1.1,
        maximum_candidate_p95_latency_ms=2_000,
        minimum_relative_improvement=0.01,
        maximum_holdout_age_seconds=86_400,
        maximum_pair_time_skew_seconds=3_600,
        maximum_canary_basis_points=500,
        maximum_manifest_ttl_seconds=86_400,
        require_complete_cost_evidence=True,
        blocking_failure_classes=(
            "budget-violation",
            "hard-invariant",
            "privacy-violation",
        ),
        non_blocking_failure_classes=(
            "contract-failed",
            "multiple_verification_failures",
            "premium-runtime-failed",
            "verification-failed",
        ),
    )


def _training_records() -> list[VerifiedOutcomeRecord]:
    records: list[VerifiedOutcomeRecord] = []
    for index in range(2):
        records.extend(
            (
                _record(
                    f"training-{index}",
                    route="premium",
                    created_at="2026-07-18T23:00:00+00:00",
                    latency_ms=1_000,
                    premium_calls=1,
                    remote_payload_chars=1_000,
                    estimated_cost_usd=0.02,
                ),
                _record(
                    f"training-{index}",
                    route="local",
                    created_at="2026-07-18T23:00:00+00:00",
                    latency_ms=400,
                    premium_calls=0,
                    remote_payload_chars=0,
                    estimated_cost_usd=0.0,
                ),
            )
        )
    return records


def _cases(count: int) -> list[PromotionCase]:
    return [
        PromotionCase(
            task_fingerprint=_digest(f"holdout-task-{index}"),
            normalized_item_sha256=_digest(f"holdout-item-{index}"),
            profile="balanced",
            capabilities=("analysis",),
            difficulty="medium",
            baseline_route="premium",
            candidate_route="local",
            order="AB" if index % 2 == 0 else "BA",
            config_sha256=CONFIG_SHA256,
            signal_provider_config_sha256=SIGNAL_PROVIDER_CONFIG_SHA256,
            runtime_plan_sha256=RUNTIME_PLAN_SHA256,
        )
        for index in range(count)
    ]


def _holdout_records(
    count: int, *, candidate_latency_ms: int = 400
) -> list[VerifiedOutcomeRecord]:
    records: list[VerifiedOutcomeRecord] = []
    for index in range(count):
        records.extend(
            (
                _record(
                    f"holdout-{index}",
                    task_name=f"holdout-task-{index}",
                    route="premium",
                    created_at="2026-07-19T01:05:00+00:00",
                    latency_ms=1_000,
                    premium_calls=1,
                    remote_payload_chars=1_000,
                    estimated_cost_usd=0.02,
                ),
                _record(
                    f"holdout-{index}",
                    task_name=f"holdout-task-{index}",
                    route="local",
                    created_at="2026-07-19T01:05:00+00:00",
                    latency_ms=candidate_latency_ms,
                    premium_calls=0,
                    remote_payload_chars=0,
                    estimated_cost_usd=0.0,
                ),
            )
        )
    return records


def _record(
    record_name: str,
    *,
    route: str,
    created_at: str,
    latency_ms: int,
    premium_calls: int,
    remote_payload_chars: int,
    estimated_cost_usd: float,
    task_name: str | None = None,
) -> VerifiedOutcomeRecord:
    task_name = task_name or record_name
    receipt_id = f"receipt-{record_name}-{route}"
    provider = "premium-a" if route == "premium" else "local-a"
    unsigned: dict[str, object] = {
        "schema_version": "1.0",
        "created_at": created_at,
        "route_receipt_id": receipt_id,
        "route_receipt_sha256": _digest(receipt_id),
        "task_fingerprint": _digest(task_name),
        "config_sha256": CONFIG_SHA256,
        "signal_provider_config_sha256": SIGNAL_PROVIDER_CONFIG_SHA256,
        "runtime_plan_sha256": RUNTIME_PLAN_SHA256,
        "profile": "balanced",
        "planned_route": route,
        "final_provider": provider,
        "capabilities": ["analysis"],
        "difficulty": "medium",
        "confidence": 0.9,
        "source": "test-fixture",
        "abstained": False,
        "outcome": "passed",
        "evidence_strength": "independent",
        "evidence_sha256": _digest(f"evidence-{record_name}-{route}"),
        "failure_class": "none",
        "latency_ms": latency_ms,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "premium_calls": premium_calls,
        "remote_payload_chars": remote_payload_chars,
        "estimated_cost_usd": estimated_cost_usd,
        "provider_runtime_sha256": _digest(provider),
        "model": "model-a",
    }
    return VerifiedOutcomeRecord.from_payload(
        {"record_id": _record_id(unsigned), **unsigned}
    )


def _record_id(unsigned: dict[str, object]) -> str:
    return f"outcome-{sha256_json(unsigned)}"


def _with_task_fingerprint(
    record: VerifiedOutcomeRecord, task_fingerprint: str
) -> VerifiedOutcomeRecord:
    unsigned = record.payload()
    unsigned.pop("record_id")
    unsigned["task_fingerprint"] = task_fingerprint
    return VerifiedOutcomeRecord.from_payload(
        {"record_id": _record_id(unsigned), **unsigned}
    )


def _mutate_record(
    record: VerifiedOutcomeRecord, **changes: object
) -> VerifiedOutcomeRecord:
    unsigned = record.payload()
    unsigned.pop("record_id")
    unsigned.update(changes)
    return VerifiedOutcomeRecord.from_payload(
        {"record_id": _record_id(unsigned), **unsigned}
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
