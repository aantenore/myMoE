from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_moe.assistant_bridge_two_phase_contracts import ArtifactDescriptor
from local_moe.paired_evidence import (
    PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE,
    PairedAttestationVerifier,
    VerifiedPairedEvidence,
)
from local_moe.paired_execution_contracts import (
    PairedOutcomeBinding,
    PairedRunClaim,
    PairedRunRoot,
)
from local_moe.paired_execution import paired_execution_harness_sha256
from local_moe.paired_execution_pricing import (
    CommandCostEvidence,
    PairedCostEvidence,
    PricingContract,
    PricingItem,
    build_cost_evidence,
)
from local_moe.route_outcomes import VerifiedOutcomeRecord
from local_moe.route_policy import load_route_policy
from local_moe.route_promotion import (
    ContentAddressedDocument,
    PromotionCase,
    PromotionGatePolicy,
    _evaluate_cell,
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
ATTESTATION_POLICY_SHA256 = "4" * 64
EXECUTOR_HARNESS_SHA256 = "5" * 64
EXECUTION_HARNESS_SHA256 = paired_execution_harness_sha256(
    executor_harness_sha256=EXECUTOR_HARNESS_SHA256,
    signal_provider_config_sha256=SIGNAL_PROVIDER_CONFIG_SHA256,
)
RUNNER_SOURCE_SHA256 = "6" * 64
PRICING = PricingContract.build(
    (
        PricingItem("local-a", "model-a", "10", "10"),
        PricingItem("premium-a", "model-a", "200", "200"),
    )
)
PRICING_SHA256 = PRICING.pricing_sha256


def _SyntheticPairedVerifier() -> PairedAttestationVerifier:
    """Exact-type lineage fixture used with a patched internal proof boundary."""

    verifier = object.__new__(PairedAttestationVerifier)
    object.__setattr__(
        verifier,
        "_trust_config",
        SimpleNamespace(
            policy=SimpleNamespace(policy_sha256=ATTESTATION_POLICY_SHA256)
        ),
    )
    object.__setattr__(verifier, "_evidence_root", Path("synthetic-cas"))
    object.__setattr__(verifier, "_bridge_config", SimpleNamespace())
    object.__setattr__(
        verifier,
        "_runner_source_sha256",
        RUNNER_SOURCE_SHA256,
    )
    object.__setattr__(
        verifier,
        "_configuration_sha256",
        _digest("synthetic-paired-verifier"),
    )
    object.__setattr__(verifier, "_sealed", True)
    return verifier


def _synthetic_verify_record(
    self: PairedAttestationVerifier,
    record: VerifiedOutcomeRecord,
    *,
    pricing: PricingContract,
) -> VerifiedPairedEvidence:
    del self, pricing
    if record.paired_run is None:
        raise VerifiedRoutingError("Synthetic fixture has no paired proof.")
    binding = PairedOutcomeBinding.from_payload(record.paired_run)
    created = datetime.fromisoformat(
        record.created_at.replace("Z", "+00:00")
    ).timestamp() + binding.ordinal
    descriptor = ArtifactDescriptor(
        media_type=PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE,
        sha256=_digest(f"proof-{record.record_id}"),
        size_bytes=1,
    )
    return VerifiedPairedEvidence(
        record=record,
        paired_outcome_binding=binding,
        receipt_descriptor=descriptor,
        verifier_ids=("synthetic-verifier",),
        candidate_created_at=created,
        latest_attestation_issued_at=created,
        earliest_attestation_expires_at=created + 300,
    )


class RoutePromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        verifier_patch = patch(
            "local_moe.route_promotion._verify_concrete_paired_record",
            _synthetic_verify_record,
        )
        verifier_patch.start()
        self.addCleanup(verifier_patch.stop)
        self.route_policy = load_route_policy(
            FIXTURES / "verified-routing-policy.json"
        )
        self.gate_policy = _gate_policy()
        self.paired_verifier = _SyntheticPairedVerifier()
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
            attestation_policy_sha256=ATTESTATION_POLICY_SHA256,
            execution_harness_sha256=EXECUTION_HARNESS_SHA256,
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_contract=PRICING.payload(),
        )

    def test_paired_attestation_verifier_is_a_final_authority_boundary(self) -> None:
        with self.assertRaisesRegex(TypeError, "final"):
            class _ForgedVerifier(PairedAttestationVerifier):
                pass

    def test_eligible_paired_holdout_emits_content_addressed_manifest(self) -> None:
        report, manifest = self._evaluate(
            _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        )

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
            attestation_policy_sha256=ATTESTATION_POLICY_SHA256,
            execution_harness_sha256=EXECUTION_HARNESS_SHA256,
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_contract=PRICING,
        )
        records = _holdout_records(20, plan_sha256=plan.plan_sha256)
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
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_holdout_disjoint", report.payload()["reason_codes"]
        )

    def test_missing_candidate_arm_counts_as_inconclusive(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        records.pop()

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn("paired_arm_completeness", report.payload()["reason_codes"])

    def test_legacy_holdout_without_paired_lineage_never_qualifies(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        unsigned = records[0].payload()
        unsigned.pop("record_id")
        unsigned.pop("paired_run")
        unsigned.pop("paired_cost")
        records[0] = VerifiedOutcomeRecord.from_payload(
            {"record_id": _record_id(unsigned), **unsigned}
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_execution_uniqueness", report.payload()["reason_codes"]
        )
        self.assertIn(
            "paired_execution_lineage", report.payload()["reason_codes"]
        )

    def test_legacy_digest_only_plan_loads_but_never_qualifies(self) -> None:
        payload = self.plan.payload()
        payload.pop("pricing_contract")
        payload.pop("attestation_policy_sha256")
        payload.pop("execution_harness_sha256")
        payload.pop("runner_source_sha256")
        payload.pop("plan_sha256")
        payload["plan_sha256"] = sha256_json(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy-plan.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            legacy_plan = load_evidence_plan(path)

        self.assertIsNone(legacy_plan.pricing_contract)
        self.assertIsNone(legacy_plan.attestation_policy_sha256)
        self.assertIsNone(legacy_plan.execution_harness_sha256)
        self.assertIsNone(legacy_plan.runner_source_sha256)
        report, manifest = evaluate_route_promotion(
            plan=legacy_plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=_holdout_records(
                20, plan_sha256=legacy_plan.plan_sha256
            ),
            evaluated_at="2026-07-19T02:00:00+00:00",
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_execution_lineage", report.payload()["reason_codes"]
        )
        provenance = next(
            check
            for check in report.payload()["checks"]
            if check["id"] == "paired_attestation_provenance"
        )
        self.assertFalse(provenance["details"]["proof_preregistered"])

    def test_forged_premium_cost_is_recalculated_from_frozen_pricing(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        index = _terminal_baseline_index(records)
        record = records[index]
        original = PairedCostEvidence.from_payload(
            record.payload()["paired_cost"]  # type: ignore[arg-type]
        )
        command = original.commands[-1]
        forged_command = CommandCostEvidence(
            provider_id=command.provider_id,
            model=command.model,
            provider_runtime_sha256=command.provider_runtime_sha256,
            prompt_tokens=command.prompt_tokens,
            completion_tokens=command.completion_tokens,
            cost_usd="0",
        )
        forged = PairedCostEvidence(
            pricing_sha256=PRICING_SHA256,
            commands=(forged_command,),
            total_cost_usd="0",
        )
        records[index] = _replace_paired_cost(
            record,
            forged,
            estimated_cost_usd=0.0,
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_execution_lineage", report.payload()["reason_codes"]
        )

    def test_final_command_metadata_must_match_outcome(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        index = _terminal_baseline_index(records)
        record = records[index]
        mismatched = build_cost_evidence(
            PRICING,
            (
                {
                    "provider_id": record.final_provider,
                    "model": record.model,
                    "provider_runtime_sha256": _digest("forged-runtime"),
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                },
            ),
        )
        records[index] = _replace_paired_cost(record, mismatched)

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_execution_lineage", report.payload()["reason_codes"]
        )

    def test_command_token_totals_must_match_outcome(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        index = _terminal_baseline_index(records)
        record = records[index]
        mismatched = build_cost_evidence(
            PRICING,
            (
                {
                    "provider_id": record.final_provider,
                    "model": record.model,
                    "provider_runtime_sha256": record.provider_runtime_sha256,
                    "prompt_tokens": record.prompt_tokens - 1,
                    "completion_tokens": record.completion_tokens + 1,
                },
            ),
        )
        records[index] = _replace_paired_cost(record, mismatched)

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_execution_lineage", report.payload()["reason_codes"]
        )

    def test_changed_order_or_normalized_item_invalidates_collected_runs(
        self,
    ) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        changed_cases = list(self.cases)
        changed_cases[0] = replace(
            changed_cases[0],
            order="BA" if changed_cases[0].order == "AB" else "AB",
            normalized_item_sha256=_digest("changed-normalized-item"),
        )
        changed_cases.sort(key=lambda case: case.task_fingerprint)
        changed_plan = build_evidence_plan(
            changed_cases,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            gate_policy=self.gate_policy,
            created_at="2026-07-19T01:00:00+00:00",
            canary_basis_points=100,
            manifest_ttl_seconds=3_600,
            assignment_salt_sha256="4" * 64,
            attestation_policy_sha256=ATTESTATION_POLICY_SHA256,
            execution_harness_sha256=EXECUTION_HARNESS_SHA256,
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_contract=PRICING,
        )

        report, manifest = evaluate_route_promotion(
            plan=changed_plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=records,
            evaluated_at="2026-07-19T02:00:00+00:00",
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_execution_lineage", report.payload()["reason_codes"]
        )

    def test_candidate_regression_is_ineligible(self) -> None:
        report, manifest = self._evaluate(
            _holdout_records(
                20,
                plan_sha256=self.plan.plan_sha256,
                candidate_latency_ms=1_200,
            )
        )

        self.assertEqual(report.payload()["status"], "ineligible")
        self.assertIsNone(manifest)
        self.assertIn(
            "paired_quality_and_efficiency", report.payload()["reason_codes"]
        )

    def test_abstained_arm_makes_evidence_inconclusive(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        baseline_index = next(
            index
            for index, record in enumerate(records)
            if record.planned_route == "premium"
        )
        records = _mutate_and_rechain(
            records, baseline_index, abstained=True
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
            attestation_policy_sha256=ATTESTATION_POLICY_SHA256,
            execution_harness_sha256=EXECUTION_HARNESS_SHA256,
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_contract=PRICING,
        )

        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=scorecard,
            training_records=future_training,
            holdout_records=_holdout_records(
                20, plan_sha256=plan.plan_sha256
            ),
            evaluated_at="2026-07-19T02:00:00+00:00",
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_chronology_and_profile", report.payload()["reason_codes"]
        )

    def test_training_attestations_must_predate_scorecard_generation(self) -> None:
        training_ids = {record.record_id for record in self.training}
        late_issued_at = datetime.fromisoformat(
            "2026-07-19T00:30:00+00:00"
        ).timestamp()

        def late_training_attestation(verifier, record, *, pricing):
            proof = _synthetic_verify_record(
                verifier,
                record,
                pricing=pricing,
            )
            if record.record_id in training_ids:
                return replace(
                    proof,
                    latest_attestation_issued_at=late_issued_at,
                )
            return proof

        with patch(
            "local_moe.route_promotion._verify_concrete_paired_record",
            late_training_attestation,
        ):
            report, manifest = self._evaluate(
                _holdout_records(20, plan_sha256=self.plan.plan_sha256)
            )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_chronology_and_profile",
            report.payload()["reason_codes"],
        )
        check = next(
            item
            for item in report.payload()["checks"]
            if item["id"] == "training_chronology_and_profile"
        )
        self.assertEqual(
            check["details"]["attestations_after_scorecard_generation"],
            len(self.training),
        )

    def test_manifest_expiry_is_bounded_by_scorecard(self) -> None:
        report, manifest = evaluate_route_promotion(
            plan=self.plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=self.scorecard,
            training_records=self.training,
            holdout_records=_holdout_records(
                20, plan_sha256=self.plan.plan_sha256
            ),
            evaluated_at="2026-07-19T23:59:50+00:00",
            paired_verifier=self.paired_verifier,
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
            holdout_records=_holdout_records(
                20, plan_sha256=self.plan.plan_sha256
            ),
            evaluated_at="2026-07-20T00:00:00+00:00",
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "plan_and_scorecard_freshness", report.payload()["reason_codes"]
        )

    def test_receipt_or_evidence_replay_is_inconclusive(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
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
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
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
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        candidate_index = next(
            index
            for index, record in enumerate(records)
            if record.planned_route == "local"
        )
        records[candidate_index] = _mutate_record(
            records[candidate_index],
            premium_calls=2,
            remote_payload_chars=2_000,
            prompt_tokens=2_080,
            estimated_cost_usd=0.021,
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "ineligible")
        self.assertIsNone(manifest)
        cell = report.payload()["cells"][0]
        self.assertEqual(cell["pairwise_premium_increases"], 1)
        self.assertEqual(cell["pairwise_egress_increases"], 1)
        self.assertEqual(cell["pairwise_cost_increases"], 1)

    def test_equal_evidence_inside_one_bound_pair_is_not_a_replay(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        run_id = str(records[0].paired_run["run_id"])  # type: ignore[index]
        pair_indexes = [
            index
            for index, record in enumerate(records)
            if record.paired_run is not None
            and str(record.paired_run["run_id"]) == run_id
        ]
        self.assertEqual(len(pair_indexes), 2)
        first, second = pair_indexes
        records = _mutate_and_rechain(
            records,
            second,
            evidence_sha256=records[first].evidence_sha256,
        )

        report, manifest = self._evaluate(records)

        self.assertEqual(report.payload()["status"], "eligible")
        self.assertIsNotNone(manifest)

    def test_exact_cost_regression_is_not_hidden_by_float_projection(self) -> None:
        case = self.cases[0]
        records = _holdout_records(1, plan_sha256=self.plan.plan_sha256)
        by_route = {record.planned_route: record for record in records}
        baseline = by_route[case.baseline_route]
        candidate = by_route[case.candidate_route]
        baseline_cost = _cost_with_exact_total(
            baseline,
            "1.000000000000000000000000000000000000000000000000000000000001",
        )
        candidate_cost = _cost_with_exact_total(
            candidate,
            "1.000000000000000000000000000000000000000000000000000000000002",
        )
        baseline = _replace_paired_cost(
            baseline,
            baseline_cost,
            estimated_cost_usd=1.0,
        )
        candidate = _replace_paired_cost(
            candidate,
            candidate_cost,
            estimated_cost_usd=1.0,
        )

        cell = _evaluate_cell(
            ((case, baseline, candidate),),
            gate_policy=self.gate_policy,
            minimum_success_rate=0.0,
            cost_weight=1.0,
        )

        self.assertEqual(cell["baseline_mean_cost_usd"], 1.0)
        self.assertEqual(cell["candidate_mean_cost_usd"], 1.0)
        self.assertEqual(cell["pairwise_cost_increases"], 1)
        self.assertFalse(cell["passed"])

    def test_real_scope_failure_code_is_blocking_by_default(self) -> None:
        records = _holdout_records(20, plan_sha256=self.plan.plan_sha256)
        task_fingerprint = records[0].task_fingerprint
        selected = [
            index
            for index, record in enumerate(records)
            if record.task_fingerprint == task_fingerprint
        ]
        selected.sort(
            key=lambda index: int(records[index].paired_run["ordinal"])  # type: ignore[index]
        )
        for index in selected:
            records = _mutate_and_rechain(
                records,
                index,
                outcome="failed",
                failure_class="candidate_scope_invalid",
            )

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
            attestation_policy_sha256=ATTESTATION_POLICY_SHA256,
            execution_harness_sha256=EXECUTION_HARNESS_SHA256,
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_contract=PRICING,
        )

        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=scorecard,
            training_records=repeated_training,
            holdout_records=_holdout_records(
                20, plan_sha256=plan.plan_sha256
            ),
            evaluated_at="2026-07-19T02:00:00+00:00",
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_effective_sample_uniqueness",
            report.payload()["reason_codes"],
        )

    def test_normalized_training_item_cannot_move_between_tasks_or_cells(
        self,
    ) -> None:
        training = _training_records()
        first_binding = PairedOutcomeBinding.from_payload(
            training[0].paired_run  # type: ignore[arg-type]
        )
        training = _rebind_training_pair(
            training,
            task_fingerprint=_digest("training-1"),
            normalized_item_sha256=first_binding.normalized_item_sha256,
            difficulty="simple",
        )
        scorecard = build_route_scorecard(
            training,
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
            attestation_policy_sha256=ATTESTATION_POLICY_SHA256,
            execution_harness_sha256=EXECUTION_HARNESS_SHA256,
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_contract=PRICING,
        )

        report, manifest = evaluate_route_promotion(
            plan=plan,
            gate_policy=self.gate_policy,
            route_policy=self.route_policy,
            scorecard=scorecard,
            training_records=training,
            holdout_records=_holdout_records(
                20,
                plan_sha256=plan.plan_sha256,
            ),
            evaluated_at="2026-07-19T02:00:00+00:00",
            paired_verifier=self.paired_verifier,
        )

        self.assertEqual(report.payload()["status"], "inconclusive")
        self.assertIsNone(manifest)
        self.assertIn(
            "training_semantic_pair_uniqueness",
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
                holdout_records=_holdout_records(
                    20, plan_sha256=self.plan.plan_sha256
                ),
                evaluated_at="2026-07-19T02:00:00+00:00",
                paired_verifier=self.paired_verifier,
            )

    def test_plan_loader_rejects_content_tampering(self) -> None:
        payload = self.plan.payload()
        payload["canary_basis_points"] = 200
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerifiedRoutingError, "plan digest"):
                load_evidence_plan(path)

    def test_plan_loader_rejects_pricing_digest_mismatch(self) -> None:
        payload = self.plan.payload()
        payload["pricing_sha256"] = "f" * 64
        payload.pop("plan_sha256")
        payload["plan_sha256"] = sha256_json(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(VerifiedRoutingError, "pricing digest"):
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
            paired_verifier=self.paired_verifier,
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


def _training_records(
    *,
    config_sha256: str = CONFIG_SHA256,
    signal_provider_config_sha256: str = SIGNAL_PROVIDER_CONFIG_SHA256,
    execution_harness_sha256: str = EXECUTION_HARNESS_SHA256,
) -> list[VerifiedOutcomeRecord]:
    records: list[VerifiedOutcomeRecord] = []
    for index in range(2):
        task_fingerprint = _digest(f"training-{index}")
        root = PairedRunRoot.build(
            plan_sha256=_digest("training-evidence-plan"),
            case_sha256=_digest(f"training-case-{index}"),
            task_fingerprint=task_fingerprint,
            normalized_item_sha256=_digest(f"training-item-{index}"),
            source_snapshot_sha256=_digest(f"training-snapshot-{index}"),
            bridge_config_sha256=config_sha256,
            executor_config_sha256=_digest("paired-test-executor"),
            execution_harness_sha256=execution_harness_sha256,
            lifecycle_config_sha256=_digest("paired-test-training-lifecycle"),
            signals_sha256=_digest(f"paired-test-training-signals-{index}"),
            runner_sha256=_digest("paired-test-runner"),
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_sha256=PRICING_SHA256,
            run_instance_nonce=_digest(f"paired-training-run-{index}"),
            order="AB",
            baseline_route="premium",
            candidate_route="local",
        )
        previous_record_id: str | None = None
        for slot in sorted(root.slots, key=lambda item: item.ordinal):
            claim = PairedRunClaim.build(root, slot)
            binding = PairedOutcomeBinding.build(
                root,
                claim,
                previous_record_id=previous_record_id,
            )
            premium = slot.route == "premium"
            record = _record(
                f"training-{index}",
                route=slot.route,
                created_at="2026-07-18T23:00:00+00:00",
                latency_ms=1_000 if premium else 400,
                premium_calls=1 if premium else 0,
                remote_payload_chars=1_000 if premium else 0,
                estimated_cost_usd=0.02 if premium else 0.001,
                paired_run=binding.payload(),
                config_sha256=config_sha256,
                signal_provider_config_sha256=(
                    signal_provider_config_sha256
                ),
            )
            records.append(record)
            previous_record_id = record.record_id
    return records


def _cases(
    count: int,
    *,
    config_sha256: str = CONFIG_SHA256,
    signal_provider_config_sha256: str = SIGNAL_PROVIDER_CONFIG_SHA256,
) -> list[PromotionCase]:
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
            config_sha256=config_sha256,
            signal_provider_config_sha256=signal_provider_config_sha256,
            runtime_plan_sha256=RUNTIME_PLAN_SHA256,
        )
        for index in range(count)
    ]


def _holdout_records(
    count: int,
    *,
    plan_sha256: str,
    candidate_latency_ms: int = 400,
    config_sha256: str = CONFIG_SHA256,
    signal_provider_config_sha256: str = SIGNAL_PROVIDER_CONFIG_SHA256,
    execution_harness_sha256: str = EXECUTION_HARNESS_SHA256,
) -> list[VerifiedOutcomeRecord]:
    records: list[VerifiedOutcomeRecord] = []
    cases = _cases(
        count,
        config_sha256=config_sha256,
        signal_provider_config_sha256=signal_provider_config_sha256,
    )
    for index, case in enumerate(cases):
        root = PairedRunRoot.build(
            plan_sha256=plan_sha256,
            case_sha256=sha256_json(case.payload()),
            task_fingerprint=case.task_fingerprint,
            normalized_item_sha256=case.normalized_item_sha256,
            source_snapshot_sha256=_digest(f"snapshot-{index}"),
            bridge_config_sha256=case.config_sha256,
            executor_config_sha256=_digest("paired-test-executor"),
            execution_harness_sha256=execution_harness_sha256,
            lifecycle_config_sha256=_digest("paired-test-lifecycle"),
            signals_sha256=_digest(f"paired-test-signals-{index}"),
            runner_sha256=_digest("paired-test-runner"),
            runner_source_sha256=RUNNER_SOURCE_SHA256,
            pricing_sha256=PRICING_SHA256,
            run_instance_nonce=_digest(f"paired-test-run-{index}"),
            order=case.order,
            baseline_route=case.baseline_route,
            candidate_route=case.candidate_route,
        )
        by_arm: dict[str, VerifiedOutcomeRecord] = {}
        previous_record_id: str | None = None
        for slot in sorted(root.slots, key=lambda item: item.ordinal):
            claim = PairedRunClaim.build(root, slot)
            binding = PairedOutcomeBinding.build(
                root,
                claim,
                previous_record_id=previous_record_id,
            )
            if slot.arm == "baseline":
                record = _record(
                    f"holdout-{index}",
                    task_name=f"holdout-task-{index}",
                    route="premium",
                    created_at="2026-07-19T01:05:00+00:00",
                    latency_ms=1_000,
                    premium_calls=1,
                    remote_payload_chars=1_000,
                    estimated_cost_usd=0.02,
                    paired_run=binding.payload(),
                    config_sha256=config_sha256,
                    signal_provider_config_sha256=(
                        signal_provider_config_sha256
                    ),
                )
            else:
                record = _record(
                    f"holdout-{index}",
                    task_name=f"holdout-task-{index}",
                    route="local",
                    created_at="2026-07-19T01:05:00+00:00",
                    latency_ms=candidate_latency_ms,
                    premium_calls=0,
                    remote_payload_chars=0,
                    estimated_cost_usd=0.001,
                    paired_run=binding.payload(),
                    config_sha256=config_sha256,
                    signal_provider_config_sha256=(
                        signal_provider_config_sha256
                    ),
                )
            by_arm[slot.arm] = record
            previous_record_id = record.record_id
        records.extend((by_arm["baseline"], by_arm["candidate"]))
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
    paired_run: dict[str, object] | None = None,
    config_sha256: str = CONFIG_SHA256,
    signal_provider_config_sha256: str = SIGNAL_PROVIDER_CONFIG_SHA256,
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
        "config_sha256": config_sha256,
        "signal_provider_config_sha256": signal_provider_config_sha256,
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
    if paired_run is not None:
        unsigned["paired_run"] = paired_run
        cost = build_cost_evidence(
            PRICING,
            (
                {
                    "provider_id": provider,
                    "model": "model-a",
                    "provider_runtime_sha256": _digest(provider),
                    "prompt_tokens": unsigned["prompt_tokens"],
                    "completion_tokens": unsigned["completion_tokens"],
                },
            ),
        )
        if float(cost.total_cost_usd) != estimated_cost_usd:
            raise AssertionError("Fixture cost does not match its pricing contract.")
        unsigned["paired_cost"] = cost.payload()
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
    unsigned.pop("paired_run", None)
    unsigned.pop("paired_cost", None)
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
    if "paired_run" in unsigned:
        provider = str(unsigned["final_provider"])
        cost = build_cost_evidence(
            PRICING,
            (
                {
                    "provider_id": provider,
                    "model": str(unsigned["model"]),
                    "provider_runtime_sha256": str(
                        unsigned["provider_runtime_sha256"]
                    ),
                    "prompt_tokens": unsigned["prompt_tokens"],
                    "completion_tokens": unsigned["completion_tokens"],
                },
            ),
        )
        unsigned["paired_cost"] = cost.payload()
    return VerifiedOutcomeRecord.from_payload(
        {"record_id": _record_id(unsigned), **unsigned}
    )


def _mutate_and_rechain(
    records: list[VerifiedOutcomeRecord],
    index: int,
    **changes: object,
) -> list[VerifiedOutcomeRecord]:
    result = list(records)
    result[index] = _mutate_record(result[index], **changes)
    raw_binding = result[index].paired_run
    if raw_binding is None:
        return result
    binding = PairedOutcomeBinding.from_payload(raw_binding)
    if binding.ordinal != 0:
        return result
    second_index = next(
        candidate_index
        for candidate_index, candidate in enumerate(result)
        if candidate_index != index
        and candidate.paired_run is not None
        and candidate.paired_run["run_id"] == binding.run_id
        and candidate.paired_run["ordinal"] == 1
    )
    second_binding = PairedOutcomeBinding.from_payload(
        result[second_index].paired_run  # type: ignore[arg-type]
    )
    root = PairedRunRoot.build(
        plan_sha256=binding.plan_sha256,
        case_sha256=binding.case_sha256,
        task_fingerprint=binding.task_fingerprint,
        normalized_item_sha256=binding.normalized_item_sha256,
        source_snapshot_sha256=binding.source_snapshot_sha256,
        bridge_config_sha256=binding.bridge_config_sha256,
        executor_config_sha256=binding.executor_config_sha256,
        execution_harness_sha256=binding.execution_harness_sha256,
        lifecycle_config_sha256=binding.lifecycle_config_sha256,
        signals_sha256=binding.signals_sha256,
        runner_sha256=binding.runner_sha256,
        runner_source_sha256=binding.runner_source_sha256,
        pricing_sha256=binding.pricing_sha256,
        run_instance_nonce=binding.run_instance_nonce,
        order=binding.order,
        baseline_route=binding.baseline_route,
        candidate_route=binding.candidate_route,
    )
    second_slot = next(
        slot for slot in root.slots if slot.slot == second_binding.slot
    )
    second_claim = PairedRunClaim.build(root, second_slot)
    rebound = PairedOutcomeBinding.build(
        root,
        second_claim,
        previous_record_id=result[index].record_id,
    )
    result[second_index] = _mutate_record(
        result[second_index],
        paired_run=rebound.payload(),
    )
    return result


def _rebind_training_pair(
    records: list[VerifiedOutcomeRecord],
    *,
    task_fingerprint: str,
    normalized_item_sha256: str,
    difficulty: str,
) -> list[VerifiedOutcomeRecord]:
    result = list(records)
    indexes = [
        index
        for index, record in enumerate(result)
        if record.task_fingerprint == task_fingerprint
    ]
    if len(indexes) != 2:
        raise AssertionError("training fixture must contain one complete pair")
    bindings = [
        PairedOutcomeBinding.from_payload(result[index].paired_run)
        for index in indexes
    ]
    template = bindings[0]
    root = PairedRunRoot.build(
        plan_sha256=template.plan_sha256,
        case_sha256=template.case_sha256,
        task_fingerprint=template.task_fingerprint,
        normalized_item_sha256=normalized_item_sha256,
        source_snapshot_sha256=template.source_snapshot_sha256,
        bridge_config_sha256=template.bridge_config_sha256,
        executor_config_sha256=template.executor_config_sha256,
        execution_harness_sha256=template.execution_harness_sha256,
        lifecycle_config_sha256=template.lifecycle_config_sha256,
        signals_sha256=template.signals_sha256,
        runner_sha256=template.runner_sha256,
        runner_source_sha256=template.runner_source_sha256,
        pricing_sha256=template.pricing_sha256,
        run_instance_nonce=template.run_instance_nonce,
        order=template.order,
        baseline_route=template.baseline_route,
        candidate_route=template.candidate_route,
    )
    previous_record_id: str | None = None
    for slot in sorted(root.slots, key=lambda item: item.ordinal):
        index = next(
            value
            for value in indexes
            if result[value].planned_route == slot.route
        )
        binding = PairedOutcomeBinding.build(
            root,
            PairedRunClaim.build(root, slot),
            previous_record_id=previous_record_id,
        )
        result[index] = _mutate_record(
            result[index],
            difficulty=difficulty,
            paired_run=binding.payload(),
        )
        previous_record_id = result[index].record_id
    return result


def _terminal_baseline_index(
    records: list[VerifiedOutcomeRecord],
) -> int:
    return next(
        index
        for index, record in enumerate(records)
        if record.planned_route == "premium"
        and record.paired_run is not None
        and record.paired_run["ordinal"] == 1
    )


def _replace_paired_cost(
    record: VerifiedOutcomeRecord,
    cost: PairedCostEvidence,
    *,
    estimated_cost_usd: float | None = None,
) -> VerifiedOutcomeRecord:
    unsigned = record.payload()
    unsigned.pop("record_id")
    unsigned["paired_cost"] = cost.payload()
    if estimated_cost_usd is not None:
        unsigned["estimated_cost_usd"] = estimated_cost_usd
    return VerifiedOutcomeRecord.from_payload(
        {"record_id": _record_id(unsigned), **unsigned}
    )


def _cost_with_exact_total(
    record: VerifiedOutcomeRecord,
    total_cost_usd: str,
) -> PairedCostEvidence:
    assert record.paired_run is not None
    assert record.final_provider is not None
    assert record.model is not None
    assert record.provider_runtime_sha256 is not None
    command = CommandCostEvidence(
        provider_id=record.final_provider,
        model=record.model,
        provider_runtime_sha256=record.provider_runtime_sha256,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        cost_usd=total_cost_usd,
    )
    return PairedCostEvidence(
        pricing_sha256=str(record.paired_run["pricing_sha256"]),
        commands=(command,),
        total_cost_usd=total_cost_usd,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
