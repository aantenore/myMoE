from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import tempfile
from typing import Iterable, Mapping, Sequence

from .paired_execution_contracts import PairedOutcomeBinding
from .paired_execution_pricing import PairedCostEvidence, PricingContract
from .paired_evidence import PairedAttestationVerifier, VerifiedPairedEvidence
from .route_outcomes import VerifiedOutcomeRecord
from .route_policy import VerifiedRoutePolicy
from .route_scorecard import RouteScorecard, build_route_scorecard
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    DIFFICULTIES,
    EVIDENCE_STRENGTHS,
    ROUTE_PLANS,
    VerifiedRoutingError,
    reject_unknown,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


_GATE_FIELDS = {
    "schema_version",
    "contract",
    "minimum_paired_tasks",
    "minimum_paired_tasks_per_cell",
    "minimum_evidence_strength",
    "minimum_confidence",
    "confidence_level",
    "maximum_candidate_latency_ratio",
    "maximum_candidate_p95_latency_ms",
    "minimum_relative_improvement",
    "maximum_holdout_age_seconds",
    "maximum_pair_time_skew_seconds",
    "maximum_canary_basis_points",
    "maximum_manifest_ttl_seconds",
    "require_complete_cost_evidence",
    "blocking_failure_classes",
    "non_blocking_failure_classes",
}
_PLAN_FIELDS = {
    "schema_version",
    "contract",
    "created_at",
    "route_policy_digest",
    "scorecard_digest",
    "training_source_digest",
    "gate_policy_digest",
    "evaluator_sha256",
    "split_sha256",
    "canary_basis_points",
    "manifest_ttl_seconds",
    "assignment_salt_sha256",
    "attestation_policy_sha256",
    "execution_harness_sha256",
    "runner_source_sha256",
    "pricing_contract",
    "pricing_sha256",
    "cases",
    "plan_sha256",
}
_PLAN_REQUIRED_FIELDS = _PLAN_FIELDS.difference(
    {
        "pricing_contract",
        "pricing_sha256",
        "attestation_policy_sha256",
        "execution_harness_sha256",
        "runner_source_sha256",
    }
)
_CASE_FIELDS = {
    "task_fingerprint",
    "normalized_item_sha256",
    "profile",
    "capabilities",
    "difficulty",
    "baseline_route",
    "candidate_route",
    "order",
    "config_sha256",
    "signal_provider_config_sha256",
    "runtime_plan_sha256",
}
_ROUTE_RANK = {"local": 0, "local_then_verify": 1, "premium": 2}
_ORDERS = {"AB", "BA"}
_MANDATORY_BLOCKING_FAILURE_CLASSES = {
    "budget-violation",
    "hard-invariant",
    "privacy-violation",
}
_EVALUATOR_DEPENDENCIES = (
    "assistant_bridge_attestation.py",
    "assistant_bridge_cas.py",
    "assistant_bridge_two_phase_config.py",
    "assistant_bridge_two_phase_contracts.py",
    "paired_evidence.py",
    "paired_execution_bridge.py",
    "paired_execution.py",
    "paired_execution_contracts.py",
    "paired_execution_pricing.py",
    "paired_execution_store.py",
    "route_outcomes.py",
    "route_policy.py",
    "route_promotion.py",
    "route_scorecard.py",
    "route_signals.py",
    "verified_routing_contracts.py",
)


@dataclass(frozen=True)
class PromotionGatePolicy:
    minimum_paired_tasks: int
    minimum_paired_tasks_per_cell: int
    minimum_evidence_strength: str
    minimum_confidence: float
    confidence_level: float
    maximum_candidate_latency_ratio: float
    maximum_candidate_p95_latency_ms: float
    minimum_relative_improvement: float
    maximum_holdout_age_seconds: int
    maximum_pair_time_skew_seconds: int
    maximum_canary_basis_points: int
    maximum_manifest_ttl_seconds: int
    require_complete_cost_evidence: bool
    blocking_failure_classes: tuple[str, ...]
    non_blocking_failure_classes: tuple[str, ...]
    schema_version: str = CONTRACT_VERSION
    contract: str = "VerifiedRoutingPromotionGatePolicy"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported promotion gate schema_version.")
        if self.contract != "VerifiedRoutingPromotionGatePolicy":
            raise VerifiedRoutingError("Promotion gate contract is unsupported.")
        for name in (
            "minimum_paired_tasks",
            "minimum_paired_tasks_per_cell",
            "maximum_holdout_age_seconds",
            "maximum_pair_time_skew_seconds",
            "maximum_canary_basis_points",
            "maximum_manifest_ttl_seconds",
        ):
            value = require_non_negative_int(getattr(self, name), name)
            if value == 0:
                raise VerifiedRoutingError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        if self.minimum_paired_tasks_per_cell > self.minimum_paired_tasks:
            raise VerifiedRoutingError(
                "minimum_paired_tasks_per_cell cannot exceed minimum_paired_tasks."
            )
        if self.maximum_canary_basis_points > 500:
            raise VerifiedRoutingError(
                "Canary rollout is capped at 500 basis points in schema 1.0."
            )
        if self.maximum_manifest_ttl_seconds > 86_400:
            raise VerifiedRoutingError(
                "Canary manifest TTL is capped at 24 hours in schema 1.0."
            )
        if self.minimum_evidence_strength not in EVIDENCE_STRENGTHS:
            raise VerifiedRoutingError("minimum_evidence_strength is unsupported.")
        for name in ("minimum_confidence", "confidence_level"):
            value = require_finite_number(
                getattr(self, name), name, minimum=0.0, maximum=1.0
            )
            object.__setattr__(self, name, value)
        if not 0.5 < self.confidence_level < 1.0:
            raise VerifiedRoutingError("confidence_level must be between 0.5 and 1.")
        object.__setattr__(
            self,
            "maximum_candidate_latency_ratio",
            require_finite_number(
                self.maximum_candidate_latency_ratio,
                "maximum_candidate_latency_ratio",
                minimum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_candidate_p95_latency_ms",
            require_finite_number(
                self.maximum_candidate_p95_latency_ms,
                "maximum_candidate_p95_latency_ms",
                minimum=0.0,
            ),
        )
        minimum_relative_improvement = require_finite_number(
            self.minimum_relative_improvement,
            "minimum_relative_improvement",
            minimum=0.0,
            maximum=1.0,
        )
        if minimum_relative_improvement <= 0:
            raise VerifiedRoutingError(
                "minimum_relative_improvement must be strictly positive."
            )
        object.__setattr__(
            self,
            "minimum_relative_improvement",
            minimum_relative_improvement,
        )
        if not isinstance(self.require_complete_cost_evidence, bool):
            raise VerifiedRoutingError(
                "require_complete_cost_evidence must be boolean."
            )
        failures = tuple(
            sorted(
                require_identifier_tuple(
                    self.blocking_failure_classes,
                    "blocking_failure_classes",
                )
            )
        )
        if not _MANDATORY_BLOCKING_FAILURE_CLASSES.issubset(failures):
            raise VerifiedRoutingError(
                "blocking_failure_classes must include the mandatory privacy, "
                "budget, and hard-invariant blockers."
            )
        object.__setattr__(self, "blocking_failure_classes", failures)
        non_blocking = tuple(
            sorted(
                require_identifier_tuple(
                    self.non_blocking_failure_classes,
                    "non_blocking_failure_classes",
                )
            )
        )
        if set(failures) & set(non_blocking):
            raise VerifiedRoutingError(
                "Blocking and non-blocking failure classes must be disjoint."
            )
        object.__setattr__(self, "non_blocking_failure_classes", non_blocking)
        object.__setattr__(self, "digest", sha256_json(self.payload()))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "minimum_paired_tasks": self.minimum_paired_tasks,
            "minimum_paired_tasks_per_cell": self.minimum_paired_tasks_per_cell,
            "minimum_evidence_strength": self.minimum_evidence_strength,
            "minimum_confidence": self.minimum_confidence,
            "confidence_level": self.confidence_level,
            "maximum_candidate_latency_ratio": (
                self.maximum_candidate_latency_ratio
            ),
            "maximum_candidate_p95_latency_ms": (
                self.maximum_candidate_p95_latency_ms
            ),
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "maximum_holdout_age_seconds": self.maximum_holdout_age_seconds,
            "maximum_pair_time_skew_seconds": self.maximum_pair_time_skew_seconds,
            "maximum_canary_basis_points": self.maximum_canary_basis_points,
            "maximum_manifest_ttl_seconds": self.maximum_manifest_ttl_seconds,
            "require_complete_cost_evidence": (
                self.require_complete_cost_evidence
            ),
            "blocking_failure_classes": list(self.blocking_failure_classes),
            "non_blocking_failure_classes": list(
                self.non_blocking_failure_classes
            ),
        }


@dataclass(frozen=True)
class PromotionCase:
    task_fingerprint: str
    normalized_item_sha256: str
    profile: str
    capabilities: tuple[str, ...]
    difficulty: str
    baseline_route: str
    candidate_route: str
    order: str
    config_sha256: str
    signal_provider_config_sha256: str
    runtime_plan_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "task_fingerprint",
            "normalized_item_sha256",
            "config_sha256",
            "signal_provider_config_sha256",
            "runtime_plan_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), name)
            )
        object.__setattr__(self, "profile", require_safe_id(self.profile, "profile"))
        capabilities = tuple(
            sorted(require_identifier_tuple(self.capabilities, "capabilities"))
        )
        object.__setattr__(self, "capabilities", capabilities)
        if self.difficulty not in DIFFICULTIES:
            raise VerifiedRoutingError("Promotion case difficulty is unsupported.")
        if self.baseline_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Promotion baseline route is unsupported.")
        if self.candidate_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Promotion candidate route is unsupported.")
        if _ROUTE_RANK[self.candidate_route] >= _ROUTE_RANK[self.baseline_route]:
            raise VerifiedRoutingError(
                "Canary v1 only permits monotone transitions toward less premium use."
            )
        if self.order not in _ORDERS:
            raise VerifiedRoutingError("Promotion case order must be AB or BA.")

    @property
    def cell_key(self) -> tuple[object, ...]:
        return (
            self.profile,
            self.capabilities,
            self.difficulty,
            self.baseline_route,
            self.candidate_route,
            self.config_sha256,
            self.signal_provider_config_sha256,
            self.runtime_plan_sha256,
        )

    def payload(self) -> dict[str, object]:
        return {
            "task_fingerprint": self.task_fingerprint,
            "normalized_item_sha256": self.normalized_item_sha256,
            "profile": self.profile,
            "capabilities": list(self.capabilities),
            "difficulty": self.difficulty,
            "baseline_route": self.baseline_route,
            "candidate_route": self.candidate_route,
            "order": self.order,
            "config_sha256": self.config_sha256,
            "signal_provider_config_sha256": (
                self.signal_provider_config_sha256
            ),
            "runtime_plan_sha256": self.runtime_plan_sha256,
        }


@dataclass(frozen=True)
class VerifiedRoutingEvidencePlan:
    created_at: str
    route_policy_digest: str
    scorecard_digest: str
    training_source_digest: str
    gate_policy_digest: str
    evaluator_sha256: str
    split_sha256: str
    canary_basis_points: int
    manifest_ttl_seconds: int
    assignment_salt_sha256: str
    attestation_policy_sha256: str | None
    execution_harness_sha256: str | None
    runner_source_sha256: str | None
    cases: tuple[PromotionCase, ...]
    plan_sha256: str
    pricing_contract: PricingContract | None = None
    pricing_sha256: str | None = None
    schema_version: str = CONTRACT_VERSION
    contract: str = "VerifiedRoutingEvidencePlan"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported evidence plan schema_version.")
        if self.contract != "VerifiedRoutingEvidencePlan":
            raise VerifiedRoutingError("Evidence plan contract is unsupported.")
        object.__setattr__(
            self, "created_at", require_utc_timestamp(self.created_at, "created_at")
        )
        for name in (
            "route_policy_digest",
            "scorecard_digest",
            "training_source_digest",
            "gate_policy_digest",
            "evaluator_sha256",
            "split_sha256",
            "assignment_salt_sha256",
            "plan_sha256",
        ):
            object.__setattr__(
                self, name, require_sha256(getattr(self, name), name)
            )
        for name in (
            "attestation_policy_sha256",
            "execution_harness_sha256",
            "runner_source_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_sha256(value, name))
        pricing_contract = self.pricing_contract
        if pricing_contract is not None:
            if not isinstance(pricing_contract, PricingContract):
                if not isinstance(pricing_contract, Mapping):
                    raise VerifiedRoutingError(
                        "pricing_contract must be a pricing contract object."
                    )
                pricing_contract = PricingContract.from_payload(
                    pricing_contract
                )
            object.__setattr__(
                self,
                "pricing_contract",
                pricing_contract,
            )
        if self.pricing_sha256 is not None:
            object.__setattr__(
                self,
                "pricing_sha256",
                require_sha256(self.pricing_sha256, "pricing_sha256"),
            )
        if pricing_contract is not None and (
            self.pricing_sha256 is None
            or self.pricing_sha256 != pricing_contract.pricing_sha256
        ):
            raise VerifiedRoutingError(
                "Evidence plan pricing digest does not match its pricing contract."
            )
        for name in ("canary_basis_points", "manifest_ttl_seconds"):
            value = require_non_negative_int(getattr(self, name), name)
            if value == 0:
                raise VerifiedRoutingError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, PromotionCase) for case in cases):
            raise VerifiedRoutingError("Evidence plan cases must be non-empty.")
        if cases != tuple(sorted(cases, key=lambda case: case.task_fingerprint)):
            raise VerifiedRoutingError("Evidence plan cases must be canonical.")
        task_ids = [case.task_fingerprint for case in cases]
        item_ids = [case.normalized_item_sha256 for case in cases]
        if len(task_ids) != len(set(task_ids)):
            raise VerifiedRoutingError("Evidence plan task fingerprints must be unique.")
        if len(item_ids) != len(set(item_ids)):
            raise VerifiedRoutingError("Evidence plan normalized items must be unique.")
        profiles = {case.profile for case in cases}
        if len(profiles) != 1:
            raise VerifiedRoutingError(
                "Schema 1.0 requires one exact profile per evidence plan."
            )
        object.__setattr__(self, "cases", cases)
        expected_split = sha256_json(
            {"cases": [case.payload() for case in cases]}
        )
        if self.split_sha256 != expected_split:
            raise VerifiedRoutingError("Evidence plan split digest is invalid.")
        if self.plan_sha256 != sha256_json(self.content_payload()):
            raise VerifiedRoutingError("Evidence plan digest is invalid.")

    @property
    def profile(self) -> str:
        return self.cases[0].profile

    def content_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "created_at": self.created_at,
            "route_policy_digest": self.route_policy_digest,
            "scorecard_digest": self.scorecard_digest,
            "training_source_digest": self.training_source_digest,
            "gate_policy_digest": self.gate_policy_digest,
            "evaluator_sha256": self.evaluator_sha256,
            "split_sha256": self.split_sha256,
            "canary_basis_points": self.canary_basis_points,
            "manifest_ttl_seconds": self.manifest_ttl_seconds,
            "assignment_salt_sha256": self.assignment_salt_sha256,
            "cases": [case.payload() for case in self.cases],
        }
        for name in (
            "attestation_policy_sha256",
            "execution_harness_sha256",
            "runner_source_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.pricing_sha256 is not None:
            payload["pricing_sha256"] = self.pricing_sha256
        if self.pricing_contract is not None:
            payload["pricing_contract"] = self.pricing_contract.payload()
        return payload

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["plan_sha256"] = self.plan_sha256
        return payload


@dataclass(frozen=True)
class ContentAddressedDocument:
    content: Mapping[str, object]
    digest_field: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        content = json.loads(json.dumps(dict(self.content), allow_nan=False))
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "digest", sha256_json(content))

    def payload(self) -> dict[str, object]:
        payload = dict(self.content)
        payload[self.digest_field] = self.digest
        return payload


def promotion_evaluator_sha256() -> str:
    package = Path(__file__).parent
    dependencies = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in _EVALUATOR_DEPENDENCIES
    }
    return sha256_json({"semantic_dependencies": dependencies})


def build_evidence_plan(
    cases: Iterable[PromotionCase | Mapping[str, object]],
    *,
    route_policy: VerifiedRoutePolicy,
    scorecard: RouteScorecard,
    gate_policy: PromotionGatePolicy,
    created_at: str,
    canary_basis_points: int,
    manifest_ttl_seconds: int,
    assignment_salt_sha256: str,
    attestation_policy_sha256: str,
    execution_harness_sha256: str,
    runner_source_sha256: str,
    pricing_contract: PricingContract | Mapping[str, object],
) -> VerifiedRoutingEvidencePlan:
    normalized = tuple(
        sorted(
            (
                case
                if isinstance(case, PromotionCase)
                else _promotion_case_from_payload(case)
                for case in cases
            ),
            key=lambda case: case.task_fingerprint,
        )
    )
    canary_basis_points = require_non_negative_int(
        canary_basis_points, "canary_basis_points"
    )
    manifest_ttl_seconds = require_non_negative_int(
        manifest_ttl_seconds, "manifest_ttl_seconds"
    )
    if not 0 < canary_basis_points <= gate_policy.maximum_canary_basis_points:
        raise VerifiedRoutingError("Requested canary size exceeds the gate policy.")
    if not 0 < manifest_ttl_seconds <= gate_policy.maximum_manifest_ttl_seconds:
        raise VerifiedRoutingError("Requested manifest TTL exceeds the gate policy.")
    normalized_pricing = (
        pricing_contract
        if isinstance(pricing_contract, PricingContract)
        else PricingContract.from_payload(pricing_contract)
    )
    content = {
        "schema_version": CONTRACT_VERSION,
        "contract": "VerifiedRoutingEvidencePlan",
        "created_at": require_utc_timestamp(created_at, "created_at"),
        "route_policy_digest": route_policy.digest,
        "scorecard_digest": scorecard.digest,
        "training_source_digest": scorecard.source_digest,
        "gate_policy_digest": gate_policy.digest,
        "evaluator_sha256": promotion_evaluator_sha256(),
        "split_sha256": sha256_json(
            {"cases": [case.payload() for case in normalized]}
        ),
        "canary_basis_points": canary_basis_points,
        "manifest_ttl_seconds": manifest_ttl_seconds,
        "assignment_salt_sha256": require_sha256(
            assignment_salt_sha256, "assignment_salt_sha256"
        ),
        "attestation_policy_sha256": require_sha256(
            attestation_policy_sha256,
            "attestation_policy_sha256",
        ),
        "execution_harness_sha256": require_sha256(
            execution_harness_sha256,
            "execution_harness_sha256",
        ),
        "runner_source_sha256": require_sha256(
            runner_source_sha256,
            "runner_source_sha256",
        ),
        "pricing_contract": normalized_pricing.payload(),
        "pricing_sha256": normalized_pricing.pricing_sha256,
        "cases": [case.payload() for case in normalized],
    }
    plan_fields = dict(content)
    plan_fields.pop("cases")
    plan_fields["pricing_contract"] = normalized_pricing
    return VerifiedRoutingEvidencePlan(
        **plan_fields,  # type: ignore[arg-type]
        cases=normalized,
        plan_sha256=sha256_json(content),
    )


def evaluate_route_promotion(
    *,
    plan: VerifiedRoutingEvidencePlan,
    gate_policy: PromotionGatePolicy,
    route_policy: VerifiedRoutePolicy,
    scorecard: RouteScorecard,
    training_records: Iterable[VerifiedOutcomeRecord | Mapping[str, object]],
    holdout_records: Iterable[VerifiedOutcomeRecord | Mapping[str, object]],
    evaluated_at: str,
    paired_verifier: PairedAttestationVerifier | None = None,
) -> tuple[ContentAddressedDocument, ContentAddressedDocument | None]:
    evaluated_at = require_utc_timestamp(evaluated_at, "evaluated_at")
    evaluated_dt = _timestamp(evaluated_at)
    training = _normalize_records(training_records, "training")
    holdout = _normalize_records(holdout_records, "holdout")
    if not training:
        raise VerifiedRoutingError("Training outcomes must be non-empty.")

    _validate_static_bindings(plan, gate_policy, route_policy, scorecard)
    (
        paired_proofs,
        proof_errors,
        proof_policy_matches,
    ) = _verify_paired_record_sets(
        training,
        holdout,
        verifier=paired_verifier,
        pricing=plan.pricing_contract,
        expected_policy_sha256=plan.attestation_policy_sha256,
        expected_execution_harness_sha256=plan.execution_harness_sha256,
        expected_runner_source_sha256=plan.runner_source_sha256,
        evaluated_at_epoch=evaluated_dt.timestamp(),
    )
    all_records = (*training, *holdout)
    proof_complete = (
        paired_verifier is not None
        and plan.attestation_policy_sha256 is not None
        and plan.execution_harness_sha256 is not None
        and plan.runner_source_sha256 is not None
        and proof_policy_matches
        and not proof_errors
        and len(paired_proofs) == len(all_records)
    )
    if proof_complete:
        training = tuple(
            paired_proofs[record.record_id].record for record in training
        )
        holdout = tuple(
            paired_proofs[record.record_id].record for record in holdout
        )
    _validate_scorecard_lineage(scorecard, training)

    checks: list[dict[str, object]] = []
    _add_check(
        checks,
        "paired_attestation_provenance",
        proof_complete,
        "inconclusive",
        {
            "verified_records": len(paired_proofs),
            "expected_records": len(all_records),
            "invalid_or_missing_proofs": len(proof_errors),
            "policy_matches_plan": proof_policy_matches,
            "proof_preregistered": all(
                value is not None
                for value in (
                    plan.attestation_policy_sha256,
                    plan.execution_harness_sha256,
                    plan.runner_source_sha256,
                )
            ),
        },
    )
    _add_check(
        checks,
        "evaluator_binding",
        plan.evaluator_sha256 == promotion_evaluator_sha256(),
        "inconclusive",
    )
    profile_policy = route_policy.profiles[plan.profile]
    evidence_floor_ok = (
        EVIDENCE_STRENGTHS.index(scorecard.minimum_evidence_strength)
        >= EVIDENCE_STRENGTHS.index(gate_policy.minimum_evidence_strength)
        and scorecard.minimum_confidence
        >= max(gate_policy.minimum_confidence, profile_policy.min_confidence)
    )
    _add_check(
        checks,
        "scorecard_evidence_floor",
        evidence_floor_ok,
        "inconclusive",
        {
            "scorecard_evidence_strength": scorecard.minimum_evidence_strength,
            "scorecard_minimum_confidence": scorecard.minimum_confidence,
        },
    )

    plan_dt = _timestamp(plan.created_at)
    scorecard_generated = _timestamp(scorecard.generated_at)
    scorecard_expires = _timestamp(scorecard.expires_at)
    timing_ok = plan_dt >= scorecard_generated and evaluated_dt < scorecard_expires
    _add_check(
        checks,
        "plan_and_scorecard_freshness",
        timing_ok,
        "inconclusive",
        {
            "plan_created_at": plan.created_at,
            "scorecard_generated_at": scorecard.generated_at,
            "scorecard_expires_at": scorecard.expires_at,
        },
    )
    training_timing_errors = sum(
        _authoritative_candidate_time(record, paired_proofs)
        > scorecard_generated
        for record in training
    )
    training_attestation_timing_errors = sum(
        paired_proofs[record.record_id].latest_attestation_issued_at
        > scorecard_generated.timestamp()
        for record in training
        if record.record_id in paired_proofs
    )
    training_profile_errors = sum(
        record.profile != plan.profile for record in training
    )
    training_receipt_ids = [record.route_receipt_id for record in training]
    training_receipt_digests = [record.route_receipt_sha256 for record in training]
    training_evidence_digests = [record.evidence_sha256 for record in training]
    training_replay_errors = (
        len(training_receipt_ids) - len(set(training_receipt_ids))
        + len(training_receipt_digests) - len(set(training_receipt_digests))
        + len(training_evidence_digests) - len(set(training_evidence_digests))
    )
    _add_check(
        checks,
        "training_chronology_and_profile",
        training_timing_errors == 0
        and training_attestation_timing_errors == 0
        and training_profile_errors == 0,
        "inconclusive",
        {
            "records_after_scorecard_generation": training_timing_errors,
            "attestations_after_scorecard_generation": (
                training_attestation_timing_errors
            ),
            "records_from_other_profiles": training_profile_errors,
        },
    )
    _add_check(
        checks,
        "training_execution_uniqueness",
        training_replay_errors == 0,
        "inconclusive",
        {"duplicate_receipt_or_evidence_digests": training_replay_errors},
    )
    training_pair_uniqueness = _training_pair_uniqueness_errors(
        training,
        paired_proofs,
    )
    _add_check(
        checks,
        "training_semantic_pair_uniqueness",
        not any(training_pair_uniqueness.values()),
        "inconclusive",
        training_pair_uniqueness,
    )

    training_tasks = {record.task_fingerprint for record in training}
    training_records_ids = {record.record_id for record in training}
    training_receipts = {record.route_receipt_id for record in training}
    holdout_tasks = {record.task_fingerprint for record in holdout}
    holdout_record_ids = {record.record_id for record in holdout}
    holdout_receipts = {record.route_receipt_id for record in holdout}
    holdout_receipt_digests = {
        record.route_receipt_sha256 for record in holdout
    }
    holdout_evidence_digests = {record.evidence_sha256 for record in holdout}
    training_normalized_items = {
        paired_proofs[record.record_id].paired_outcome_binding.normalized_item_sha256
        for record in training
        if record.record_id in paired_proofs
    }
    holdout_normalized_items = {
        paired_proofs[record.record_id].paired_outcome_binding.normalized_item_sha256
        for record in holdout
        if record.record_id in paired_proofs
    }
    overlap_counts = {
        "task_fingerprints": len(training_tasks & holdout_tasks),
        "record_ids": len(training_records_ids & holdout_record_ids),
        "route_receipt_ids": len(training_receipts & holdout_receipts),
        "route_receipt_digests": len(
            {record.route_receipt_sha256 for record in training}
            & {record.route_receipt_sha256 for record in holdout}
        ),
        "evidence_digests": len(
            {record.evidence_sha256 for record in training}
            & {record.evidence_sha256 for record in holdout}
        ),
        "normalized_item_sha256": len(
            training_normalized_items & holdout_normalized_items
        ),
    }
    _add_check(
        checks,
        "training_holdout_disjoint",
        not any(overlap_counts.values()),
        "inconclusive",
        overlap_counts,
    )
    holdout_evidence_replay_errors = _holdout_evidence_replay_errors(holdout)
    holdout_replay_errors = (
        len(holdout) - len(holdout_receipts)
        + len(holdout) - len(holdout_receipt_digests)
        + holdout_evidence_replay_errors
    )
    _add_check(
        checks,
        "holdout_execution_uniqueness",
        holdout_replay_errors == 0,
        "inconclusive",
        {
            "duplicate_receipt_or_unbound_evidence_digests": (
                holdout_replay_errors
            )
        },
    )
    paired_payloads = [
        paired_proofs[record.record_id].paired_outcome_binding.payload()
        for record in holdout
        if record.record_id in paired_proofs
    ]
    paired_run_ids = [str(item["run_id"]) for item in paired_payloads]
    paired_claim_ids = [str(item["claim_sha256"]) for item in paired_payloads]
    paired_binding_ids = [
        str(item["binding_sha256"]) for item in paired_payloads
    ]
    paired_run_counts: dict[str, int] = {}
    for run_id in paired_run_ids:
        paired_run_counts[run_id] = paired_run_counts.get(run_id, 0) + 1
    paired_replay_errors = (
        len(holdout) - len(paired_payloads)
        + len(paired_claim_ids) - len(set(paired_claim_ids))
        + len(paired_binding_ids) - len(set(paired_binding_ids))
        + sum(count != 2 for count in paired_run_counts.values())
    )
    _add_check(
        checks,
        "paired_execution_uniqueness",
        paired_replay_errors == 0,
        "inconclusive",
        {
            "records_without_paired_binding": len(holdout)
            - len(paired_payloads),
            "duplicate_or_incomplete_paired_executions": paired_replay_errors,
        },
    )

    planned_tasks = {case.task_fingerprint for case in plan.cases}
    exact_task_set = holdout_tasks == planned_tasks
    _add_check(
        checks,
        "intention_to_treat_task_set",
        exact_task_set,
        "inconclusive",
        {
            "planned_tasks": len(planned_tasks),
            "observed_tasks": len(holdout_tasks),
            "missing_tasks": len(planned_tasks - holdout_tasks),
            "unexpected_tasks": len(holdout_tasks - planned_tasks),
        },
    )

    holdout_by_task: dict[str, list[VerifiedOutcomeRecord]] = {}
    for record in holdout:
        holdout_by_task.setdefault(record.task_fingerprint, []).append(record)
    pairs: list[tuple[PromotionCase, VerifiedOutcomeRecord, VerifiedOutcomeRecord]] = []
    pair_errors = 0
    paired_lineage_errors = 0
    evidence_errors = 0
    freshness_errors = 0
    pair_time_skew_errors = 0
    plan_by_task = {case.task_fingerprint: case for case in plan.cases}
    minimum_rank = EVIDENCE_STRENGTHS.index(gate_policy.minimum_evidence_strength)
    minimum_confidence = max(
        gate_policy.minimum_confidence, profile_policy.min_confidence
    )
    for task_fingerprint in sorted(planned_tasks):
        case = plan_by_task[task_fingerprint]
        records = holdout_by_task.get(task_fingerprint, [])
        by_route: dict[str, VerifiedOutcomeRecord] = {}
        if len(records) != 2:
            pair_errors += 1
            continue
        for record in records:
            if record.planned_route in by_route:
                pair_errors += 1
                break
            by_route[record.planned_route] = record
        if set(by_route) != {case.baseline_route, case.candidate_route}:
            pair_errors += 1
            continue
        baseline = by_route[case.baseline_route]
        candidate = by_route[case.candidate_route]
        if not _pair_static_matches_plan(case, baseline, candidate):
            pair_errors += 1
            continue
        if not _paired_lineage_matches_plan(
            case,
            baseline,
            candidate,
            plan_sha256=plan.plan_sha256,
            pricing_contract=plan.pricing_contract,
        ):
            paired_lineage_errors += 1
            continue
        for record in (baseline, candidate):
            if (
                EVIDENCE_STRENGTHS.index(record.evidence_strength) < minimum_rank
                or record.confidence < minimum_confidence
                or record.abstained
                or record.outcome == "inconclusive"
            ):
                evidence_errors += 1
            created = _authoritative_candidate_time(record, paired_proofs)
            age = (evaluated_dt - created).total_seconds()
            if (
                created < plan_dt
                or created > evaluated_dt
                or age >= gate_policy.maximum_holdout_age_seconds
            ):
                freshness_errors += 1
        baseline_time = _authoritative_candidate_time(
            baseline,
            paired_proofs,
        )
        candidate_time = _authoritative_candidate_time(
            candidate,
            paired_proofs,
        )
        if (
            candidate_time <= baseline_time
            if case.order == "AB"
            else baseline_time <= candidate_time
        ):
            pair_time_skew_errors += 1
        elif (
            abs((candidate_time - baseline_time).total_seconds())
            > gate_policy.maximum_pair_time_skew_seconds
        ):
            pair_time_skew_errors += 1
        pairs.append((case, baseline, candidate))
    _add_check(
        checks,
        "paired_arm_completeness",
        (
            pair_errors == 0
            and paired_lineage_errors == 0
            and len(pairs) == len(plan.cases)
        ),
        "inconclusive",
        {
            "complete_pairs": len(pairs),
            "invalid_or_missing_pairs": pair_errors,
            "invalid_paired_lineage": paired_lineage_errors,
        },
    )
    _add_check(
        checks,
        "paired_execution_lineage",
        paired_lineage_errors == 0,
        "inconclusive",
        {"invalid_paired_lineage": paired_lineage_errors},
    )
    _add_check(
        checks,
        "holdout_evidence_floor",
        evidence_errors == 0,
        "inconclusive",
        {"records_below_floor": evidence_errors},
    )
    _add_check(
        checks,
        "holdout_freshness",
        freshness_errors == 0 and pair_time_skew_errors == 0,
        "inconclusive",
        {
            "records_outside_window": freshness_errors,
            "pairs_over_time_skew_limit": pair_time_skew_errors,
        },
    )

    cell_pairs: dict[
        tuple[object, ...],
        list[tuple[PromotionCase, VerifiedOutcomeRecord, VerifiedOutcomeRecord]],
    ] = {}
    for pair in pairs:
        cell_pairs.setdefault(pair[0].cell_key, []).append(pair)
    cell_counts = [len(items) for items in cell_pairs.values()]
    coverage_ok = (
        len(pairs) >= gate_policy.minimum_paired_tasks
        and len(cell_pairs) == len({case.cell_key for case in plan.cases})
        and bool(cell_counts)
        and min(cell_counts) >= gate_policy.minimum_paired_tasks_per_cell
    )
    _add_check(
        checks,
        "paired_sample_coverage",
        coverage_ok,
        "inconclusive",
        {
            "paired_tasks": len(pairs),
            "cells": len(cell_pairs),
            "minimum_cell_tasks": min(cell_counts) if cell_counts else 0,
        },
    )

    scorecard_coverage_errors = 0
    scorecard_quality_errors = 0
    training_effective_sample_errors = 0
    for cell_key in sorted(cell_pairs, key=_cell_sort_key):
        case = cell_pairs[cell_key][0][0]
        for route in (case.baseline_route, case.candidate_route):
            entry = scorecard.conservative_entry(
                config_sha256=case.config_sha256,
                signal_provider_config_sha256=(
                    case.signal_provider_config_sha256
                ),
                runtime_plan_sha256=case.runtime_plan_sha256,
                route=route,
                capabilities=case.capabilities,
                difficulty=case.difficulty,
            )
            if entry is None or entry.verified_samples < profile_policy.min_samples:
                scorecard_coverage_errors += 1
            elif entry.success_rate < profile_policy.min_success_rate:
                scorecard_quality_errors += 1
            qualifying_training = [
                record
                for record in training
                if _training_record_matches_cell(
                    record,
                    case,
                    route,
                    scorecard=scorecard,
                )
            ]
            unique_training_tasks = {
                record.task_fingerprint for record in qualifying_training
            }
            unique_training_items = {
                paired_proofs[record.record_id]
                .paired_outcome_binding.normalized_item_sha256
                for record in qualifying_training
                if record.record_id in paired_proofs
            }
            if (
                len(unique_training_tasks) != len(qualifying_training)
                or len(unique_training_items) != len(qualifying_training)
                or len(unique_training_tasks) < profile_policy.min_samples
            ):
                training_effective_sample_errors += 1
    _add_check(
        checks,
        "scorecard_cell_coverage",
        scorecard_coverage_errors == 0,
        "inconclusive",
        {"missing_or_thin_route_cells": scorecard_coverage_errors},
    )
    _add_check(
        checks,
        "scorecard_cell_quality",
        scorecard_quality_errors == 0,
        "ineligible",
        {"route_cells_below_profile_floor": scorecard_quality_errors},
    )
    _add_check(
        checks,
        "training_effective_sample_uniqueness",
        training_effective_sample_errors == 0,
        "inconclusive",
        {"route_cells_with_repeated_or_thin_tasks": training_effective_sample_errors},
    )

    cell_reports: list[dict[str, object]] = []
    quality_failures = 0
    cost_completeness_failures = 0
    blocking_failures = 0
    for cell_key in sorted(cell_pairs, key=_cell_sort_key):
        items = cell_pairs[cell_key]
        report = _evaluate_cell(
            items,
            gate_policy=gate_policy,
            minimum_success_rate=profile_policy.min_success_rate,
            cost_weight=profile_policy.weights.cost,
        )
        cell_reports.append(report)
        if not bool(report["passed"]):
            quality_failures += 1
        cost_completeness_failures += int(
            not bool(report["cost_evidence_complete"])
        )
        blocking_failures += int(report["blocking_failure_count"])
    _add_check(
        checks,
        "cost_evidence_completeness",
        cost_completeness_failures == 0,
        "inconclusive",
        {"cells_with_incomplete_cost": cost_completeness_failures},
    )
    _add_check(
        checks,
        "hard_invariant_failures",
        blocking_failures == 0,
        "ineligible",
        {"blocking_outcomes": blocking_failures},
    )
    _add_check(
        checks,
        "paired_quality_and_efficiency",
        quality_failures == 0 and bool(cell_reports),
        "ineligible",
        {"failing_cells": quality_failures},
    )

    failed_kinds = {
        str(check["failure_kind"])
        for check in checks
        if not bool(check["passed"])
    }
    if "inconclusive" in failed_kinds:
        status = "inconclusive"
    elif "ineligible" in failed_kinds:
        status = "ineligible"
    else:
        status = "eligible"
    reasons = sorted(
        str(check["id"]) for check in checks if not bool(check["passed"])
    )
    paired_proof_set_sha256 = sha256_json(
        {
            "proofs": [
                {
                    "record_id": record_id,
                    "receipt": proof.receipt_descriptor.payload(),
                    "paired_binding_sha256": (
                        proof.paired_outcome_binding.binding_sha256
                    ),
                    "verifier_ids": list(proof.verifier_ids),
                }
                for record_id, proof in sorted(paired_proofs.items())
            ]
        }
    )
    paired_verifier_sha256 = (
        None
        if paired_verifier is None
        else paired_verifier.configuration_sha256
    )
    report_content: dict[str, object] = {
        "schema_version": CONTRACT_VERSION,
        "contract": "VerifiedRoutingPromotionReport",
        "evaluated_at": evaluated_at,
        "status": status,
        "promotion_eligible": status == "eligible",
        "reason_codes": reasons,
        "checks": checks,
        "lineage": {
            "plan_sha256": plan.plan_sha256,
            "gate_policy_digest": gate_policy.digest,
            "route_policy_digest": route_policy.digest,
            "scorecard_digest": scorecard.digest,
            "training_source_digest": scorecard.source_digest,
            "holdout_source_digest": _record_set_digest(holdout),
            "evaluator_sha256": promotion_evaluator_sha256(),
            "pricing_sha256": plan.pricing_sha256,
            "attestation_policy_sha256": plan.attestation_policy_sha256,
            "execution_harness_sha256": plan.execution_harness_sha256,
            "runner_source_sha256": plan.runner_source_sha256,
            "paired_attestation_verifier_sha256": paired_verifier_sha256,
            "paired_proof_set_sha256": paired_proof_set_sha256,
        },
        "coverage": {
            "planned_tasks": len(plan.cases),
            "paired_tasks": len(pairs),
            "cells": len(cell_reports),
        },
        "cells": cell_reports,
        "requested_canary": {
            "basis_points": plan.canary_basis_points,
            "ttl_seconds": plan.manifest_ttl_seconds,
        },
        "runtime_effect": {
            "applied": False,
            "authority": "structural_eligibility_only",
            "producer_authenticity": "not_attested",
        },
    }
    report = ContentAddressedDocument(report_content, "report_sha256")
    if status != "eligible":
        return report, None

    requested_expiry = evaluated_dt + timedelta(
        seconds=plan.manifest_ttl_seconds
    )
    holdout_valid_until = min(
        _authoritative_candidate_time(record, paired_proofs)
        + timedelta(seconds=gate_policy.maximum_holdout_age_seconds)
        for record in holdout
    )
    effective_expiry = min(
        requested_expiry,
        scorecard_expires,
        holdout_valid_until,
    )
    expires_at = effective_expiry.replace(microsecond=0).isoformat()
    enabled_cells = [
        {
            key: cell[key]
            for key in (
                "profile",
                "capabilities",
                "difficulty",
                "baseline_route",
                "candidate_route",
                "config_sha256",
                "signal_provider_config_sha256",
                "runtime_plan_sha256",
                "paired_tasks",
                "candidate_success_rate",
                "candidate_success_ci_lower",
            )
        }
        for cell in cell_reports
    ]
    manifest_content: dict[str, object] = {
        "schema_version": CONTRACT_VERSION,
        "contract": "VerifiedRoutingCanaryManifest",
        "current_mode": "shadow",
        "target_mode": "canary",
        "authority": "structural_eligibility_only",
        "producer_authenticity": "not_attested",
        "applied": False,
        "not_before": evaluated_at,
        "expires_at": expires_at,
        "evidence_valid_until": holdout_valid_until.replace(
            microsecond=0
        ).isoformat(),
        "canary_basis_points": plan.canary_basis_points,
        "assignment_salt_sha256": plan.assignment_salt_sha256,
        "lineage": {
            "plan_sha256": plan.plan_sha256,
            "report_sha256": report.digest,
            "gate_policy_digest": gate_policy.digest,
            "route_policy_digest": route_policy.digest,
            "scorecard_digest": scorecard.digest,
            "training_source_digest": scorecard.source_digest,
            "evaluator_sha256": promotion_evaluator_sha256(),
            "pricing_sha256": plan.pricing_sha256,
            "attestation_policy_sha256": plan.attestation_policy_sha256,
            "execution_harness_sha256": plan.execution_harness_sha256,
            "runner_source_sha256": plan.runner_source_sha256,
            "paired_attestation_verifier_sha256": paired_verifier_sha256,
            "paired_proof_set_sha256": paired_proof_set_sha256,
        },
        "enabled_cells": enabled_cells,
        "invariants": {
            "monotone_less_premium_only": True,
            "privacy_budget_and_capability_guards_preserved": True,
            "runtime_integration_required_before_application": True,
            "trusted_signature_required_before_runtime_consumption": True,
        },
    }
    return report, ContentAddressedDocument(
        manifest_content, "manifest_sha256"
    )


def load_promotion_gate_policy(path: str | Path) -> PromotionGatePolicy:
    raw = _load_strict_json(Path(path))
    if not isinstance(raw, dict):
        raise VerifiedRoutingError("Promotion gate policy must be an object.")
    data = dict(raw)
    reject_unknown(data, _GATE_FIELDS, "promotion gate policy")
    missing = sorted(_GATE_FIELDS.difference(data))
    if missing:
        raise VerifiedRoutingError(
            f"Missing promotion gate policy fields: {', '.join(missing)}."
        )
    failures = data["blocking_failure_classes"]
    if not isinstance(failures, list):
        raise VerifiedRoutingError("blocking_failure_classes must be a list.")
    data["blocking_failure_classes"] = tuple(failures)
    return PromotionGatePolicy(**data)  # type: ignore[arg-type]


def load_evidence_plan(path: str | Path) -> VerifiedRoutingEvidencePlan:
    raw = _load_strict_json(Path(path))
    if not isinstance(raw, dict):
        raise VerifiedRoutingError("Evidence plan must be an object.")
    data = dict(raw)
    reject_unknown(data, _PLAN_FIELDS, "evidence plan")
    missing = sorted(_PLAN_REQUIRED_FIELDS.difference(data))
    if missing:
        raise VerifiedRoutingError(
            f"Missing evidence plan fields: {', '.join(missing)}."
        )
    raw_cases = data.pop("cases")
    if not isinstance(raw_cases, list):
        raise VerifiedRoutingError("Evidence plan cases must be a list.")
    cases = tuple(_promotion_case_from_payload(case) for case in raw_cases)
    if "pricing_contract" in data:
        pricing_contract = data["pricing_contract"]
        if not isinstance(pricing_contract, dict):
            raise VerifiedRoutingError(
                "Evidence plan pricing_contract must be an object."
            )
        data["pricing_contract"] = PricingContract.from_payload(
            pricing_contract
        )
    for name in (
        "attestation_policy_sha256",
        "execution_harness_sha256",
        "runner_source_sha256",
    ):
        data.setdefault(name, None)
    return VerifiedRoutingEvidencePlan(
        **data,  # type: ignore[arg-type]
        cases=cases,
    )


def load_promotion_cases(path: str | Path) -> tuple[PromotionCase, ...]:
    raw = _load_strict_json(Path(path))
    if not isinstance(raw, list):
        raise VerifiedRoutingError("Promotion cases JSON must be a list.")
    return tuple(_promotion_case_from_payload(item) for item in raw)


def load_promotion_outcome_payloads(
    path: str | Path,
) -> list[dict[str, object]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records: list[dict[str, object]] = []
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise VerifiedRoutingError(f"Unable to read outcomes: {source}.") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            raw = _loads_strict_json(
                line, f"{source} line {line_number}"
            )
            if not isinstance(raw, dict):
                raise VerifiedRoutingError(
                    f"Outcome line {line_number} must be an object."
                )
            records.append(dict(raw))
        return records
    raw = _load_strict_json(source)
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and set(raw) == {"records"}:
        items = raw["records"]
    else:
        raise VerifiedRoutingError(
            "Outcome JSON must be a list or an object containing only records."
        )
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise VerifiedRoutingError("Promotion outcomes must be a list of objects.")
    return [dict(item) for item in items]


def load_strict_promotion_json(path: str | Path) -> object:
    return _load_strict_json(Path(path))


def write_content_addressed_json(
    path: str | Path,
    document: ContentAddressedDocument | VerifiedRoutingEvidencePlan,
) -> None:
    payload = document.payload()
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == encoded:
            return
        raise VerifiedRoutingError(
            f"Refusing to replace existing content-addressed file: {destination}."
        )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise VerifiedRoutingError(
                    f"Concurrent writer produced different content: {destination}."
                )
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _evaluate_cell(
    pairs: Sequence[
        tuple[PromotionCase, VerifiedOutcomeRecord, VerifiedOutcomeRecord]
    ],
    *,
    gate_policy: PromotionGatePolicy,
    minimum_success_rate: float,
    cost_weight: float,
) -> dict[str, object]:
    case = pairs[0][0]
    baselines = [pair[1] for pair in pairs]
    candidates = [pair[2] for pair in pairs]
    baseline_successes = sum(_record_passed(record) for record in baselines)
    candidate_successes = sum(_record_passed(record) for record in candidates)
    samples = len(pairs)
    baseline_success_rate = baseline_successes / samples
    candidate_success_rate = candidate_successes / samples
    ci_lower, ci_upper = _wilson_interval(
        candidate_successes, samples, gate_policy.confidence_level
    )
    candidate_only_failures = sum(
        _record_passed(baseline) and not _record_passed(candidate)
        for baseline, candidate in zip(baselines, candidates)
    )
    blocking_failure_count = sum(
        _failure_is_blocking(record, gate_policy)
        for record in [*baselines, *candidates]
    )
    baseline_latency = _mean([record.latency_ms for record in baselines])
    candidate_latency = _mean([record.latency_ms for record in candidates])
    baseline_p95 = _p95([record.latency_ms for record in baselines])
    candidate_p95 = _p95([record.latency_ms for record in candidates])
    baseline_tokens = _mean(
        [record.prompt_tokens + record.completion_tokens for record in baselines]
    )
    candidate_tokens = _mean(
        [record.prompt_tokens + record.completion_tokens for record in candidates]
    )
    baseline_premium = _mean([record.premium_calls for record in baselines])
    candidate_premium = _mean([record.premium_calls for record in candidates])
    baseline_egress = _mean([record.remote_payload_chars for record in baselines])
    candidate_egress = _mean([record.remote_payload_chars for record in candidates])
    baseline_exact_costs = tuple(_exact_paired_cost(record) for record in baselines)
    candidate_exact_costs = tuple(_exact_paired_cost(record) for record in candidates)
    cost_complete = all(
        value is not None
        for value in (*baseline_exact_costs, *candidate_exact_costs)
    )
    baseline_cost_total = (
        _decimal_sum(value for value in baseline_exact_costs if value is not None)
        if cost_complete
        else None
    )
    candidate_cost_total = (
        _decimal_sum(value for value in candidate_exact_costs if value is not None)
        if cost_complete
        else None
    )
    # Exact Decimal totals decide eligibility.  Float values are only a legacy,
    # human-readable projection in the generated report.
    baseline_cost = (
        float(_decimal_mean(baseline_cost_total, samples))
        if baseline_cost_total is not None
        else None
    )
    candidate_cost = (
        float(_decimal_mean(candidate_cost_total, samples))
        if candidate_cost_total is not None
        else None
    )
    cost_required = gate_policy.require_complete_cost_evidence or cost_weight > 0
    cost_evidence_complete = cost_complete or not cost_required
    latency_ratio = (
        candidate_p95 / baseline_p95
        if baseline_p95 > 0
        else (1.0 if candidate_p95 == 0 else math.inf)
    )
    monotone_cost = (
        cost_evidence_complete
        and (
            baseline_cost_total is None
            or candidate_cost_total is None
            or candidate_cost_total <= baseline_cost_total
        )
    )
    pairwise_premium_increases = sum(
        candidate.premium_calls > baseline.premium_calls
        for baseline, candidate in zip(baselines, candidates)
    )
    pairwise_egress_increases = sum(
        candidate.remote_payload_chars > baseline.remote_payload_chars
        for baseline, candidate in zip(baselines, candidates)
    )
    pairwise_cost_increases = sum(
        candidate_cost is not None
        and baseline_cost is not None
        and candidate_cost > baseline_cost
        for baseline_cost, candidate_cost in zip(
            baseline_exact_costs,
            candidate_exact_costs,
        )
    )
    improved_dimensions: list[str] = []
    if candidate_successes > baseline_successes:
        improved_dimensions.append("success")
    threshold = gate_policy.minimum_relative_improvement
    for name, baseline_value, candidate_value in (
        ("latency", baseline_latency, candidate_latency),
        ("tokens", baseline_tokens, candidate_tokens),
        ("premium_calls", baseline_premium, candidate_premium),
        ("egress", baseline_egress, candidate_egress),
    ):
        if _relative_improvement(baseline_value, candidate_value) >= threshold:
            improved_dimensions.append(name)
    if (
        baseline_cost_total is not None
        and candidate_cost_total is not None
        and _decimal_relative_improvement_at_least(
            baseline_cost_total,
            candidate_cost_total,
            threshold,
        )
    ):
        improved_dimensions.append("cost")
    passed = (
        candidate_only_failures == 0
        and candidate_success_rate >= minimum_success_rate
        and ci_lower >= minimum_success_rate
        and blocking_failure_count == 0
        and cost_evidence_complete
        and latency_ratio <= gate_policy.maximum_candidate_latency_ratio
        and candidate_p95 <= gate_policy.maximum_candidate_p95_latency_ms
        and candidate_premium <= baseline_premium
        and candidate_egress <= baseline_egress
        and monotone_cost
        and pairwise_premium_increases == 0
        and pairwise_egress_increases == 0
        and pairwise_cost_increases == 0
        and bool(improved_dimensions)
    )
    return {
        "profile": case.profile,
        "capabilities": list(case.capabilities),
        "difficulty": case.difficulty,
        "baseline_route": case.baseline_route,
        "candidate_route": case.candidate_route,
        "config_sha256": case.config_sha256,
        "signal_provider_config_sha256": case.signal_provider_config_sha256,
        "runtime_plan_sha256": case.runtime_plan_sha256,
        "paired_tasks": samples,
        "baseline_success_rate": _rounded(baseline_success_rate),
        "candidate_success_rate": _rounded(candidate_success_rate),
        "candidate_success_ci_lower": _rounded(ci_lower),
        "candidate_success_ci_upper": _rounded(ci_upper),
        "candidate_only_failures": candidate_only_failures,
        "baseline_non_successes": samples - baseline_successes,
        "candidate_non_successes": samples - candidate_successes,
        "baseline_p95_latency_ms": _rounded(baseline_p95),
        "candidate_p95_latency_ms": _rounded(candidate_p95),
        "candidate_latency_ratio": _rounded(latency_ratio),
        "baseline_mean_tokens": _rounded(baseline_tokens),
        "candidate_mean_tokens": _rounded(candidate_tokens),
        "baseline_mean_premium_calls": _rounded(baseline_premium),
        "candidate_mean_premium_calls": _rounded(candidate_premium),
        "baseline_mean_egress_chars": _rounded(baseline_egress),
        "candidate_mean_egress_chars": _rounded(candidate_egress),
        "baseline_mean_cost_usd": (
            None if baseline_cost is None else _rounded(baseline_cost)
        ),
        "candidate_mean_cost_usd": (
            None if candidate_cost is None else _rounded(candidate_cost)
        ),
        "pairwise_premium_increases": pairwise_premium_increases,
        "pairwise_egress_increases": pairwise_egress_increases,
        "pairwise_cost_increases": pairwise_cost_increases,
        "cost_evidence_complete": cost_evidence_complete,
        "blocking_failure_count": blocking_failure_count,
        "improved_dimensions": improved_dimensions,
        "passed": passed,
    }


def _validate_static_bindings(
    plan: VerifiedRoutingEvidencePlan,
    gate_policy: PromotionGatePolicy,
    route_policy: VerifiedRoutePolicy,
    scorecard: RouteScorecard,
) -> None:
    bindings = {
        "route policy": (plan.route_policy_digest, route_policy.digest),
        "scorecard": (plan.scorecard_digest, scorecard.digest),
        "training source": (
            plan.training_source_digest,
            scorecard.source_digest,
        ),
        "gate policy": (plan.gate_policy_digest, gate_policy.digest),
    }
    mismatched = [name for name, values in bindings.items() if values[0] != values[1]]
    if mismatched:
        raise VerifiedRoutingError(
            f"Evidence plan binding mismatch: {', '.join(mismatched)}."
        )
    if plan.profile not in route_policy.profiles:
        raise VerifiedRoutingError("Evidence plan profile is not in the route policy.")
    if plan.canary_basis_points > gate_policy.maximum_canary_basis_points:
        raise VerifiedRoutingError("Evidence plan canary size exceeds the gate policy.")
    if plan.manifest_ttl_seconds > gate_policy.maximum_manifest_ttl_seconds:
        raise VerifiedRoutingError("Evidence plan TTL exceeds the gate policy.")


def _validate_scorecard_lineage(
    scorecard: RouteScorecard,
    training: Sequence[VerifiedOutcomeRecord],
) -> None:
    if _record_set_digest(training) != scorecard.source_digest:
        raise VerifiedRoutingError(
            "Scorecard source digest does not match supplied training outcomes."
        )
    generated = _timestamp(scorecard.generated_at)
    expires = _timestamp(scorecard.expires_at)
    ttl_seconds = int((expires - generated).total_seconds())
    rebuilt = build_route_scorecard(
        training,
        minimum_evidence_strength=scorecard.minimum_evidence_strength,
        minimum_confidence=scorecard.minimum_confidence,
        generated_at=scorecard.generated_at,
        ttl_seconds=ttl_seconds,
    )
    if rebuilt.digest != scorecard.digest:
        raise VerifiedRoutingError(
            "Scorecard aggregates do not match supplied training outcomes."
        )


def _pair_static_matches_plan(
    case: PromotionCase,
    baseline: VerifiedOutcomeRecord,
    candidate: VerifiedOutcomeRecord,
) -> bool:
    expected = (
        case.task_fingerprint,
        case.profile,
        case.capabilities,
        case.difficulty,
        case.config_sha256,
        case.signal_provider_config_sha256,
        case.runtime_plan_sha256,
    )
    for record in (baseline, candidate):
        observed = (
            record.task_fingerprint,
            record.profile,
            record.capabilities,
            record.difficulty,
            record.config_sha256,
            record.signal_provider_config_sha256,
            record.runtime_plan_sha256,
        )
        if observed != expected:
            return False
    return (
        baseline.source == candidate.source
        and baseline.confidence == candidate.confidence
        and baseline.route_receipt_sha256 != candidate.route_receipt_sha256
    )


def _holdout_evidence_replay_errors(
    records: Sequence[VerifiedOutcomeRecord],
) -> int:
    """Allow shared evidence only inside one fully bound AB/BA execution.

    A deterministic or independent verifier can legitimately produce identical
    evidence for both routes.  Distinct durable claims, bindings, and route
    receipts carry replay protection; evidence equality alone is not a replay.
    """

    groups: dict[str, list[VerifiedOutcomeRecord]] = {}
    for record in records:
        groups.setdefault(record.evidence_sha256, []).append(record)
    errors = 0
    for group in groups.values():
        if len(group) == 1:
            continue
        bindings = [record.paired_run for record in group]
        if len(group) != 2 or any(binding is None for binding in bindings):
            errors += len(group) - 1
            continue
        run_ids = {str(binding["run_id"]) for binding in bindings if binding}
        claims = {str(binding["claim_sha256"]) for binding in bindings if binding}
        binding_ids = {
            str(binding["binding_sha256"]) for binding in bindings if binding
        }
        receipts = {record.route_receipt_sha256 for record in group}
        if not (
            len(run_ids) == 1
            and len(claims) == 2
            and len(binding_ids) == 2
            and len(receipts) == 2
        ):
            errors += len(group) - 1
    return errors


def _training_pair_uniqueness_errors(
    records: Sequence[VerifiedOutcomeRecord],
    proofs: Mapping[str, VerifiedPairedEvidence],
) -> dict[str, int]:
    """Require one globally unique semantic item per complete training pair."""

    by_item: dict[
        str,
        list[tuple[VerifiedOutcomeRecord, PairedOutcomeBinding]],
    ] = {}
    task_items: dict[str, set[str]] = {}
    task_runs: dict[str, set[str]] = {}
    run_items: dict[str, set[str]] = {}
    nonce_items: dict[str, set[str]] = {}
    missing_proofs = 0
    for record in records:
        proof = proofs.get(record.record_id)
        if proof is None:
            missing_proofs += 1
            continue
        binding = proof.paired_outcome_binding
        item = binding.normalized_item_sha256
        by_item.setdefault(item, []).append((record, binding))
        task_items.setdefault(binding.task_fingerprint, set()).add(item)
        task_runs.setdefault(binding.task_fingerprint, set()).add(binding.run_id)
        run_items.setdefault(binding.run_id, set()).add(item)
        nonce_items.setdefault(binding.run_instance_nonce, set()).add(item)

    invalid_groups = 0
    for item, group in by_item.items():
        records_by_slot = {
            binding.slot: (record, binding) for record, binding in group
        }
        tasks = {binding.task_fingerprint for _, binding in group}
        run_ids = {binding.run_id for _, binding in group}
        nonces = {binding.run_instance_nonce for _, binding in group}
        cells = {
            (
                record.profile,
                record.config_sha256,
                record.signal_provider_config_sha256,
                record.runtime_plan_sha256,
                record.capabilities,
                record.difficulty,
            )
            for record, _ in group
        }
        claims = {binding.claim_sha256 for _, binding in group}
        bindings = {binding.binding_sha256 for _, binding in group}
        record_ids = {record.record_id for record, _ in group}
        structurally_complete = (
            len(group) == 2
            and len(records_by_slot) == 2
            and set(records_by_slot) == {"A", "B"}
            and len(tasks) == 1
            and len(run_ids) == 1
            and len(nonces) == 1
            and len(cells) == 1
            and len(claims) == 2
            and len(bindings) == 2
            and len(record_ids) == 2
            and all(
                binding.normalized_item_sha256 == item
                and binding.task_fingerprint == record.task_fingerprint
                and binding.route == record.planned_route
                for record, binding in group
            )
        )
        if not structurally_complete:
            invalid_groups += 1
            continue
        _, baseline = records_by_slot["A"]
        _, candidate = records_by_slot["B"]
        first_record, first = min(group, key=lambda value: value[1].ordinal)
        second_record, second = max(group, key=lambda value: value[1].ordinal)
        if not (
            baseline.arm == "baseline"
            and baseline.route == baseline.baseline_route
            and candidate.arm == "candidate"
            and candidate.route == candidate.candidate_route
            and baseline.baseline_route == candidate.baseline_route
            and baseline.candidate_route == candidate.candidate_route
            and baseline.order == candidate.order
            and {first.ordinal, second.ordinal} == {0, 1}
            and first.previous_record_id is None
            and second.previous_record_id == first_record.record_id
            and first_record.record_id != second_record.record_id
        ):
            invalid_groups += 1

    return {
        "records_without_paired_proof": missing_proofs,
        "invalid_normalized_item_pairs": invalid_groups,
        "tasks_with_multiple_items": sum(
            len(items) != 1 for items in task_items.values()
        ),
        "tasks_with_multiple_runs": sum(
            len(runs) != 1 for runs in task_runs.values()
        ),
        "runs_with_multiple_items": sum(
            len(items) != 1 for items in run_items.values()
        ),
        "run_nonces_with_multiple_items": sum(
            len(items) != 1 for items in nonce_items.values()
        ),
    }


def _paired_lineage_matches_plan(
    case: PromotionCase,
    baseline: VerifiedOutcomeRecord,
    candidate: VerifiedOutcomeRecord,
    *,
    plan_sha256: str,
    pricing_contract: PricingContract | None,
) -> bool:
    if (
        pricing_contract is None
        or baseline.paired_run is None
        or candidate.paired_run is None
        or baseline.paired_cost is None
        or candidate.paired_cost is None
    ):
        return False
    try:
        baseline_binding = PairedOutcomeBinding.from_payload(
            baseline.paired_run
        )
        candidate_binding = PairedOutcomeBinding.from_payload(
            candidate.paired_run
        )
        baseline_cost = PairedCostEvidence.from_payload(
            baseline.payload()["paired_cost"],  # type: ignore[arg-type]
            pricing=pricing_contract,
        )
        candidate_cost = PairedCostEvidence.from_payload(
            candidate.payload()["paired_cost"],  # type: ignore[arg-type]
            pricing=pricing_contract,
        )
    except VerifiedRoutingError:
        return False
    pricing_sha256 = pricing_contract.pricing_sha256
    if (
        baseline_cost.pricing_sha256 != pricing_sha256
        or candidate_cost.pricing_sha256 != pricing_sha256
        or not _paired_cost_matches_outcome(baseline_cost, baseline)
        or not _paired_cost_matches_outcome(candidate_cost, candidate)
    ):
        return False
    case_sha256 = sha256_json(case.payload())
    shared_expected = (
        plan_sha256,
        case_sha256,
        case.task_fingerprint,
        case.normalized_item_sha256,
        case.config_sha256,
        case.order,
        case.baseline_route,
        case.candidate_route,
    )
    for binding in (baseline_binding, candidate_binding):
        observed = (
            binding.plan_sha256,
            binding.case_sha256,
            binding.task_fingerprint,
            binding.normalized_item_sha256,
            binding.bridge_config_sha256,
            binding.order,
            binding.baseline_route,
            binding.candidate_route,
        )
        if observed != shared_expected:
            return False
    shared_lineage = (
        baseline_binding.run_id,
        baseline_binding.source_snapshot_sha256,
        baseline_binding.executor_config_sha256,
        baseline_binding.lifecycle_config_sha256,
        baseline_binding.signals_sha256,
        baseline_binding.runner_sha256,
        baseline_binding.pricing_sha256,
        baseline_binding.run_instance_nonce,
    )
    if baseline_binding.pricing_sha256 != pricing_sha256 or shared_lineage != (
        candidate_binding.run_id,
        candidate_binding.source_snapshot_sha256,
        candidate_binding.executor_config_sha256,
        candidate_binding.lifecycle_config_sha256,
        candidate_binding.signals_sha256,
        candidate_binding.runner_sha256,
        candidate_binding.pricing_sha256,
        candidate_binding.run_instance_nonce,
    ):
        return False
    if (
        baseline_binding.arm != "baseline"
        or baseline_binding.slot != "A"
        or baseline_binding.route != case.baseline_route
        or candidate_binding.arm != "candidate"
        or candidate_binding.slot != "B"
        or candidate_binding.route != case.candidate_route
        or baseline_binding.claim_sha256 == candidate_binding.claim_sha256
    ):
        return False
    first, second = (
        (baseline_binding, candidate_binding)
        if case.order == "AB"
        else (candidate_binding, baseline_binding)
    )
    first_record, second_record = (
        (baseline, candidate)
        if case.order == "AB"
        else (candidate, baseline)
    )
    return (
        first.ordinal == 0
        and second.ordinal == 1
        and first.previous_record_id is None
        and second.previous_record_id == first_record.record_id
        and second_record.record_id != first_record.record_id
        and _timestamp(first_record.created_at)
        <= _timestamp(second_record.created_at)
    )


def _paired_cost_matches_outcome(
    cost: PairedCostEvidence,
    record: VerifiedOutcomeRecord,
) -> bool:
    final_command = cost.commands[-1]
    return (
        sum(command.prompt_tokens for command in cost.commands)
        == record.prompt_tokens
        and sum(command.completion_tokens for command in cost.commands)
        == record.completion_tokens
        and final_command.provider_id == record.final_provider
        and final_command.model == record.model
        and final_command.provider_runtime_sha256
        == record.provider_runtime_sha256
    )


def _training_record_matches_cell(
    record: VerifiedOutcomeRecord,
    case: PromotionCase,
    route: str,
    *,
    scorecard: RouteScorecard,
) -> bool:
    return (
        record.profile == case.profile
        and record.config_sha256 == case.config_sha256
        and record.signal_provider_config_sha256
        == case.signal_provider_config_sha256
        and record.runtime_plan_sha256 == case.runtime_plan_sha256
        and record.planned_route == route
        and record.capabilities == case.capabilities
        and record.difficulty == case.difficulty
        and EVIDENCE_STRENGTHS.index(record.evidence_strength)
        >= EVIDENCE_STRENGTHS.index(scorecard.minimum_evidence_strength)
        and record.confidence >= scorecard.minimum_confidence
        and not record.abstained
        and record.outcome != "inconclusive"
    )


def _record_passed(record: VerifiedOutcomeRecord) -> bool:
    return record.outcome == "passed" and not record.abstained


def _failure_is_blocking(
    record: VerifiedOutcomeRecord,
    gate_policy: PromotionGatePolicy,
) -> bool:
    if record.failure_class == "none":
        return False
    if record.failure_class in gate_policy.blocking_failure_classes:
        return True
    return record.failure_class not in gate_policy.non_blocking_failure_classes


def _verify_concrete_paired_record(
    verifier: PairedAttestationVerifier,
    record: VerifiedOutcomeRecord,
    *,
    pricing: PricingContract,
) -> VerifiedPairedEvidence:
    """Narrow internal authority boundary, patched only by gate-isolation tests."""

    return PairedAttestationVerifier.verify_record(
        verifier,
        record,
        pricing=pricing,
    )


def _verify_paired_record_sets(
    training: Sequence[VerifiedOutcomeRecord],
    holdout: Sequence[VerifiedOutcomeRecord],
    *,
    verifier: PairedAttestationVerifier | None,
    pricing: PricingContract | None,
    expected_policy_sha256: str | None,
    expected_execution_harness_sha256: str | None,
    expected_runner_source_sha256: str | None,
    evaluated_at_epoch: float,
) -> tuple[dict[str, VerifiedPairedEvidence], tuple[str, ...], bool]:
    """Rebuild every authoritative row from DSSE/CAS before gate decisions."""

    if verifier is not None and type(verifier) is not PairedAttestationVerifier:
        raise TypeError("paired_verifier must be PairedAttestationVerifier or None.")
    if verifier is None or pricing is None:
        return {}, ("paired-verifier-or-pricing-missing",), False
    policy_matches = (
        expected_policy_sha256 is not None
        and expected_execution_harness_sha256 is not None
        and expected_runner_source_sha256 is not None
        and verifier.trust_config.policy.policy_sha256 == expected_policy_sha256
        and verifier.runner_source_sha256 == expected_runner_source_sha256
    )
    proofs: dict[str, VerifiedPairedEvidence] = {}
    errors: list[str] = []
    for record in (*training, *holdout):
        try:
            proof = _verify_concrete_paired_record(
                verifier,
                record,
                pricing=pricing,
            )
            if proof.latest_attestation_issued_at < proof.candidate_created_at:
                raise VerifiedRoutingError(
                    "Paired attestation predates its signed candidate binding."
                )
            if proof.latest_attestation_issued_at > evaluated_at_epoch:
                raise VerifiedRoutingError(
                    "Paired attestation was issued after evaluation time."
                )
            if (
                proof.paired_outcome_binding.execution_harness_sha256
                != expected_execution_harness_sha256
                or proof.paired_outcome_binding.runner_source_sha256
                != expected_runner_source_sha256
            ):
                raise VerifiedRoutingError(
                    "Paired proof does not match the preregistered execution harness."
                )
            proofs[record.record_id] = proof
        except (ValueError, OSError) as exc:
            errors.append(
                sha256_json(
                    {
                        "record_id": record.record_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            )
    return proofs, tuple(sorted(errors)), policy_matches


def _normalize_records(
    records: Iterable[VerifiedOutcomeRecord | Mapping[str, object]],
    label: str,
) -> tuple[VerifiedOutcomeRecord, ...]:
    normalized = tuple(
        record
        if isinstance(record, VerifiedOutcomeRecord)
        else VerifiedOutcomeRecord.from_payload(record)
        for record in records
    )
    record_ids = [record.record_id for record in normalized]
    if len(record_ids) != len(set(record_ids)):
        raise VerifiedRoutingError(f"{label} outcome record_ids must be unique.")
    return tuple(sorted(normalized, key=lambda record: record.record_id))


def _record_set_digest(records: Sequence[VerifiedOutcomeRecord]) -> str:
    ordered = sorted(records, key=lambda item: item.record_id)
    return sha256_json({"records": [record.payload() for record in ordered]})


def _promotion_case_from_payload(raw: object) -> PromotionCase:
    if not isinstance(raw, Mapping):
        raise VerifiedRoutingError("Promotion case must be an object.")
    data = dict(raw)
    reject_unknown(data, _CASE_FIELDS, "promotion case")
    missing = sorted(_CASE_FIELDS.difference(data))
    if missing:
        raise VerifiedRoutingError(
            f"Missing promotion case fields: {', '.join(missing)}."
        )
    capabilities = data.pop("capabilities")
    if not isinstance(capabilities, list):
        raise VerifiedRoutingError("Promotion case capabilities must be a list.")
    return PromotionCase(
        **data,  # type: ignore[arg-type]
        capabilities=tuple(capabilities),
    )


def _add_check(
    checks: list[dict[str, object]],
    check_id: str,
    passed: bool,
    failure_kind: str,
    details: Mapping[str, object] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "failure_kind": failure_kind,
            "details": dict(details or {}),
        }
    )


def _load_strict_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerifiedRoutingError(f"Unable to read JSON file: {path}.") from exc
    return _loads_strict_json(text, str(path))


def _loads_strict_json(text: str, label: str) -> object:
    def reject_constant(value: str) -> object:
        raise VerifiedRoutingError(f"Non-finite JSON number is forbidden: {value}.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise VerifiedRoutingError(f"Duplicate JSON key is forbidden: {key}.")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise VerifiedRoutingError(f"Invalid JSON in {label}.") from exc


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _authoritative_candidate_time(
    record: VerifiedOutcomeRecord,
    proofs: Mapping[str, VerifiedPairedEvidence],
) -> datetime:
    proof = proofs.get(record.record_id)
    if proof is None:
        return _timestamp(record.created_at)
    return datetime.fromtimestamp(
        proof.candidate_created_at,
        tz=timezone.utc,
    )


def _wilson_interval(
    successes: int,
    total: int,
    confidence_level: float,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    probability = successes / total
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    denominator = 1.0 + (z * z / total)
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean(values: Sequence[float | int]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _exact_paired_cost(record: VerifiedOutcomeRecord) -> Decimal | None:
    if record.paired_cost is None:
        return None
    try:
        evidence = PairedCostEvidence.from_payload(
            record.payload()["paired_cost"]  # type: ignore[arg-type]
        )
    except VerifiedRoutingError:
        return None
    return Decimal(evidence.total_cost_usd)


def _decimal_sum(values: Iterable[Decimal]) -> Decimal:
    items = tuple(values)
    with localcontext() as context:
        context.prec = 160
        return sum(items, Decimal(0))


def _decimal_mean(total: Decimal, samples: int) -> Decimal:
    if samples <= 0:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 160
        return total / Decimal(samples)


def _decimal_relative_improvement_at_least(
    baseline: Decimal,
    candidate: Decimal,
    threshold: float,
) -> bool:
    if baseline <= 0:
        return False
    with localcontext() as context:
        context.prec = 160
        return baseline - candidate >= baseline * Decimal(str(threshold))


def _p95(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - candidate) / baseline


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _cell_sort_key(value: tuple[object, ...]) -> str:
    return json.dumps(value, sort_keys=True)
