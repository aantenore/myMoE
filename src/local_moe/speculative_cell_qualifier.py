"""Deterministic, non-authorizing speculative-cell qualification."""

from __future__ import annotations

from math import ceil
from statistics import median
from typing import Iterable

from .speculative_cell_contracts import (
    REGIMES,
    SpeculativeCellContractError,
    SpeculativeQualificationPlan,
    SpeculativeQualificationReceipt,
    SpeculativeTrial,
)
from .verified_routing_contracts import sha256_json


def expected_trial_order(
    plan: SpeculativeQualificationPlan,
    case_sha256: str,
    regime: str,
    repetition: int,
) -> str:
    """Return the preregistered AB/BA order with balanced position parity."""

    try:
        case_index = plan.case_sha256s.index(case_sha256)
        regime_index = plan.required_regimes.index(regime)
    except ValueError as exc:
        raise SpeculativeCellContractError(
            "Trial case or regime is outside the frozen plan."
        ) from exc
    if repetition < 0 or repetition >= plan.policy.trials_per_case:
        raise SpeculativeCellContractError("Trial repetition is outside the plan.")
    seed_parity = int(plan.order_seed_sha256[-1], 16) % 2
    parity = (seed_parity + case_index + regime_index + repetition) % 2
    return "AB" if parity == 0 else "BA"


def expected_trial_sequence_index(
    plan: SpeculativeQualificationPlan,
    case_sha256: str,
    regime: str,
    repetition: int,
) -> int:
    """Return the frozen global position for one stateless paired trial."""

    try:
        case_index = plan.case_sha256s.index(case_sha256)
        regime_index = plan.required_regimes.index(regime)
    except ValueError as exc:
        raise SpeculativeCellContractError(
            "Trial case or regime is outside the frozen plan."
        ) from exc
    if repetition < 0 or repetition >= plan.policy.trials_per_case:
        raise SpeculativeCellContractError("Trial repetition is outside the plan.")
    return (
        case_index * len(plan.required_regimes) + regime_index
    ) * plan.policy.trials_per_case + repetition


def validate_speculative_plan_implementation(
    plan: SpeculativeQualificationPlan,
) -> None:
    """Fail if a valid plan targets another installed adapter contract."""

    if not isinstance(plan, SpeculativeQualificationPlan):
        raise SpeculativeCellContractError("Qualification plan is invalid.")
    from .llama_cpp_speculative_adapter import (
        LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
    )

    if plan.execution.adapter_contract_sha256 != LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256:
        raise SpeculativeCellContractError(
            "Plan adapter does not match this qualifier implementation."
        )


def qualify_speculative_cell(
    plan: SpeculativeQualificationPlan,
    trials: Iterable[SpeculativeTrial],
) -> SpeculativeQualificationReceipt:
    """Evaluate an exact speculative cell without authorizing its activation."""

    validate_speculative_plan_implementation(plan)
    observations = tuple(trials)
    if len(observations) > plan.expected_trial_count:
        raise SpeculativeCellContractError("Evidence exceeds the frozen trial plan.")

    by_key: dict[tuple[str, str, int], SpeculativeTrial] = {}
    order_mismatches = 0
    schedule_mismatches = 0
    for observed_index, trial in enumerate(observations):
        if not isinstance(trial, SpeculativeTrial):
            raise SpeculativeCellContractError("Trial evidence is invalid.")
        if trial.plan_sha256 != plan.digest:
            raise SpeculativeCellContractError("Trial is bound to a different plan.")
        if trial.case_sha256 not in plan.case_sha256s:
            raise SpeculativeCellContractError("Trial case is outside the plan.")
        if trial.regime not in plan.required_regimes:
            raise SpeculativeCellContractError("Trial regime is outside the plan.")
        if trial.repetition >= plan.policy.trials_per_case:
            raise SpeculativeCellContractError("Trial repetition is outside the plan.")
        if trial.baseline.cell_sha256 != plan.baseline.digest:
            raise SpeculativeCellContractError("Baseline measurement changed cells.")
        if trial.candidate.cell_sha256 != plan.candidate.digest:
            raise SpeculativeCellContractError("Candidate measurement changed cells.")
        key = (trial.case_sha256, trial.regime, trial.repetition)
        if key in by_key:
            raise SpeculativeCellContractError("Trial evidence contains duplicates.")
        by_key[key] = trial
        if trial.order != expected_trial_order(plan, *key):
            order_mismatches += 1
        if (
            trial.sequence_index != expected_trial_sequence_index(plan, *key)
            or trial.sequence_index != observed_index
        ):
            schedule_mismatches += 1

    expected_keys = {
        (case_sha256, regime, repetition)
        for case_sha256 in plan.case_sha256s
        for regime in plan.required_regimes
        for repetition in range(plan.policy.trials_per_case)
    }
    missing = expected_keys - set(by_key)
    counts = {
        regime: sum(trial.regime == regime for trial in observations)
        for regime in REGIMES
    }
    failed_arms = sum(
        int(not trial.baseline.success) + int(not trial.candidate.success)
        for trial in observations
    )
    output_mismatches = sum(
        trial.baseline.success
        and trial.candidate.success
        and (
            trial.baseline.output_sha256 != trial.candidate.output_sha256
            or trial.baseline.predicted_tokens != trial.candidate.predicted_tokens
        )
        for trial in observations
    )
    evidence_sha256 = sha256_json(
        {
            "schema_version": "1.0",
            "plan_sha256": plan.digest,
            "trial_sha256s": [trial.digest for trial in observations],
        }
    )

    abstention_reasons: set[str] = set()
    if missing:
        abstention_reasons.add("evidence_incomplete")
    if order_mismatches or not _is_counterbalanced(observations):
        abstention_reasons.add("evidence_not_counterbalanced")
    if schedule_mismatches:
        abstention_reasons.add("evidence_schedule_mismatch")
    if failed_arms:
        abstention_reasons.add("evidence_execution_failed")
    if abstention_reasons:
        return _receipt(
            plan,
            evidence_sha256=evidence_sha256,
            decision="abstained",
            reasons=abstention_reasons,
            observations=observations,
            counts=counts,
            failed_arms=failed_arms,
            output_mismatches=output_mismatches,
        )

    regime_metrics = {
        regime: _regime_metrics(
            tuple(trial for trial in observations if trial.regime == regime)
        )
        for regime in REGIMES
    }
    candidate_peak = max(
        int(trial.candidate.peak_memory_bytes) for trial in observations
    )
    acceptance_missing = any(
        metrics["acceptance_rate"] is None for metrics in regime_metrics.values()
    )

    if acceptance_missing:
        return _receipt(
            plan,
            evidence_sha256=evidence_sha256,
            decision="abstained",
            reasons={"candidate_acceptance_missing"},
            observations=observations,
            counts=counts,
            failed_arms=0,
            output_mismatches=output_mismatches,
            regime_metrics=regime_metrics,
            candidate_peak_memory_bytes=candidate_peak,
        )

    rejection_reasons: set[str] = set()
    if output_mismatches:
        rejection_reasons.add("exact_output_mismatch")
    if any(
        float(metrics["median_speedup_ratio"])
        < plan.policy.minimum_median_speedup_ratio
        for metrics in regime_metrics.values()
    ):
        rejection_reasons.add("median_speedup_below_threshold")
    if any(
        float(metrics["p95_latency_ratio"]) > plan.policy.maximum_p95_latency_ratio
        for metrics in regime_metrics.values()
    ):
        rejection_reasons.add("p95_latency_regression")
    if any(
        float(metrics["p95_ttft_ratio"]) > plan.policy.maximum_p95_ttft_ratio
        for metrics in regime_metrics.values()
    ):
        rejection_reasons.add("p95_ttft_regression")
    if candidate_peak > plan.policy.maximum_candidate_peak_memory_bytes:
        rejection_reasons.add("candidate_memory_budget_exceeded")
    if any(
        float(metrics["acceptance_rate"]) < plan.policy.minimum_acceptance_rate
        for metrics in regime_metrics.values()
    ):
        rejection_reasons.add("candidate_acceptance_below_threshold")

    return _receipt(
        plan,
        evidence_sha256=evidence_sha256,
        decision="rejected" if rejection_reasons else "qualified",
        reasons=rejection_reasons,
        observations=observations,
        counts=counts,
        failed_arms=0,
        output_mismatches=output_mismatches,
        regime_metrics=regime_metrics,
        candidate_peak_memory_bytes=candidate_peak,
    )


def _throughput(arm) -> float:
    return float(arm.predicted_tokens) * 1000.0 / float(arm.predicted_ms)


def _regime_metrics(
    observations: tuple[SpeculativeTrial, ...],
) -> dict[str, float | None]:
    if not observations:
        raise SpeculativeCellContractError("Regime evidence is incomplete.")
    speedups = tuple(
        _throughput(trial.candidate) / _throughput(trial.baseline)
        for trial in observations
    )
    baseline_p95 = _p95(
        float(trial.baseline.total_latency_ms) for trial in observations
    )
    candidate_p95 = _p95(
        float(trial.candidate.total_latency_ms) for trial in observations
    )
    baseline_ttft_p95 = _p95(float(trial.baseline.ttft_ms) for trial in observations)
    candidate_ttft_p95 = _p95(float(trial.candidate.ttft_ms) for trial in observations)
    draft_pairs = tuple(
        (
            trial.candidate.draft_generated_tokens,
            trial.candidate.draft_accepted_tokens,
        )
        for trial in observations
    )
    acceptance_missing = any(
        generated is None or accepted is None for generated, accepted in draft_pairs
    )
    generated_total = sum(int(generated or 0) for generated, _ in draft_pairs)
    accepted_total = sum(int(accepted or 0) for _, accepted in draft_pairs)
    return {
        "median_speedup_ratio": median(speedups),
        "p95_latency_ratio": candidate_p95 / baseline_p95,
        "p95_ttft_ratio": candidate_ttft_p95 / baseline_ttft_p95,
        "acceptance_rate": (
            None
            if acceptance_missing or generated_total == 0
            else accepted_total / generated_total
        ),
    }


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered or ordered[0] <= 0:
        raise SpeculativeCellContractError("Latency evidence is invalid.")
    return ordered[ceil(0.95 * len(ordered)) - 1]


def _is_counterbalanced(observations: tuple[SpeculativeTrial, ...]) -> bool:
    for regime in REGIMES:
        orders = [trial.order for trial in observations if trial.regime == regime]
        if orders and abs(orders.count("AB") - orders.count("BA")) > 1:
            return False
    return True


def _receipt(
    plan: SpeculativeQualificationPlan,
    *,
    evidence_sha256: str,
    decision: str,
    reasons: set[str],
    observations: tuple[SpeculativeTrial, ...],
    counts: dict[str, int],
    failed_arms: int,
    output_mismatches: int,
    regime_metrics: dict[str, dict[str, float | None]] | None = None,
    candidate_peak_memory_bytes: int | None = None,
) -> SpeculativeQualificationReceipt:
    cold = regime_metrics.get("cold") if regime_metrics is not None else None
    warm = regime_metrics.get("warm") if regime_metrics is not None else None

    def metric(values: dict[str, float | None] | None, name: str) -> float | None:
        if values is None or values[name] is None:
            return None
        return float(values[name])

    return SpeculativeQualificationReceipt(
        plan_sha256=plan.digest,
        evidence_sha256=evidence_sha256,
        decision=decision,
        reason_codes=tuple(sorted(reasons)),
        expected_trials=plan.expected_trial_count,
        expected_cases=len(plan.case_sha256s),
        trials_per_case=plan.policy.trials_per_case,
        observed_trials=len(observations),
        unique_cases=len({trial.case_sha256 for trial in observations}),
        cold_trials=counts["cold"],
        warm_trials=counts["warm"],
        failed_arms=failed_arms,
        output_mismatches=output_mismatches,
        cold_median_speedup_ratio=metric(cold, "median_speedup_ratio"),
        cold_p95_latency_ratio=metric(cold, "p95_latency_ratio"),
        cold_p95_ttft_ratio=metric(cold, "p95_ttft_ratio"),
        cold_candidate_acceptance_rate=metric(cold, "acceptance_rate"),
        warm_median_speedup_ratio=metric(warm, "median_speedup_ratio"),
        warm_p95_latency_ratio=metric(warm, "p95_latency_ratio"),
        warm_p95_ttft_ratio=metric(warm, "p95_ttft_ratio"),
        warm_candidate_acceptance_rate=metric(warm, "acceptance_rate"),
        candidate_peak_memory_bytes=candidate_peak_memory_bytes,
    )
