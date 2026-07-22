from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from local_moe.llama_cpp_speculative_adapter import (
    LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
    parse_llama_cpp_completion,
)
from local_moe.speculative_cell_contracts import (
    SpeculativeArmMeasurement,
    SpeculativeCellBinding,
    SpeculativeCellContractError,
    SpeculativeExecutionBinding,
    SpeculativeQualificationPlan,
    SpeculativeQualificationPolicy,
    SpeculativeTrial,
)
from local_moe.speculative_cell_qualifier import (
    expected_trial_order,
    expected_trial_sequence_index,
    qualify_speculative_cell,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "outputs" / "speculative-cell-qualifier-contract.json"
PRIVATE_OUTPUT_MARKER = "PRIVATE-GENERATED-OUTPUT-MUST-NOT-APPEAR"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _plan() -> SpeculativeQualificationPlan:
    return SpeculativeQualificationPlan(
        plan_id="synthetic-ngram-cell",
        execution=SpeculativeExecutionBinding(
            runtime_revision_sha256=_sha("llama.cpp-synthetic-revision"),
            runtime_binary_sha256=_sha("synthetic-runtime-binary"),
            runtime_binding_manifest_sha256=_sha("synthetic-binding-manifest"),
            hardware_sha256=_sha("synthetic-unified-memory-machine"),
            target_model_sha256=_sha("synthetic-target-model"),
            shared_runtime_config_sha256=_sha("synthetic-shared-runtime-config"),
            request_policy_sha256=_sha("synthetic-text-only-request-policy"),
            regime_protocol_sha256=_sha("synthetic-cold-warm-protocol"),
            harness_sha256=_sha("synthetic-harness"),
            collector_sha256=_sha("synthetic-speed-bench-collector"),
            adapter_contract_sha256=LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
        ),
        baseline=SpeculativeCellBinding(
            cell_id="synthetic-baseline",
            speculation_config_sha256=_sha("speculation-none"),
            speculation_mode="none",
        ),
        candidate=SpeculativeCellBinding(
            cell_id="synthetic-ngram-simple",
            speculation_config_sha256=_sha("speculation-ngram-simple"),
            speculation_mode="ngram-simple",
        ),
        workload_sha256=_sha("synthetic-public-workload"),
        case_sha256s=tuple(sorted(_sha(f"public-case-{index}") for index in range(4))),
        order_seed_sha256=_sha("synthetic-order-seed"),
        policy=SpeculativeQualificationPolicy(
            trials_per_case=4,
            minimum_median_speedup_ratio=1.10,
            maximum_p95_latency_ratio=1.0,
            maximum_p95_ttft_ratio=1.05,
            minimum_acceptance_rate=0.05,
            maximum_candidate_peak_memory_bytes=2_000,
        ),
    )


def _arm(
    cell_sha256: str,
    *,
    candidate: bool,
    case_index: int,
    repetition: int,
    regime: str,
) -> SpeculativeArmMeasurement:
    cold_penalty = 100.0 if regime == "cold" else 0.0
    jitter = float(case_index * 7 + repetition * 3)
    if candidate:
        predicted_ms = 690.0 + cold_penalty + jitter
        total_ms = 835.0 + cold_penalty + jitter
        ttft_ms = 88.0 + cold_penalty / 5 + jitter / 10
        peak = 1_550
        draft_generated = 160
        draft_accepted = 112
    else:
        predicted_ms = 1_000.0 + cold_penalty + jitter
        total_ms = 1_120.0 + cold_penalty + jitter
        ttft_ms = 96.0 + cold_penalty / 5 + jitter / 10
        peak = 1_100
        draft_generated = None
        draft_accepted = None
    return SpeculativeArmMeasurement(
        cell_sha256=cell_sha256,
        success=True,
        output_sha256=_sha(f"stable-output-{case_index}"),
        ttft_ms=ttft_ms,
        total_latency_ms=total_ms,
        predicted_tokens=128,
        predicted_ms=predicted_ms,
        peak_memory_bytes=peak,
        draft_generated_tokens=draft_generated,
        draft_accepted_tokens=draft_accepted,
    )


def _trials(plan: SpeculativeQualificationPlan) -> tuple[SpeculativeTrial, ...]:
    observations = []
    for case_index, case_sha256 in enumerate(plan.case_sha256s):
        for regime in plan.required_regimes:
            for repetition in range(plan.policy.trials_per_case):
                observations.append(
                    SpeculativeTrial(
                        plan_sha256=plan.digest,
                        sequence_index=expected_trial_sequence_index(
                            plan, case_sha256, regime, repetition
                        ),
                        case_sha256=case_sha256,
                        repetition=repetition,
                        regime=regime,
                        order=expected_trial_order(
                            plan, case_sha256, regime, repetition
                        ),
                        baseline=_arm(
                            plan.baseline.digest,
                            candidate=False,
                            case_index=case_index,
                            repetition=repetition,
                            regime=regime,
                        ),
                        candidate=_arm(
                            plan.candidate.digest,
                            candidate=True,
                            case_index=case_index,
                            repetition=repetition,
                            regime=regime,
                        ),
                    )
                )
    return tuple(observations)


def run_benchmark() -> dict[str, Any]:
    plan = _plan()
    trials = _trials(plan)
    qualified = qualify_speculative_cell(plan, trials)

    regressed_trials = tuple(
        replace(
            trial,
            candidate=replace(
                trial.candidate,
                output_sha256=_sha(f"changed-output-{index}"),
                predicted_ms=1_400.0,
                total_latency_ms=1_500.0,
                ttft_ms=140.0,
                peak_memory_bytes=2_500,
                digest="",
            ),
            digest="",
        )
        for index, trial in enumerate(trials)
    )
    rejected = qualify_speculative_cell(plan, regressed_trials)
    abstained = qualify_speculative_cell(plan, trials[:-1])

    warm_regressed_trials = tuple(
        replace(
            trial,
            candidate=replace(
                trial.candidate,
                predicted_ms=2_000.0,
                total_latency_ms=1_900.0,
                ttft_ms=200.0,
                digest="",
            ),
            digest="",
        )
        if trial.regime == "warm"
        else trial
        for trial in trials
    )
    warm_regressed = qualify_speculative_cell(plan, warm_regressed_trials)

    schedule_drift_trials = list(trials)
    schedule_drift_trials[0], schedule_drift_trials[1] = (
        schedule_drift_trials[1],
        schedule_drift_trials[0],
    )
    schedule_drift = qualify_speculative_cell(plan, schedule_drift_trials)

    adapter_measurement = parse_llama_cpp_completion(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": PRIVATE_OUTPUT_MARKER,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 32},
            "timings": {
                "predicted_n": 32,
                "predicted_ms": 200.0,
                "draft_n": 40,
                "draft_n_accepted": 30,
            },
        },
        cell_sha256=plan.candidate.digest,
        ttft_ms=25.0,
        total_latency_ms=230.0,
        peak_memory_bytes=1_500,
    )
    agentic_surface_rejected = False
    try:
        parse_llama_cpp_completion(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": "synthetic-call"}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"completion_tokens": 1},
                "timings": {"predicted_n": 1, "predicted_ms": 10.0},
            },
            cell_sha256=plan.candidate.digest,
            ttft_ms=5.0,
            total_latency_ms=12.0,
            peak_memory_bytes=1_500,
        )
    except SpeculativeCellContractError:
        agentic_surface_rejected = True

    scenarios = {
        "qualified_cell": qualified.payload(),
        "regressed_cell": rejected.payload(),
        "warm_only_regression": warm_regressed.payload(),
        "incomplete_evidence": abstained.payload(),
        "schedule_drift": schedule_drift.payload(),
        "llama_cpp_adapter": {
            "adapter_sha256": LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
            "output_retained": False,
            "output_sha256": adapter_measurement.output_sha256,
            "draft_generated_tokens": adapter_measurement.draft_generated_tokens,
            "draft_accepted_tokens": adapter_measurement.draft_accepted_tokens,
            "agentic_surface_rejected": agentic_surface_rejected,
        },
    }
    serialized = json.dumps(scenarios, allow_nan=False, sort_keys=True)
    criteria = {
        "qualified_exact_cell_passes_every_gate": (
            qualified.decision == "qualified" and not qualified.reason_codes
        ),
        "qualified_receipt_never_activates_runtime": (
            not qualified.activation_authorized
            and qualified.authority == "host_attested_unsigned_advisory"
        ),
        "cold_and_warm_evidence_is_complete": (
            qualified.observed_trials == qualified.expected_trials == 32
            and qualified.cold_trials == 16
            and qualified.warm_trials == 16
        ),
        "paired_speedup_is_measured_not_declared": (
            (qualified.cold_median_speedup_ratio or 0) >= 1.10
            and (qualified.warm_median_speedup_ratio or 0) >= 1.10
            and (qualified.cold_p95_latency_ratio or 2) <= 1.0
            and (qualified.warm_p95_latency_ratio or 2) <= 1.0
        ),
        "quality_memory_latency_and_ttft_regressions_reject": (
            rejected.decision == "rejected"
            and {
                "exact_output_mismatch",
                "candidate_memory_budget_exceeded",
                "p95_latency_regression",
                "p95_ttft_regression",
            }.issubset(rejected.reason_codes)
        ),
        "incomplete_preregistered_evidence_abstains": (
            abstained.decision == "abstained"
            and "evidence_incomplete" in abstained.reason_codes
        ),
        "warm_regression_cannot_be_masked_by_cold_speedup": (
            warm_regressed.decision == "rejected"
            and {
                "median_speedup_below_threshold",
                "p95_latency_regression",
                "p95_ttft_regression",
            }.issubset(warm_regressed.reason_codes)
            and (warm_regressed.cold_median_speedup_ratio or 0) >= 1.10
            and (warm_regressed.warm_median_speedup_ratio or 1) < 1.0
        ),
        "global_schedule_drift_abstains": (
            schedule_drift.decision == "abstained"
            and "evidence_schedule_mismatch" in schedule_drift.reason_codes
        ),
        "acceptance_rate_is_separate_evidence": (
            qualified.cold_candidate_acceptance_rate == 0.7
            and qualified.warm_candidate_acceptance_rate == 0.7
        ),
        "adapter_keeps_only_output_digest": (
            PRIVATE_OUTPUT_MARKER not in serialized
            and scenarios["llama_cpp_adapter"]["output_retained"] is False
        ),
        "adapter_rejects_agentic_surfaces": agentic_surface_rejected,
        "baseline_and_candidate_bind_same_runtime_target_hardware": (
            plan.execution.runtime_revision_sha256
            == _sha("llama.cpp-synthetic-revision")
            and plan.execution.target_model_sha256 == _sha("synthetic-target-model")
            and plan.execution.hardware_sha256
            == _sha("synthetic-unified-memory-machine")
        ),
        "ab_ba_order_is_preregistered": all(
            trial.order
            == expected_trial_order(
                plan, trial.case_sha256, trial.regime, trial.repetition
            )
            for trial in trials
        ),
    }
    pass_count = sum(criteria.values())
    return {
        "schema_version": "1.0",
        "contract": "speculative_cell_qualifier",
        "benchmark": "deterministic_contract_fixture",
        "plan": {
            "plan_sha256": plan.digest,
            "baseline_cell_sha256": plan.baseline.digest,
            "candidate_cell_sha256": plan.candidate.digest,
            "candidate_mode": plan.candidate.speculation_mode,
            "case_count": len(plan.case_sha256s),
            "expected_trials": plan.expected_trial_count,
        },
        "scenarios": scenarios,
        "criteria": criteria,
        "pass_count": pass_count,
        "check_count": len(criteria),
        "contract_checks_passed": pass_count == len(criteria),
        "limits": [
            "synthetic_model_free_contract_fixture",
            "does_not_start_stop_or_contact_a_runtime",
            "does_not_establish_live_speedup_or_memory_savings",
            "host_attested_unsigned_evidence_cannot_prove_measurement_honesty",
            "qualified_receipt_is_advisory_and_never_authorizes_activation",
            "exact_output_hash_equality_is_not_universal_semantic_equivalence",
            "stable_digests_can_reveal_workload_equality",
            "stateful_cross_request_speculation_modes_are_not_supported_in_alpha",
        ],
    }


def render_report(report: dict[str, Any]) -> str:
    return (
        json.dumps(
            report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Speculative Cell Qualifier benchmark."
    )
    parser.add_argument("--out", help="Optional report path to write or verify.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the regenerated report byte-for-byte with the artifact.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    rendered = render_report(report)
    destination = Path(args.out) if args.out else DEFAULT_ARTIFACT
    if args.check:
        if destination.read_bytes() != rendered.encode("utf-8"):
            raise SystemExit("Speculative Cell Qualifier artifact is out of date.")
    elif args.out:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rendered.encode("utf-8"))
    else:
        print(rendered, end="")
    if not report["contract_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
