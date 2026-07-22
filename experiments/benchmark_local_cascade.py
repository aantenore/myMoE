from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from local_moe.local_cascade import run_local_cascade
from local_moe.local_cascade_contracts import (
    LocalCascadeAttemptRequestV1,
    LocalCascadeAttemptResultV1,
    LocalCascadeConfigV1,
    LocalCascadeTaskV1,
    LocalCascadeTokenCountV1,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "local-cascade.example.json"
DEFAULT_ARTIFACT = ROOT / "outputs" / "local-cascade-contract-benchmark.json"

RAW_CONTEXT = "\n".join(
    (
        "build log: dependency cache restored",
        "build log: 128 unchanged checks passed",
        "build log: warning from an unrelated optional example",
        "task contract: classify the bounded change",
        "task contract: return a decision and its evidence",
        "build log: repeated progress line 01",
        "build log: repeated progress line 02",
        "build log: repeated progress line 03",
    )
)
SELECTED_CONTEXT = "\n".join(
    (
        "task contract: classify the bounded change",
        "task contract: return a decision and its evidence",
    )
)
RAW_TOOL_OUTPUT = "\n".join(
    (
        "test_a passed in 0.10s",
        "test_b passed in 0.11s",
        "test_c failed: expected decision field",
        "full stack frame 1",
        "full stack frame 2",
        "full stack frame 3",
        "summary: 1 failed, 2 passed",
    )
)
FILTERED_TOOL_OUTPUT = "\n".join(
    (
        "test_c failed: expected decision field",
        "summary: 1 failed, 2 passed",
    )
)

FROZEN_TASKS = (
    LocalCascadeTaskV1(
        task_id="classify-small-change",
        kind="classification",
        instruction="Classify the bounded local change using the supplied contract.",
        output_format="text",
    ),
    LocalCascadeTaskV1(
        task_id="summarize-local-result",
        kind="summarization",
        instruction="Summarize the frozen local result and cite deterministic evidence.",
        output_format="text",
    ),
    LocalCascadeTaskV1(
        task_id="extract-local-decision",
        kind="extraction",
        instruction="Extract the decision and evidence from the frozen local fixture.",
        output_format="text",
    ),
    LocalCascadeTaskV1(
        task_id="reject-unverified-result",
        kind="classification",
        instruction="Reject any candidate that cannot satisfy the frozen verifier.",
        output_format="text",
    ),
)


def _tokens(source: str, count: int | None) -> LocalCascadeTokenCountV1:
    return LocalCascadeTokenCountV1(source=source, count=count)


def _completed(
    content: str,
    *,
    input_tokens: tuple[str, int | None],
    output_tokens: tuple[str, int | None],
) -> LocalCascadeAttemptResultV1:
    return LocalCascadeAttemptResultV1(
        status="completed",
        content=content,
        input_tokens=_tokens(*input_tokens),
        output_tokens=_tokens(*output_tokens),
    )


def _abstained() -> LocalCascadeAttemptResultV1:
    return LocalCascadeAttemptResultV1(
        status="abstained",
        content=None,
        input_tokens=LocalCascadeTokenCountV1.unknown(),
        output_tokens=LocalCascadeTokenCountV1.unknown(),
    )


FIXTURE_RESULTS = {
    ("classify-small-change", "utility"): _completed(
        "decision=accept; evidence=the deterministic contract passed.",
        input_tokens=("actual", 48),
        output_tokens=("actual", 12),
    ),
    ("summarize-local-result", "utility"): _completed(
        "The result is probably fine.",
        input_tokens=("estimated", 60),
        output_tokens=("estimated", 8),
    ),
    ("summarize-local-result", "resident-generalist"): _completed(
        "decision=accept; evidence=the summary matches the frozen result.",
        input_tokens=("actual", 75),
        output_tokens=("actual", 20),
    ),
    ("extract-local-decision", "utility"): _abstained(),
    ("extract-local-decision", "resident-generalist"): _completed(
        "decision=accept; evidence=both required fields were extracted.",
        input_tokens=("estimated", 80),
        output_tokens=("estimated", 24),
    ),
    ("reject-unverified-result", "utility"): _completed(
        "The candidate omitted its contract fields.",
        input_tokens=("actual", 50),
        output_tokens=("actual", 5),
    ),
    ("reject-unverified-result", "resident-generalist"): _completed(
        "decision=accept; evidence=unverified assertion.",
        input_tokens=("estimated", 70),
        output_tokens=("estimated", 8),
    ),
    ("reject-unverified-result", "cold-specialist"): _completed(
        "No acceptable evidence was available.",
        input_tokens=("unknown", None),
        output_tokens=("unknown", None),
    ),
}


class _StepClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        current = self._value
        self._value += 0.001
        return current


class _FixtureAttemptPort:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def attempt(
        self,
        request: LocalCascadeAttemptRequestV1,
    ) -> LocalCascadeAttemptResultV1:
        self.requests.append(request.payload())
        try:
            return FIXTURE_RESULTS[(request.task.task_id, request.tier.tier_id)]
        except KeyError as exc:
            raise AssertionError("Frozen attempt fixture is incomplete.") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field: {key}")
        value[key] = item
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> LocalCascadeConfigV1:
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON value: {value}")
        ),
    )
    return LocalCascadeConfigV1.from_payload(raw)


def _byte_reduction(raw: str, reduced: str) -> dict[str, object]:
    raw_bytes = len(raw.encode("utf-8"))
    reduced_bytes = len(reduced.encode("utf-8"))
    return {
        "measurement_unit": "utf8_bytes",
        "raw_bytes": raw_bytes,
        "retained_bytes": reduced_bytes,
        "reduction_bytes": raw_bytes - reduced_bytes,
        "retained_ratio": round(reduced_bytes / raw_bytes, 6),
    }


def _scenario_payload(run: Any) -> dict[str, object]:
    return {
        "status": run.receipt.status,
        "selected_tier_id": run.receipt.selected_tier_id,
        "attempt_count": run.receipt.attempt_count,
        "attempts": [
            {
                "attempt_number": attempt.attempt_number,
                "tier_id": attempt.tier_id,
                "attempt_status": attempt.attempt_status,
                "verification_status": attempt.verification_status,
                "verifier_reason_codes": list(attempt.verifier_reason_codes),
                "input_tokens": attempt.input_tokens.payload(),
                "output_tokens": attempt.output_tokens.payload(),
            }
            for attempt in run.receipt.attempts
        ],
        "token_observations": run.receipt.token_totals.payload(),
        "accepted_content_in_report": False,
    }


def run_benchmark(
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Run deterministic contracts only; no model, network, tool, or write call."""

    config = load_config(config_path)
    attempt_port = _FixtureAttemptPort()
    runs = [
        run_local_cascade(
            task,
            config,
            attempt_port,
            clock=_StepClock(),
        )
        for task in FROZEN_TASKS
    ]
    scenarios = {
        task.task_id: _scenario_payload(run)
        for task, run in zip(FROZEN_TASKS, runs, strict=True)
    }

    token_fields = (
        "actual_input_tokens",
        "actual_output_tokens",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "unknown_input_attempts",
        "unknown_output_attempts",
    )
    token_observations = {
        field: sum(getattr(run.receipt.token_totals, field) for run in runs)
        for field in token_fields
    }

    attempts = [attempt for run in runs for attempt in run.receipt.attempts]
    tier_attempts = Counter(attempt.tier_id for attempt in attempts)
    passed_verifications = sum(
        attempt.verification_status == "passed" for attempt in attempts
    )
    failed_content_verifications = sum(
        attempt.attempt_status == "completed"
        and attempt.verification_status == "escalate"
        for attempt in attempts
    )
    non_content_escalations = sum(
        attempt.attempt_status != "completed"
        and attempt.verification_status == "escalate"
        for attempt in attempts
    )
    passed_runs = sum(run.receipt.status == "passed" for run in runs)
    simulated_premium_calls_after_local = len(runs) - passed_runs

    serialized_requests = json.dumps(
        attempt_port.requests,
        allow_nan=False,
        sort_keys=True,
    )
    serialized_report_scenarios = json.dumps(
        scenarios,
        allow_nan=False,
        sort_keys=True,
    )
    tier_sequences = {
        task.task_id: [attempt.tier_id for attempt in run.receipt.attempts]
        for task, run in zip(FROZEN_TASKS, runs, strict=True)
    }
    expected_sequences = {
        "classify-small-change": ["utility"],
        "summarize-local-result": ["utility", "resident-generalist"],
        "extract-local-decision": ["utility", "resident-generalist"],
        "reject-unverified-result": [
            "utility",
            "resident-generalist",
            "cold-specialist",
        ],
    }

    reduction_observations = {
        "context_selection": _byte_reduction(RAW_CONTEXT, SELECTED_CONTEXT),
        "command_aware_tool_output_filter": _byte_reduction(
            RAW_TOOL_OUTPUT,
            FILTERED_TOOL_OUTPUT,
        ),
        "aggregation_policy": "not_aggregated_across_surfaces",
    }
    verifier_observations = {
        "passed_attempts": passed_verifications,
        "failed_completed_attempts": failed_content_verifications,
        "non_content_escalations": non_content_escalations,
        "passed_runs": passed_runs,
        "exhausted_runs": sum(run.receipt.status == "exhausted" for run in runs),
    }
    local_attempt_observations = {
        "total": len(attempts),
        "by_tier": dict(sorted(tier_attempts.items())),
        "execution": "sequential_cheapest_first",
    }
    premium_counterfactual = {
        "mode": "simulated_counterfactual_no_premium_service_called",
        "actual_premium_calls": 0,
        "simulated_premium_only_baseline_calls": len(runs),
        "simulated_premium_calls_after_local": simulated_premium_calls_after_local,
        "simulated_premium_calls_avoided": (
            len(runs) - simulated_premium_calls_after_local
        ),
    }

    criteria = {
        "tiers_are_tried_cheapest_first_and_stop_after_acceptance": (
            tier_sequences == expected_sequences
        ),
        "only_deterministically_verified_content_is_accepted": (
            passed_verifications == 3
            and failed_content_verifications == 4
            and non_content_escalations == 1
            and passed_runs == 3
        ),
        "failed_content_is_not_forwarded_to_the_next_tier": (
            "content" not in serialized_requests
            and "probably fine" not in serialized_requests
            and "unverified assertion" not in serialized_requests
        ),
        "report_is_metadata_only": (
            "deterministic contract passed" not in serialized_report_scenarios
            and "summary matches the frozen result" not in serialized_report_scenarios
        ),
        "local_token_sources_remain_separate": (
            token_observations
            == {
                "actual_input_tokens": 173,
                "actual_output_tokens": 37,
                "estimated_input_tokens": 210,
                "estimated_output_tokens": 40,
                "unknown_input_attempts": 2,
                "unknown_output_attempts": 2,
            }
        ),
        "context_and_tool_output_reductions_are_byte_scoped": (
            reduction_observations["context_selection"]["measurement_unit"]
            == "utf8_bytes"
            and reduction_observations["context_selection"]["reduction_bytes"] > 0
            and reduction_observations["command_aware_tool_output_filter"][
                "measurement_unit"
            ]
            == "utf8_bytes"
            and reduction_observations["command_aware_tool_output_filter"][
                "reduction_bytes"
            ]
            > 0
        ),
        "premium_numbers_are_explicitly_simulated": (
            premium_counterfactual["actual_premium_calls"] == 0
            and premium_counterfactual["simulated_premium_calls_avoided"] == 3
            and premium_counterfactual["simulated_premium_calls_after_local"] == 1
        ),
    }
    pass_count = sum(criteria.values())

    return {
        "schema_version": "1.0",
        "contract": "local_cascade_contract_benchmark",
        "benchmark_kind": "deterministic_offline_contract_fixture",
        "configuration": {
            "cascade_id": config.cascade_id,
            "role_refs": [tier.model_ref for tier in config.ordered_tiers],
            "downloads_performed": 0,
            "model_invocations_performed": 0,
            "network_calls_performed": 0,
            "tool_calls_performed": 0,
            "write_operations_performed": 0,
        },
        "scenarios": scenarios,
        "local_attempt_observations": local_attempt_observations,
        "verifier_observations": verifier_observations,
        "local_token_observations": {
            **token_observations,
            "comparison_policy": (
                "actual, estimated, and unknown observations are separate; "
                "input and output directions are not added into one headline"
            ),
        },
        "context_and_tool_output_reduction": reduction_observations,
        "premium_counterfactual": premium_counterfactual,
        "paired_runner_extension": {
            "status": "contract_only_not_executed",
            "required_same_inputs": [
                "frozen_task_set",
                "verifier_contract",
                "pass_criteria",
                "tool_authority",
                "context_source",
            ],
            "measure_separately": [
                "actual_local_input_tokens",
                "actual_local_output_tokens",
                "actual_premium_input_tokens",
                "actual_premium_output_tokens",
                "unknown_token_observations",
                "accepted_outcomes",
                "end_to_end_latency",
            ],
            "adapter_boundary": (
                "inject a real local attempt port and a separately metered premium "
                "runner without changing tasks, verifiers, or pass criteria"
            ),
        },
        "criteria": criteria,
        "pass_count": pass_count,
        "check_count": len(criteria),
        "contract_checks_passed": pass_count == len(criteria),
        "limits": [
            "not_a_live_model_quality_claim",
            "not_a_live_cost_claim",
            "not_a_frontier_token_savings_claim",
            "percentages_from_different_reduction_layers_are_non_additive",
            "overall_savings_require_a_paired_end_to_end_benchmark",
            "paired_runs_must_use_identical_tasks_verifiers_and_pass_criteria",
            "fixture_token_counts_are_source_labeled_observations",
            "context_and_tool_output_reduction_uses_bytes_not_tokens",
        ],
    }


def render_report(report: dict[str, object]) -> str:
    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic offline LocalCascade contract benchmark."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="LocalCascade configuration to validate and exercise.",
    )
    parser.add_argument(
        "--out",
        help="Optional report path to write or verify.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the regenerated report byte-for-byte with the artifact.",
    )
    args = parser.parse_args()

    report = run_benchmark(Path(args.config))
    rendered = render_report(report)
    destination = Path(args.out) if args.out else DEFAULT_ARTIFACT
    if args.check:
        try:
            current = destination.read_bytes()
        except OSError as exc:
            raise SystemExit(f"Unable to read benchmark artifact: {exc}") from exc
        if current != rendered.encode("utf-8"):
            raise SystemExit("LocalCascade benchmark artifact is out of date.")
    elif args.out:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rendered.encode("utf-8"))
    else:
        print(rendered, end="")

    if not report["contract_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
