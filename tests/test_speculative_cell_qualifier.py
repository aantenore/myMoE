from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from local_moe.llama_cpp_speculative_adapter import (
    LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
    llama_cpp_failure_measurement,
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
    speculative_plan_from_payload,
    speculative_receipt_from_payload,
    speculative_trial_from_payload,
)
from local_moe.speculative_cell_qualifier import (
    expected_trial_order,
    expected_trial_sequence_index,
    qualify_speculative_cell,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _plan(**policy_changes) -> SpeculativeQualificationPlan:
    baseline = SpeculativeCellBinding(
        cell_id="target-baseline",
        speculation_config_sha256=_sha("baseline-speculation-config"),
        speculation_mode="none",
    )
    candidate = SpeculativeCellBinding(
        cell_id="target-ngram",
        speculation_config_sha256=_sha("candidate-speculation-config"),
        speculation_mode="ngram-simple",
    )
    policy = SpeculativeQualificationPolicy(
        trials_per_case=2,
        maximum_candidate_peak_memory_bytes=2_000,
        **policy_changes,
    )
    return SpeculativeQualificationPlan(
        plan_id="m4-ngram-qualification",
        execution=SpeculativeExecutionBinding(
            runtime_revision_sha256=_sha("llama.cpp-b1234"),
            runtime_binary_sha256=_sha("llama-server-binary"),
            runtime_binding_manifest_sha256=_sha("runtime-binding-manifest"),
            hardware_sha256=_sha("apple-m4-24g"),
            target_model_sha256=_sha("target-model"),
            shared_runtime_config_sha256=_sha("shared-runtime-config"),
            request_policy_sha256=_sha("text-only-request-policy"),
            regime_protocol_sha256=_sha("cold-warm-protocol"),
            harness_sha256=_sha("paired-harness"),
            collector_sha256=_sha("speed-bench-collector"),
            adapter_contract_sha256=LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
        ),
        baseline=baseline,
        candidate=candidate,
        workload_sha256=_sha("public-workload-v1"),
        case_sha256s=tuple(sorted((_sha("case-a"), _sha("case-b")))),
        order_seed_sha256=_sha("order-seed"),
        policy=policy,
    )


def _arm(
    cell_sha256: str,
    *,
    candidate: bool,
    output_sha256: str | None = None,
    success: bool = True,
    error_code: str | None = None,
    predicted_ms: float | None = None,
    total_latency_ms: float | None = None,
    ttft_ms: float | None = None,
    peak_memory_bytes: int | None = None,
    include_draft: bool = True,
) -> SpeculativeArmMeasurement:
    if not success:
        return SpeculativeArmMeasurement(
            cell_sha256=cell_sha256,
            success=False,
            error_code=error_code or "runtime_failure",
        )
    return SpeculativeArmMeasurement(
        cell_sha256=cell_sha256,
        success=True,
        output_sha256=output_sha256 or _sha("same-output"),
        ttft_ms=ttft_ms if ttft_ms is not None else (90.0 if candidate else 100.0),
        total_latency_ms=(
            total_latency_ms
            if total_latency_ms is not None
            else (850.0 if candidate else 1_100.0)
        ),
        predicted_tokens=100,
        predicted_ms=(
            predicted_ms
            if predicted_ms is not None
            else (700.0 if candidate else 1_000.0)
        ),
        peak_memory_bytes=(
            peak_memory_bytes
            if peak_memory_bytes is not None
            else (1_500 if candidate else 1_000)
        ),
        draft_generated_tokens=(120 if candidate and include_draft else None),
        draft_accepted_tokens=(72 if candidate and include_draft else None),
    )


def _trials(plan: SpeculativeQualificationPlan) -> tuple[SpeculativeTrial, ...]:
    return tuple(
        SpeculativeTrial(
            plan_sha256=plan.digest,
            sequence_index=expected_trial_sequence_index(
                plan, case, regime, repetition
            ),
            case_sha256=case,
            repetition=repetition,
            regime=regime,
            order=expected_trial_order(plan, case, regime, repetition),
            baseline=_arm(plan.baseline.digest, candidate=False, include_draft=False),
            candidate=_arm(plan.candidate.digest, candidate=True),
        )
        for case in plan.case_sha256s
        for regime in plan.required_regimes
        for repetition in range(plan.policy.trials_per_case)
    )


class SpeculativeCellQualifierTests(unittest.TestCase):
    def test_contracts_round_trip_and_reject_tampering(self) -> None:
        plan = _plan()
        trial = _trials(plan)[0]
        receipt = qualify_speculative_cell(plan, _trials(plan))

        self.assertEqual(speculative_plan_from_payload(plan.payload()), plan)
        self.assertEqual(speculative_trial_from_payload(trial.payload()), trial)
        self.assertEqual(speculative_receipt_from_payload(receipt.payload()), receipt)

        with self.assertRaises(SpeculativeCellContractError):
            speculative_plan_from_payload(plan.payload() | {"unexpected": True})
        with self.assertRaisesRegex(SpeculativeCellContractError, "digest"):
            replace(plan, workload_sha256=_sha("changed"))

    def test_exact_cell_is_one_shared_execution_plus_speculation_delta(self) -> None:
        plan = _plan()
        with self.assertRaisesRegex(SpeculativeCellContractError, "digest"):
            replace(
                plan.execution,
                hardware_sha256=_sha("other-hardware"),
            )
        with self.assertRaisesRegex(
            SpeculativeCellContractError, "configs must differ"
        ):
            replace(
                plan,
                candidate=replace(
                    plan.candidate,
                    speculation_config_sha256=(plan.baseline.speculation_config_sha256),
                    digest="",
                ),
                digest="",
            )
        with self.assertRaisesRegex(SpeculativeCellContractError, "Draft-model"):
            replace(
                plan.candidate,
                speculation_mode="draft-simple",
                draft_model_sha256=None,
                digest="",
            )
        with self.assertRaisesRegex(SpeculativeCellContractError, "not supported"):
            replace(
                plan.candidate,
                speculation_mode="ngram-mod",
                digest="",
            )

    def test_complete_paired_evidence_qualifies_but_never_activates(self) -> None:
        plan = _plan()
        receipt = qualify_speculative_cell(plan, _trials(plan))

        self.assertEqual(receipt.decision, "qualified")
        self.assertEqual(receipt.reason_codes, ())
        self.assertEqual(receipt.observed_trials, 8)
        self.assertEqual(receipt.expected_cases, 2)
        self.assertEqual(receipt.trials_per_case, 2)
        self.assertEqual(receipt.unique_cases, 2)
        self.assertEqual(receipt.cold_trials, 4)
        self.assertEqual(receipt.warm_trials, 4)
        self.assertGreater(receipt.cold_median_speedup_ratio or 0, 1.4)
        self.assertGreater(receipt.warm_median_speedup_ratio or 0, 1.4)
        self.assertEqual(receipt.cold_candidate_acceptance_rate, 0.6)
        self.assertEqual(receipt.warm_candidate_acceptance_rate, 0.6)
        self.assertFalse(receipt.activation_authorized)

    def test_output_change_speed_regression_and_memory_are_rejected(self) -> None:
        plan = _plan()
        trials = list(_trials(plan))
        changed_candidate = _arm(
            plan.candidate.digest,
            candidate=True,
            output_sha256=_sha("different-output"),
            predicted_ms=1_100.0,
            total_latency_ms=1_300.0,
            ttft_ms=120.0,
            peak_memory_bytes=2_100,
        )
        trials[0] = replace(trials[0], candidate=changed_candidate, digest="")

        receipt = qualify_speculative_cell(plan, trials)

        self.assertEqual(receipt.decision, "rejected")
        self.assertIn("exact_output_mismatch", receipt.reason_codes)
        self.assertIn("candidate_memory_budget_exceeded", receipt.reason_codes)
        self.assertIn("p95_latency_regression", receipt.reason_codes)
        self.assertIn("p95_ttft_regression", receipt.reason_codes)

    def test_every_regime_must_pass_without_cold_masking_warm(self) -> None:
        plan = _plan()
        trials = tuple(
            replace(
                trial,
                candidate=_arm(
                    plan.candidate.digest,
                    candidate=True,
                    predicted_ms=200.0 if trial.regime == "cold" else 2_000.0,
                    total_latency_ms=(300.0 if trial.regime == "cold" else 1_900.0),
                    ttft_ms=50.0 if trial.regime == "cold" else 200.0,
                ),
                digest="",
            )
            for trial in _trials(plan)
        )

        receipt = qualify_speculative_cell(plan, trials)

        self.assertEqual(receipt.decision, "rejected")
        self.assertIn("median_speedup_below_threshold", receipt.reason_codes)
        self.assertIn("p95_latency_regression", receipt.reason_codes)
        self.assertIn("p95_ttft_regression", receipt.reason_codes)
        self.assertGreater(receipt.cold_median_speedup_ratio or 0, 4.0)
        self.assertLess(receipt.warm_median_speedup_ratio or 1, 1.0)

    def test_tiny_positive_speedup_is_rejected_without_rounding_error(self) -> None:
        plan = _plan()
        trials = tuple(
            replace(
                trial,
                candidate=_arm(
                    plan.candidate.digest,
                    candidate=True,
                    predicted_ms=10_000_000_000.0,
                ),
                digest="",
            )
            for trial in _trials(plan)
        )

        receipt = qualify_speculative_cell(plan, trials)

        self.assertEqual(receipt.decision, "rejected")
        self.assertIn("median_speedup_below_threshold", receipt.reason_codes)
        self.assertGreater(receipt.cold_median_speedup_ratio or 0, 0)

    def test_incomplete_failed_or_unbalanced_evidence_abstains(self) -> None:
        plan = _plan()
        incomplete = qualify_speculative_cell(plan, _trials(plan)[:-1])
        self.assertEqual(incomplete.decision, "abstained")
        self.assertIn("evidence_incomplete", incomplete.reason_codes)

        trials = list(_trials(plan))
        trials[0] = replace(
            trials[0],
            candidate=_arm(
                plan.candidate.digest,
                candidate=True,
                success=False,
                error_code="runtime_timeout",
            ),
            digest="",
        )
        failed = qualify_speculative_cell(plan, trials)
        self.assertIn("evidence_execution_failed", failed.reason_codes)

        trials = list(_trials(plan))
        trials[0] = replace(
            trials[0],
            order="BA" if trials[0].order == "AB" else "AB",
            digest="",
        )
        unbalanced = qualify_speculative_cell(plan, trials)
        self.assertIn("evidence_not_counterbalanced", unbalanced.reason_codes)

        trials = list(_trials(plan))
        trials[0], trials[1] = trials[1], trials[0]
        reordered = qualify_speculative_cell(plan, trials)
        self.assertIn("evidence_schedule_mismatch", reordered.reason_codes)

    def test_missing_acceptance_metrics_abstain(self) -> None:
        plan = _plan()
        trials = tuple(
            replace(
                trial,
                candidate=_arm(
                    plan.candidate.digest,
                    candidate=True,
                    include_draft=False,
                ),
                digest="",
            )
            for trial in _trials(plan)
        )
        receipt = qualify_speculative_cell(plan, trials)
        self.assertEqual(receipt.decision, "abstained")
        self.assertEqual(receipt.reason_codes, ("candidate_acceptance_missing",))

    def test_duplicate_or_wrong_cell_evidence_is_invalid(self) -> None:
        plan = _plan()
        trials = _trials(plan)
        with self.assertRaisesRegex(SpeculativeCellContractError, "duplicates"):
            qualify_speculative_cell(plan, trials[:-1] + (trials[0],))
        wrong = replace(
            trials[0],
            candidate=replace(
                trials[0].candidate,
                cell_sha256=_sha("other-cell"),
                digest="",
            ),
            digest="",
        )
        with self.assertRaisesRegex(SpeculativeCellContractError, "changed cells"):
            qualify_speculative_cell(plan, (wrong,) + trials[1:])

    def test_llama_cpp_adapter_retains_only_output_digest(self) -> None:
        plan = _plan()
        secret_output = "private generated answer"
        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": secret_output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 12},
            "timings": {
                "predicted_n": 12,
                "predicted_ms": 240.0,
                "draft_n": 20,
                "draft_n_accepted": 15,
            },
        }
        measurement = parse_llama_cpp_completion(
            payload,
            cell_sha256=plan.candidate.digest,
            ttft_ms=30.0,
            total_latency_ms=280.0,
            peak_memory_bytes=1_500,
        )

        self.assertNotIn(secret_output, str(measurement.payload()))
        self.assertEqual(
            measurement.output_sha256,
            _sha(
                '{"content":"private generated answer","finish_reason":"stop",'
                '"schema_version":"1.0","surface":"chat"}'
            ),
        )
        self.assertEqual(measurement.draft_accepted_tokens, 15)
        self.assertEqual(len(LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256), 64)

    def test_llama_cpp_adapter_rejects_inconsistent_counts(self) -> None:
        plan = _plan()
        payload = {
            "choices": [{"text": "answer", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 11},
            "timings": {"predicted_n": 12, "predicted_ms": 240.0},
        }
        with self.assertRaisesRegex(SpeculativeCellContractError, "disagree"):
            parse_llama_cpp_completion(
                payload,
                cell_sha256=plan.baseline.digest,
                ttft_ms=30.0,
                total_latency_ms=280.0,
                peak_memory_bytes=1_000,
            )
        failure = llama_cpp_failure_measurement(
            cell_sha256=plan.baseline.digest,
            error_code="bounded_transport_failure",
        )
        self.assertFalse(failure.success)
        self.assertIsNone(failure.output_sha256)

    def test_text_adapter_rejects_agentic_or_reasoning_surfaces(self) -> None:
        plan = _plan()
        base = {
            "usage": {"completion_tokens": 1},
            "timings": {"predicted_n": 1, "predicted_ms": 10.0},
        }
        for message in (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-a"}],
            },
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "hidden",
            },
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SpeculativeCellContractError, "text-only"):
                    parse_llama_cpp_completion(
                        base
                        | {"choices": [{"message": message, "finish_reason": "stop"}]},
                        cell_sha256=plan.candidate.digest,
                        ttft_ms=5.0,
                        total_latency_ms=12.0,
                        peak_memory_bytes=1_500,
                    )

    def test_receipt_parser_rejects_impossible_qualified_state(self) -> None:
        plan = _plan()
        receipt = qualify_speculative_cell(plan, _trials(plan))
        with self.assertRaises(SpeculativeCellContractError):
            replace(
                receipt,
                observed_trials=0,
                cold_trials=0,
                warm_trials=0,
                unique_cases=0,
                failed_arms=999,
                digest="",
            )
        with self.assertRaisesRegex(SpeculativeCellContractError, "regime counts"):
            replace(receipt, cold_trials=1, warm_trials=7, digest="")
        with self.assertRaisesRegex(SpeculativeCellContractError, "case counts"):
            replace(receipt, unique_cases=1, digest="")
        with self.assertRaisesRegex(SpeculativeCellContractError, "expected counts"):
            replace(receipt, trials_per_case=1, digest="")

        trials = list(_trials(plan))
        trials[0] = replace(
            trials[0],
            candidate=_arm(
                plan.candidate.digest,
                candidate=True,
                output_sha256=_sha("different-output"),
            ),
            digest="",
        )
        rejected = qualify_speculative_cell(plan, trials)
        with self.assertRaisesRegex(SpeculativeCellContractError, "Rejected"):
            replace(rejected, output_mismatches=0, digest="")


if __name__ == "__main__":
    unittest.main()
