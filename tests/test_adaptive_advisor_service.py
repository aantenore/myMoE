from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_moe.adaptive_advisor_service import (
    ADVISOR_RECEIPT_CONTRACT,
    MAX_TASK_BYTES,
    MAX_TASK_CHARS,
    AdaptiveAdvisorReceipt,
    AdvisorServiceError,
    advisor_presentation_payload,
    evaluate_advisor,
)
from local_moe.adaptive_selector import (
    AdaptiveAdvice,
    CandidateAssessment,
    build_adaptive_request,
)
from local_moe.resource_snapshot import build_resource_snapshot
from local_moe.verified_routing_contracts import sha256_json


SHA_A, SHA_B, SHA_C = (character * 64 for character in "abc")
EVALUATED_AT = "2026-07-21T12:00:00+00:00"


def _request():
    return build_adaptive_request(
        exact_request_fingerprint=SHA_A,
        intent_family_sha256=None,
        workload_id="coding.edit",
        required_capabilities=("code",),
        required_tool_surfaces=("workspace",),
        risk_class="compute_only",
        required_context_tokens=4096,
        evaluation_contract_sha256=SHA_B,
        profile="balanced",
        evaluated_at=EVALUATED_AT,
    )


def _snapshot():
    return build_resource_snapshot(
        system="Linux",
        os_release="test",
        machine="x86_64",
        cpu_count=8,
        cpu_identity_sha256=SHA_A,
        memory_topology="system",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        effective_memory_limit_bytes=16 * 1024**3,
        swap_used_bytes=0,
        accelerator_kind="none",
        accelerator_identity_sha256=None,
        runtime_environment_sha256=SHA_B,
        captured_at=EVALUATED_AT,
        source={"fixture": "service"},
    )


def _candidate(*, reasons: tuple[str, ...] = ()) -> CandidateAssessment:
    if reasons:
        return CandidateAssessment(
            cell_id="coder-local",
            passport_sha256=SHA_A,
            hard_eligible=False,
            pareto_eligible=False,
            rejection_codes=reasons,
            success_rate=None,
            p95_latency_ms=None,
            memory_pool=None,
            placement=None,
            effective_peak_host_memory_bytes=None,
            effective_peak_unified_memory_bytes=None,
            effective_peak_accelerator_memory_bytes=None,
            utility=None,
        )
    return CandidateAssessment(
        cell_id="coder-local",
        passport_sha256=SHA_A,
        hard_eligible=True,
        pareto_eligible=True,
        rejection_codes=(),
        success_rate=0.95,
        p95_latency_ms=250,
        memory_pool="host",
        placement="cpu",
        effective_peak_host_memory_bytes=2 * 1024**3,
        effective_peak_unified_memory_bytes=None,
        effective_peak_accelerator_memory_bytes=None,
        utility=0.9,
    )


def _advice(request, snapshot, *, reasons: tuple[str, ...] = ()) -> AdaptiveAdvice:
    recommended = not reasons
    return AdaptiveAdvice(
        catalog_sha256=SHA_C,
        request_sha256=request.digest,
        resource_snapshot_sha256=snapshot.digest,
        evaluated_at=request.evaluated_at,
        profile=request.profile,
        status="recommended" if recommended else "abstained",
        selected_cell_id="coder-local" if recommended else None,
        candidates=(_candidate(reasons=reasons),),
        reason_codes=(
            ("advisory_only", "pareto_frontier_selected")
            if recommended
            else ("advisory_only", "no_eligible_cell")
        ),
    )


def _receipt(*, reasons: tuple[str, ...] = ()) -> AdaptiveAdvisorReceipt:
    request, snapshot = _request(), _snapshot()
    advice = _advice(request, snapshot, reasons=reasons)
    if not reasons:
        state = "recommended_now"
    elif set(reasons).issubset({"model_unavailable", "capability_gap"}):
        state = "not_available_now"
    else:
        state = "not_enough_evidence"
    return AdaptiveAdvisorReceipt(
        request=request,
        advice=advice,
        task_chars=12,
        display_state=state,
    )


class AdaptiveAdvisorServiceTests(unittest.TestCase):
    def test_public_service_does_not_accept_synthetic_time_or_snapshot_hooks(self) -> None:
        parameters = inspect.signature(evaluate_advisor).parameters
        self.assertNotIn("evaluated_at", parameters)
        self.assertNotIn("snapshot", parameters)

    def test_evaluate_builds_a_complete_content_addressed_receipt(self) -> None:
        secret = "private task"
        snapshot = _snapshot()
        catalog = SimpleNamespace(digest=SHA_C)

        def advise(catalog_value, snapshot_value, request_value):
            self.assertIs(catalog_value, catalog)
            self.assertIs(snapshot_value, snapshot)
            return _advice(request_value, snapshot_value)

        with (
            patch(
                "local_moe.adaptive_advisor_service.read_bounded_regular_file",
                return_value=b'{"suite":"local"}',
            ),
            patch(
                "local_moe.adaptive_advisor_service.load_cell_catalog",
                return_value=catalog,
            ),
            patch(
                "local_moe.adaptive_advisor_service.collect_resource_snapshot",
                return_value=snapshot,
            ) as collect,
            patch(
                "local_moe.adaptive_advisor_service.now_utc",
                return_value=EVALUATED_AT,
            ),
            patch(
                "local_moe.adaptive_advisor_service.advise_cell",
                side_effect=advise,
            ),
        ):
            receipt = evaluate_advisor(
                catalog_path=Path("catalog.json"),
                evaluation_contract_path=Path("evaluation.json"),
                task_text=secret,
                workload_id="coding.edit",
                required_capabilities=("code",),
                required_tool_surfaces=("workspace",),
                risk_class="compute_only",
                context_tokens=4096,
                profile="balanced",
            )

        collect.assert_called_once_with()
        payload = receipt.payload()
        self.assertEqual(payload["contract"], ADVISOR_RECEIPT_CONTRACT)
        self.assertEqual(payload["request"], receipt.request.payload())
        self.assertEqual(payload["advice"], receipt.advice.payload())
        self.assertEqual(payload["task_chars"], len(secret))
        self.assertEqual(payload["display_state"], "recommended_now")
        self.assertEqual(
            payload["request"]["exact_request_fingerprint"],
            hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn(secret, str(payload))
        content = dict(payload)
        digest = content.pop("digest")
        self.assertEqual(digest, sha256_json(content))
        with self.assertRaises(FrozenInstanceError):
            receipt.task_chars = 99  # type: ignore[misc]

    def test_receipt_rejects_nested_tamper_even_with_recomputed_envelope(self) -> None:
        receipt = _receipt()
        tampered_request = replace(
            receipt.request,
            exact_request_fingerprint=SHA_C,
            digest="",
        )
        forged_content = {
            "schema_version": receipt.schema_version,
            "contract": receipt.contract,
            "request": tampered_request.payload(),
            "advice": receipt.advice.payload(),
            "task_chars": receipt.task_chars,
            "display_state": receipt.display_state,
        }
        with self.assertRaises(AdvisorServiceError) as failure:
            AdaptiveAdvisorReceipt(
                request=tampered_request,
                advice=receipt.advice,
                task_chars=receipt.task_chars,
                display_state=receipt.display_state,
                digest=sha256_json(forged_content),
            )
        self.assertEqual(failure.exception.code, "receipt_binding_invalid")

    def test_display_classification_is_fail_closed_for_unknown_or_mixed_codes(self) -> None:
        conclusive = _receipt(reasons=("model_unavailable", "capability_gap"))
        unknown = _receipt(reasons=("future_boundary",))
        mixed = _receipt(reasons=("model_unavailable", "measurement_unknown"))

        self.assertEqual(conclusive.display_state, "not_available_now")
        self.assertEqual(unknown.display_state, "not_enough_evidence")
        self.assertEqual(mixed.display_state, "not_enough_evidence")
        for receipt in (conclusive, unknown, mixed):
            presentation = advisor_presentation_payload(receipt)
            self.assertEqual(presentation["display_state"], receipt.display_state)
            self.assertLessEqual(len(presentation["badges"]), 3)
            self.assertEqual(presentation["receipt"], receipt.payload())

    def test_task_and_evaluation_contract_errors_have_stable_safe_codes(self) -> None:
        common = dict(
            catalog_path="catalog.json",
            evaluation_contract_path="evaluation.json",
            workload_id="coding.edit",
            required_capabilities=("code",),
            required_tool_surfaces=(),
            risk_class="compute_only",
            context_tokens=4096,
            profile="balanced",
        )
        with self.assertRaises(AdvisorServiceError) as empty_task:
            evaluate_advisor(task_text="   ", **common)
        self.assertEqual(empty_task.exception.code, "task_invalid")

        with patch(
            "local_moe.adaptive_advisor_service.read_bounded_regular_file",
            return_value=b"",
        ):
            with self.assertRaises(AdvisorServiceError) as empty_contract:
                evaluate_advisor(task_text="safe", **common)
        self.assertEqual(
            empty_contract.exception.code, "evaluation_contract_invalid"
        )

        for oversized_task in (
            "x" * (MAX_TASK_CHARS + 1),
            "\U0001f642" * (MAX_TASK_BYTES // 4 + 1),
        ):
            with self.assertRaises(AdvisorServiceError) as oversized:
                evaluate_advisor(task_text=oversized_task, **common)
            self.assertEqual(oversized.exception.code, "task_too_large")
            self.assertNotIn(oversized_task[:32], str(oversized.exception))


if __name__ == "__main__":
    unittest.main()
