from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_moe.adaptive_advisor_service import AdaptiveAdvisorReceipt
from local_moe.adaptive_execution_gate import (
    MAX_ADVISOR_RECEIPT_BYTES,
    AdaptiveCellExecutionPolicy,
    AdaptiveExecutionGateError,
    adaptive_advisor_receipt_from_payload,
    adaptive_cell_execution_preview_receipt_from_payload,
    adaptive_execution_policy_from_payload,
    load_adaptive_advisor_receipt,
    preview_cell_execution,
)
from local_moe.adaptive_selector import (
    AdaptiveAdvice,
    CandidateAssessment,
    build_adaptive_request,
)


SHA_A, SHA_B, SHA_C, SHA_D = (character * 64 for character in "abcd")
SOURCE_TIME = "2026-07-21T12:00:00+00:00"
FRESH_TIME = "2026-07-21T12:01:00+00:00"


def _candidate(
    *,
    cell_id: str = "coder-local",
    passport_sha256: str = SHA_A,
    recommended: bool = True,
) -> CandidateAssessment:
    return CandidateAssessment(
        cell_id=cell_id,
        passport_sha256=passport_sha256,
        hard_eligible=recommended,
        pareto_eligible=recommended,
        rejection_codes=() if recommended else ("model_unavailable",),
        success_rate=0.95 if recommended else None,
        p95_latency_ms=250 if recommended else None,
        memory_pool="host" if recommended else None,
        placement="cpu" if recommended else None,
        effective_peak_host_memory_bytes=2 * 1024**3 if recommended else None,
        effective_peak_unified_memory_bytes=None,
        effective_peak_accelerator_memory_bytes=None,
        utility=0.9 if recommended else None,
    )


def _receipt(
    task: str,
    *,
    evaluated_at: str = SOURCE_TIME,
    catalog_sha256: str = SHA_C,
    evaluation_contract_sha256: str = SHA_B,
    snapshot_sha256: str = SHA_D,
    cell_id: str = "coder-local",
    passport_sha256: str = SHA_A,
    risk_class: str = "compute_only",
    tool_surfaces: tuple[str, ...] = (),
    recommended: bool = True,
) -> AdaptiveAdvisorReceipt:
    request = build_adaptive_request(
        exact_request_fingerprint=hashlib.sha256(task.encode("utf-8")).hexdigest(),
        intent_family_sha256=None,
        workload_id="coding.edit",
        required_capabilities=("code",),
        required_tool_surfaces=tool_surfaces,
        risk_class=risk_class,
        required_context_tokens=4096,
        evaluation_contract_sha256=evaluation_contract_sha256,
        profile="balanced",
        evaluated_at=evaluated_at,
    )
    candidate = _candidate(
        cell_id=cell_id,
        passport_sha256=passport_sha256,
        recommended=recommended,
    )
    advice = AdaptiveAdvice(
        catalog_sha256=catalog_sha256,
        request_sha256=request.digest,
        resource_snapshot_sha256=snapshot_sha256,
        evaluated_at=evaluated_at,
        profile=request.profile,
        status="recommended" if recommended else "abstained",
        selected_cell_id=cell_id if recommended else None,
        candidates=(candidate,),
        reason_codes=(
            ("advisory_only", "pareto_frontier_selected")
            if recommended
            else ("advisory_only", "no_eligible_cell")
        ),
    )
    return AdaptiveAdvisorReceipt(
        request=request,
        advice=advice,
        task_chars=len(task),
        display_state="recommended_now" if recommended else "not_available_now",
    )


def _policy(*, age: int = 120) -> AdaptiveCellExecutionPolicy:
    return AdaptiveCellExecutionPolicy(
        max_source_receipt_age_seconds=age,
        allowed_risk_classes=("compute_only",),
        max_tool_surfaces=0,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class AdaptiveExecutionContractTests(unittest.TestCase):
    def test_policy_is_content_addressed_and_cannot_widen_v1(self) -> None:
        policy = _policy()
        self.assertEqual(
            adaptive_execution_policy_from_payload(policy.payload()), policy
        )
        for changes in (
            {"mode": "execute"},
            {"allowed_risk_classes": ["compute_only", "low"]},
            {"max_tool_surfaces": 1},
            {"max_source_receipt_age_seconds": 121},
        ):
            payload = {**policy.payload(), **changes, "digest": policy.digest}
            with (
                self.subTest(changes=changes),
                self.assertRaises(AdaptiveExecutionGateError),
            ):
                adaptive_execution_policy_from_payload(payload)

    def test_advisor_parser_round_trips_and_verifies_every_nested_digest(self) -> None:
        receipt = _receipt("private task")
        self.assertEqual(
            adaptive_advisor_receipt_from_payload(receipt.payload()), receipt
        )

        tampered = receipt.payload()
        tampered["advice"]["candidates"][0]["passport_sha256"] = SHA_B
        with self.assertRaises(AdaptiveExecutionGateError):
            adaptive_advisor_receipt_from_payload(tampered)

        unknown = receipt.payload()
        unknown["request"]["demand"]["future_field"] = True
        with self.assertRaises(AdaptiveExecutionGateError):
            adaptive_advisor_receipt_from_payload(unknown)

    def test_file_loader_rejects_duplicate_nonfinite_and_linked_json(self) -> None:
        receipt = _receipt("private task")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "receipt.json"
            _write_json(valid, receipt.payload())
            self.assertEqual(load_adaptive_advisor_receipt(valid), receipt)

            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schema_version":"1.0","schema_version":"1.0"}',
                encoding="utf-8",
            )
            with self.assertRaises(AdaptiveExecutionGateError):
                load_adaptive_advisor_receipt(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaises(AdaptiveExecutionGateError):
                load_adaptive_advisor_receipt(nonfinite)

            oversized = root / "oversized.json"
            oversized.write_bytes(b"x" * (MAX_ADVISOR_RECEIPT_BYTES + 1))
            with self.assertRaises(AdaptiveExecutionGateError):
                load_adaptive_advisor_receipt(oversized)

            if os.name != "nt":
                linked = root / "linked.json"
                linked.symlink_to(valid)
                with self.assertRaises(AdaptiveExecutionGateError):
                    load_adaptive_advisor_receipt(linked)

                special = root / "special.json"
                os.mkfifo(special)
                with self.assertRaises(AdaptiveExecutionGateError):
                    load_adaptive_advisor_receipt(special)

    def test_preview_passes_only_for_same_fresh_cell_passport_and_exact_task(
        self,
    ) -> None:
        secret = "SECRET task that must not be persisted"
        source = _receipt(secret)
        fresh = _receipt(secret, evaluated_at=FRESH_TIME)
        policy = _policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, policy_path = root / "receipt.json", root / "policy.json"
            _write_json(source_path, source.payload())
            _write_json(policy_path, policy.payload())
            with patch(
                "local_moe.adaptive_execution_gate.evaluate_advisor",
                return_value=fresh,
            ) as evaluate:
                preview = preview_cell_execution(
                    source_path,
                    secret,
                    root / "catalog.json",
                    root / "evaluation.json",
                    policy_path,
                )

        self.assertEqual(preview.status, "admission_passed")
        self.assertEqual(preview.reason_codes, ())
        self.assertEqual(preview.source_advisor_receipt_sha256, source.digest)
        self.assertEqual(preview.fresh_advisor_receipt_sha256, fresh.digest)
        self.assertEqual(preview.source_selected_cell_id, "coder-local")
        self.assertEqual(preview.fresh_selected_cell_id, "coder-local")
        self.assertEqual(preview.source_passport_sha256, SHA_A)
        self.assertEqual(preview.fresh_passport_sha256, SHA_A)
        self.assertFalse(preview.applied)
        self.assertFalse(preview.authorizes_execution)
        self.assertFalse(preview.network_used)
        self.assertEqual(preview.model_invocations, 0)
        self.assertNotIn(secret, json.dumps(preview.payload()))
        self.assertEqual(
            adaptive_cell_execution_preview_receipt_from_payload(preview.payload()),
            preview,
        )
        self.assertEqual(evaluate.call_args.kwargs["required_tool_surfaces"], ())
        self.assertEqual(evaluate.call_args.kwargs["risk_class"], "compute_only")

    def test_preview_blocks_all_task_policy_and_lineage_drift_without_fallback(
        self,
    ) -> None:
        source = _receipt(
            "source task",
            evaluated_at=SOURCE_TIME,
            risk_class="low",
            tool_surfaces=("workspace",),
        )
        fresh = _receipt(
            "different task text",
            evaluated_at="2026-07-21T12:03:00+00:00",
            catalog_sha256=SHA_D,
            evaluation_contract_sha256=SHA_C,
            cell_id="other-cell",
            passport_sha256=SHA_B,
            risk_class="low",
            tool_surfaces=("workspace",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, policy_path = root / "receipt.json", root / "policy.json"
            _write_json(source_path, source.payload())
            _write_json(policy_path, _policy().payload())
            with patch(
                "local_moe.adaptive_execution_gate.evaluate_advisor",
                return_value=fresh,
            ):
                preview = preview_cell_execution(
                    source_path,
                    "different task text",
                    root / "catalog.json",
                    root / "evaluation.json",
                    policy_path,
                )

        self.assertEqual(preview.status, "admission_blocked")
        self.assertEqual(
            preview.reason_codes,
            (
                "catalog_drift",
                "evaluation_contract_drift",
                "risk_class_blocked",
                "selected_cell_changed",
                "selected_passport_changed",
                "source_receipt_expired",
                "task_fingerprint_mismatch",
                "task_size_mismatch",
                "tool_surface_blocked",
            ),
        )
        self.assertEqual(preview.fresh_selected_cell_id, "other-cell")

    def test_preview_blocks_non_recommendation_and_future_receipt(self) -> None:
        task = "private task"
        source = _receipt(
            task,
            evaluated_at="2026-07-21T12:02:00+00:00",
            recommended=False,
        )
        fresh = _receipt(
            task,
            evaluated_at=FRESH_TIME,
            recommended=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, policy_path = root / "receipt.json", root / "policy.json"
            _write_json(source_path, source.payload())
            _write_json(policy_path, _policy().payload())
            with patch(
                "local_moe.adaptive_execution_gate.evaluate_advisor",
                return_value=fresh,
            ):
                preview = preview_cell_execution(
                    source_path,
                    task,
                    root / "catalog.json",
                    root / "evaluation.json",
                    policy_path,
                )
        self.assertEqual(
            preview.reason_codes,
            (
                "fresh_admission_blocked",
                "source_receipt_from_future",
                "source_receipt_not_recommended",
            ),
        )

    def test_preview_rechecks_fresh_request_semantics_and_policy(self) -> None:
        task = "private task"
        source = _receipt(task)
        fresh = _receipt(
            task,
            evaluated_at=FRESH_TIME,
            risk_class="write_local",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, policy_path = root / "receipt.json", root / "policy.json"
            _write_json(source_path, source.payload())
            _write_json(policy_path, _policy().payload())
            with patch(
                "local_moe.adaptive_execution_gate.evaluate_advisor",
                return_value=fresh,
            ):
                preview = preview_cell_execution(
                    source_path,
                    task,
                    root / "catalog.json",
                    root / "evaluation.json",
                    policy_path,
                )

        self.assertEqual(preview.status, "admission_blocked")
        self.assertEqual(
            preview.reason_codes,
            ("request_semantics_changed", "risk_class_blocked"),
        )

    def test_preview_receipt_rejects_authority_even_with_recomputed_digest(
        self,
    ) -> None:
        policy = _policy()
        source = _receipt("private task")
        fresh = _receipt("private task", evaluated_at=FRESH_TIME)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, policy_path = root / "receipt.json", root / "policy.json"
            _write_json(source_path, source.payload())
            _write_json(policy_path, policy.payload())
            with patch(
                "local_moe.adaptive_execution_gate.evaluate_advisor",
                return_value=fresh,
            ):
                preview = preview_cell_execution(
                    source_path,
                    "private task",
                    root / "catalog.json",
                    root / "evaluation.json",
                    policy_path,
                )
        with self.assertRaises(AdaptiveExecutionGateError):
            replace(preview, authorizes_execution=True, digest="")


if __name__ == "__main__":
    unittest.main()
