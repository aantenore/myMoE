from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_moe.adaptive_execution_gate import AdaptiveCellExecutionPreviewReceipt
from local_moe.bound_cell_run import (
    ModelIdentityProbe,
    resolve_bound_cell_target,
    run_bound_cell,
)
from local_moe.bound_cell_run_contracts import (
    BoundCellRunContractError,
    bound_cell_run_receipt_from_payload,
)
from tests.bound_cell_run_lease_fakes import (
    FakeLeaseStore,
    claim_for,
    preview_evaluation,
    resource_snapshot,
)
from tests.test_runtime_binding_inspector import (
    GENERIC_BACKEND,
    NOW,
    _InspectionFixture,
)


SHA = "a" * 64


class _StableTransport:
    def __init__(self, model: str) -> None:
        self.model = model
        self.probes = 0
        self.invocations = 0

    def probe_models(self, **_kwargs: object) -> ModelIdentityProbe:
        self.probes += 1
        return ModelIdentityProbe.from_ids([self.model], maximum=4)

    def invoke(self, **_kwargs: object) -> str:
        self.invocations += 1
        return "PRIVATE-RESPONSE-BODY"


class _InvalidResponseWithDriftTransport(_StableTransport):
    def probe_models(self, **_kwargs: object) -> ModelIdentityProbe:
        self.probes += 1
        models = [self.model] if self.probes == 1 else [self.model, "other-model"]
        return ModelIdentityProbe.from_ids(models, maximum=4)

    def invoke(self, **_kwargs: object) -> str:
        self.invocations += 1
        return "\ud800"


class BoundCellRunIntegrationTests(unittest.TestCase):
    @staticmethod
    def _preview(target: object, task: str) -> AdaptiveCellExecutionPreviewReceipt:
        return AdaptiveCellExecutionPreviewReceipt(
            source_advisor_receipt_sha256=SHA,
            source_request_sha256=SHA,
            fresh_advisor_receipt_sha256=SHA,
            fresh_request_sha256=SHA,
            policy_sha256=SHA,
            evaluated_at=NOW.isoformat(),
            source_selected_cell_id=target.request.cell_id,
            fresh_selected_cell_id=target.request.cell_id,
            source_passport_sha256=target.passport.digest,
            fresh_passport_sha256=target.passport.digest,
            fresh_resource_snapshot_sha256=SHA,
            status="admission_passed",
            reason_codes=(),
            task_chars=len(task),
        )

    def test_real_binding_inspector_completes_one_metadata_only_run(self) -> None:
        task = "PRIVATE-TASK-BODY"
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _InspectionFixture(Path(temporary), GENERIC_BACKEND)
            fixture.make_verified()
            target = resolve_bound_cell_target(fixture.request_path)
            transport = _StableTransport(target.expert.model)
            output_path = fixture.root / "run-receipt.json"
            snapshot = resource_snapshot()
            preview = self._preview(target, task)
            claim = claim_for(
                preview, snapshot, passport_sha256=target.passport.digest
            )
            lease_store = FakeLeaseStore()

            def previewer(
                *_args: object, resource_snapshot: object
            ) -> object:
                self.assertIs(resource_snapshot, snapshot)
                return preview_evaluation(preview, snapshot)

            with patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ):
                result = run_bound_cell(
                    fixture.root / "advisor.json",
                    task,
                    fixture.catalog_path,
                    fixture.root / "evaluation.json",
                    fixture.root / "policy.json",
                    fixture.request_path,
                    confirmed=True,
                    transport=transport,
                    previewer=previewer,
                    snapshot_collector=lambda: snapshot,
                    lease_store=lease_store,
                    clock=lambda: NOW,
                    publication_path=output_path,
                )

            rendered = json.dumps(result.receipt.payload(), sort_keys=True)
            self.assertEqual(result.receipt.status, "completed")
            self.assertEqual(transport.probes, 2)
            self.assertEqual(transport.invocations, 1)
            self.assertEqual(result.receipt.endpoint_probe_requests, 2)
            self.assertEqual(result.receipt.invocation_attempts, 1)
            self.assertEqual(result.response_text, "PRIVATE-RESPONSE-BODY")
            self.assertEqual(len(result.publication_protected_roots), 2)
            self.assertIn(fixture.request_path, result.publication_inputs)
            self.assertNotIn(task, rendered)
            self.assertNotIn("PRIVATE-RESPONSE-BODY", rendered)
            self.assertEqual(
                bound_cell_run_receipt_from_payload(result.receipt.payload()),
                result.receipt,
            )
            unknown = {**result.receipt.payload(), "unexpected": True}
            with self.assertRaises(BoundCellRunContractError):
                bound_cell_run_receipt_from_payload(unknown)

            impossible_attempt = {
                **result.receipt.payload(),
                "status": "failed",
                "reason_codes": ["transport_failed"],
                "task_bytes": 0,
                "preview_sha256": None,
                "selected_cell_id": None,
                "invocation_attempts": 1,
                "endpoint_probe_requests": 0,
                "delivery_status": "attempted_unknown",
                "response_sha256": None,
                "response_bytes": None,
                "response_chars": None,
                "digest": "",
            }
            with self.assertRaises(BoundCellRunContractError):
                bound_cell_run_receipt_from_payload(impossible_attempt)

            impossible_block = {
                **result.receipt.payload(),
                "status": "blocked",
                "reason_codes": ["model_probe_failed"],
                "invocation_attempts": 0,
                "endpoint_probe_requests": 2,
                "delivery_status": "not_attempted",
                "response_sha256": None,
                "response_bytes": None,
                "response_chars": None,
                "digest": "",
            }
            with self.assertRaises(BoundCellRunContractError):
                bound_cell_run_receipt_from_payload(impossible_block)

    def test_invalid_utf8_text_and_post_probe_drift_still_produce_a_receipt(
        self,
    ) -> None:
        task = "PRIVATE-TASK-BODY"
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _InspectionFixture(Path(temporary), GENERIC_BACKEND)
            fixture.make_verified()
            target = resolve_bound_cell_target(fixture.request_path)
            transport = _InvalidResponseWithDriftTransport(target.expert.model)
            snapshot = resource_snapshot()
            preview = self._preview(target, task)
            claim = claim_for(
                preview, snapshot, passport_sha256=target.passport.digest
            )

            with patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ):
                result = run_bound_cell(
                    fixture.root / "advisor.json",
                    task,
                    fixture.catalog_path,
                    fixture.root / "evaluation.json",
                    fixture.root / "policy.json",
                    fixture.request_path,
                    confirmed=True,
                    transport=transport,
                    previewer=lambda *_args, **_kwargs: preview_evaluation(
                        preview, snapshot
                    ),
                    snapshot_collector=lambda: snapshot,
                    lease_store=FakeLeaseStore(),
                    clock=lambda: NOW,
                    publication_path=fixture.root / "run-receipt.json",
                )

            self.assertEqual(result.receipt.status, "invalidated")
            self.assertIn("response_invalid", result.receipt.reason_codes)
            self.assertIn("model_identity_changed", result.receipt.reason_codes)
            self.assertIsNone(result.receipt.response_sha256)
            self.assertIsNone(result.response_text)

    def test_wall_clock_rollback_after_inference_invalidates_without_losing_receipt(
        self,
    ) -> None:
        task = "PRIVATE-TASK-BODY"
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _InspectionFixture(Path(temporary), GENERIC_BACKEND)
            fixture.make_verified()
            target = resolve_bound_cell_target(fixture.request_path)
            transport = _StableTransport(target.expert.model)
            wall_times = iter((NOW, NOW, NOW, NOW - timedelta(seconds=1)))
            monotonic_times = iter((10.0, 11.0))
            snapshot = resource_snapshot()
            preview = self._preview(target, task)
            claim = claim_for(
                preview, snapshot, passport_sha256=target.passport.digest
            )

            with patch(
                "local_moe.bound_cell_run.cooperative_resource_claim_from_preview",
                return_value=claim,
            ):
                result = run_bound_cell(
                    fixture.root / "advisor.json",
                    task,
                    fixture.catalog_path,
                    fixture.root / "evaluation.json",
                    fixture.root / "policy.json",
                    fixture.request_path,
                    confirmed=True,
                    transport=transport,
                    previewer=lambda *_args, **_kwargs: preview_evaluation(
                        preview, snapshot
                    ),
                    snapshot_collector=lambda: snapshot,
                    lease_store=FakeLeaseStore(),
                    clock=lambda: next(wall_times),
                    monotonic_clock=lambda: next(monotonic_times),
                    publication_path=fixture.root / "run-receipt.json",
                )

            self.assertEqual(result.receipt.status, "invalidated")
            self.assertIn("clock_invalid", result.receipt.reason_codes)
            self.assertEqual(result.receipt.elapsed_ms, 1000)
            self.assertEqual(result.receipt.started_at, result.receipt.completed_at)
            self.assertEqual(result.response_text, "PRIVATE-RESPONSE-BODY")


if __name__ == "__main__":
    unittest.main()
