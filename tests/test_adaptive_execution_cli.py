from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import unittest

from local_moe.adaptive_execution_cli import main as cell_execution_main
from local_moe.adaptive_execution_gate import AdaptiveCellExecutionPreviewReceipt
from local_moe.adaptive_advisor_service import MAX_TASK_BYTES
from local_moe.bound_cell_run_contracts import BoundCellRunReceipt
from local_moe.bound_cell_run_envelope import (
    BoundCellRunEnvelopeV2,
    bound_cell_run_envelope_v2_from_payload,
)
from local_moe.cli import main as mymoe_main
from local_moe.cooperative_resource_lease_contracts import (
    CooperativeResourceClaim,
    CooperativeResourceLeaseAdmissionReceipt,
    CooperativeResourceLeaseReleaseReceipt,
    CooperativeResourceLeaseTransitionReceipt,
)


SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, SHA_F = (character * 64 for character in "abcdef")
SHA_0, SHA_1 = (character * 64 for character in "01")
EVALUATED_AT = "2026-07-21T12:01:00+00:00"
COMPLETED_AT = "2026-07-21T12:01:01+00:00"


def _preview(*, passed: bool) -> AdaptiveCellExecutionPreviewReceipt:
    return AdaptiveCellExecutionPreviewReceipt(
        source_advisor_receipt_sha256=SHA_A,
        source_request_sha256=SHA_B,
        fresh_advisor_receipt_sha256=SHA_C,
        fresh_request_sha256=SHA_D,
        policy_sha256=SHA_E,
        evaluated_at=EVALUATED_AT,
        source_selected_cell_id="coder-local",
        fresh_selected_cell_id="coder-local" if passed else "other-cell",
        source_passport_sha256=SHA_A,
        fresh_passport_sha256=SHA_A if passed else SHA_B,
        fresh_resource_snapshot_sha256=SHA_F,
        status="admission_passed" if passed else "admission_blocked",
        reason_codes=()
        if passed
        else ("selected_cell_changed", "selected_passport_changed"),
        task_chars=12,
    )


def _arguments() -> list[str]:
    return [
        "preview",
        "--receipt",
        "receipt.json",
        "--task-stdin",
        "--catalog",
        "adaptive-cells.json",
        "--evaluation-contract",
        "adaptive-evaluation-contract.json",
        "--policy",
        "adaptive-execution-policy.json",
        "--json",
    ]


def _run_arguments(receipt_out: Path, *, confirm: bool = True) -> list[str]:
    arguments = [
        "run",
        "--receipt",
        "receipt.json",
        "--task-stdin",
        "--catalog",
        "adaptive-cells.json",
        "--evaluation-contract",
        "adaptive-evaluation-contract.json",
        "--policy",
        "adaptive-execution-policy.json",
        "--binding-request",
        "cell-binding-request.json",
        "--receipt-out",
        str(receipt_out),
    ]
    if confirm:
        arguments.append("--confirm")
    return arguments


def _run_receipt(*, completed: bool) -> BoundCellRunReceipt:
    if not completed:
        return BoundCellRunReceipt(
            policy_sha256=SHA_A,
            status="blocked",
            reason_codes=("confirmation_required",),
            started_at=EVALUATED_AT,
            completed_at=COMPLETED_AT,
            confirmed=False,
            task_sha256=SHA_B,
            task_bytes=12,
        )
    return BoundCellRunReceipt(
        policy_sha256=SHA_A,
        status="completed",
        reason_codes=(),
        started_at=EVALUATED_AT,
        completed_at=COMPLETED_AT,
        confirmed=True,
        task_sha256=SHA_B,
        task_bytes=12,
        preview_sha256=SHA_C,
        selected_cell_id="coder-local",
        passport_sha256=SHA_D,
        declaration_sha256=SHA_E,
        expert_id="coder-expert",
        pre_binding_bundle_sha256=SHA_F,
        pre_binding_request_sha256=SHA_0,
        pre_binding_manifest_sha256=SHA_1,
        pre_inspection_receipt_sha256=SHA_A,
        post_binding_bundle_sha256=SHA_F,
        post_binding_request_sha256=SHA_0,
        post_binding_manifest_sha256=SHA_1,
        post_inspection_receipt_sha256=SHA_A,
        pre_config_source_sha256=SHA_B,
        post_config_source_sha256=SHA_B,
        pre_model_identity_set_sha256=SHA_C,
        post_model_identity_set_sha256=SHA_C,
        response_sha256=SHA_F,
        response_bytes=21,
        response_chars=21,
        invocation_attempts=1,
        endpoint_probe_requests=2,
        delivery_status="response_received",
    )


def _completed_envelope() -> BoundCellRunEnvelopeV2:
    run = _run_receipt(completed=True)
    claim = CooperativeResourceClaim(
        preview_sha256=run.preview_sha256,
        candidate_sha256=SHA_E,
        passport_sha256=run.passport_sha256,
        resource_snapshot_sha256=SHA_F,
        resource_class_sha256=SHA_0,
        catalog_sha256=SHA_1,
        profile_sha256=SHA_A,
        pool="system",
        system_claim_bytes=600,
        accelerator_claim_bytes=0,
        accelerator_identity_sha256=None,
        safety_reserve_bytes=100,
    )
    admission = CooperativeResourceLeaseAdmissionReceipt(
        policy_sha256=SHA_A,
        claim_sha256=claim.digest,
        resource_snapshot_sha256=claim.resource_snapshot_sha256,
        coordination_domain_sha256=SHA_B,
        status="acquired",
        reason_codes=(),
        evaluated_at=EVALUATED_AT,
        lease_id="lease-1",
        lease_token_sha256=SHA_E,
        active_leases_before=0,
        active_leases_after=1,
        reaped_leases=0,
        system_available_bytes=1_000,
        accelerator_available_bytes=None,
        active_system_claim_bytes=0,
        active_accelerator_claim_bytes=0,
        requested_system_claim_bytes=claim.system_claim_bytes,
        requested_accelerator_claim_bytes=claim.accelerator_claim_bytes,
        safety_reserve_bytes=claim.safety_reserve_bytes,
        applied_system_reserve_bytes=claim.safety_reserve_bytes,
        applied_accelerator_reserve_bytes=0,
    )
    transition = CooperativeResourceLeaseTransitionReceipt(
        policy_sha256=admission.policy_sha256,
        admission_receipt_sha256=admission.digest,
        claim_sha256=claim.digest,
        coordination_domain_sha256=admission.coordination_domain_sha256,
        lease_id=admission.lease_id,
        state="delivery_armed",
        transition_applied=True,
        reason_codes=(),
        transitioned_at=EVALUATED_AT,
    )
    release = CooperativeResourceLeaseReleaseReceipt(
        policy_sha256=admission.policy_sha256,
        admission_receipt_sha256=admission.digest,
        claim_sha256=claim.digest,
        coordination_domain_sha256=admission.coordination_domain_sha256,
        lease_id=admission.lease_id,
        status="released",
        reason_codes=(),
        delivery_status="response_received",
        released_at=COMPLETED_AT,
        active_leases_after=0,
    )
    return BoundCellRunEnvelopeV2(
        run_receipt=run,
        lease_claim=claim,
        lease_admission_receipt=admission,
        lease_transition_receipt=transition,
        lease_release_receipt=release,
    )


def _run_result(
    *,
    completed: bool,
    response_text: str | None,
    interruption: BaseException | None = None,
) -> SimpleNamespace:
    envelope = (
        _completed_envelope()
        if completed
        else BoundCellRunEnvelopeV2(run_receipt=_run_receipt(completed=False))
    )
    return SimpleNamespace(
        receipt=envelope.run_receipt,
        envelope=envelope,
        response_text=response_text,
        interruption=interruption,
    )


class AdaptiveExecutionCliTests(unittest.TestCase):
    def test_json_preview_has_stable_exit_codes_and_never_echoes_task(self) -> None:
        secret = "SECRET task that must never be emitted"
        for passed, expected_status in ((True, 0), (False, 1)):
            stdout, stderr = StringIO(), StringIO()
            with (
                patch("sys.stdin", StringIO(secret)),
                patch(
                    "local_moe.adaptive_execution_cli.preview_cell_execution",
                    return_value=_preview(passed=passed),
                ) as preview,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                status = cell_execution_main(_arguments())
            self.assertEqual(status, expected_status)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                payload["status"],
                "admission_passed" if passed else "admission_blocked",
            )
            self.assertFalse(payload["applied"])
            self.assertFalse(payload["authorizes_execution"])
            self.assertEqual(payload["model_invocations"], 0)
            self.assertNotIn(secret, stdout.getvalue())
            self.assertEqual(preview.call_args.kwargs["task_text"], secret)

    def test_invalid_invocation_and_oversize_stdin_exit_two_before_preview(
        self,
    ) -> None:
        stderr = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            status = cell_execution_main(["preview"])
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stderr.getvalue())["code"], "invocation_invalid")

        with (
            patch("sys.stdin", StringIO("x" * (MAX_TASK_BYTES + 1))),
            patch("local_moe.adaptive_execution_cli.preview_cell_execution") as preview,
            redirect_stdout(StringIO()),
            redirect_stderr(stderr := StringIO()),
        ):
            status = cell_execution_main(_arguments())
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stderr.getvalue())["code"], "task_too_large")
        preview.assert_not_called()

    def test_task_file_preserves_exact_bytes_without_shell_redirection(self) -> None:
        task_text = "exact UTF-8 task: caffè\n"
        with tempfile.TemporaryDirectory() as temporary:
            task_path = Path(temporary) / "task.txt"
            task_path.write_bytes(task_text.encode("utf-8"))
            arguments = _arguments()
            stdin_index = arguments.index("--task-stdin")
            arguments[stdin_index : stdin_index + 1] = [
                "--task-file",
                str(task_path),
            ]
            with (
                patch(
                    "local_moe.adaptive_execution_cli.preview_cell_execution",
                    return_value=_preview(passed=True),
                ) as preview,
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                status = cell_execution_main(arguments)

        self.assertEqual(status, 0)
        self.assertEqual(preview.call_args.kwargs["task_text"], task_text)

    def test_root_dispatch_and_help_expose_preview_without_apply_surface(self) -> None:
        root_stdout = StringIO()
        with patch("sys.argv", ["mymoe", "--help"]), redirect_stdout(root_stdout):
            with self.assertRaises(SystemExit) as root_exit:
                mymoe_main()
        self.assertEqual(root_exit.exception.code, 0)
        self.assertIn("cell-exec", root_stdout.getvalue())

        help_stdout = StringIO()
        with (
            patch("sys.argv", ["mymoe", "cell-exec", "preview", "--help"]),
            redirect_stdout(help_stdout),
        ):
            with self.assertRaises(SystemExit) as help_exit:
                mymoe_main()
        self.assertEqual(help_exit.exception.code, 0)
        help_text = help_stdout.getvalue()
        for option in (
            "--receipt",
            "--task-file",
            "--task-stdin",
            "--catalog",
            "--evaluation-contract",
            "--policy",
            "--json",
        ):
            self.assertIn(option, help_text)
        self.assertNotIn("--apply", help_text)
        self.assertNotIn("--execute", help_text)
        self.assertNotIn("--tool", help_text)

    def test_run_separates_answer_from_private_receipt_and_requires_confirm(
        self,
    ) -> None:
        secret_task = "SECRET task input"
        secret_response = "SECRET model response"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_out = root / "run-receipt.json"
            completed = _run_result(
                completed=True,
                response_text=secret_response,
            )
            with (
                patch("sys.stdin", StringIO(secret_task)),
                patch(
                    "local_moe.adaptive_execution_cli._run_publication_boundary",
                    return_value=(receipt_out, (), ()),
                ),
                patch(
                    "local_moe.adaptive_execution_cli.run_bound_cell",
                    return_value=completed,
                ) as run,
                redirect_stdout(stdout := StringIO()),
                redirect_stderr(stderr := StringIO()),
            ):
                status = cell_execution_main(_run_arguments(receipt_out))

            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), secret_response)
            self.assertIn("process identity", stderr.getvalue())
            persisted = receipt_out.read_text(encoding="utf-8")
            payload = json.loads(persisted)
            parsed = bound_cell_run_envelope_v2_from_payload(payload)
            self.assertEqual(parsed, completed.envelope)
            self.assertEqual(payload["contract"], "BoundCellRunEnvelopeV2")
            self.assertEqual(payload["schema_version"], "2.0")
            self.assertEqual(payload["run_receipt"]["contract"], "BoundCellRunReceipt")
            self.assertEqual(payload["run_receipt"]["schema_version"], "1.0")
            self.assertEqual(
                payload["run_receipt_sha256"],
                payload["run_receipt"]["digest"],
            )
            self.assertNotIn(secret_task, persisted)
            self.assertNotIn(secret_response, persisted)
            self.assertNotIn("RAW_LEASE_TOKEN_super_secret_991", persisted)
            for status_fragment in (
                "Cooperative lease: released",
                f"Envelope SHA-256: {completed.envelope.digest}",
                f"Nested v1 receipt SHA-256: {completed.receipt.digest}",
                "cooperative accounting only",
                "not a RAM/VRAM reservation or runtime lifecycle control",
            ):
                self.assertIn(status_fragment, stderr.getvalue())
            self.assertTrue(run.call_args.kwargs["confirmed"])
            self.assertEqual(run.call_args.kwargs["task_text"], secret_task)
            self.assertEqual(run.call_args.kwargs["publication_path"], receipt_out)
            self.assertEqual(list(root.glob(".*.mymoe-pending-*")), [])

            blocked_out = root / "blocked-receipt.json"
            blocked = _run_result(
                completed=False,
                response_text=None,
            )
            with (
                patch("sys.stdin", StringIO(secret_task)),
                patch(
                    "local_moe.adaptive_execution_cli._run_publication_boundary",
                    return_value=(blocked_out, (), ()),
                ),
                patch(
                    "local_moe.adaptive_execution_cli.run_bound_cell",
                    return_value=blocked,
                ) as run,
                redirect_stdout(stdout := StringIO()),
                redirect_stderr(StringIO()),
            ):
                status = cell_execution_main(_run_arguments(blocked_out, confirm=False))
            self.assertEqual(status, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(run.call_args.kwargs["confirmed"])

    def test_run_keeps_a_finalized_recovery_journal_if_publication_loses_a_race(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_out = root / "run-receipt.json"
            result = _run_result(
                completed=True,
                response_text="response-body",
            )
            with (
                patch("sys.stdin", StringIO("task-body")),
                patch(
                    "local_moe.adaptive_execution_cli._run_publication_boundary",
                    return_value=(receipt_out, (), ()),
                ),
                patch(
                    "local_moe.adaptive_execution_cli.run_bound_cell",
                    return_value=result,
                ),
                patch(
                    "local_moe.adaptive_execution_cli._publish_run_envelope",
                    side_effect=OSError("synthetic publication race"),
                ),
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
            ):
                status = cell_execution_main(_run_arguments(receipt_out))

            self.assertEqual(status, 2)
            self.assertFalse(receipt_out.exists())
            journals = list(root.glob(".*.mymoe-pending-*"))
            self.assertEqual(len(journals), 1)
            journal = journals[0].read_text(encoding="utf-8")
            entries = [json.loads(line) for line in journal.splitlines()]
            self.assertEqual(
                [entry["state"] for entry in entries], ["reserved", "finalized"]
            )
            self.assertEqual(
                [entry["schema_version"] for entry in entries], ["2.0", "2.0"]
            )
            finalized = entries[1]
            self.assertEqual(
                finalized["envelope"]["contract"], "BoundCellRunEnvelopeV2"
            )
            self.assertEqual(
                finalized["envelope"]["run_receipt"]["contract"],
                "BoundCellRunReceipt",
            )
            self.assertEqual(
                bound_cell_run_envelope_v2_from_payload(finalized["envelope"]),
                result.envelope,
            )
            self.assertIn('"state": "reserved"', journal)
            self.assertIn('"state": "finalized"', journal)
            self.assertIn('"status": "completed"', journal)
            self.assertNotIn("task-body", journal)
            self.assertNotIn("response-body", journal)
            self.assertNotIn("RAW_LEASE_TOKEN_super_secret_991", journal)

    def test_run_persists_receipt_before_reraising_an_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_out = root / "run-receipt.json"
            interruption = KeyboardInterrupt()
            result = _run_result(
                completed=True,
                response_text="response-body",
                interruption=interruption,
            )
            with (
                patch("sys.stdin", StringIO("task-body")),
                patch(
                    "local_moe.adaptive_execution_cli._run_publication_boundary",
                    return_value=(receipt_out, (), ()),
                ),
                patch(
                    "local_moe.adaptive_execution_cli.run_bound_cell",
                    return_value=result,
                ),
                redirect_stdout(stdout := StringIO()),
                redirect_stderr(StringIO()),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    cell_execution_main(_run_arguments(receipt_out))

            self.assertTrue(receipt_out.exists())
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(list(root.glob(".*.mymoe-pending-*")), [])

    def test_run_completes_short_stdout_writes_and_ignores_broken_status_stream(
        self,
    ) -> None:
        class ShortBuffer:
            def __init__(self) -> None:
                self.value = bytearray()

            def write(self, value: bytes) -> int:
                accepted = min(3, len(value))
                self.value.extend(value[:accepted])
                return accepted

            def flush(self) -> None:
                return None

        class BrokenStatusStream:
            def write(self, _value: str) -> int:
                raise BrokenPipeError()

            def flush(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_out = root / "run-receipt.json"
            response = "exact caffè response"
            result = _run_result(
                completed=True,
                response_text=response,
            )
            short_buffer = ShortBuffer()
            with (
                patch("sys.stdin", StringIO("task-body")),
                patch("sys.stdout", SimpleNamespace(buffer=short_buffer)),
                patch("sys.stderr", BrokenStatusStream()),
                patch(
                    "local_moe.adaptive_execution_cli._run_publication_boundary",
                    return_value=(receipt_out, (), ()),
                ),
                patch(
                    "local_moe.adaptive_execution_cli.run_bound_cell",
                    return_value=result,
                ),
            ):
                status = cell_execution_main(_run_arguments(receipt_out))

            self.assertEqual(status, 0)
            self.assertEqual(bytes(short_buffer.value), response.encode("utf-8"))
            self.assertTrue(receipt_out.exists())

    def test_run_help_exposes_one_shot_boundary_without_tool_or_lifecycle_flags(
        self,
    ) -> None:
        help_stdout = StringIO()
        with (
            patch("sys.argv", ["mymoe", "cell-exec", "run", "--help"]),
            redirect_stdout(help_stdout),
        ):
            with self.assertRaises(SystemExit) as help_exit:
                mymoe_main()
        self.assertEqual(help_exit.exception.code, 0)
        help_text = help_stdout.getvalue()
        for option in (
            "--receipt",
            "--task-file",
            "--task-stdin",
            "--catalog",
            "--evaluation-contract",
            "--policy",
            "--binding-request",
            "--receipt-out",
            "--confirm",
        ):
            self.assertIn(option, help_text)
        for forbidden in ("--tool", "--start", "--stop", "--retry", "--fallback"):
            self.assertNotIn(forbidden, help_text)
        self.assertIn("BoundCellRunEnvelopeV2", help_text)
        self.assertIn("metadata-only", help_text)


if __name__ == "__main__":
    unittest.main()
