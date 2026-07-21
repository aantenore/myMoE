from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from local_moe.adaptive_execution_cli import main as cell_execution_main
from local_moe.adaptive_execution_gate import AdaptiveCellExecutionPreviewReceipt
from local_moe.adaptive_advisor_service import MAX_TASK_BYTES
from local_moe.cli import main as mymoe_main


SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, SHA_F = (character * 64 for character in "abcdef")
EVALUATED_AT = "2026-07-21T12:01:00+00:00"


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


if __name__ == "__main__":
    unittest.main()
