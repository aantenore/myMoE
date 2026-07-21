from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import IO, Sequence

from .adaptive_advisor_service import MAX_TASK_BYTES
from .adaptive_execution_gate import (
    AdaptiveCellExecutionPreviewReceipt,
    AdaptiveExecutionGateError,
    preview_cell_execution,
)
from .cell_contracts import CellContractError
from .secure_files import read_bounded_regular_file


class AdaptiveExecutionCliError(ValueError):
    """Stable, task-safe command failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _ExecutionArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise AdaptiveExecutionCliError(
            "invocation_invalid", "Invalid adaptive cell execution invocation."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _ExecutionArgumentParser(
        prog="mymoe cell-exec",
        description=(
            "Preview a fresh, exact local-cell admission without executing or "
            "authorizing anything."
        ),
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser(
        "preview",
        help="Re-verify one persisted advisor receipt against current evidence.",
        allow_abbrev=False,
    )
    preview.add_argument(
        "--receipt", required=True, help="Persisted Adaptive Advisor receipt JSON."
    )
    task_group = preview.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task-file",
        help="Read the exact task from one bounded regular non-link UTF-8 file.",
    )
    task_group.add_argument(
        "--task-stdin",
        action="store_true",
        help="Read the exact bounded UTF-8 task from standard input.",
    )
    preview.add_argument(
        "--catalog", required=True, help="Current Adaptive Cell catalog JSON."
    )
    preview.add_argument(
        "--evaluation-contract",
        required=True,
        help="Current evaluation contract whose byte digest must remain unchanged.",
    )
    preview.add_argument(
        "--policy",
        required=True,
        help="Strict dry-run Adaptive Cell Execution policy JSON.",
    )
    preview.add_argument(
        "--json", action="store_true", dest="json_output", help="Emit the receipt."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command != "preview":
            raise AdaptiveExecutionCliError(
                "invocation_invalid", "Adaptive cell execution command is required."
            )
        task_text = _load_task(args)
        receipt = preview_cell_execution(
            source_receipt_path=args.receipt,
            task_text=task_text,
            catalog_path=args.catalog,
            evaluation_contract_path=args.evaluation_contract,
            policy_path=args.policy,
        )
        args.task_file = None
        del task_text
        rendered = (
            json.dumps(receipt.payload(), indent=2, sort_keys=True)
            if args.json_output
            else render_human(receipt)
        )
    except AdaptiveExecutionCliError as exc:
        _render_error(exc.code, str(exc))
        return 2
    except AdaptiveExecutionGateError as exc:
        _render_error(exc.code, str(exc))
        return 2
    except Exception:
        _render_error(
            "cell_execution_preview_failed",
            "Adaptive cell execution preview failed safely.",
        )
        return 2

    print(rendered)
    return 0 if receipt.status == "admission_passed" else 1


def render_human(receipt: AdaptiveCellExecutionPreviewReceipt) -> str:
    if receipt.status == "admission_passed":
        heading = f"Admission passed: {receipt.fresh_selected_cell_id}"
        reasons = "No blockers found."
    else:
        heading = "Admission blocked"
        reasons = "; ".join(receipt.reason_codes)
    return "\n".join(
        (
            heading,
            f"Why: {reasons}",
            (
                "Boundary: this is a dry-run preview. It does not authorize or "
                "apply execution."
            ),
            (
                "No model invocation, network access, runtime lifecycle action, "
                "or tool surface was used."
            ),
            f"Receipt SHA-256: {receipt.digest}",
            f"Fresh snapshot SHA-256: {receipt.fresh_resource_snapshot_sha256}",
        )
    )


def _read_task_stdin(stream: IO[object]) -> str:
    reader = getattr(stream, "buffer", stream)
    try:
        value = reader.read(MAX_TASK_BYTES + 1)
    except (OSError, TypeError, ValueError) as exc:
        raise AdaptiveExecutionCliError(
            "task_input_invalid", "Task input could not be read from standard input."
        ) from exc
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise AdaptiveExecutionCliError(
                "task_input_invalid", "Task input must be valid UTF-8 text."
            ) from exc
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise AdaptiveExecutionCliError(
            "task_input_invalid", "Task input could not be read from standard input."
        )
    if len(encoded) > MAX_TASK_BYTES:
        raise AdaptiveExecutionCliError(
            "task_too_large", "Task input exceeds the byte limit."
        )
    return _decode_task(encoded)


def _load_task(args: argparse.Namespace) -> str:
    if args.task_file is not None:
        try:
            value = read_bounded_regular_file(
                Path(args.task_file),
                maximum_bytes=MAX_TASK_BYTES,
                label="task input",
            )
        except CellContractError as exc:
            raise AdaptiveExecutionCliError(
                "task_input_invalid", "Task file could not be verified."
            ) from exc
        return _decode_task(value)
    if args.task_stdin:
        return _read_task_stdin(sys.stdin)
    raise AdaptiveExecutionCliError("task_input_invalid", "Task input is required.")


def _decode_task(encoded: bytes) -> str:
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdaptiveExecutionCliError(
            "task_input_invalid", "Task input must be valid UTF-8 text."
        ) from exc


def _render_error(code: str, message: str) -> None:
    print(
        json.dumps({"code": code, "message": message}, sort_keys=True),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdaptiveExecutionCliError",
    "build_parser",
    "main",
    "render_human",
]
