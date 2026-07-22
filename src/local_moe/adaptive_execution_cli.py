from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import IO, Sequence

from .adaptive_advisor_cli import (
    AdvisorCliError,
    ProtectedRootIdentity,
    _write_output,
    capture_protected_root_identity,
)
from .adaptive_advisor_service import MAX_TASK_BYTES
from .adaptive_execution_gate import (
    AdaptiveCellExecutionPreviewReceipt,
    AdaptiveExecutionGateError,
    preview_cell_execution,
)
from .bound_cell_run import BoundCellRunResult, run_bound_cell
from .bound_cell_run_contracts import (
    BoundCellRunContractError,
    BoundCellRunPolicy,
)
from .bound_cell_run_envelope import BoundCellRunEnvelopeV2
from .cell_contracts import CellContractError
from .runtime_binding_inspector import (
    RuntimeBindingInspectionError,
    load_cell_binding_inspect_request,
)
from .secure_files import read_bounded_regular_file


EXIT_COMPLETED = 0
EXIT_BLOCKED = 1
EXIT_INVALID = 2
EXIT_FAILED = 3
EXIT_INVALIDATED = 4


class AdaptiveExecutionCliError(ValueError):
    """Stable, task-safe command failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class _RunRecoveryJournal:
    path: Path
    identity: tuple[int, int]
    size: int

    def finalize(self, envelope: BoundCellRunEnvelopeV2) -> None:
        encoded = (
            json.dumps(
                {
                    "contract": "BoundCellRunRecoveryJournal",
                    "schema_version": "2.0",
                    "state": "finalized",
                    "envelope": envelope.payload(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            before = self.path.lstat()
            if (
                stat.S_ISLNK(before.st_mode)
                or (before.st_dev, before.st_ino) != self.identity
            ):
                raise AdaptiveExecutionCliError(
                    "receipt_journal_changed",
                    "Run receipt recovery journal changed before finalization.",
                )
            descriptor = os.open(self.path, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != self.identity
                or opened.st_size != self.size
            ):
                raise AdaptiveExecutionCliError(
                    "receipt_journal_changed",
                    "Run receipt recovery journal could not be verified.",
                )
            _write_descriptor(descriptor, encoded)
            os.fsync(descriptor)
            self.size += len(encoded)
        except AdaptiveExecutionCliError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise AdaptiveExecutionCliError(
                "receipt_journal_failed",
                "Run receipt recovery journal could not be finalized.",
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def remove(self) -> None:
        try:
            current = self.path.lstat()
            if (
                stat.S_ISLNK(current.st_mode)
                or (current.st_dev, current.st_ino) != self.identity
            ):
                return
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return


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
            "Preview exact local-cell admission or explicitly run one already-"
            "resident, compute-only loopback cell once."
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

    run = commands.add_parser(
        "run",
        help="Invoke one evidence-bound, already-resident local cell once.",
        description=(
            "Recheck one exact recommendation and declared cell binding, then "
            "make at most one tool-free loopback model request."
        ),
        allow_abbrev=False,
    )
    run.add_argument(
        "--receipt", required=True, help="Persisted Adaptive Advisor receipt JSON."
    )
    run_task_group = run.add_mutually_exclusive_group(required=True)
    run_task_group.add_argument(
        "--task-file",
        help="Read the exact task from one bounded regular non-link UTF-8 file.",
    )
    run_task_group.add_argument(
        "--task-stdin",
        action="store_true",
        help="Read the exact bounded UTF-8 task from standard input.",
    )
    run.add_argument(
        "--catalog", required=True, help="Current Adaptive Cell catalog JSON."
    )
    run.add_argument(
        "--evaluation-contract",
        required=True,
        help="Current evaluation contract whose byte digest must remain unchanged.",
    )
    run.add_argument(
        "--policy",
        required=True,
        help="Strict dry-run Adaptive Cell Execution policy JSON.",
    )
    run.add_argument(
        "--binding-request",
        required=True,
        help="Bounded CellBindingInspectRequest for the selected local cell.",
    )
    run.add_argument(
        "--receipt-out",
        required=True,
        help=(
            "Publish the metadata-only BoundCellRunEnvelopeV2 evidence to one "
            "new private file; existing files are never replaced."
        ),
    )
    run.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "Authorize only this one at-most-once model request. Omitting it "
            "produces a blocked receipt before endpoint traffic."
        ),
    )
    run.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request loopback timeout, from greater than 0 through 120 seconds.",
    )
    run.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Bounded provider output-token request, from 1 through 32768.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        task_text = _load_task(args)
        if args.command == "preview":
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
            print(rendered)
            return (
                EXIT_COMPLETED if receipt.status == "admission_passed" else EXIT_BLOCKED
            )
        if args.command == "run":
            return _run_once(args, task_text)
        raise AdaptiveExecutionCliError(
            "invocation_invalid", "Adaptive cell execution command is required."
        )
    except AdaptiveExecutionCliError as exc:
        _render_error(exc.code, str(exc))
        return EXIT_INVALID
    except AdaptiveExecutionGateError as exc:
        _render_error(exc.code, str(exc))
        return EXIT_INVALID
    except BoundCellRunContractError as exc:
        _render_error(
            exc.code,
            "Bound cell run contract validation failed safely.",
        )
        return EXIT_INVALID
    except RuntimeBindingInspectionError as exc:
        _render_error(
            getattr(exc, "code", "binding_request_invalid"),
            "Bound cell run inputs could not be verified safely.",
        )
        return EXIT_INVALID
    except AdvisorCliError as exc:
        _render_error(
            getattr(exc, "code", "receipt_publish_failed"),
            "Bound cell run receipt could not be published safely.",
        )
        return EXIT_INVALID
    except Exception:
        _render_error(
            "cell_execution_failed",
            "Adaptive cell execution failed safely.",
        )
        return EXIT_INVALID


def _run_once(args: argparse.Namespace, task_text: str) -> int:
    output_path, protected_inputs, protected_roots = _run_publication_boundary(args)
    _preflight_receipt_output(output_path, protected_inputs, protected_roots)
    policy = BoundCellRunPolicy(
        timeout_seconds=args.timeout_seconds,
        max_task_bytes=MAX_TASK_BYTES,
        max_output_tokens=args.max_output_tokens,
    )
    journal = _reserve_run_journal(
        output_path,
        protected_inputs=protected_inputs,
        protected_roots=protected_roots,
    )
    result = run_bound_cell(
        source_advisor_receipt_path=args.receipt,
        task_text=task_text,
        catalog_path=args.catalog,
        evaluation_contract_path=args.evaluation_contract,
        adaptive_policy_path=args.policy,
        binding_request_path=args.binding_request,
        confirmed=args.confirm,
        policy=policy,
        publication_path=output_path,
    )
    journal.finalize(result.envelope)
    refreshed_output, refreshed_inputs, refreshed_roots = _run_publication_boundary(
        args
    )
    if refreshed_output != output_path:
        raise AdaptiveExecutionCliError(
            "receipt_output_changed",
            "Run receipt output changed before publication.",
        )
    args.task_file = None
    del task_text
    actual_inputs = _merge_paths(
        _merge_paths(tuple(protected_inputs), tuple(refreshed_inputs)),
        tuple(getattr(result, "publication_inputs", ())),
    )
    actual_roots = _merge_root_identities(
        _merge_root_identities(tuple(protected_roots), tuple(refreshed_roots)),
        tuple(getattr(result, "publication_protected_roots", ())),
    )
    _publish_run_envelope(
        output_path,
        result.envelope,
        protected_inputs=actual_inputs,
        protected_roots=actual_roots,
    )
    journal.remove()
    interruption = getattr(result, "interruption", None)
    if interruption is not None:
        _render_run_status(result, output_path)
        raise interruption
    if result.response_text is not None:
        _write_response_stdout(result.response_text)
    _render_run_status(result, output_path)
    return {
        "completed": EXIT_COMPLETED,
        "blocked": EXIT_BLOCKED,
        "failed": EXIT_FAILED,
        "invalidated": EXIT_INVALIDATED,
    }[result.receipt.status]


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


def _run_publication_boundary(
    args: argparse.Namespace,
) -> tuple[Path, tuple[Path, ...], tuple[ProtectedRootIdentity, ...]]:
    binding_request_path = _absolute_path(args.binding_request)
    binding_request = load_cell_binding_inspect_request(binding_request_path)
    binding_root = binding_request_path.parent
    protected_roots = tuple(
        capture_protected_root_identity(binding_root / relative)
        for relative in (
            binding_request.runtime_root,
            binding_request.model_artifact_root,
        )
    )
    inputs = [
        _absolute_path(args.receipt),
        _absolute_path(args.catalog),
        _absolute_path(args.evaluation_contract),
        _absolute_path(args.policy),
        binding_request_path,
        binding_root / binding_request.catalog_path,
        binding_root / binding_request.runtime_config_path,
    ]
    if args.task_file is not None:
        inputs.append(_absolute_path(args.task_file))
    return (
        _absolute_path(args.receipt_out),
        tuple(inputs),
        protected_roots,
    )


def _preflight_receipt_output(
    path: Path,
    protected_inputs: Sequence[Path],
    protected_roots: Sequence[ProtectedRootIdentity],
) -> None:
    """Reject predictable publication failures before any endpoint traffic.

    The hardened no-clobber publisher repeats the authoritative checks after
    the model call. This preflight only keeps an already-existing target,
    obvious input alias, or protected-root destination from causing a known
    post-invocation publication failure.
    """

    try:
        parent_metadata = path.parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            raise AdaptiveExecutionCliError(
                "receipt_parent_invalid",
                "Run receipt parent must be one real existing directory.",
            )
        if os.path.lexists(path):
            raise AdaptiveExecutionCliError(
                "receipt_exists",
                "Run receipt output already exists; overwrite is forbidden.",
            )
        canonical_parent = path.parent.resolve(strict=True)
        canonical_output = canonical_parent / path.name
        for protected in protected_inputs:
            if canonical_output == protected.resolve(strict=False):
                raise AdaptiveExecutionCliError(
                    "receipt_aliases_input",
                    "Run receipt output must differ from every input.",
                )
        for identity in protected_roots:
            root = identity.path.resolve(strict=True)
            try:
                canonical_output.relative_to(root)
            except ValueError:
                continue
            raise AdaptiveExecutionCliError(
                "receipt_path_conflict",
                "Run receipt output must stay outside inspected artifact roots.",
            )
    except AdaptiveExecutionCliError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AdaptiveExecutionCliError(
            "receipt_output_invalid",
            "Run receipt output could not be verified.",
        ) from exc


def _reserve_run_journal(
    output_path: Path,
    *,
    protected_inputs: Sequence[Path],
    protected_roots: Sequence[ProtectedRootIdentity],
) -> _RunRecoveryJournal:
    journal_path = output_path.with_name(
        f".{output_path.name}.mymoe-pending-{secrets.token_hex(16)}"
    )
    encoded = (
        json.dumps(
            {
                "contract": "BoundCellRunRecoveryJournal",
                "schema_version": "2.0",
                "state": "reserved",
                "invocation_status": "not_started_at_reservation",
                "recovery_note": (
                    "If no finalized receipt follows, a later invocation may have "
                    "an unknown delivery outcome."
                ),
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_output(
        journal_path,
        encoded,
        protected_inputs=protected_inputs,
        protected_root_identities=protected_roots,
    )
    try:
        metadata = journal_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600)
            or metadata.st_size != len(encoded)
        ):
            raise OSError("unsafe journal")
    except OSError as exc:
        raise AdaptiveExecutionCliError(
            "receipt_journal_failed",
            "Run receipt recovery journal could not be verified.",
        ) from exc
    return _RunRecoveryJournal(
        path=journal_path,
        identity=(metadata.st_dev, metadata.st_ino),
        size=len(encoded),
    )


def _merge_paths(first: tuple[Path, ...], second: tuple[Path, ...]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for item in (*first, *second):
        absolute = _absolute_path(item)
        key = os.path.normcase(os.fspath(absolute))
        if key not in seen:
            result.append(absolute)
            seen.add(key)
    return tuple(result)


def _merge_root_identities(
    first: tuple[ProtectedRootIdentity, ...],
    second: tuple[ProtectedRootIdentity, ...],
) -> tuple[ProtectedRootIdentity, ...]:
    result: list[ProtectedRootIdentity] = []
    for identity in (*first, *second):
        if identity not in result:
            result.append(identity)
    return tuple(result)


def _write_descriptor(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short journal write")
        offset += written


def _write_response_stdout(response: str) -> None:
    encoded = response.encode("utf-8")
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        offset = 0
        while offset < len(response):
            written = sys.stdout.write(response[offset:])
            if (
                isinstance(written, bool)
                or not isinstance(written, int)
                or written <= 0
            ):
                raise OSError("standard output accepted a short text write")
            offset += written
        sys.stdout.flush()
        return
    offset = 0
    while offset < len(encoded):
        written = stream.write(encoded[offset:])
        if isinstance(written, bool) or not isinstance(written, int) or written <= 0:
            raise OSError("standard output accepted a short byte write")
        offset += written
    stream.flush()


def _publish_run_envelope(
    path: Path,
    envelope: BoundCellRunEnvelopeV2,
    *,
    protected_inputs: Sequence[Path],
    protected_roots: Sequence[ProtectedRootIdentity],
) -> None:
    encoded = (
        json.dumps(
            envelope.payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_output(
        path,
        encoded,
        protected_inputs=protected_inputs,
        protected_root_identities=protected_roots,
    )


def _render_run_status(result: BoundCellRunResult, output_path: Path) -> None:
    receipt = result.receipt
    envelope = result.envelope
    if receipt.status == "completed":
        summary = "Bound cell run completed with stable sampled bindings."
    elif receipt.status == "blocked":
        summary = "Bound cell run blocked before model invocation."
    elif receipt.status == "failed":
        summary = "Bound cell run attempted once but did not return a valid response."
    else:
        summary = (
            "Bound cell run returned a response or attempt, but post-run evidence "
            "could not support a stable-completion claim."
        )
    try:
        print(
            "\n".join(
                (
                    summary,
                    "Reasons: " + (", ".join(receipt.reason_codes) or "none"),
                    f"Delivery: {receipt.delivery_status}",
                    f"Cooperative lease: {_render_lease_status(envelope)}",
                    f"Evidence envelope: {output_path}",
                    f"Envelope SHA-256: {envelope.digest}",
                    f"Nested v1 receipt SHA-256: {receipt.digest}",
                    (
                        "Boundary: the resident loopback process identity and semantic "
                        "correctness of the response are not verified."
                    ),
                    (
                        "Resource boundary: the lease is cooperative accounting only; "
                        "it is not a RAM/VRAM reservation or runtime lifecycle control."
                    ),
                )
            ),
            file=sys.stderr,
        )
    except (BrokenPipeError, OSError, ValueError):
        return


def _render_lease_status(envelope: BoundCellRunEnvelopeV2) -> str:
    release = envelope.lease_release_receipt
    if release is not None:
        return release.status
    transition = envelope.lease_transition_receipt
    if transition is not None and transition.transition_applied:
        return transition.state
    admission = envelope.lease_admission_receipt
    if admission is not None:
        return admission.status
    if envelope.lease_error_code is not None:
        return envelope.lease_error_code
    return "not_acquired"


def _absolute_path(value: object) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise AdaptiveExecutionCliError(
            "invocation_invalid", "Cell execution paths could not be resolved."
        ) from exc


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
    "EXIT_BLOCKED",
    "EXIT_COMPLETED",
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_INVALIDATED",
    "build_parser",
    "main",
    "render_human",
]
