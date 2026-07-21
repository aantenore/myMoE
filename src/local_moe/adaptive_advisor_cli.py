from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import IO, Sequence

from .adaptive_advisor_service import (
    MAX_EVALUATION_CONTRACT_BYTES,
    MAX_TASK_BYTES,
    AdaptiveAdvisorReceipt,
    AdvisorServiceError,
    evaluate_advisor,
)
from .cell_contracts import CellContractError
from .secure_files import read_bounded_regular_file


_REASON_TEXT = {
    "accelerator_identity_unknown": "accelerator identity is not verified",
    "accelerator_memory_headroom_insufficient": "accelerator memory headroom is insufficient",
    "accelerator_memory_unknown": "accelerator memory headroom is unknown",
    "advisory_only": "the result is read-only advice",
    "capability_gap": "required capabilities are unavailable",
    "context_window_exceeded": "the requested context exceeds the verified window",
    "harness_availability_unknown": "harness availability is unknown",
    "harness_expected_identity_unknown": "expected harness identity is unknown",
    "harness_identity_mismatch": "harness identity does not match",
    "harness_unavailable": "the harness is unavailable",
    "host_memory_headroom_insufficient": "host memory headroom is insufficient",
    "host_memory_unknown": "host memory headroom is unknown",
    "insufficient_samples": "too few qualified evaluation samples are available",
    "machine_not_supported": "the current machine architecture is unsupported",
    "measurement_demand_mismatch": "measurements do not cover this exact demand",
    "measurement_evaluation_contract_mismatch": "measurements use a different evaluation contract",
    "measurement_expired": "measurements have expired",
    "measurement_from_future": "measurement time is inconsistent",
    "measurement_not_applicable": "measurements do not match this machine resource class",
    "measurement_placement_mismatch": "measured and declared placement differ",
    "measurement_unknown": "qualified measurements are missing",
    "model_availability_unknown": "model availability is unknown",
    "model_expected_identity_unknown": "expected model identity is unknown",
    "model_identity_mismatch": "model identity does not match",
    "model_unavailable": "the model is unavailable",
    "no_eligible_cell": "no cell passed every hard verification boundary",
    "observation_expired": "runtime observations have expired",
    "observation_from_future": "runtime observation time is inconsistent",
    "offline_not_supported": "offline operation is not supported",
    "pareto_frontier_selected": "the verified Pareto frontier produced a selection",
    "quality_floor_not_met": "verified quality is below the selected profile floor",
    "ranking_metric_unknown": "a ranking metric is unknown",
    "resource_class_unknown": "the current machine resource class is incomplete",
    "resource_estimate_unknown": "resource estimates are missing",
    "risk_class_not_supported": "the requested risk class is unsupported",
    "runtime_availability_unknown": "runtime availability is unknown",
    "runtime_expected_identity_unknown": "expected runtime identity is unknown",
    "runtime_identity_mismatch": "runtime identity does not match",
    "runtime_unavailable": "the runtime is unavailable",
    "snapshot_from_future": "the resource snapshot time is inconsistent",
    "snapshot_stale": "the resource snapshot is stale",
    "swap_limit_exceeded": "current swap use exceeds the profile limit",
    "swap_usage_unknown": "current swap use is unknown",
    "system_not_supported": "the current operating system is unsupported",
    "tool_contract_availability_unknown": "tool contract availability is unknown",
    "tool_contract_expected_identity_unknown": "expected tool contract identity is unknown",
    "tool_contract_identity_mismatch": "tool contract identity does not match",
    "tool_contract_unavailable": "the tool contract is unavailable",
    "tool_surface_gap": "required tool surfaces are unavailable",
    "unified_memory_headroom_insufficient": "unified memory headroom is insufficient",
    "unified_memory_unavailable": "unified memory is unavailable",
    "unified_memory_unknown": "unified memory headroom is unknown",
}

_NO_SIDE_EFFECTS = (
    "No model invocation, network access, download, runtime start or stop, or "
    "configuration change was used."
)


class AdvisorCliError(ValueError):
    """Stable, task-safe command failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _AdvisorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise AdvisorCliError(
            "invocation_invalid", "Invalid adaptive advisor invocation."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _AdvisorArgumentParser(
        prog="mymoe advisor",
        description=(
            "Recommend a verified offline cell from live local resources without "
            "starting or changing anything."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--catalog", required=True, help="Adaptive cell catalog JSON.")
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        help="Task text; only its SHA-256 and character count enter the receipt.",
    )
    task_group.add_argument(
        "--task-file",
        help="Read task text from one bounded regular non-link UTF-8 file.",
    )
    task_group.add_argument(
        "--task-stdin",
        action="store_true",
        help="Read bounded UTF-8 task text from standard input.",
    )
    parser.add_argument("--workload", required=True, help="Stable workload identifier.")
    parser.add_argument(
        "--capability",
        action="append",
        required=True,
        help="Required capability; repeat for more than one.",
    )
    parser.add_argument(
        "--tool-surface",
        action="append",
        default=[],
        help="Required tool surface; repeat for more than one.",
    )
    parser.add_argument("--risk-class", required=True)
    parser.add_argument("--context-tokens", required=True, type=int)
    parser.add_argument(
        "--evaluation-contract",
        required=True,
        help="Bounded regular file whose SHA-256 qualifies evaluation evidence.",
    )
    profile_group = parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument(
        "--goal",
        dest="profile",
        help="Advisor goal declared by the catalog.",
    )
    profile_group.add_argument(
        "--profile",
        dest="profile",
        help="Advisor profile declared by the catalog.",
    )
    parser.add_argument("--intent-family-sha256")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--out", help="Atomically create a private, metadata-only receipt file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        task_text = _load_task(args)
        receipt = evaluate_advisor(
            catalog_path=args.catalog,
            evaluation_contract_path=args.evaluation_contract,
            task_text=task_text,
            workload_id=args.workload,
            required_capabilities=args.capability,
            required_tool_surfaces=args.tool_surface,
            risk_class=args.risk_class,
            context_tokens=args.context_tokens,
            profile=args.profile,
            intent_family_sha256=args.intent_family_sha256,
        )
        args.task = None
        del task_text

        rendered = (
            json.dumps(receipt.payload(), indent=2, sort_keys=True)
            if args.json_output
            else render_human(receipt)
        )
        if args.out:
            protected_inputs = [
                Path(args.catalog),
                Path(args.evaluation_contract),
            ]
            if args.task_file:
                protected_inputs.append(Path(args.task_file))
            _write_output(
                Path(args.out),
                f"{rendered}\n".encode("utf-8"),
                protected_inputs=protected_inputs,
            )
    except AdvisorCliError as exc:
        _render_error(exc.code, str(exc))
        return 2
    except AdvisorServiceError as exc:
        _render_error(exc.code, str(exc))
        return 2
    except Exception:
        _render_error("advisor_failed", "Adaptive advisor command failed safely.")
        return 2

    print(rendered)
    return 0


def render_human(receipt: AdaptiveAdvisorReceipt) -> str:
    advice = receipt.advice
    if advice.status == "recommended":
        heading = f"Recommended now: {advice.selected_cell_id}"
        selected = next(
            item for item in advice.candidates if item.cell_id == advice.selected_cell_id
        )
        tradeoff = (
            f"Trade-off: profile '{advice.profile}' selected verified evidence with "
            f"{_selected_metrics(selected)}."
        )
        evidence_lines: list[str] = []
    else:
        headings = {
            "not_available_now": "Not available now",
            "not_enough_evidence": "Not enough verified evidence",
        }
        heading = headings[receipt.display_state]
        tradeoff = (
            "Trade-off: safety over speculation; missing or incompatible evidence "
            "was not filled in."
        )
        evidence_lines = [
            f"  - {candidate.cell_id}: "
            f"{_render_reasons(candidate.rejection_codes)}"
            for candidate in advice.candidates
        ]

    lines = [
        heading,
        f"Why: {_render_reasons(advice.reason_codes)}",
        tradeoff,
    ]
    if evidence_lines:
        lines.append("Verified boundaries by candidate:")
        lines.extend(evidence_lines)
    lines.extend(
        [
            (
                "Boundary: this receipt is read-only advice. It does not authorize "
                "or apply execution."
            ),
            _NO_SIDE_EFFECTS,
            f"Receipt SHA-256: {receipt.digest}",
            f"Advice SHA-256: {advice.digest}",
            f"Request SHA-256: {receipt.request.digest}",
            f"Catalog SHA-256: {advice.catalog_sha256}",
            f"Snapshot SHA-256: {advice.resource_snapshot_sha256}",
        ]
    )
    return "\n".join(lines)


def _selected_metrics(candidate: object) -> str:
    success = getattr(candidate, "success_rate", None)
    latency = getattr(candidate, "p95_latency_ms", None)
    memory = getattr(candidate, "effective_total_memory_bytes", None)
    parts = []
    if success is not None:
        parts.append(f"verified success {success * 100:.1f}%")
    if latency is not None:
        parts.append(f"p95 latency {latency:.0f} ms")
    if memory is not None:
        parts.append(f"peak memory {_format_bytes(memory)}")
    return ", ".join(parts) if parts else "verified ranking evidence"


def _format_bytes(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _render_reasons(reason_codes: Sequence[str]) -> str:
    return "; ".join(
        _REASON_TEXT.get(code, f"unrecognized boundary ({code})")
        for code in sorted(reason_codes)
    )


def _load_task(args: argparse.Namespace) -> str:
    if args.task is not None:
        return args.task
    if args.task_file is not None:
        try:
            value = read_bounded_regular_file(
                args.task_file,
                maximum_bytes=MAX_TASK_BYTES,
                label="task input",
            )
        except CellContractError as exc:
            raise AdvisorCliError(
                "task_input_invalid", "Task file could not be verified."
            ) from exc
        return _decode_task(value)
    if args.task_stdin:
        return _read_task_stdin(sys.stdin)
    raise AdvisorCliError("task_input_invalid", "Task input is required.")


def _read_task_stdin(stream: IO[object]) -> str:
    reader = getattr(stream, "buffer", stream)
    try:
        value = reader.read(MAX_TASK_BYTES + 1)
    except (OSError, TypeError, ValueError) as exc:
        raise AdvisorCliError(
            "task_input_invalid", "Task input could not be read from standard input."
        ) from exc
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise AdvisorCliError(
                "task_input_invalid", "Task input must be valid UTF-8 text."
            ) from exc
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise AdvisorCliError(
            "task_input_invalid", "Task input could not be read from standard input."
        )
    if len(encoded) > MAX_TASK_BYTES:
        raise AdvisorCliError("task_too_large", "Task input exceeds the byte limit.")
    return _decode_task(encoded)


def _decode_task(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdvisorCliError(
            "task_input_invalid", "Task input must be valid UTF-8 text."
        ) from exc


def _write_output(
    path: Path, value: bytes, *, protected_inputs: Sequence[Path]
) -> None:
    if any(_same_location(path, protected) for protected in protected_inputs):
        raise AdvisorCliError(
            "output_aliases_input", "Advisor output must differ from every input file."
        )
    if (
        path.name in {"", ".", ".."}
        or any(part == ".." for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in path.name)
    ):
        raise AdvisorCliError(
            "output_invalid", "Advisor output must name one new file."
        )
    if os.name == "nt":
        _validate_windows_output_name(path.name)
        _write_output_windows(path, value)
        return
    _write_output_posix(path, value, protected_inputs=protected_inputs)


def _write_output_posix(
    path: Path,
    value: bytes,
    *,
    protected_inputs: Sequence[Path],
) -> None:
    parent = path.parent
    try:
        parent_before = parent.lstat()
    except OSError as exc:
        raise AdvisorCliError(
            "output_parent_invalid", "Advisor output parent must already exist."
        ) from exc
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise AdvisorCliError(
            "output_parent_invalid",
            "Advisor output parent must be a real existing directory.",
        )
    try:
        canonical_parent = parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AdvisorCliError(
            "output_parent_invalid", "Advisor output parent could not be verified."
        ) from exc
    canonical_target = canonical_parent / path.name
    if any(
        _same_location(canonical_target, protected) for protected in protected_inputs
    ):
        raise AdvisorCliError(
            "output_aliases_input", "Advisor output must differ from every input file."
        )

    descriptors = _open_posix_directory_chain(canonical_parent)
    parent_descriptor = descriptors[-1]
    try:
        parent_identity = os.fstat(parent_descriptor)
    except OSError as exc:
        for item in reversed(descriptors):
            try:
                os.close(item)
            except OSError:
                pass
        raise AdvisorCliError(
            "output_parent_invalid", "Advisor output parent could not be pinned."
        ) from exc
    try:
        os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        for item in reversed(descriptors):
            try:
                os.close(item)
            except OSError:
                pass
        raise AdvisorCliError(
            "output_invalid", "Advisor output could not be inspected safely."
        ) from exc
    else:
        for item in reversed(descriptors):
            try:
                os.close(item)
            except OSError:
                pass
        raise AdvisorCliError(
            "output_exists", "Advisor output already exists; overwrite is forbidden."
        )

    temporary_name = f".mymoe-advisor-{secrets.token_hex(16)}.tmp"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    published = False
    staged_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, value)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise AdvisorCliError(
                "output_invalid", "Advisor output staging file is unsafe."
            )
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise AdvisorCliError(
                "output_permissions_invalid",
                "Advisor output staging permissions are unsafe.",
            )
        staged_identity = (opened.st_dev, opened.st_ino)
        _require_directory_identity(canonical_parent, parent_identity)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as exc:
            raise AdvisorCliError(
                "output_exists", "Advisor output already exists; overwrite is forbidden."
            ) from exc
        published_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (published_stat.st_dev, published_stat.st_ino) != staged_identity:
            raise AdvisorCliError(
                "output_verification_failed", "Advisor output identity is invalid."
            )
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = ""
        os.fsync(parent_descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        verified = _read_exact_descriptor(descriptor, maximum_bytes=max(len(value), 1))
        if verified != value:
            raise AdvisorCliError(
                "output_verification_failed", "Advisor output verification failed."
            )
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1:
            raise AdvisorCliError(
                "output_permissions_invalid", "Advisor output permissions are unsafe."
            )
        _require_directory_identity(canonical_parent, parent_identity)
    except AdvisorCliError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AdvisorCliError(
            "output_publish_failed", "Advisor output could not be published safely."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            _unlink_matching_at(
                parent_descriptor,
                temporary_name,
                expected=staged_identity,
            )
        for item in reversed(descriptors):
            try:
                os.close(item)
            except OSError:
                pass
        if published:
            # A published receipt is never removed after an uncertain error.
            # The directory-relative identity checks ensure cleanup cannot
            # follow a replacement pathname.
            pass


def _write_output_windows(path: Path, value: bytes) -> None:
    from . import _win32_fs

    parent = path.parent.absolute()
    path = parent / path.name
    pinned: list[tuple[int, object]] = []
    target_descriptor = -1
    staged_identity = None
    try:
        parent_before = parent.lstat()
        if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(
            parent_before.st_mode
        ):
            raise AdvisorCliError(
                "output_parent_invalid",
                "Advisor output parent must be a real existing directory.",
            )
        current = Path(parent.absolute().anchor)
        descriptor, identity = _win32_fs.open_nofollow_fd(
            current,
            directory=True,
            writable=False,
            share_delete=False,
        )
        pinned.append((descriptor, identity))
        for component in parent.absolute().parts[1:]:
            if component:
                current = current / component
            descriptor, identity = _win32_fs.open_nofollow_fd(
                current,
                directory=True,
                writable=False,
                share_delete=False,
            )
            pinned.append((descriptor, identity))
        if os.path.lexists(path):
            raise AdvisorCliError(
                "output_exists", "Advisor output already exists; overwrite is forbidden."
            )
        temporary = parent / f".mymoe-advisor-{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            _write_all(descriptor, value)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise AdvisorCliError(
                    "output_invalid", "Advisor output staging file is unsafe."
                )
            staged_identity = _win32_fs.identity_from_fd(descriptor)
            if staged_identity.is_reparse_point:
                raise AdvisorCliError(
                    "output_invalid", "Advisor output staging file is unsafe."
                )
        finally:
            os.close(descriptor)
        try:
            _win32_fs.move_no_replace(temporary, path)
        except FileExistsError as exc:
            raise AdvisorCliError(
                "output_exists", "Advisor output already exists; overwrite is forbidden."
            ) from exc
        target_descriptor, target_identity = _win32_fs.open_nofollow_fd(
            path,
            directory=False,
            writable=False,
            share_delete=False,
        )
        if staged_identity is None or not staged_identity.same_file_as(target_identity):
            raise AdvisorCliError(
                "output_verification_failed",
                "Advisor output identity changed before publication.",
            )
        observed = os.fstat(target_descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise AdvisorCliError(
                "output_verification_failed", "Advisor output identity is invalid."
            )
        verified = _read_exact_descriptor(
            target_descriptor,
            maximum_bytes=max(len(value), 1),
        )
        if verified != value:
            raise AdvisorCliError(
                "output_verification_failed", "Advisor output verification failed."
            )
    except AdvisorCliError:
        raise
    except (CellContractError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AdvisorCliError(
            "output_publish_failed", "Advisor output could not be published safely."
        ) from exc
    finally:
        if target_descriptor >= 0:
            try:
                os.close(target_descriptor)
            except OSError:
                pass
        try:
            temporary.unlink()
        except (NameError, FileNotFoundError):
            pass
        except OSError:
            pass
        for descriptor, _ in reversed(pinned):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validate_windows_output_name(name: str) -> None:
    stem = name.rstrip(" .").split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{index}" for index in range(1, 10))
    reserved.update(f"LPT{index}" for index in range(1, 10))
    if (
        not name
        or name.endswith((" ", "."))
        or ":" in name
        or stem in reserved
        or len(name.encode("utf-16-le")) // 2 > 255
    ):
        raise AdvisorCliError(
            "output_invalid", "Advisor output uses an ambiguous Windows filename."
        )


def _open_posix_directory_chain(path: Path) -> list[int]:
    if (
        not path.is_absolute()
        or not path.anchor
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        raise AdvisorCliError(
            "output_parent_invalid", "Secure output directories are unavailable."
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY")
        | getattr(os, "O_NOFOLLOW")
    )
    descriptors: list[int] = []
    try:
        current = os.open(path.anchor, flags)
        descriptors.append(current)
        for component in path.parts[1:]:
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        return descriptors
    except OSError as exc:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise AdvisorCliError(
            "output_parent_invalid", "Advisor output parent could not be pinned."
        ) from exc


def _require_directory_identity(path: Path, expected: os.stat_result) -> None:
    try:
        observed = path.stat()
    except OSError as exc:
        raise AdvisorCliError(
            "output_parent_changed", "Advisor output parent changed during publish."
        ) from exc
    if (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)) != (
        expected.st_dev,
        expected.st_ino,
        stat.S_IFMT(expected.st_mode),
    ):
        raise AdvisorCliError(
            "output_parent_changed", "Advisor output parent changed during publish."
        )


def _read_exact_descriptor(descriptor: int, *, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(64 * 1024, maximum_bytes - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > maximum_bytes:
            raise OSError("output exceeds verification bound")
        chunks.append(chunk)


def _unlink_matching_at(
    parent_descriptor: int,
    name: str,
    *,
    expected: tuple[int, int] | None,
) -> None:
    if expected is None:
        return
    try:
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (observed.st_dev, observed.st_ino) == expected:
            os.unlink(name, dir_fd=parent_descriptor)
    except (FileNotFoundError, OSError):
        return


def _write_all(descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write made no progress")
        remaining = remaining[written:]


def _same_location(left: Path, right: Path) -> bool:
    try:
        left_value = os.path.normcase(os.path.abspath(os.fspath(left)))
        right_value = os.path.normcase(os.path.abspath(os.fspath(right)))
    except (OSError, TypeError, ValueError) as exc:
        raise AdvisorCliError(
            "output_invalid", "Advisor output path could not be validated."
        ) from exc
    if left_value == right_value:
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def _render_error(code: str, message: str) -> None:
    print(
        json.dumps(
            {"error": "advisor_error", "code": code, "message": message},
            sort_keys=True,
        ),
        file=sys.stderr,
    )


__all__ = [
    "AdvisorCliError",
    "MAX_EVALUATION_CONTRACT_BYTES",
    "MAX_TASK_BYTES",
    "build_parser",
    "main",
    "render_human",
]


if __name__ == "__main__":
    raise SystemExit(main())
