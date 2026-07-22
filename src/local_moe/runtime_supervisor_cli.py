"""Foreground CLI for one explicitly confirmed process-bound runtime session.

The production session factory is imported only after argument validation and
confirmation.  This keeps help and validation usable without the optional
runtime-supervisor dependency and gives tests a narrow injection seam.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
from typing import Callable, ContextManager, Mapping, Protocol, Sequence


EXIT_OK = 0
EXIT_INVALID = 2
EXIT_LIFECYCLE_FAILED = 3
EXIT_CLEANUP_UNVERIFIED = 4

_PUBLIC_CODE = re.compile(r"\A[a-z][a-z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_PUBLIC_CORE_CODES = frozenset(
    {
        "already_started",
        "binding_changed",
        "binding_not_verified",
        "cleanup_unverified",
        "endpoint_ambiguous",
        "endpoint_in_use",
        "endpoint_owner_mismatch",
        "endpoint_ownership_lost",
        "health_probe_failed",
        "process_exited",
        "process_identity_changed",
        "process_observation_failed",
        "runtime_dependency_unavailable",
        "runtime_lease_conflict",
        "runtime_profile_unsupported",
    }
)
_FINAL_LIFECYCLE_FAILURE_CODES = frozenset({"binding_changed"})


class RuntimeSupervisorSession(Protocol):
    @property
    def cleanup_unknown(self) -> bool: ...

    def start(self) -> object: ...

    def inspect(self) -> object: ...

    def stop(self) -> object: ...


class RuntimeSupervisorSessionFactory(Protocol):
    def __call__(self, request_path: Path) -> RuntimeSupervisorSession: ...


WaitForStop = Callable[[threading.Event, float], bool]
SignalInstaller = Callable[["_StopRequest"], ContextManager[None]]


class RuntimeSupervisorCliError(RuntimeError):
    """Stable public CLI failure without private runtime details."""

    def __init__(self, code: str, message: str, *, exit_code: int) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(message)


class _RuntimeSupervisorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise RuntimeSupervisorCliError(
            "invocation_invalid",
            "Invalid runtime supervisor invocation.",
            exit_code=EXIT_INVALID,
        )


@dataclass
class _StopRequest:
    event: threading.Event
    signal_name: str | None = None


@dataclass(frozen=True)
class _LifecycleResult:
    started: object
    inspected: object
    stopped: object
    stop_reason: str


def build_parser() -> argparse.ArgumentParser:
    boundary = (
        "Foreground ownership only: explicit confirmation is required; the CLI "
        "never daemonizes or attaches; teardown targets the process group it launched."
    )
    parser = _RuntimeSupervisorArgumentParser(
        prog="mymoe-runtime",
        description="Run one verified process-bound local runtime session.",
        epilog=boundary,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit stable sanitized JSON containing no paths, argv, or raw process data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser(
        "check",
        help="Start, validate once, and stop one verified runtime.",
        description="Start, validate once, and stop one verified runtime.",
        epilog=boundary,
        allow_abbrev=False,
    )
    _add_common_arguments(check)

    supervise = commands.add_parser(
        "supervise",
        help="Own and validate one runtime in the foreground until SIGINT or SIGTERM.",
        description=(
            "Own and periodically validate one runtime in the foreground until "
            "SIGINT or SIGTERM requests verified cleanup."
        ),
        epilog=boundary,
        allow_abbrev=False,
    )
    _add_common_arguments(supervise)
    supervise.add_argument(
        "--inspection-interval-seconds",
        type=_inspection_interval,
        default=2.0,
        metavar="SECONDS",
        help="Seconds between continuity inspections (0.05 to 60; default: 2).",
    )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--binding-request",
        required=True,
        metavar="PATH",
        help="Verified CellBindingInspectRequest JSON used to resolve the exact launch.",
    )
    parser.add_argument(
        "--confirm",
        required=True,
        action="store_true",
        help="Explicitly authorize start and verified cleanup for this invocation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        default=argparse.SUPPRESS,
        help="Emit stable sanitized JSON containing no paths, argv, or raw process data.",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: RuntimeSupervisorSessionFactory | None = None,
    wait_for_stop: WaitForStop | None = None,
    signal_installer: SignalInstaller | None = None,
) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in raw
    command = _command(raw)
    try:
        args = build_parser().parse_args(raw)
        command = str(args.command)
        json_output = bool(args.json_output)
        factory = session_factory or _load_default_session_factory()
        request_path = _absolute_path(args.binding_request)
        if command == "check":
            lifecycle = _run_check(
                factory,
                request_path=request_path,
            )
        elif command == "supervise":
            lifecycle = _run_supervise(
                factory,
                request_path=request_path,
                inspection_interval_seconds=args.inspection_interval_seconds,
                wait_for_stop=wait_for_stop or _default_wait_for_stop,
                signal_installer=signal_installer or _install_stop_signals,
            )
        else:  # pragma: no cover - argparse enforces the closed command set.
            raise RuntimeSupervisorCliError(
                "invocation_invalid",
                "Invalid runtime supervisor invocation.",
                exit_code=EXIT_INVALID,
            )
        payload = _success_payload(command, lifecycle)
    except RuntimeSupervisorCliError as exc:
        _emit_error(
            command,
            code=exc.code,
            message=str(exc),
            json_output=json_output,
        )
        return exc.exit_code
    except KeyboardInterrupt:
        _emit_error(
            command,
            code="interrupted",
            message="Runtime supervision was interrupted before verified cleanup.",
            json_output=json_output,
        )
        return EXIT_LIFECYCLE_FAILED
    except Exception as exc:
        public = _public_lifecycle_error(exc)
        _emit_error(
            command,
            code=public.code,
            message=str(public),
            json_output=json_output,
        )
        return public.exit_code

    _emit_success(payload, json_output=json_output)
    return EXIT_OK


def _run_check(
    factory: RuntimeSupervisorSessionFactory,
    *,
    request_path: Path,
) -> _LifecycleResult:
    session = _create_session(factory, request_path)
    started: object = None
    inspected: object = None
    primary_error: BaseException | None = None
    try:
        started = session.start()
        inspected = session.inspect()
    except BaseException as exc:
        primary_error = exc
    stopped, cleanup_error = _stop_session(session)
    _require_verified_cleanup(session, stopped, cleanup_error)
    if primary_error is not None:
        raise _public_lifecycle_error(primary_error)
    _require_valid_final_session(stopped)
    return _LifecycleResult(started, inspected, stopped, "check_complete")


def _run_supervise(
    factory: RuntimeSupervisorSessionFactory,
    *,
    request_path: Path,
    inspection_interval_seconds: float,
    wait_for_stop: WaitForStop,
    signal_installer: SignalInstaller,
) -> _LifecycleResult:
    session = _create_session(factory, request_path)
    stop_request = _StopRequest(threading.Event())
    started: object = None
    inspected: object = None
    primary_error: BaseException | None = None
    try:
        with signal_installer(stop_request):
            started = session.start()
            inspected = session.inspect()
            while not wait_for_stop(
                stop_request.event,
                inspection_interval_seconds,
            ):
                inspected = session.inspect()
    except BaseException as exc:
        primary_error = exc
    stopped, cleanup_error = _stop_session(session)
    _require_verified_cleanup(session, stopped, cleanup_error)
    if primary_error is not None:
        raise _public_lifecycle_error(primary_error)
    _require_valid_final_session(stopped)
    stop_reason = "signal" if stop_request.signal_name is not None else "requested"
    return _LifecycleResult(started, inspected, stopped, stop_reason)


def _create_session(
    factory: RuntimeSupervisorSessionFactory,
    request_path: Path,
) -> RuntimeSupervisorSession:
    try:
        return factory(request_path)
    except RuntimeSupervisorCliError:
        raise
    except Exception as exc:
        raise _public_lifecycle_error(exc) from exc


def _stop_session(
    session: RuntimeSupervisorSession,
) -> tuple[object, BaseException | None]:
    try:
        return session.stop(), None
    except BaseException as exc:
        return None, exc


def _require_verified_cleanup(
    session: RuntimeSupervisorSession,
    stopped: object,
    cleanup_error: BaseException | None,
) -> None:
    if cleanup_error is not None:
        raise RuntimeSupervisorCliError(
            "cleanup_unverified",
            "Runtime cleanup could not be verified.",
            exit_code=EXIT_CLEANUP_UNVERIFIED,
        ) from cleanup_error
    try:
        cleanup_unknown = session.cleanup_unknown
    except Exception as exc:
        raise RuntimeSupervisorCliError(
            "cleanup_unverified",
            "Runtime cleanup could not be verified.",
            exit_code=EXIT_CLEANUP_UNVERIFIED,
        ) from exc
    if cleanup_unknown is not False or _final_state(stopped) != "stopped":
        raise RuntimeSupervisorCliError(
            "cleanup_unverified",
            "Runtime cleanup could not be verified.",
            exit_code=EXIT_CLEANUP_UNVERIFIED,
        )


def _require_valid_final_session(stopped: object) -> None:
    failures = _final_reason_codes(stopped).intersection(
        _FINAL_LIFECYCLE_FAILURE_CODES
    )
    if not failures:
        return
    code = sorted(failures)[0]
    raise RuntimeSupervisorCliError(
        code,
        "Runtime lifecycle validation failed during verified shutdown.",
        exit_code=EXIT_LIFECYCLE_FAILED,
    )


def _final_reason_codes(value: object, *, depth: int = 0) -> frozenset[str]:
    if depth > 4 or value is None:
        return frozenset()
    if isinstance(value, Mapping):
        raw = value.get("reason_codes")
        if isinstance(raw, (tuple, list)):
            return frozenset(code for code in raw if isinstance(code, str))
        for key in ("receipt", "final_receipt"):
            if key in value:
                nested = _final_reason_codes(value[key], depth=depth + 1)
                if nested:
                    return nested
        return frozenset()
    raw = getattr(value, "reason_codes", None)
    if isinstance(raw, (tuple, list)):
        return frozenset(code for code in raw if isinstance(code, str))
    for name in ("receipt", "final_receipt"):
        nested = _final_reason_codes(getattr(value, name, None), depth=depth + 1)
        if nested:
            return nested
    if isinstance(value, (tuple, list)):
        for item in reversed(value):
            nested = _final_reason_codes(item, depth=depth + 1)
            if nested:
                return nested
    return frozenset()


def _final_state(value: object, *, depth: int = 0) -> str | None:
    if depth > 4 or value is None:
        return None
    if isinstance(value, Mapping):
        state = value.get("state")
        if isinstance(state, str):
            return state
        for key in ("receipt", "final_receipt"):
            if key in value:
                nested = _final_state(value[key], depth=depth + 1)
                if nested is not None:
                    return nested
        return None
    state = getattr(value, "state", None)
    if isinstance(state, str):
        return state
    for name in ("receipt", "final_receipt"):
        nested_value = getattr(value, name, None)
        nested = _final_state(nested_value, depth=depth + 1)
        if nested is not None:
            return nested
    if isinstance(value, (tuple, list)):
        for item in reversed(value):
            nested = _final_state(item, depth=depth + 1)
            if nested is not None:
                return nested
    return None


def _load_default_session_factory() -> RuntimeSupervisorSessionFactory:
    try:
        from .runtime_supervisor import create_runtime_supervisor_session
    except (AttributeError, ImportError) as exc:
        raise RuntimeSupervisorCliError(
            "runtime_dependency_unavailable",
            "Runtime supervisor dependencies are unavailable.",
            exit_code=EXIT_LIFECYCLE_FAILED,
        ) from exc
    if not callable(create_runtime_supervisor_session):
        raise RuntimeSupervisorCliError(
            "runtime_dependency_unavailable",
            "Runtime supervisor dependencies are unavailable.",
            exit_code=EXIT_LIFECYCLE_FAILED,
        )
    return create_runtime_supervisor_session


@contextmanager
def _install_stop_signals(stop_request: _StopRequest):
    previous: dict[signal.Signals, object] = {}

    def request_stop(signum: int, _frame: object) -> None:
        try:
            stop_request.signal_name = signal.Signals(signum).name
        except ValueError:
            stop_request.signal_name = "SIGNAL"
        stop_request.event.set()

    signals = tuple(
        item
        for item in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None))
        if item is not None
    )
    try:
        for item in signals:
            previous[item] = signal.getsignal(item)
            signal.signal(item, request_stop)
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def _default_wait_for_stop(event: threading.Event, timeout: float) -> bool:
    return event.wait(timeout)


def _inspection_interval(value: str) -> float:
    try:
        rendered = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid inspection interval") from exc
    if not 0.05 <= rendered <= 60 or rendered != rendered:
        raise argparse.ArgumentTypeError("invalid inspection interval")
    return rendered


def _absolute_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeSupervisorCliError(
            "invocation_invalid",
            "Invalid runtime supervisor invocation.",
            exit_code=EXIT_INVALID,
        )
    return Path(os.path.abspath(os.path.expanduser(value)))


def _public_lifecycle_error(exc: BaseException) -> RuntimeSupervisorCliError:
    if isinstance(exc, RuntimeSupervisorCliError):
        return exc
    if isinstance(exc, KeyboardInterrupt):
        return RuntimeSupervisorCliError(
            "interrupted",
            "Runtime supervision was interrupted.",
            exit_code=EXIT_LIFECYCLE_FAILED,
        )
    raw_code = getattr(exc, "code", "")
    code = str(raw_code) if isinstance(raw_code, str) else ""
    if code not in _PUBLIC_CORE_CODES or _PUBLIC_CODE.fullmatch(code) is None:
        code = "lifecycle_failed"
    if code == "cleanup_unverified":
        return RuntimeSupervisorCliError(
            code,
            "Runtime cleanup could not be verified.",
            exit_code=EXIT_CLEANUP_UNVERIFIED,
        )
    if code == "runtime_dependency_unavailable":
        message = "Runtime supervisor dependencies are unavailable."
    else:
        message = "Runtime lifecycle operation failed safely."
    return RuntimeSupervisorCliError(
        code,
        message,
        exit_code=EXIT_LIFECYCLE_FAILED,
    )


def _success_payload(
    command: str,
    lifecycle: _LifecycleResult,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "command": command,
        "status": "completed" if command == "check" else "stopped",
        "confirmation": "explicit",
        "foreground": True,
        "daemonized": False,
        "attach_allowed": False,
        "process_target_policy": "owned_process_group_only",
        "cleanup_verified": True,
        "stop_reason": lifecycle.stop_reason,
        "start": _public_result(lifecycle.started),
        "inspection": _public_result(lifecycle.inspected),
        "stop": _public_result(lifecycle.stopped),
    }


def _public_result(value: object, *, depth: int = 0) -> object:
    """Return only validated status metadata and content digests."""

    if depth > 4 or value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [_public_result(item, depth=depth + 1) for item in value]

    result: dict[str, object] = {}
    getter = value.get if isinstance(value, Mapping) else lambda name: getattr(value, name, None)
    for name in ("state", "status"):
        candidate = getter(name)
        if isinstance(candidate, str) and _PUBLIC_CODE.fullmatch(candidate):
            result[name] = candidate
    reasons = getter("reason_codes")
    if isinstance(reasons, (tuple, list)) and all(
        isinstance(item, str) and _PUBLIC_CODE.fullmatch(item) for item in reasons
    ):
        result["reason_codes"] = sorted(set(reasons))
    digest = getter("digest")
    if isinstance(digest, str) and _SHA256.fullmatch(digest):
        result["digest"] = digest
    for name in ("receipt", "evidence", "final_receipt"):
        nested = getter(name)
        if nested is not None and nested is not value:
            result[name] = _public_result(nested, depth=depth + 1)
    if not result:
        stable = hashlib.sha256(type(value).__name__.encode("utf-8")).hexdigest()
        return {"observed": True, "type_sha256": stable}
    return result


def _emit_success(payload: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return
    print(
        f"mymoe-runtime {payload['command']}: {payload['status']} "
        "(cleanup verified)"
    )


def _emit_error(
    command: str,
    *,
    code: str,
    message: str,
    json_output: bool,
) -> None:
    payload = {
        "schema_version": "1.0",
        "command": command,
        "status": "error",
        "error": {"code": code, "message": message},
        "foreground": True,
        "daemonized": False,
        "attach_allowed": False,
        "process_target_policy": "owned_process_group_only",
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=sys.stderr)
    else:
        print(f"mymoe-runtime {command}: {code}", file=sys.stderr)


def _command(argv: Sequence[str]) -> str:
    return next((item for item in argv if item in {"check", "supervise"}), "unknown")


if __name__ == "__main__":
    raise SystemExit(main())
