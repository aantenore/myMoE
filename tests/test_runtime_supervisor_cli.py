from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
from io import StringIO
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from local_moe import runtime_supervisor_cli as cli


PRIVATE_PATH = "/private/runtime/secret-model.gguf"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Receipt:
    state: str
    digest: str
    reason_codes: tuple[str, ...] = ()
    private_path: str = PRIVATE_PATH
    runtime_pid: int = 4242


@dataclass(frozen=True)
class _Evidence:
    digest: str
    private_path: str = PRIVATE_PATH
    argv: tuple[str, ...] = ("/private/runtime", "--secret")


@dataclass(frozen=True)
class _StartResult:
    evidence: _Evidence
    receipt: _Receipt
    private_path: str = PRIVATE_PATH


class _CoreError(RuntimeError):
    def __init__(self, code: str, detail: str = PRIVATE_PATH) -> None:
        self.code = code
        super().__init__(detail)


class _Session:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cleanup_unknown = False
        self.start_error: BaseException | None = None
        self.inspect_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.stop_state = "stopped"
        self.stop_reason_codes: tuple[str, ...] = ()

    def start(self) -> object:
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error
        return _StartResult(
            _Evidence(_sha("start-evidence")),
            _Receipt("ready", _sha("ready-receipt")),
        )

    def inspect(self) -> object:
        self.calls.append("inspect")
        if self.inspect_error is not None:
            raise self.inspect_error
        return _Evidence(_sha(f"inspection-{self.calls.count('inspect')}"))

    def stop(self) -> object:
        self.calls.append("stop")
        if self.stop_error is not None:
            raise self.stop_error
        return _Receipt(
            self.stop_state,
            _sha(f"stop-{self.stop_state}"),
            (
                ("cleanup_unverified",)
                if self.stop_state == "unknown_blocking"
                else self.stop_reason_codes
            ),
        )


class _Factory:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.calls: list[Path] = []

    def __call__(self, request_path: Path) -> _Session:
        self.calls.append(request_path)
        return self.session


def _invoke(
    argv: list[str],
    *,
    factory: _Factory | None = None,
    wait_for_stop=None,
    signal_installer=None,
) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = cli.main(
            argv,
            session_factory=factory,
            wait_for_stop=wait_for_stop,
            signal_installer=signal_installer,
        )
    return status, stdout.getvalue(), stderr.getvalue()


class RuntimeSupervisorCliTests(unittest.TestCase):
    def test_help_exposes_only_check_and_foreground_supervise(self) -> None:
        parser = cli.build_parser()
        help_text = parser.format_help()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, cli.argparse._SubParsersAction)
        )

        self.assertEqual(set(subparsers.choices), {"check", "supervise"})
        self.assertIn("explicit confirmation", help_text)
        self.assertIn("never daemonizes", help_text)
        self.assertIn("teardown targets the process group it launched", help_text)
        for name in ("check", "supervise"):
            command_help = subparsers.choices[name].format_help()
            self.assertIn("--binding-request PATH", command_help)
            self.assertNotIn("--state-directory", command_help)
            self.assertIn("--confirm", command_help)

    def test_missing_confirmation_is_rejected_before_lazy_core_import(self) -> None:
        with patch.object(cli, "_load_default_session_factory") as loader:
            status, stdout, stderr = _invoke(
                ["--json", "check", "--binding-request", PRIVATE_PATH]
            )

        self.assertEqual(status, cli.EXIT_INVALID)
        self.assertEqual(stdout, "")
        self.assertIn('"code": "invocation_invalid"', stderr)
        self.assertNotIn(PRIVATE_PATH, stderr)
        loader.assert_not_called()

    def test_check_runs_start_inspect_stop_and_emits_only_sanitized_metadata(self) -> None:
        session = _Session()
        factory = _Factory(session)
        status, stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=factory,
        )

        self.assertEqual(status, cli.EXIT_OK)
        self.assertEqual(stderr, "")
        self.assertEqual(session.calls, ["start", "inspect", "stop"])
        self.assertEqual(factory.calls[0], Path(PRIVATE_PATH))
        self.assertIn('"cleanup_verified": true', stdout)
        self.assertIn('"state": "ready"', stdout)
        self.assertIn('"state": "stopped"', stdout)
        for private in (PRIVATE_PATH, "--secret", "4242"):
            self.assertNotIn(private, stdout)

    def test_start_failure_still_stops_and_sanitizes_the_core_error(self) -> None:
        session = _Session()
        session.start_error = _CoreError("endpoint_in_use")

        status, stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=_Factory(session),
        )

        self.assertEqual(status, cli.EXIT_LIFECYCLE_FAILED)
        self.assertEqual(stdout, "")
        self.assertEqual(session.calls, ["start", "stop"])
        self.assertIn('"code": "endpoint_in_use"', stderr)
        self.assertNotIn(PRIVATE_PATH, stderr)

    def test_inspection_failure_still_performs_verified_stop(self) -> None:
        session = _Session()
        session.inspect_error = _CoreError("process_identity_changed")

        status, _stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=_Factory(session),
        )

        self.assertEqual(status, cli.EXIT_LIFECYCLE_FAILED)
        self.assertEqual(session.calls, ["start", "inspect", "stop"])
        self.assertIn('"code": "process_identity_changed"', stderr)

    def test_binding_drift_during_verified_stop_is_a_lifecycle_failure(self) -> None:
        session = _Session()
        session.stop_reason_codes = ("binding_changed",)

        status, stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=_Factory(session),
        )

        self.assertEqual(status, cli.EXIT_LIFECYCLE_FAILED)
        self.assertEqual(stdout, "")
        self.assertFalse(session.cleanup_unknown)
        self.assertEqual(session.calls, ["start", "inspect", "stop"])
        self.assertIn('"code": "binding_changed"', stderr)
        self.assertNotIn('"cleanup_verified": true', stderr)

    def test_cleanup_error_overrides_primary_failure_and_is_never_reported_safe(self) -> None:
        session = _Session()
        session.inspect_error = _CoreError("process_exited")
        session.stop_error = _CoreError("cleanup_unverified")
        session.cleanup_unknown = True

        status, stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=_Factory(session),
        )

        self.assertEqual(status, cli.EXIT_CLEANUP_UNVERIFIED)
        self.assertEqual(stdout, "")
        self.assertIn('"code": "cleanup_unverified"', stderr)
        self.assertNotIn('"cleanup_verified": true', stderr)

    def test_unknown_blocking_final_receipt_is_cleanup_failure(self) -> None:
        session = _Session()
        session.stop_state = "unknown_blocking"
        session.cleanup_unknown = True

        status, _stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=_Factory(session),
        )

        self.assertEqual(status, cli.EXIT_CLEANUP_UNVERIFIED)
        self.assertIn('"code": "cleanup_unverified"', stderr)

    def test_supervise_is_foreground_and_inspects_until_stop_is_requested(self) -> None:
        session = _Session()
        waits: list[float] = []
        entered: list[str] = []

        @contextmanager
        def signal_guard(_request):
            entered.append("enter")
            try:
                yield
            finally:
                entered.append("exit")

        def waiter(_event, timeout: float) -> bool:
            waits.append(timeout)
            return len(waits) == 2

        status, stdout, stderr = _invoke(
            [
                "--json",
                "supervise",
                "--binding-request",
                PRIVATE_PATH,
                "--inspection-interval-seconds",
                "0.25",
                "--confirm",
            ],
            factory=_Factory(session),
            wait_for_stop=waiter,
            signal_installer=signal_guard,
        )

        self.assertEqual(status, cli.EXIT_OK)
        self.assertEqual(stderr, "")
        self.assertEqual(
            session.calls,
            ["start", "inspect", "inspect", "stop"],
        )
        self.assertEqual(waits, [0.25, 0.25])
        self.assertEqual(entered, ["enter", "exit"])
        self.assertIn('"foreground": true', stdout)
        self.assertIn('"daemonized": false', stdout)
        self.assertIn(
            '"process_target_policy": "owned_process_group_only"', stdout
        )

    def test_signal_guard_handles_sigint_and_restores_both_handlers(self) -> None:
        stop_request = cli._StopRequest(cli.threading.Event())
        installed: dict[object, object] = {}
        prior = object()

        def install(signum, handler):
            installed[signum] = handler

        with (
            patch.object(cli.signal, "getsignal", return_value=prior),
            patch.object(cli.signal, "signal", side_effect=install) as signal_call,
        ):
            with cli._install_stop_signals(stop_request):
                handler = installed[signal.SIGINT]
                handler(signal.SIGINT, None)  # type: ignore[operator]

        self.assertTrue(stop_request.event.is_set())
        self.assertEqual(stop_request.signal_name, "SIGINT")
        restored = [
            call.args
            for call in signal_call.call_args_list
            if len(call.args) == 2 and call.args[1] is prior
        ]
        self.assertEqual(
            {item[0] for item in restored},
            {signal.SIGINT, signal.SIGTERM},
        )

    def test_invalid_interval_is_rejected_without_creating_a_session(self) -> None:
        session = _Session()
        factory = _Factory(session)

        status, _stdout, stderr = _invoke(
            [
                "--json",
                "supervise",
                "--binding-request",
                PRIVATE_PATH,
                "--inspection-interval-seconds",
                "0",
                "--confirm",
            ],
            factory=factory,
        )

        self.assertEqual(status, cli.EXIT_INVALID)
        self.assertEqual(factory.calls, [])
        self.assertNotIn(PRIVATE_PATH, stderr)

    def test_unknown_exception_type_and_message_are_not_disclosed(self) -> None:
        session = _Session()
        session.start_error = ValueError(f"secret failure at {PRIVATE_PATH}")

        status, _stdout, stderr = _invoke(
            [
                "--json",
                "check",
                "--binding-request",
                PRIVATE_PATH,
                "--confirm",
            ],
            factory=_Factory(session),
        )

        self.assertEqual(status, cli.EXIT_LIFECYCLE_FAILED)
        self.assertIn('"code": "lifecycle_failed"', stderr)
        self.assertNotIn("ValueError", stderr)
        self.assertNotIn(PRIVATE_PATH, stderr)


if __name__ == "__main__":
    unittest.main()
