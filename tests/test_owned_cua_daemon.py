from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Iterator
import unittest
from unittest.mock import Mock, patch

import local_moe.desktop_capability as desktop
from local_moe.desktop_capability import (
    DesktopCapabilityConfig,
    _OwnedCuaDaemon,
    _desktop_session_policy,
    _desktop_user_policy,
)
from local_moe.extensions import McpServerDefinition
from local_moe.tool_runner import ToolExecutionError


_DAEMON_PID = 8123


class OwnedCuaDaemonTests(unittest.TestCase):
    def test_launches_exact_bounded_contract_and_removes_owned_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = _valid_status()
            with _patched_runtime(Path(tmp), status) as harness:
                daemon = _OwnedCuaDaemon(
                    harness.config,
                    harness.server,
                    harness.base_environment,
                )

                proxy, receipt = daemon.start(timeout_seconds=2.5)
                launch_argv, launch_options = harness.launches[0]
                root = Path(launch_options["cwd"])
                socket_path = root / "daemon.sock"
                user_policy_path = root / "user-policy.yml"
                session_policy_path = root / "session-policy.yml"
                expected_user_policy = _desktop_user_policy(harness.config)
                expected_session_policy = _desktop_session_policy()

                self.assertEqual(
                    launch_argv,
                    [
                        str(harness.binary),
                        "serve",
                        "--embedded",
                        "--socket",
                        str(socket_path),
                        "--permission-mode",
                        "bounded",
                        "--session-policy",
                        str(session_policy_path),
                        "--approve-session-policy",
                        "--no-overlay",
                    ],
                )
                self.assertEqual(launch_options["stdin"], subprocess.DEVNULL)
                self.assertEqual(launch_options["stdout"], subprocess.DEVNULL)
                self.assertEqual(launch_options["stderr"], subprocess.DEVNULL)
                self.assertTrue(launch_options["start_new_session"])
                self.assertEqual(
                    launch_options["env"],
                    {
                        **harness.base_environment,
                        "CUA_DRIVER_EMBEDDED": "1",
                        "CUA_DRIVER_POLICY_FILE": str(user_policy_path),
                    },
                )
                self.assertEqual(user_policy_path.read_bytes(), expected_user_policy)
                self.assertEqual(
                    session_policy_path.read_bytes(),
                    expected_session_policy,
                )
                if os.name == "posix":
                    self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
                    self.assertEqual(
                        stat.S_IMODE(user_policy_path.stat().st_mode),
                        0o600,
                    )
                    self.assertEqual(
                        stat.S_IMODE(session_policy_path.stat().st_mode),
                        0o600,
                    )

                harness.wait_for_status.assert_called_once()
                wait_args = harness.wait_for_status.call_args
                self.assertEqual(
                    wait_args.args,
                    (harness.binary, socket_path, launch_options["env"]),
                )
                self.assertGreater(wait_args.kwargs["timeout_seconds"], 0)
                self.assertLessEqual(wait_args.kwargs["timeout_seconds"], 2.5)
                harness.verify_private_socket.assert_called_once_with(socket_path)
                harness.resolve_process_identity.assert_called_once_with(_DAEMON_PID)
                self.assertEqual(proxy.command, str(harness.binary))
                self.assertEqual(
                    proxy.args,
                    ("mcp", "--embedded", "--socket", str(socket_path)),
                )
                self.assertEqual(proxy.cwd, str(root))
                self.assertEqual(receipt["permission_mode"], "bounded")
                self.assertTrue(receipt["socket_owner_verified"])
                self.assertTrue(receipt["daemon_process_verified"])
                self.assertEqual(receipt["user_policy_sha256"], "d" * 64)
                self.assertEqual(receipt["session_policy_sha256"], "e" * 64)
                self.assertEqual(
                    receipt["user_policy_source_sha256"],
                    hashlib.sha256(expected_user_policy).hexdigest(),
                )
                self.assertEqual(
                    receipt["session_policy_source_sha256"],
                    hashlib.sha256(expected_session_policy).hexdigest(),
                )

                daemon.close()

                self.assertFalse(root.exists())
                self.assertEqual(harness.process.wait_timeouts, [3])
                self.assertEqual(
                    [call[0][1] for call in harness.bounded_calls],
                    ["revoke", "stop"],
                )
                self.assertEqual(
                    harness.bounded_calls[0][0],
                    [
                        str(harness.binary),
                        "revoke",
                        "--all",
                        "--socket",
                        str(socket_path),
                    ],
                )
                self.assertEqual(
                    harness.bounded_calls[1][0],
                    [
                        str(harness.binary),
                        "stop",
                        "--socket",
                        str(socket_path),
                    ],
                )
                self.assertTrue(
                    all(
                        environment == harness.base_environment
                        and timeout_seconds == 2
                        for _, environment, timeout_seconds in harness.bounded_calls
                    )
                )
                daemon.close()
                self.assertEqual(len(harness.bounded_calls), 2)

    def test_invalid_effective_policy_tears_down_process_and_temp_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = {**_valid_status(), "permission mode": "unbounded"}
            with _patched_runtime(Path(tmp), status) as harness:
                daemon = _OwnedCuaDaemon(
                    harness.config,
                    harness.server,
                    harness.base_environment,
                )

                with self.assertRaisesRegex(ToolExecutionError, "policy status"):
                    daemon.start()

                root = Path(harness.launches[0][1]["cwd"])
                self.assertFalse(root.exists())
                self.assertEqual(harness.process.returncode, 0)
                self.assertEqual(
                    [call[0][1] for call in harness.bounded_calls],
                    ["revoke", "stop"],
                )
                daemon.close()
                self.assertEqual(len(harness.bounded_calls), 2)


class _Harness:
    def __init__(self, root: Path, status: dict[str, str]) -> None:
        self.root = root
        self.binary = root / "cua-driver"
        self.binary.write_bytes(b"qualified-cua-driver")
        self.binary.chmod(0o700)
        self.server, self.config = _server_and_config(self.binary)
        self.base_environment = {
            "PATH": str(root),
            "CUA_DRIVER_RS_TELEMETRY_ENABLED": "false",
            "CUA_DRIVER_RS_UPDATE_CHECK": "false",
        }
        self.status = status
        self.process = _FakeProcess()
        self.launches: list[tuple[list[str], dict[str, Any]]] = []
        self.bounded_calls: list[
            tuple[list[str], dict[str, str], float]
        ] = []
        self.wait_for_status: Mock
        self.verify_private_socket: Mock
        self.resolve_process_identity: Mock

    def popen(self, argv: list[str], **options: Any) -> _FakeProcess:
        self.launches.append((list(argv), dict(options)))
        return self.process

    def run_bounded(
        self,
        argv: list[str],
        environment: dict[str, str],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        self.bounded_calls.append(
            (list(argv), dict(environment), timeout_seconds)
        )
        if len(argv) > 1 and argv[1] == "stop":
            self.process.returncode = 0
        return subprocess.CompletedProcess(argv, 0, "", "")


class _FakeProcess:
    pid = _DAEMON_PID

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0


class _OsProxy:
    name = "posix"

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


class _SubprocessProxy:
    DEVNULL = subprocess.DEVNULL
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, harness: _Harness) -> None:
        self._harness = harness

    def Popen(self, argv: list[str], **options: Any) -> _FakeProcess:
        return self._harness.popen(argv, **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(subprocess, name)


class _TempfileProxy:
    def __init__(self, root: Path) -> None:
        self._root = root

    def gettempdir(self) -> str:
        return str(self._root)

    def mkdtemp(self, *, prefix: str, dir: str) -> str:
        del dir
        return tempfile.mkdtemp(prefix=prefix, dir=self._root)


@contextmanager
def _patched_runtime(
    root: Path,
    status: dict[str, str],
) -> Iterator[_Harness]:
    harness = _Harness(root, status)
    process_identity = {
        "pid": _DAEMON_PID,
        "name": "cua-driver",
        "started_at": "1753084800.000000",
        "executable_sha256": harness.config.provider_executable_sha256,
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(desktop, "os", _OsProxy()))
        stack.enter_context(
            patch.object(desktop, "subprocess", _SubprocessProxy(harness))
        )
        stack.enter_context(
            patch.object(desktop, "tempfile", _TempfileProxy(root))
        )
        stack.enter_context(
            patch.object(desktop, "_resolve_executable", return_value=harness.binary)
        )
        harness.verify_private_socket = stack.enter_context(
            patch.object(desktop, "_verify_private_socket")
        )
        harness.resolve_process_identity = stack.enter_context(
            patch.object(
                desktop,
                "_resolve_process_identity",
                return_value=process_identity,
            )
        )
        stack.enter_context(
            patch.object(desktop, "_run_bounded", side_effect=harness.run_bounded)
        )
        harness.wait_for_status = stack.enter_context(
            patch.object(
                _OwnedCuaDaemon,
                "_wait_for_status",
                return_value=dict(status),
            )
        )
        yield harness


def _server_and_config(
    binary: Path,
) -> tuple[McpServerDefinition, DesktopCapabilityConfig]:
    provider_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    server = McpServerDefinition(
        name="desktop-local",
        description="Owned daemon lifecycle fixture",
        command=str(binary),
        args=("mcp",),
        enabled=True,
        risk_class="identity_access",
        capabilities=("desktop", "tools"),
        transport="stdio",
        cwd=".",
        env={},
        timeout_seconds=3,
        allowed_tools=("get_window_state",),
        desktop_capability={
            "enabled": True,
            "provider": "cua_driver",
            "version": "0.10.0",
            "provider_executable_sha256": provider_digest,
            "telemetry_enabled": False,
            "tool_schema_sha256": {"get_window_state": "c" * 64},
            "target": {
                "id": "offline-editor",
                "pid": 4242,
                "window_id": 17,
                "process_name": "Offline Editor",
                "process_started_at": "1753084800.000000",
                "process_executable_sha256": "b" * 64,
            },
        },
    )
    return server, DesktopCapabilityConfig.from_server(server)


def _valid_status() -> dict[str, str]:
    return {
        "pid": str(_DAEMON_PID),
        "permission mode": "bounded (trusted_startup_configuration)",
        "user policy": "configured=true, active=true, valid=true",
        "user policy sha256": "d" * 64,
        "managed policy": "configured=false, active=false, valid=true",
        "session policy": "configured=true, approved_at_startup=true, valid=true",
        "session policy sha256": "e" * 64,
    }


if __name__ == "__main__":
    unittest.main()
