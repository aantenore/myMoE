from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import local_moe.assistant_bridge_runtime as bridge_runtime
from local_moe.assistant_bridge_runtime import (
    ExecutableChangedError,
    ProcessCleanupError,
    ProcessExecutionPolicy,
    ProcessTreeUnavailableError,
    execute_process,
    fingerprint_environment,
    inspect_executable,
    resolve_executable,
    runtime_capabilities,
)


STRICT_EXECUTION_AVAILABLE = runtime_capabilities().strict_tree_supported


class AssistantBridgeRuntimeIdentityTests(unittest.TestCase):
    def test_environment_fingerprint_is_order_independent_and_value_sensitive(
        self,
    ) -> None:
        first = fingerprint_environment({"B": "two", "A": "one"})
        reordered = fingerprint_environment({"A": "one", "B": "two"})
        changed = fingerprint_environment({"A": "one", "B": "three"})

        self.assertEqual(first.sha256, reordered.sha256)
        self.assertNotEqual(first.sha256, changed.sha256)
        self.assertEqual(first.variable_count, 2)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_resolved_absolute_executable_survives_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "bridge-tool").symlink_to("/bin/echo")
            (second / "bridge-tool").symlink_to("/usr/bin/false")
            original_env = {"PATH": str(first)}
            identity = resolve_executable("bridge-tool", env=original_env)

            swapped_env = {"PATH": str(second)}
            result = execute_process(
                identity,
                ("first",),
                env=swapped_env,
                timeout_seconds=2.0,
            )

        self.assertEqual(Path(identity.resolved_path), Path("/bin/echo").resolve())
        self.assertEqual(result.code, "completed")
        self.assertEqual(result.stdout, b"first\n")
        self.assertEqual(
            result.environment.sha256,
            fingerprint_environment(swapped_env).sha256,
        )

    @unittest.skipUnless(
        STRICT_EXECUTION_AVAILABLE, "strict process-tree control unavailable"
    )
    def test_inspection_binds_hash_and_bounded_version_metadata(self) -> None:
        identity = inspect_executable(
            sys.executable,
            version_args=("--version",),
            version_timeout_seconds=3.0,
        )

        self.assertTrue(Path(identity.resolved_path).is_absolute())
        self.assertEqual(len(identity.sha256), 64)
        self.assertGreater(identity.size_bytes, 0)
        self.assertIsNotNone(identity.version)
        assert identity.version is not None
        self.assertEqual(identity.version.status, "completed")
        self.assertIn("Python", identity.version.text)
        self.assertEqual(len(identity.version.output_sha256), 64)

    def test_missing_psutil_contract_is_explicit(self) -> None:
        with patch.object(bridge_runtime, "_psutil", None):
            capabilities = runtime_capabilities()
            self.assertFalse(capabilities.psutil_available)
            self.assertIn("psutil", capabilities.detached_descendant_contract)
            if os.name == "nt":
                identity = resolve_executable(sys.executable)
                with self.assertRaises(ProcessTreeUnavailableError):
                    execute_process(identity, timeout_seconds=1.0)

    def test_optional_psutil_capability_is_observable_when_present(self) -> None:
        with patch.object(bridge_runtime, "_psutil", object()):
            capabilities = runtime_capabilities()

        self.assertTrue(capabilities.psutil_available)
        self.assertIn("recursively observes", capabilities.detached_descendant_contract)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_changed_executable_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "bridge-tool"
            self._write_executable(executable, "first")
            identity = resolve_executable(executable)
            self._write_executable(executable, "second")

            with self.assertRaises(ExecutableChangedError):
                execute_process(identity, timeout_seconds=1.0)

    @staticmethod
    def _write_executable(path: Path, marker: str) -> None:
        path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{marker}'\n", encoding="utf-8")
        path.chmod(0o755)


@unittest.skipUnless(
    STRICT_EXECUTION_AVAILABLE, "strict process-tree control unavailable"
)
class AssistantBridgeRuntimeProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python = resolve_executable(sys.executable)

    def test_deadline_includes_blocked_stdin_write(self) -> None:
        started = time.monotonic()
        result = execute_process(
            self.python,
            ("-c", "import time; time.sleep(30)"),
            stdin=b"x" * (4 * 1024 * 1024),
            timeout_seconds=0.20,
            policy=ProcessExecutionPolicy(
                stdin_limit_bytes=8 * 1024 * 1024,
                cleanup_grace_seconds=0.10,
                cleanup_kill_seconds=0.50,
            ),
        )
        elapsed = time.monotonic() - started

        self.assertEqual(result.code, "timed_out")
        self.assertTrue(result.timed_out)
        self.assertLess(elapsed, 1.50)
        self.assertLess(result.execution_duration_ms, 750)
        self.assertTrue(result.cleanup.verified)
        self.assertTrue(result.cleanup.pipe_threads_joined)

    @unittest.skipIf(os.name == "nt", "POSIX process-group liveness assertion")
    def test_child_and_grandchild_holding_pipes_are_reaped(self) -> None:
        grandchild_code = "import time; time.sleep(30)"
        child_code = (
            "import subprocess,sys,time;"
            f"p=subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]);"
            "print(p.pid,flush=True);time.sleep(30)"
        )
        parent_code = (
            "import subprocess,sys,time;"
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],"
            "stdout=sys.stdout,stderr=sys.stderr);"
            "print(p.pid,flush=True);time.sleep(0.15)"
        )

        result = execute_process(
            self.python,
            ("-c", parent_code),
            timeout_seconds=2.0,
            policy=ProcessExecutionPolicy(
                pipe_settle_seconds=0.05,
                cleanup_grace_seconds=0.10,
                cleanup_kill_seconds=0.75,
            ),
        )

        pids = [int(line) for line in result.stdout.decode().splitlines()]
        self.assertEqual(len(pids), 2)
        self.assertEqual(result.code, "completed")
        self.assertTrue(result.cleanup.verified)
        self.assertTrue(result.cleanup.pipe_threads_joined)
        self.assertTrue(result.cleanup.process_group_verified)
        if runtime_capabilities().psutil_available:
            self.assertTrue(result.cleanup.psutil_verified)
            self.assertGreaterEqual(result.cleanup.observed_descendants, 2)
        for pid in pids:
            self.assertFalse(self._pid_alive(pid), f"process {pid} survived cleanup")

    def test_stdout_overflow_is_bounded_and_terminates_tree(self) -> None:
        limit = 4096
        result = execute_process(
            self.python,
            (
                "-c",
                "import sys,time;sys.stdout.buffer.write(b'x'*65536);sys.stdout.flush();time.sleep(30)",
            ),
            timeout_seconds=2.0,
            policy=ProcessExecutionPolicy(
                stdout_limit_bytes=limit,
                stderr_limit_bytes=limit,
                cleanup_grace_seconds=0.10,
                cleanup_kill_seconds=0.50,
            ),
        )

        self.assertEqual(result.code, "stdout_limit_exceeded")
        self.assertEqual(len(result.stdout), limit)
        self.assertGreater(result.stdout_bytes, limit)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.cleanup.verified)

    def test_fast_process_finishes_well_inside_deadline(self) -> None:
        result = execute_process(
            self.python,
            ("-c", "print('ready')"),
            timeout_seconds=1.0,
        )

        self.assertEqual(result.code, "completed")
        self.assertEqual(result.stdout, b"ready\n")
        self.assertFalse(result.timed_out)
        self.assertLess(result.execution_duration_ms, 900)
        self.assertTrue(result.cleanup.verified)

    @unittest.skipIf(os.name == "nt", "POSIX process-group failure injection")
    def test_unverified_cleanup_fails_closed(self) -> None:
        with (
            patch.object(bridge_runtime, "_process_group_alive", return_value=True),
            patch.object(bridge_runtime, "_signal_process_group"),
        ):
            with self.assertRaises(ProcessCleanupError) as failure:
                execute_process(
                    self.python,
                    ("-c", "pass"),
                    timeout_seconds=1.0,
                    policy=ProcessExecutionPolicy(
                        cleanup_grace_seconds=0.0,
                        cleanup_kill_seconds=0.05,
                    ),
                )

        self.assertFalse(failure.exception.details["verified"])
        self.assertFalse(failure.exception.details["process_group_verified"])

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True
        return waited == 0


if __name__ == "__main__":
    unittest.main()
