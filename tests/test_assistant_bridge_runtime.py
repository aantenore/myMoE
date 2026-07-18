from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import local_moe.assistant_bridge_runtime as bridge_runtime
from local_moe.assistant_bridge_runtime import (
    ExecutableChangedError,
    LauncherChainChangedError,
    LauncherChainError,
    ProcessCleanupError,
    ProcessExecutionPolicy,
    ProcessLaunchError,
    ProcessLaunchLifecycleError,
    ProcessLaunchNotAuthorizedError,
    ProcessLaunchPermit,
    ProcessTreeUnavailableError,
    RuntimeCapabilities,
    execute_process,
    fingerprint_environment,
    inspect_executable,
    resolve_executable,
    resolve_launcher_chain,
    runtime_capabilities,
    validate_environment_name,
)


STRICT_EXECUTION_AVAILABLE = runtime_capabilities().strict_tree_supported


class AssistantBridgeRuntimeIdentityTests(unittest.TestCase):
    def test_regular_file_attestation_requests_binary_mode_when_available(
        self,
    ) -> None:
        binary_flag = 1 << 28
        with patch.object(
            bridge_runtime.os,
            "O_BINARY",
            binary_flag,
            create=True,
        ):
            flags = bridge_runtime._regular_file_open_flags()

        self.assertTrue(flags & binary_flag)

    def _assert_isolated_python_succeeds(self, source: str) -> None:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", source],
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.fail("Isolated FIFO regression process exceeded its safety timeout")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

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

    @unittest.skipIf(os.name == "nt", "POSIX symlink fixture")
    def test_launch_symlink_retarget_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            launch_path = Path(tmp) / "bridge-tool"
            launch_path.symlink_to("/bin/echo")
            identity = resolve_executable(launch_path)
            launch_path.unlink()
            launch_path.symlink_to("/usr/bin/false")

            with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                with self.assertRaises(ExecutableChangedError):
                    execute_process(identity, timeout_seconds=1.0)

            popen.assert_not_called()

    @unittest.skipIf(os.name == "nt", "POSIX symlink fixture")
    def test_launch_symlink_retarget_during_popen_is_rejected_and_reaped(
        self,
    ) -> None:
        sleep = shutil.which("sleep")
        replacement = shutil.which("false")
        if sleep is None or replacement is None:
            self.skipTest("POSIX sleep/false fixtures unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            launch_path = Path(tmp) / "bridge-tool"
            launch_path.symlink_to(sleep)
            identity = resolve_executable(launch_path)
            spawned: list[object] = []
            real_popen = bridge_runtime.subprocess.Popen

            def retarget_after_spawn(*args: object, **kwargs: object) -> object:
                process = real_popen(*args, **kwargs)
                spawned.append(process)
                launch_path.unlink()
                launch_path.symlink_to(replacement)
                return process

            with patch.object(
                bridge_runtime.subprocess,
                "Popen",
                side_effect=retarget_after_spawn,
            ):
                with self.assertRaisesRegex(
                    ProcessLaunchError,
                    "executable identity",
                ):
                    execute_process(identity, ("30",), timeout_seconds=2.0)

            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO fixture")
    def test_fifo_executable_replacement_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bridge-tool"
            source = (
                "import os\n"
                "from pathlib import Path\n"
                "from local_moe.assistant_bridge_runtime import "
                "ExecutableChangedError, ProcessExecutionPolicy, execute_process, "
                "resolve_executable\n"
                f"target = Path({str(target)!r})\n"
                "target.write_text('#!/bin/sh\\nexit 0\\n', encoding='utf-8')\n"
                "target.chmod(0o700)\n"
                "identity = resolve_executable(target)\n"
                "target.unlink()\n"
                "os.mkfifo(target, 0o700)\n"
                "try:\n"
                "    execute_process(identity, timeout_seconds=1.0, "
                "policy=ProcessExecutionPolicy(require_tree_isolation=False))\n"
                "except ExecutableChangedError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('FIFO executable replacement was accepted')\n"
            )
            self._assert_isolated_python_succeeds(source)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO fixture")
    def test_fifo_shebang_read_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "launcher.py"
            os.mkfifo(target, 0o600)
            source = (
                "from pathlib import Path\n"
                "import local_moe.assistant_bridge_runtime as runtime\n"
                "from local_moe.assistant_bridge_runtime import LauncherChainError\n"
                f"target = Path({str(target)!r})\n"
                "try:\n"
                "    runtime._read_shebang(target)\n"
                "except LauncherChainError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('FIFO shebang was accepted')\n"
            )
            self._assert_isolated_python_succeeds(source)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO fixture")
    def test_fifo_undeclared_script_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "launcher.py"
            os.mkfifo(target, 0o600)
            source = (
                "import sys\n"
                "from pathlib import Path\n"
                "from local_moe.assistant_bridge_runtime import "
                "LauncherChainError, resolve_executable, resolve_launcher_chain\n"
                f"target = Path({str(target)!r})\n"
                "try:\n"
                "    resolve_launcher_chain(resolve_executable(sys.executable), "
                "(str(target),), cwd=target.parent, strict=True)\n"
                "except LauncherChainError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('FIFO script argument was accepted')\n"
            )
            self._assert_isolated_python_succeeds(source)

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

    @unittest.skipUnless(
        STRICT_EXECUTION_AVAILABLE, "observed process-tree control unavailable"
    )
    def test_public_executable_payload_omits_local_and_version_details(self) -> None:
        identity = inspect_executable(sys.executable, version_args=("--version",))

        public_payload = identity.payload()
        private_payload = identity.binding_payload()
        public = json.dumps(public_payload, sort_keys=True)

        self.assertNotIn(identity.requested, public)
        self.assertNotIn(identity.resolved_path, public)
        self.assertNotIn("--version", public)
        assert identity.version is not None
        self.assertNotIn(identity.version.text, public)
        self.assertEqual(private_payload["requested"], identity.requested)
        self.assertEqual(private_payload["resolved_path"], identity.resolved_path)
        self.assertEqual(private_payload["version"]["args"], ["--version"])
        self.assertEqual(len(identity.payload()["resolved_path_sha256"]), 64)
        self.assertNotIn("mtime_ns", identity.payload())
        self.assertIn("mtime_ns", identity.binding_payload())
        rendered_repr = repr(identity)
        self.assertNotIn(identity.requested, rendered_repr)
        self.assertNotIn(identity.resolved_path, rendered_repr)
        self.assertNotIn(identity.version.text, rendered_repr)

    def test_dangerous_loader_and_runtime_environment_is_rejected(self) -> None:
        for name in (
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "GIT_CONFIG_COUNT",
            "BASH_ENV",
            "NODE_OPTIONS",
            "PYTHONPATH",
            "RUBYOPT",
            "SSLKEYLOGFILE",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "injection policy"):
                    validate_environment_name(name)
                with self.assertRaisesRegex(ValueError, "injection policy"):
                    fingerprint_environment({name: "attacker-controlled"})

        validate_environment_name("SSL_CERT_FILE")
        validate_environment_name("LANG")

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

    def test_capability_payload_does_not_claim_hard_containment(self) -> None:
        payload = runtime_capabilities().payload()

        self.assertFalse(payload["hard_containment_supported"])
        self.assertFalse(payload["race_free_launch_binding"])
        self.assertEqual(
            payload["launch_change_detection"],
            "pre_and_post_process_creation",
        )
        self.assertIn("observed_tree_cleanup_supported", payload)
        self.assertIn("strict_tree_supported", payload)
        self.assertEqual(
            payload["schema_version"],
            "assistant-bridge-runtime-capabilities/v1",
        )

    def test_runtime_capabilities_accepts_legacy_and_observed_names(self) -> None:
        legacy = RuntimeCapabilities("test", True, False, True, "legacy contract")
        legacy_keyword = RuntimeCapabilities(
            platform="test",
            posix_process_groups=True,
            psutil_available=False,
            strict_tree_supported=True,
            detached_descendant_contract="legacy contract",
        )
        observed = RuntimeCapabilities(
            platform="test",
            posix_process_groups=True,
            psutil_available=False,
            observed_tree_cleanup_supported=True,
            detached_descendant_contract="observed contract",
        )

        self.assertTrue(legacy.observed_tree_cleanup_supported)
        self.assertTrue(legacy_keyword.strict_tree_supported)
        self.assertTrue(observed.strict_tree_supported)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_changed_executable_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "bridge-tool"
            self._write_executable(executable, "first")
            identity = resolve_executable(executable)
            self._write_executable(executable, "second")

            with self.assertRaises(ExecutableChangedError):
                execute_process(identity, timeout_seconds=1.0)

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture")
    def test_source_swap_during_launch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "bridge-tool"
            self._write_executable(executable, "attested")
            identity = resolve_executable(executable)
            real_popen = bridge_runtime.subprocess.Popen

            def swap_source(*args: object, **kwargs: object) -> object:
                self._write_executable(executable, "swapped")
                return real_popen(*args, **kwargs)

            with patch.object(
                bridge_runtime.subprocess,
                "Popen",
                side_effect=swap_source,
            ):
                with self.assertRaisesRegex(
                    ProcessLaunchError,
                    "executable identity",
                ):
                    execute_process(identity, timeout_seconds=2.0)

    @unittest.skipIf(os.name == "nt", "POSIX shebang fixture")
    def test_script_launcher_keeps_relative_companion_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            companion = root / "companion.txt"
            companion.write_text("companion-visible", encoding="utf-8")
            launcher = root / "bridge-tool"
            launcher.write_text(
                "#!/bin/sh\n"
                'root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                'printf "%s\\n" "$(cat "$root/companion.txt")"\n',
                encoding="utf-8",
            )
            launcher.chmod(0o755)

            result = execute_process(
                resolve_executable(launcher),
                timeout_seconds=2.0,
            )

        self.assertEqual(result.code, "completed")
        self.assertEqual(result.stdout, b"companion-visible\n")

    @unittest.skipIf(os.name == "nt", "POSIX direct shebang semantics")
    def test_strict_direct_script_chain_preserves_zero_and_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            companion = root / "companion.txt"
            companion.write_text("bound-companion", encoding="utf-8")
            launcher = root / "bridge-tool"
            launcher.write_text(
                "#!/bin/sh\n"
                'root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                'printf "%s|%s\\n" "$0" "$(cat "$root/companion.txt")"\n',
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            identity = resolve_executable(launcher)
            chain = resolve_launcher_chain(
                identity,
                cwd=root,
                companions=(companion,),
                strict=True,
            )

            result = execute_process(
                identity,
                cwd=root,
                timeout_seconds=2.0,
                launcher_chain=chain,
                policy=ProcessExecutionPolicy(require_launcher_chain=True),
            )

        self.assertEqual(result.code, "completed")
        self.assertEqual(
            result.stdout.decode().strip(),
            f"{launcher}|bound-companion",
        )

    @unittest.skipIf(os.name == "nt", "POSIX env shebang fixture")
    def test_env_shebang_binds_env_launcher_and_resolved_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / "bridge-tool"
            launcher.write_text(
                "#!/usr/bin/env python3\nprint('env-chain')\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            identity = resolve_executable(launcher)

            chain = resolve_launcher_chain(identity, cwd=root, strict=True)

        self.assertIsNotNone(chain.env_launcher)
        self.assertIsNotNone(chain.interpreter)
        assert chain.env_launcher is not None
        assert chain.interpreter is not None
        self.assertEqual(Path(chain.env_launcher.resolved_path).name, "env")
        self.assertIn("python", Path(chain.interpreter.resolved_path).name.lower())

    def test_interpreter_driven_chain_preserves_entrypoint_and_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            companion = root / "companion.txt"
            companion.write_text("cross-platform-companion", encoding="utf-8")
            entrypoint = root / "wrapper.py"
            entrypoint.write_text(
                "from pathlib import Path\n"
                "print(Path(__file__).with_name('companion.txt').read_text())\n",
                encoding="utf-8",
            )
            identity = resolve_executable(sys.executable)
            chain = resolve_launcher_chain(
                identity,
                (str(entrypoint),),
                entrypoint=entrypoint,
                companions=(companion,),
                cwd=root,
                strict=True,
            )

            result = execute_process(
                identity,
                (str(entrypoint),),
                cwd=root,
                timeout_seconds=2.0,
                launcher_chain=chain,
                policy=ProcessExecutionPolicy(require_launcher_chain=True),
            )

        self.assertEqual(result.code, "completed")
        self.assertEqual(
            result.stdout.decode("utf-8").splitlines(),
            ["cross-platform-companion"],
        )

    def test_strict_mode_rejects_undeclared_script_entrypoint(self) -> None:
        for filename, content in (
            ("wrapper.py", "print('must-not-run')\n"),
            ("wrapper", "#!/usr/bin/env python3\nprint('must-not-run')\n"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                entrypoint = root / filename
                entrypoint.write_text(content, encoding="utf-8")
                identity = resolve_executable(sys.executable)

                with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(LauncherChainError, "attested"):
                        execute_process(
                            identity,
                            (str(entrypoint),),
                            cwd=root,
                            timeout_seconds=1.0,
                            policy=ProcessExecutionPolicy(require_launcher_chain=True),
                        )

                popen.assert_not_called()

    def test_strict_chain_rejects_additional_undeclared_executable_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "entrypoint.py"
            entrypoint.write_text("print('entrypoint')\n", encoding="utf-8")
            helper_script = root / "helper.js"
            helper_script.write_text("console.log('helper')\n", encoding="utf-8")
            native_helper = root / (
                "native-helper.exe" if os.name == "nt" else "native-helper"
            )
            native_helper.write_text("native helper\n", encoding="utf-8")
            native_helper.chmod(0o700)
            identity = resolve_executable(sys.executable)

            for extra in (
                str(helper_script),
                f"--loader={helper_script}",
                str(native_helper),
            ):
                with self.subTest(extra=extra), self.assertRaisesRegex(
                    LauncherChainError,
                    "entrypoint or companion",
                ):
                    resolve_launcher_chain(
                        identity,
                        (str(entrypoint), extra),
                        entrypoint=entrypoint,
                        cwd=root,
                        strict=True,
                    )

    def test_declared_helper_drift_fails_before_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "entrypoint.py"
            entrypoint.write_text("print('entrypoint')\n", encoding="utf-8")
            helper = root / "helper.js"
            helper.write_text("console.log('planned')\n", encoding="utf-8")
            identity = resolve_executable(sys.executable)
            args = (str(entrypoint), f"--loader={helper}")
            chain = resolve_launcher_chain(
                identity,
                args,
                entrypoint=entrypoint,
                companions=(helper,),
                cwd=root,
                strict=True,
            )
            helper.write_text("console.log('drifted')\n", encoding="utf-8")
            reservations = 0

            def reserve():
                nonlocal reservations
                reservations += 1
                return None

            with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(
                    LauncherChainChangedError,
                    "changed",
                ):
                    execute_process(
                        identity,
                        args,
                        cwd=root,
                        timeout_seconds=1.0,
                        launcher_chain=chain,
                        reserve_launch=reserve,
                        policy=ProcessExecutionPolicy(
                            require_tree_isolation=False,
                            require_psutil=False,
                            require_launcher_chain=True,
                        ),
                    )

            self.assertEqual(reservations, 0)
            popen.assert_not_called()

    def test_strict_policy_rejects_missing_chain_for_native_executable(self) -> None:
        identity = resolve_executable(sys.executable)

        with patch.object(bridge_runtime.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(LauncherChainError, "explicitly attested"):
                execute_process(
                    identity,
                    ("-c", "pass"),
                    timeout_seconds=1.0,
                    policy=ProcessExecutionPolicy(
                        require_tree_isolation=False,
                        require_launcher_chain=True,
                    ),
                )

        popen.assert_not_called()

    def test_strict_policy_rejects_non_strict_launcher_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "wrapper.py"
            entrypoint.write_text("print('must-not-run')\n", encoding="utf-8")
            identity = resolve_executable(sys.executable)
            chain = resolve_launcher_chain(
                identity,
                (str(entrypoint),),
                entrypoint=entrypoint,
                cwd=root,
                strict=False,
            )

            with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(LauncherChainError, "strict"):
                    execute_process(
                        identity,
                        (str(entrypoint),),
                        cwd=root,
                        timeout_seconds=1.0,
                        launcher_chain=chain,
                        policy=ProcessExecutionPolicy(require_launcher_chain=True),
                    )

            popen.assert_not_called()

    def test_companion_replacement_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "wrapper.py"
            entrypoint.write_text("print('must-not-run')\n", encoding="utf-8")
            companion = root / "companion.txt"
            companion.write_text("attested", encoding="utf-8")
            identity = resolve_executable(sys.executable)
            chain = resolve_launcher_chain(
                identity,
                (str(entrypoint),),
                entrypoint=entrypoint,
                companions=(companion,),
                cwd=root,
            )
            companion.write_text("replaced", encoding="utf-8")

            with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                with self.assertRaises(LauncherChainChangedError):
                    execute_process(
                        identity,
                        (str(entrypoint),),
                        cwd=root,
                        timeout_seconds=1.0,
                        launcher_chain=chain,
                        policy=ProcessExecutionPolicy(require_launcher_chain=True),
                    )

            popen.assert_not_called()

    def test_entrypoint_replacement_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "wrapper.py"
            entrypoint.write_text("print('attested')\n", encoding="utf-8")
            identity = resolve_executable(sys.executable)
            chain = resolve_launcher_chain(
                identity,
                (str(entrypoint),),
                entrypoint=entrypoint,
                cwd=root,
            )
            entrypoint.write_text("print('replacement')\n", encoding="utf-8")

            with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                with self.assertRaises(LauncherChainChangedError):
                    execute_process(
                        identity,
                        (str(entrypoint),),
                        cwd=root,
                        timeout_seconds=1.0,
                        launcher_chain=chain,
                        policy=ProcessExecutionPolicy(require_launcher_chain=True),
                    )

            popen.assert_not_called()

    def test_companion_replacement_during_popen_is_rejected_and_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "wrapper.py"
            entrypoint.write_text(
                "import time; time.sleep(30)\n",
                encoding="utf-8",
            )
            companion = root / "companion.txt"
            companion.write_text("attested", encoding="utf-8")
            identity = resolve_executable(sys.executable)
            chain = resolve_launcher_chain(
                identity,
                (str(entrypoint),),
                entrypoint=entrypoint,
                companions=(companion,),
                cwd=root,
            )
            spawned: list[object] = []
            real_popen = bridge_runtime.subprocess.Popen

            def replace_after_spawn(*args: object, **kwargs: object) -> object:
                process = real_popen(*args, **kwargs)
                spawned.append(process)
                companion.write_text("replaced", encoding="utf-8")
                return process

            with patch.object(
                bridge_runtime.subprocess,
                "Popen",
                side_effect=replace_after_spawn,
            ):
                expected_error = (
                    (ProcessLaunchError, ProcessCleanupError)
                    if os.name == "nt"
                    else ProcessLaunchError
                )
                with self.assertRaises(expected_error) as raised:
                    execute_process(
                        identity,
                        (str(entrypoint),),
                        cwd=root,
                        timeout_seconds=2.0,
                        launcher_chain=chain,
                        policy=ProcessExecutionPolicy(require_launcher_chain=True),
                    )

            if isinstance(raised.exception, ProcessLaunchError):
                self.assertIn("launcher chain", str(raised.exception))
            else:
                self.assertFalse(raised.exception.details["verified"])

            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]

    def test_entrypoint_replacement_during_popen_is_rejected_and_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "wrapper.py"
            entrypoint.write_text(
                "import time; time.sleep(30)\n",
                encoding="utf-8",
            )
            identity = resolve_executable(sys.executable)
            chain = resolve_launcher_chain(
                identity,
                (str(entrypoint),),
                entrypoint=entrypoint,
                cwd=root,
            )
            spawned: list[object] = []
            real_popen = bridge_runtime.subprocess.Popen

            def replace_after_spawn(*args: object, **kwargs: object) -> object:
                process = real_popen(*args, **kwargs)
                spawned.append(process)
                entrypoint.write_text("print('replacement')\n", encoding="utf-8")
                return process

            with patch.object(
                bridge_runtime.subprocess,
                "Popen",
                side_effect=replace_after_spawn,
            ):
                with self.assertRaisesRegex(ProcessLaunchError, "launcher chain"):
                    execute_process(
                        identity,
                        (str(entrypoint),),
                        cwd=root,
                        timeout_seconds=2.0,
                        launcher_chain=chain,
                        policy=ProcessExecutionPolicy(require_launcher_chain=True),
                    )

            self.assertEqual(len(spawned), 1)
            self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]

    @unittest.skipIf(os.name == "nt", "POSIX shebang interpreter fixture")
    def test_shebang_interpreter_replacement_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpreter = root / "python-copy"
            shutil.copy2(sys.executable, interpreter)
            interpreter.chmod(0o755)
            launcher = root / "bridge-tool"
            launcher.write_text(
                f"#!{interpreter}\nprint('must-not-run')\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            companion = root / "companion.txt"
            companion.write_text("attested", encoding="utf-8")
            identity = resolve_executable(launcher)
            chain = resolve_launcher_chain(
                identity,
                companions=(companion,),
                cwd=root,
                strict=True,
            )
            interpreter.write_bytes(interpreter.read_bytes() + b"replacement")

            with patch.object(bridge_runtime.subprocess, "Popen") as popen:
                with self.assertRaises(ExecutableChangedError):
                    execute_process(
                        identity,
                        cwd=root,
                        timeout_seconds=1.0,
                        launcher_chain=chain,
                        policy=ProcessExecutionPolicy(require_launcher_chain=True),
                    )

            popen.assert_not_called()

    def test_working_directory_swap_during_launch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            displaced = root / "displaced"
            real_popen = bridge_runtime.subprocess.Popen

            def swap_workspace(*args: object, **kwargs: object) -> object:
                workspace.rename(displaced)
                workspace.mkdir()
                return real_popen(*args, **kwargs)

            with patch.object(
                bridge_runtime.subprocess,
                "Popen",
                side_effect=swap_workspace,
            ):
                with self.assertRaisesRegex(ProcessLaunchError, "working directory"):
                    execute_process(
                        resolve_executable(sys.executable),
                        ("-c", "import time; time.sleep(30)"),
                        cwd=workspace,
                        timeout_seconds=2.0,
                    )

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

    def test_tracker_constructor_failure_reaps_owned_process(self) -> None:
        spawned: list[object] = []
        real_popen = bridge_runtime.subprocess.Popen

        def capture(*args: object, **kwargs: object) -> object:
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        with (
            patch.object(bridge_runtime.subprocess, "Popen", side_effect=capture),
            patch.object(
                bridge_runtime,
                "_ProcessTracker",
                side_effect=RuntimeError("tracker setup failed"),
            ),
        ):
            with self.assertRaises(ProcessCleanupError) as failure:
                execute_process(
                    self.python,
                    ("-c", "import time; time.sleep(30)"),
                    timeout_seconds=2.0,
                )

        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]
        self.assertTrue(failure.exception.details["fallback_used"])
        self.assertTrue(failure.exception.details["root_reaped"])

    def test_event_and_worker_setup_failures_reap_owned_process(self) -> None:
        for target, replacement in (
            ("Event", RuntimeError("event setup failed")),
            ("Thread", RuntimeError("worker setup failed")),
        ):
            with self.subTest(target=target):
                spawned: list[object] = []
                real_popen = bridge_runtime.subprocess.Popen

                def capture(*args: object, **kwargs: object) -> object:
                    process = real_popen(*args, **kwargs)
                    spawned.append(process)
                    return process

                with (
                    patch.object(
                        bridge_runtime.subprocess,
                        "Popen",
                        side_effect=capture,
                    ),
                    patch.object(
                        bridge_runtime.threading,
                        target,
                        side_effect=replacement,
                    ),
                ):
                    with self.assertRaises(ProcessCleanupError) as failure:
                        execute_process(
                            self.python,
                            ("-c", "import time; time.sleep(30)"),
                            timeout_seconds=2.0,
                        )

                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]
                self.assertTrue(failure.exception.details["fallback_used"])
                self.assertTrue(failure.exception.details["root_reaped"])

    def test_cleanup_exception_runs_tracker_independent_emergency_reaper(self) -> None:
        spawned: list[object] = []
        real_popen = bridge_runtime.subprocess.Popen

        def capture(*args: object, **kwargs: object) -> object:
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        with (
            patch.object(bridge_runtime.subprocess, "Popen", side_effect=capture),
            patch.object(
                bridge_runtime,
                "_cleanup_process_tree",
                side_effect=RuntimeError("cleanup observation failed"),
            ),
        ):
            with self.assertRaises(ProcessCleanupError) as failure:
                execute_process(
                    self.python,
                    ("-c", "import time; time.sleep(30)"),
                    timeout_seconds=2.0,
                )

        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]
        self.assertTrue(failure.exception.details["fallback_used"])
        self.assertTrue(failure.exception.details["root_reaped"])

    @unittest.skipUnless(
        runtime_capabilities().psutil_available,
        "psutil observation-loss regression requires psutil",
    )
    def test_required_observation_loss_aborts_quickly_and_reaps(self) -> None:
        spawned: list[object] = []
        observe_calls: dict[int, int] = {}
        real_popen = bridge_runtime.subprocess.Popen
        real_observe = bridge_runtime._ProcessTracker.observe

        def capture(*args: object, **kwargs: object) -> object:
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        def lose_observation(tracker: object) -> None:
            real_observe(tracker)  # type: ignore[arg-type]
            key = id(tracker)
            observe_calls[key] = observe_calls.get(key, 0) + 1
            if observe_calls[key] >= 2:
                tracker._observation_failed = True  # type: ignore[attr-defined]

        started = time.monotonic()
        with (
            patch.object(bridge_runtime.subprocess, "Popen", side_effect=capture),
            patch.object(
                bridge_runtime._ProcessTracker,
                "observe",
                new=lose_observation,
            ),
        ):
            with self.assertRaises(ProcessCleanupError) as failure:
                execute_process(
                    self.python,
                    ("-c", "import time; time.sleep(30)"),
                    timeout_seconds=10.0,
                    policy=ProcessExecutionPolicy(
                        require_psutil=True,
                        cleanup_grace_seconds=0.05,
                        cleanup_kill_seconds=0.5,
                    ),
                )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.5)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]
        self.assertFalse(failure.exception.details["observation_verified"])
        self.assertTrue(failure.exception.details["root_reaped"])

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
        self.assertEqual(result.stdout.decode("utf-8").splitlines(), ["ready"])
        self.assertFalse(result.timed_out)
        self.assertLess(result.execution_duration_ms, 900)
        self.assertTrue(result.cleanup.verified)
        self.assertFalse(result.cleanup.payload()["hard_containment"])
        self.assertEqual(
            result.cleanup.payload()["verification_scope"],
            "observed_process_tree",
        )

    def test_launch_permit_reserves_then_commits_around_popen(self) -> None:
        events: list[str] = []
        real_popen = bridge_runtime.subprocess.Popen

        def reserve() -> ProcessLaunchPermit:
            events.append("reserve")
            return ProcessLaunchPermit(
                commit_after_popen=lambda: events.append("commit"),
                release_after_popen_failure=lambda: events.append("release"),
            )

        def launch(*args: object, **kwargs: object) -> object:
            events.append("popen")
            return real_popen(*args, **kwargs)

        with patch.object(
            bridge_runtime.subprocess,
            "Popen",
            side_effect=launch,
        ):
            result = execute_process(
                self.python,
                ("-c", "pass"),
                timeout_seconds=1.0,
                reserve_launch=reserve,
            )

        self.assertTrue(result.ok)
        self.assertEqual(events, ["reserve", "popen", "commit"])

    def test_denied_launch_permit_never_calls_popen(self) -> None:
        with patch.object(bridge_runtime.subprocess, "Popen") as popen:
            with self.assertRaises(ProcessLaunchNotAuthorizedError):
                execute_process(
                    self.python,
                    ("-c", "pass"),
                    timeout_seconds=1.0,
                    reserve_launch=lambda: None,
                )

        popen.assert_not_called()

    def test_popen_failure_releases_without_committing_permit(self) -> None:
        events: list[str] = []
        permit = ProcessLaunchPermit(
            commit_after_popen=lambda: events.append("commit"),
            release_after_popen_failure=lambda: events.append("release"),
        )

        with patch.object(
            bridge_runtime.subprocess,
            "Popen",
            side_effect=OSError("fixed diagnostic"),
        ):
            with self.assertRaises(ProcessLaunchError):
                execute_process(
                    self.python,
                    ("-c", "pass"),
                    timeout_seconds=1.0,
                    reserve_launch=lambda: permit,
                )

        self.assertEqual(events, ["release"])

    def test_commit_failure_never_releases_and_reaps_the_process(self) -> None:
        events: list[str] = []
        spawned: list[object] = []
        real_popen = bridge_runtime.subprocess.Popen

        def launch(*args: object, **kwargs: object) -> object:
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        def fail_commit() -> None:
            events.append("commit")
            raise RuntimeError("sensitive commit failure")

        permit = ProcessLaunchPermit(
            commit_after_popen=fail_commit,
            release_after_popen_failure=lambda: events.append("release"),
        )
        with patch.object(
            bridge_runtime.subprocess,
            "Popen",
            side_effect=launch,
        ):
            with self.assertRaisesRegex(
                ProcessLaunchLifecycleError,
                "commit failed",
            ) as raised:
                execute_process(
                    self.python,
                    ("-c", "import time; time.sleep(30)"),
                    timeout_seconds=2.0,
                    reserve_launch=lambda: permit,
                )

        self.assertEqual(events, ["commit"])
        self.assertNotIn("sensitive commit failure", str(raised.exception))
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())  # type: ignore[attr-defined]

    def test_release_failure_keeps_a_sanitized_pending_reservation(self) -> None:
        permit = ProcessLaunchPermit(
            commit_after_popen=lambda: None,
            release_after_popen_failure=lambda: (_ for _ in ()).throw(
                RuntimeError("sensitive lease token")
            ),
        )
        with patch.object(
            bridge_runtime.subprocess,
            "Popen",
            side_effect=OSError("fixed diagnostic"),
        ):
            with self.assertRaisesRegex(
                ProcessLaunchLifecycleError,
                "release failed",
            ) as raised:
                execute_process(
                    self.python,
                    ("-c", "pass"),
                    timeout_seconds=1.0,
                    reserve_launch=lambda: permit,
                )

        self.assertNotIn("sensitive lease token", str(raised.exception))

    def test_partial_pipe_read_failure_is_never_completed(self) -> None:
        def fail_reader(
            stream: object,
            state: object,
            wake: object,
        ) -> None:
            del stream
            state.update(b"partial")  # type: ignore[attr-defined]
            state.fail()  # type: ignore[attr-defined]
            wake.set()  # type: ignore[attr-defined]

        with patch.object(bridge_runtime, "_read_pipe", side_effect=fail_reader):
            result = execute_process(
                self.python,
                ("-c", "print('unobserved')"),
                timeout_seconds=1.0,
            )

        self.assertEqual(result.code, "io_failed")
        self.assertFalse(result.ok)

    def test_late_pipe_failure_overrides_nonzero_exit_taxonomy(self) -> None:
        def late_fail_reader(
            stream: object,
            state: object,
            wake: object,
        ) -> None:
            if stream is not None:
                chunk = stream.read()  # type: ignore[attr-defined]
                if chunk:
                    state.update(chunk)  # type: ignore[attr-defined]
            state.finish()  # type: ignore[attr-defined]
            time.sleep(0.08)
            state.fail()  # type: ignore[attr-defined]
            wake.set()  # type: ignore[attr-defined]

        with patch.object(
            bridge_runtime,
            "_read_pipe",
            side_effect=late_fail_reader,
        ):
            result = execute_process(
                self.python,
                ("-c", "import sys; print('partial'); sys.exit(3)"),
                timeout_seconds=1.0,
                policy=ProcessExecutionPolicy(pipe_settle_seconds=0.01),
            )

        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.code, "io_failed")
        self.assertFalse(result.ok)

    def test_partial_stdin_write_failure_is_never_completed(self) -> None:
        def fail_writer(
            stream: object,
            content: bytes,
            state: object,
            wake: object,
        ) -> None:
            del stream, content
            state.count = 1  # type: ignore[attr-defined]
            state.fail()  # type: ignore[attr-defined]
            wake.set()  # type: ignore[attr-defined]

        with patch.object(bridge_runtime, "_write_stdin", side_effect=fail_writer):
            result = execute_process(
                self.python,
                ("-c", "import time; time.sleep(0.1)"),
                stdin=b"request body",
                timeout_seconds=1.0,
            )

        self.assertEqual(result.code, "io_failed")
        self.assertFalse(result.ok)

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

    def test_psutil_observation_error_fails_closed(self) -> None:
        class FakePsutil:
            class Error(Exception):
                pass

            class NoSuchProcess(Error):
                pass

            class ZombieProcess(Error):
                pass

            STATUS_ZOMBIE = "zombie"

            class Process:
                def __init__(self, pid: int) -> None:
                    self.pid = pid

                def create_time(self) -> float:
                    return 1.0

                def children(self, *, recursive: bool) -> list[object]:
                    assert recursive
                    raise FakePsutil.Error("observation denied")

                def is_running(self) -> bool:
                    return True

                def status(self) -> str:
                    raise FakePsutil.Error("liveness denied")

                def terminate(self) -> None:
                    raise FakePsutil.Error("termination denied")

                def kill(self) -> None:
                    raise FakePsutil.Error("kill denied")

        with patch.object(bridge_runtime, "_psutil", FakePsutil):
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
        self.assertFalse(failure.exception.details["psutil_verified"])

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
