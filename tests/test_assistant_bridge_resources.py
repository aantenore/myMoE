from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import local_moe.assistant_bridge_resources as resources
from local_moe.assistant_bridge_resources import (
    RESOURCE_CONTROLS,
    VerifierResourcePolicy,
    build_verifier_resource_enforcement_report,
    build_verifier_resource_plan,
    verifier_resource_capabilities,
)
from local_moe.assistant_bridge_runtime import resolve_executable


def _minimal_environment() -> dict[str, str]:
    return {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}


def _python_identity():
    return resolve_executable(sys.executable, env=_minimal_environment())


def _required_strengths(**overrides: str) -> dict[str, str]:
    values = {name: "unsupported" for name in RESOURCE_CONTROLS}
    values["workspace_growth"] = "post_run"
    values.update(overrides)
    return values


class VerifierResourceContractTests(unittest.TestCase):
    def test_default_policy_requires_only_portable_process_hard_controls(self) -> None:
        policy = VerifierResourcePolicy()

        self.assertEqual(policy.required_strengths["cpu_time"], "process_hard")
        self.assertEqual(policy.required_strengths["file_size"], "process_hard")
        self.assertEqual(policy.required_strengths["open_files"], "process_hard")
        self.assertEqual(policy.required_strengths["memory"], "unsupported")
        self.assertEqual(policy.required_strengths["processes"], "unsupported")
        self.assertEqual(
            set(policy.payload()["required_strengths"]), set(RESOURCE_CONTROLS)
        )

    def test_linux_claims_kernel_strength_only_after_verified_systemd_probe(self) -> None:
        identity = _python_identity()
        policy = VerifierResourcePolicy()
        with (
            mock.patch.object(
                resources, "_linux_cgroup_v2_available", return_value=True
            ),
            mock.patch.object(
                resources, "_attest_os_owned_executable", return_value=identity
            ),
            mock.patch.object(
                resources, "_resolve_supervisor_executable", return_value=identity
            ),
            mock.patch.object(
                resources,
                "_systemd_user_environment",
                return_value={"XDG_RUNTIME_DIR": "/run/user/1000"},
            ),
            mock.patch.object(
                resources, "_probe_systemd_user_scope", return_value="f" * 64
            ),
        ):
            capability = verifier_resource_capabilities(
                policy, platform_name="linux"
            )

        self.assertEqual(capability.backend, "linux-systemd-cgroup-v2")
        self.assertEqual(capability.strength("memory"), "kernel_hard")
        self.assertEqual(capability.strength("processes"), "kernel_hard")
        self.assertEqual(capability.strength("cpu_quota"), "kernel_hard")
        self.assertEqual(capability.probe_sha256, "f" * 64)
        self.assertEqual(
            capability.payload()["environment_keys"], ["XDG_RUNTIME_DIR"]
        )

    def test_linux_fallback_never_labels_process_limits_as_kernel_hard(self) -> None:
        with mock.patch.object(
            resources, "_linux_cgroup_v2_available", return_value=False
        ):
            capability = verifier_resource_capabilities(
                VerifierResourcePolicy(), platform_name="linux"
            )

        self.assertEqual(capability.backend, "linux-setrlimit")
        self.assertEqual(capability.strength("cpu_time"), "process_hard")
        self.assertEqual(capability.strength("memory"), "unsupported")
        self.assertIn("not verified", capability.reason)

    def test_required_kernel_control_fails_closed_on_setrlimit_host(self) -> None:
        policy = VerifierResourcePolicy(
            required_strengths=_required_strengths(memory="kernel_hard")
        )
        capability = verifier_resource_capabilities(
            policy, platform_name="darwin"
        )
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=temporary,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", "raise SystemExit(0)"),
                command_binding_sha256="a" * 64,
                environment=_minimal_environment(),
                sandbox_ready=True,
            )

        self.assertFalse(plan.runnable)
        self.assertEqual(
            plan.reason, "required_resource_strength_unavailable:memory"
        )

    def test_windows_job_api_presence_is_not_reported_as_bound_enforcement(self) -> None:
        with mock.patch.object(
            resources, "_windows_job_objects_available", return_value=True
        ):
            capability = verifier_resource_capabilities(
                VerifierResourcePolicy(), platform_name="win32"
            )
        self.assertEqual(capability.backend, "windows-job-object-unbound")
        self.assertFalse(capability.supported)
        self.assertEqual(capability.strength("memory"), "unsupported")
        self.assertIn("no attested launcher adapter", capability.reason)

        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            plan = build_verifier_resource_plan(
                VerifierResourcePolicy(),
                capability,
                workspace=temporary,
                command_executable=None,
                command_argv=(),
                command_binding_sha256="b" * 64,
                environment=_minimal_environment(),
                sandbox_ready=False,
            )

        self.assertFalse(plan.runnable)
        self.assertEqual(plan.reason, "filesystem_network_sandbox_unavailable")

    def test_supervisor_source_is_bound_inline_and_never_carried_in_environment(self) -> None:
        policy = VerifierResourcePolicy()
        capability = verifier_resource_capabilities(
            policy, platform_name="darwin"
        )
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=temporary,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", "raise SystemExit(0)"),
                command_binding_sha256="9" * 64,
                environment=_minimal_environment(),
                sandbox_ready=True,
            )

        self.assertTrue(plan.runnable)
        self.assertEqual(plan.argv[2], resources._POSIX_RESOURCE_SUPERVISOR)
        self.assertNotIn("MYMOE_RESOURCE_SUPERVISOR", plan.environment)
        self.assertNotIn(
            resources._POSIX_RESOURCE_SUPERVISOR,
            plan.environment.values(),
        )
        assert plan.launcher_chain is not None
        self.assertEqual(plan.launcher_chain.argv, plan.argv)

    def test_systemd_probe_requires_kernel_readback_and_exit_propagation(self) -> None:
        readback = {
            "cpu_max": "200000 100000",
            "memory_max": "2147483648",
            "pids_max": "256",
            "scope_membership": "transient_non_root_scope",
            "verified": True,
        }
        completed = subprocess.CompletedProcess(
            args=(),
            returncode=resources._SYSTEMD_PROBE_EXIT,
            stdout=json.dumps(readback).encode("utf-8"),
            stderr=b"",
        )
        environment = {
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        }
        with mock.patch.object(subprocess, "run", return_value=completed) as run:
            digest = resources._probe_systemd_user_scope(
                "/usr/bin/systemd-run",
                supervisor=_python_identity(),
                policy=VerifierResourcePolicy(),
                environment=environment,
            )

        self.assertIsNotNone(digest)
        argv = run.call_args.args[0]
        self.assertIn("--scope", argv)
        self.assertIn("--collect", argv)
        self.assertIn("--pipe", argv)
        self.assertIn("--expand-environment=no", argv)
        self.assertIn("--property=RuntimeMaxSec=5000ms", argv)
        self.assertIn("--property=CPUQuota=200%", argv)
        self.assertIn("--property=MemoryMax=2147483648", argv)
        self.assertIn("--property=TasksMax=256", argv)
        self.assertEqual(
            {
                key: run.call_args.kwargs["env"][key]
                for key in environment
            },
            environment,
        )

    def test_systemd_probe_rejects_success_without_sentinel_exit(self) -> None:
        completed = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=b'{"cpu_max":"200000 100000","memory_max":"2147483648",'
            b'"pids_max":"256","verified":true}',
            stderr=b"",
        )
        with mock.patch.object(subprocess, "run", return_value=completed):
            digest = resources._probe_systemd_user_scope(
                "/usr/bin/systemd-run",
                supervisor=_python_identity(),
                policy=VerifierResourcePolicy(),
                environment={"XDG_RUNTIME_DIR": "/run/user/1000"},
            )

        self.assertIsNone(digest)

    def test_systemd_probe_rejects_unattested_scope_membership(self) -> None:
        for membership in ("", "cgroup_root", "non_scope_cgroup"):
            with self.subTest(membership=membership):
                readback = {
                    "cpu_max": "200000 100000",
                    "memory_max": "2147483648",
                    "pids_max": "256",
                    "scope_membership": membership,
                    "verified": True,
                }
                completed = subprocess.CompletedProcess(
                    args=(),
                    returncode=resources._SYSTEMD_PROBE_EXIT,
                    stdout=json.dumps(readback).encode("utf-8"),
                    stderr=b"",
                )
                with mock.patch.object(subprocess, "run", return_value=completed):
                    digest = resources._probe_systemd_user_scope(
                        "/usr/bin/systemd-run",
                        supervisor=_python_identity(),
                        policy=VerifierResourcePolicy(),
                        environment={"XDG_RUNTIME_DIR": "/run/user/1000"},
                    )

                self.assertIsNone(digest)

    def test_systemd_scope_probe_rejects_empty_root_and_non_scope_paths(self) -> None:
        expected = {
            "cpu_quota_percent": 200,
            "memory_bytes": 2 * 1024 * 1024 * 1024,
            "processes": 256,
            "runtime_max_milliseconds": 5000,
            "sentinel_exit": resources._SYSTEMD_PROBE_EXIT,
        }
        with tempfile.TemporaryDirectory(prefix="mymoe-cgroup-probe-") as temporary:
            base = Path(temporary)
            proc_cgroup = base / "self.cgroup"
            cgroup_root = base / "cgroup"
            cgroup_root.mkdir()

            def write_controls(path: Path) -> None:
                path.mkdir(parents=True, exist_ok=True)
                path.joinpath("cpu.max").write_text(
                    "200000 100000", encoding="ascii"
                )
                path.joinpath("memory.max").write_text(
                    "2147483648", encoding="ascii"
                )
                path.joinpath("pids.max").write_text("256", encoding="ascii")

            write_controls(cgroup_root)
            write_controls(cgroup_root / "user.slice" / "plain")
            scope = cgroup_root / "user.slice" / "run-test.scope"
            write_controls(scope)
            source = resources._SYSTEMD_SCOPE_PROBE.replace(
                'Path("/proc/self/cgroup")', f"Path({str(proc_cgroup)!r})", 1
            ).replace(
                'Path("/sys/fs/cgroup")', f"Path({str(cgroup_root)!r})", 1
            )
            self.assertNotEqual(source, resources._SYSTEMD_SCOPE_PROBE)

            def run_probe(cgroup_line: str) -> subprocess.CompletedProcess[bytes]:
                proc_cgroup.write_text(cgroup_line, encoding="ascii")
                return subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        source,
                        json.dumps(expected, sort_keys=True),
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    check=False,
                    timeout=5,
                )

            for label, cgroup_line in (
                ("empty", ""),
                ("root", "0::/\n"),
                ("non_scope", "0::/user.slice/plain\n"),
            ):
                with self.subTest(label=label):
                    completed = run_probe(cgroup_line)
                    self.assertNotEqual(
                        completed.returncode, resources._SYSTEMD_PROBE_EXIT
                    )

            completed = run_probe("0::/user.slice/run-test.scope\n")
            self.assertEqual(completed.returncode, resources._SYSTEMD_PROBE_EXIT)
            readback = json.loads(completed.stdout.decode("utf-8"))
            self.assertEqual(
                readback["scope_membership"], "transient_non_root_scope"
            )
            self.assertNotIn(str(cgroup_root), completed.stdout.decode("utf-8"))

    def test_report_keeps_output_cleanup_and_workspace_controls_distinct(self) -> None:
        policy = VerifierResourcePolicy(workspace_growth_bytes=8)
        capability = verifier_resource_capabilities(
            policy, platform_name="darwin"
        )
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            root = Path(temporary)
            root.joinpath("baseline.txt").write_bytes(b"base")
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=root,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", "raise SystemExit(0)"),
                command_binding_sha256="c" * 64,
                environment=_minimal_environment(),
                sandbox_ready=True,
            )
            root.joinpath("result.txt").write_bytes(b"ok")
            report = build_verifier_resource_enforcement_report(
                plan,
                workspace=root,
                stdout_bytes=2,
                stderr_bytes=0,
                stdout_limit_bytes=1024,
                stderr_limit_bytes=1024,
                stdout_truncated=False,
                stderr_truncated=False,
                cleanup={
                    "verified": True,
                    "verification_scope": "observed_process_tree",
                    "hard_containment": False,
                    "observed_descendants": 1,
                },
            )

        payload = report.payload()
        self.assertTrue(payload["compliant"])
        self.assertFalse(payload["output_capture"]["resource_control"])
        self.assertFalse(payload["tree_cleanup"]["resource_control"])
        self.assertFalse(payload["tree_cleanup"]["hard_containment"])
        self.assertTrue(payload["workspace_growth"]["resource_control"])
        self.assertFalse(
            payload["workspace_growth"]["quota_enforced_during_execution"]
        )
        self.assertEqual(payload["workspace_growth"]["growth_bytes"], 2)

    def test_report_refuses_compliance_when_required_strength_is_missing(self) -> None:
        policy = VerifierResourcePolicy(
            required=False,
            required_strengths=_required_strengths(memory="kernel_hard"),
        )
        capability = verifier_resource_capabilities(
            policy, platform_name="darwin"
        )
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=temporary,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", "raise SystemExit(0)"),
                command_binding_sha256="8" * 64,
                environment=_minimal_environment(),
                sandbox_ready=True,
            )
            report = build_verifier_resource_enforcement_report(
                plan,
                workspace=temporary,
                stdout_bytes=0,
                stderr_bytes=0,
                stdout_limit_bytes=1024,
                stderr_limit_bytes=1024,
                stdout_truncated=False,
                stderr_truncated=False,
                cleanup={
                    "verified": True,
                    "verification_scope": "observed_process_tree",
                    "hard_containment": False,
                    "observed_descendants": 0,
                },
            )

        self.assertTrue(plan.runnable)
        self.assertFalse(report.compliant)
        self.assertEqual(report.missing_required_controls, ("memory",))
        self.assertFalse(report.payload()["required_strengths_satisfied"])

    def test_post_run_workspace_growth_violation_is_not_hidden(self) -> None:
        policy = VerifierResourcePolicy(workspace_growth_bytes=1)
        capability = verifier_resource_capabilities(
            policy, platform_name="darwin"
        )
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            root = Path(temporary)
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=root,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", "raise SystemExit(0)"),
                command_binding_sha256="d" * 64,
                environment=_minimal_environment(),
                sandbox_ready=True,
            )
            root.joinpath("too-large.txt").write_bytes(b"xx")
            report = build_verifier_resource_enforcement_report(
                plan,
                workspace=root,
                stdout_bytes=0,
                stderr_bytes=0,
                stdout_limit_bytes=1024,
                stderr_limit_bytes=1024,
                stdout_truncated=False,
                stderr_truncated=False,
                cleanup={
                    "verified": True,
                    "verification_scope": "observed_process_tree",
                    "hard_containment": False,
                    "observed_descendants": 0,
                },
            )

        self.assertFalse(report.compliant)
        self.assertFalse(report.payload()["workspace_growth"]["within_bound"])

    @unittest.skipIf(os.name == "nt", "symlink creation may require privilege")
    def test_workspace_accounting_never_follows_file_or_directory_links(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            base = Path(temporary)
            workspace = base / "workspace"
            workspace.mkdir()
            workspace.joinpath("local.txt").write_bytes(b"abc")
            external_file = base / "external.txt"
            external_file.write_bytes(b"external-content")
            external_directory = base / "external-directory"
            external_directory.mkdir()
            external_directory.joinpath("large.txt").write_bytes(b"x" * 100)
            workspace.joinpath("file-link").symlink_to(external_file)
            workspace.joinpath("directory-link").symlink_to(
                external_directory,
                target_is_directory=True,
            )

            observed = resources.measure_workspace_bytes(workspace)

        self.assertEqual(observed, 3)

    def test_workspace_accounting_fails_closed_when_walk_reports_an_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            def denied_walk(*args, **kwargs):
                onerror = kwargs.get("onerror")
                self.assertIsNotNone(onerror)
                assert onerror is not None
                onerror(PermissionError("simulated unreadable directory"))
                return ()

            with mock.patch.object(resources.os, "walk", side_effect=denied_walk):
                with self.assertRaisesRegex(
                    resources.VerifierResourceError,
                    "could not traverse",
                ):
                    resources.measure_workspace_bytes(temporary)

    @unittest.skipIf(os.name == "nt", "POSIX directory permissions are required")
    def test_workspace_accounting_fails_closed_on_unreadable_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            root = Path(temporary)
            blocked = root / "blocked"
            blocked.mkdir()
            blocked.joinpath("hidden.txt").write_bytes(b"hidden")
            blocked.chmod(0)
            try:
                try:
                    list(blocked.iterdir())
                except PermissionError:
                    pass
                else:
                    self.skipTest(
                        "current identity can traverse mode-000 directories"
                    )
                with self.assertRaisesRegex(
                    resources.VerifierResourceError,
                    "could not traverse",
                ):
                    resources.measure_workspace_bytes(root)
            finally:
                blocked.chmod(0o700)

    def test_workspace_accounting_rejects_root_identity_replacement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-") as temporary:
            base = Path(temporary)
            root = base / "workspace"
            root.mkdir()
            root.joinpath("local.txt").write_bytes(b"abc")
            moved = base / "original-workspace"
            real_walk = os.walk

            def replacing_walk(*args, **kwargs):
                yield from real_walk(*args, **kwargs)
                root.rename(moved)
                root.mkdir()

            with mock.patch.object(resources.os, "walk", replacing_walk):
                with self.assertRaisesRegex(
                    resources.VerifierResourceError,
                    "root identity changed",
                ):
                    resources.measure_workspace_bytes(root)


@unittest.skipUnless(sys.platform == "darwin", "requires macOS setrlimit")
class MacOSVerifierResourceLiveTests(unittest.TestCase):
    def test_fixed_supervisor_applies_inherited_cpu_file_and_nofile_limits(self) -> None:
        policy = VerifierResourcePolicy(
            cpu_time_seconds=3,
            file_size_bytes=1024 * 1024,
            open_files=64,
        )
        capability = verifier_resource_capabilities(policy)
        self.assertEqual(capability.backend, "darwin-setrlimit")
        self.assertEqual(capability.strength("memory"), "unsupported")
        self.assertEqual(capability.strength("processes"), "unsupported")
        child = (
            "import json,resource;"
            "print(json.dumps({"
            "'cpu':resource.getrlimit(resource.RLIMIT_CPU)[0],"
            "'fsize':resource.getrlimit(resource.RLIMIT_FSIZE)[0],"
            "'nofile':resource.getrlimit(resource.RLIMIT_NOFILE)[0]}))"
        )
        environment = _minimal_environment()
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-live-") as temporary:
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=temporary,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", child),
                command_binding_sha256="e" * 64,
                environment=environment,
                sandbox_ready=True,
            )
            self.assertTrue(plan.runnable)
            assert plan.executable is not None
            completed = subprocess.run(
                [plan.executable.launch_path, *plan.argv],
                cwd=temporary,
                env=dict(plan.environment),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )

        observed = json.loads(completed.stdout)
        self.assertLessEqual(observed["cpu"], policy.cpu_time_seconds)
        self.assertLessEqual(observed["fsize"], policy.file_size_bytes)
        self.assertLessEqual(observed["nofile"], policy.open_files)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux systemd")
class LinuxVerifierResourceLiveTests(unittest.TestCase):
    def test_scope_wall_backstop_terminates_a_sleeping_process(self) -> None:
        policy = VerifierResourcePolicy()
        capability = verifier_resource_capabilities(policy)
        if capability.backend != "linux-systemd-cgroup-v2":
            self.skipTest("verified systemd user-scope backend is unavailable")
        environment = _minimal_environment()
        child = (
            "import os,time;"
            "open('resource-child.pid','w').write(str(os.getpid()));"
            "time.sleep(5)"
        )
        with tempfile.TemporaryDirectory(prefix="mymoe-resource-live-") as temporary:
            root = Path(temporary)
            plan = build_verifier_resource_plan(
                policy,
                capability,
                workspace=root,
                command_executable=_python_identity(),
                command_argv=("-I", "-c", child),
                command_binding_sha256="7" * 64,
                environment=environment,
                sandbox_ready=True,
                wall_time_seconds=1.0,
            )
            self.assertTrue(plan.runnable)
            assert plan.executable is not None
            completed = subprocess.run(
                [plan.executable.launch_path, *plan.argv],
                cwd=root,
                env=dict(plan.environment),
                check=False,
                capture_output=True,
                timeout=4,
            )
            self.assertNotEqual(completed.returncode, 0)
            pid_path = root / "resource-child.pid"
            self.assertTrue(pid_path.is_file())
            pid = int(pid_path.read_text(encoding="ascii"))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            else:
                self.fail("systemd scope child survived its wall-time backstop")


if __name__ == "__main__":
    unittest.main()
