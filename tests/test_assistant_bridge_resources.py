from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
                resources, "_probe_systemd_user_scope", return_value=True
            ),
        ):
            capability = verifier_resource_capabilities(
                policy, platform_name="linux"
            )

        self.assertEqual(capability.backend, "linux-systemd-cgroup-v2")
        self.assertEqual(capability.strength("memory"), "kernel_hard")
        self.assertEqual(capability.strength("processes"), "kernel_hard")
        self.assertEqual(capability.strength("cpu_quota"), "kernel_hard")

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

    def test_windows_job_contract_does_not_bypass_missing_command_sandbox(self) -> None:
        with mock.patch.object(
            resources, "_windows_job_objects_available", return_value=True
        ):
            capability = verifier_resource_capabilities(
                VerifierResourcePolicy(), platform_name="win32"
            )
        self.assertEqual(capability.backend, "windows-job-object-contract")
        self.assertEqual(capability.strength("memory"), "kernel_hard")

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


if __name__ == "__main__":
    unittest.main()
