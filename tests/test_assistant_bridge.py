from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, patch

import local_moe.assistant_bridge as assistant_bridge_module
import local_moe.assistant_bridge_runtime as assistant_bridge_runtime
from local_moe.assistant_bridge import (
    AssistantBridgeError,
    AssistantBridgeRunner,
    GitIdentity,
    ProviderAdapterRegistry,
    VerificationEvidence,
    attest_workspace,
    build_assistant_task,
    build_codex_command_plan,
    build_escalation_capsule,
    default_provider_adapter_registry,
    execute_codex_command,
    load_assistant_bridge_config,
    load_assistant_task,
    load_verification_evidence,
    plan_assistant_route,
)
from local_moe.app_config import load_app_config
from local_moe.assistant_bridge_workspace import (
    WorkspaceScopePolicy,
    WorkspaceWriteCapability,
    snapshot_workspace,
)


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SPEC_SHA256 = (
    "bbd7d15a562a8eb4ffc6b52cf6f89855589c5047e89d5d22c23645e912ff1e30"
)


class AssistantBridgeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _load_test_assistant_bridge_config()

    def test_python_runner_attestation_uses_binary_descriptors(self) -> None:
        binary_flag = 1 << 28
        platform_binary_flag = getattr(os, "O_BINARY", 0)
        real_open = os.open
        observed_flags: list[int] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runner.py"
            content = b"print('raw')\r\n# \x1a marker\r\n"
            path.write_bytes(content)

            def open_with_observed_flags(raw_path, flags, *args, **kwargs):
                observed_flags.append(flags)
                return real_open(
                    raw_path,
                    (flags & ~binary_flag) | platform_binary_flag,
                    *args,
                    **kwargs,
                )

            with (
                patch.object(
                    assistant_bridge_module.os,
                    "O_BINARY",
                    binary_flag,
                    create=True,
                ),
                patch.object(
                    assistant_bridge_module.os,
                    "open",
                    side_effect=open_with_observed_flags,
                ),
            ):
                size, _, digest = assistant_bridge_module._attest_python_runner_file(
                    path
                )

        self.assertEqual(size, len(content))
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertTrue(observed_flags[0] & binary_flag)
        self.assertIn(
            'getattr(os, "O_BINARY", 0)',
            assistant_bridge_module._PYTHON_UNITTEST_BOOTSTRAP,
        )

    def test_workspace_synthetic_git_identity_is_configurable(self) -> None:
        self.assertEqual(
            self.config.workspace.scope.synthetic_git_identity,
            GitIdentity(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads(
                (ROOT / "configs" / "assistant-bridge.json").read_text(
                    encoding="utf-8"
                )
            )
            raw["workspace"]["synthetic_git_identity"] = {
                "name": "Configured Materializer",
                "email": "configured@localhost",
            }
            path = Path(tmp) / "assistant-bridge.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            configured = load_assistant_bridge_config(path)

        self.assertEqual(
            configured.workspace.scope.synthetic_git_identity,
            GitIdentity(
                name="Configured Materializer",
                email="configured@localhost",
            ),
        )
        descriptor = configured.workspace.effective_descriptor()
        self.assertIn("synthetic_git_identity_sha256", descriptor)
        self.assertNotIn("Configured Materializer", json.dumps(descriptor))
        self.assertNotIn("configured@localhost", json.dumps(descriptor))

    def test_loads_task_and_strict_evidence_contract_fixtures(self) -> None:
        task = load_assistant_task(
            ROOT / "tests" / "fixtures" / "assistant-bridge.task.json"
        )
        evidence = load_verification_evidence(
            ROOT / "tests" / "fixtures" / "assistant-bridge.verification-pass.json"
        )

        self.assertEqual(task.task_id, "fixture-safe-refactor")
        self.assertTrue(task.allow_remote_workspace)
        self.assertEqual(evidence[0].kind, "external")
        self.assertEqual(evidence[0].verifier_spec_sha256, EXTERNAL_SPEC_SHA256)

    def test_resource_policy_is_loaded_and_bound_into_command_verifier_plans(self) -> None:
        policy = self.config.verifier_resources
        self.assertTrue(policy.required)
        self.assertEqual(policy.cpu_time_seconds, 900)
        self.assertEqual(policy.memory_bytes, 2 * 1024 * 1024 * 1024)
        self.assertEqual(policy.required_strengths["memory"], "unsupported")
        isolation = assistant_bridge_module.verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not isolation.supported:
            self.skipTest(isolation.reason)

        spec = next(
            item for item in self.config.command_verifiers if item.kind == "command"
        )
        plan = assistant_bridge_module._build_verifier_plan(
            spec,
            workspace=ROOT,
            runtime_policy=self.config.runtime,
            isolation_policy=self.config.verifier_isolation,
            resource_policy=policy,
        )

        self.assertIsNotNone(plan.resources)
        assert plan.resources is not None
        self.assertTrue(plan.resources.runnable)
        self.assertEqual(plan.resources.policy, policy)
        self.assertEqual(
            plan.resources.wall_time_milliseconds,
            spec.timeout_seconds * 1000,
        )
        self.assertFalse(plan.payload()["resources"]["complete_resource_containment"])

    def test_resource_policy_rejects_unknown_and_incomplete_strength_contracts(self) -> None:
        for mutate, message in (
            (
                lambda raw: raw["verification"]["resources"].__setitem__(
                    "unknown_control", 1
                ),
                "Unknown verification.resources",
            ),
            (
                lambda raw: raw["verification"]["resources"].__setitem__(
                    "required_strengths", {"cpu_time": "process_hard"}
                ),
                "declare every resource control",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                raw = json.loads(
                    (ROOT / "configs" / "assistant-bridge.json").read_text(
                        encoding="utf-8"
                    )
                )
                mutate(raw)
                path = Path(tmp) / "assistant-bridge.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(AssistantBridgeError, message):
                    load_assistant_bridge_config(path)

    def test_verifier_planning_uses_one_fresh_resource_snapshot_per_batch(self) -> None:
        first = next(
            item for item in self.config.command_verifiers if item.kind == "command"
        )
        second = replace(first, id="project-test-suite-second")
        snapshots = (object(), object())
        observed: list[object] = []

        def capture_build(spec, **kwargs):
            observed.append(kwargs["resource_capability_snapshot"])
            return spec.id

        with patch.object(
            assistant_bridge_module,
            "verifier_resource_capabilities",
            side_effect=snapshots,
        ) as probe, patch.object(
            assistant_bridge_module,
            "_build_verifier_plan",
            side_effect=capture_build,
        ):
            first_batch = assistant_bridge_module._build_verifier_plan_batch(
                (first, second),
                workspace=ROOT,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
                resource_policy=self.config.verifier_resources,
            )
            second_batch = assistant_bridge_module._build_verifier_plan_batch(
                (first, second),
                workspace=ROOT,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
                resource_policy=self.config.verifier_resources,
            )

        self.assertEqual(first_batch, (first.id, second.id))
        self.assertEqual(second_batch, (first.id, second.id))
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(observed[:2], [snapshots[0], snapshots[0]])
        self.assertEqual(observed[2:], [snapshots[1], snapshots[1]])

    def test_verifier_execution_reattests_resources_once_per_batch(self) -> None:
        first_spec = next(
            item for item in self.config.command_verifiers if item.kind == "command"
        )
        second_spec = replace(first_spec, id="project-test-suite-second")
        resource_plan = SimpleNamespace(policy=self.config.verifier_resources)
        plans = (
            SimpleNamespace(spec=first_spec, resources=resource_plan),
            SimpleNamespace(spec=second_spec, resources=resource_plan),
        )
        snapshots = (object(), object())

        def capture_run(plan, **kwargs):
            return kwargs["resource_capability_snapshot"]

        with patch.object(
            assistant_bridge_module,
            "verifier_resource_capabilities",
            side_effect=snapshots,
        ) as probe, patch.object(
            assistant_bridge_module,
            "_run_bound_verifier",
            side_effect=capture_run,
        ):
            first_batch = assistant_bridge_module._run_bound_verifier_batch(
                plans,
                task=build_assistant_task("Verify a batch."),
                workspace=attest_workspace(ROOT),
                verifier_workspace=ROOT,
            )
            second_batch = assistant_bridge_module._run_bound_verifier_batch(
                plans,
                task=build_assistant_task("Verify a new batch."),
                workspace=attest_workspace(ROOT),
                verifier_workspace=ROOT,
            )

        self.assertEqual(probe.call_count, 2)
        self.assertEqual(first_batch, (snapshots[0], snapshots[0]))
        self.assertEqual(second_batch, (snapshots[1], snapshots[1]))

    def test_task_and_receipt_ids_bind_the_complete_contract(self) -> None:
        first = build_assistant_task(
            "Preserve behavior.",
            constraints=("Keep API A.",),
            allow_remote=True,
        )
        second = build_assistant_task(
            "Preserve behavior.",
            constraints=("Keep API B.",),
            allow_remote=True,
        )
        first_receipt = plan_assistant_route(first, self.config, workspace=ROOT)
        second_receipt = plan_assistant_route(second, self.config, workspace=ROOT)

        self.assertNotEqual(first.task_id, second.task_id)
        self.assertNotEqual(first.task_fingerprint, second.task_fingerprint)
        self.assertNotEqual(first_receipt.receipt_id, second_receipt.receipt_id)

    def test_receipt_binds_runtime_override_and_contains_no_task_text(self) -> None:
        task = build_assistant_task(
            "Private objective that must not appear in a receipt.",
            required_capabilities=("code",),
            allow_remote=True,
            allow_remote_workspace=True,
        )
        ollama = plan_assistant_route(
            task,
            self.config,
            workspace=ROOT,
            local_provider_override="ollama",
        )
        lmstudio = plan_assistant_route(
            task,
            self.config,
            workspace=ROOT,
            local_provider_override="lmstudio",
        )
        rendered = json.dumps(ollama.payload(), sort_keys=True)

        self.assertNotEqual(ollama.receipt_id, lmstudio.receipt_id)
        self.assertNotEqual(
            ollama.local_runtime["runtime_sha256"],
            lmstudio.local_runtime["runtime_sha256"],
        )
        self.assertNotIn(task.objective, rendered)
        self.assertIn(task.objective_sha256, rendered)

    def test_profiles_remain_policy_distinct(self) -> None:
        routes = {
            profile: plan_assistant_route(
                build_assistant_task(
                    "Implement a local change.",
                    profile=profile,
                    required_capabilities=("code",),
                    allow_remote=True,
                    allow_remote_workspace=True,
                ),
                self.config,
                workspace=ROOT,
            ).route
            for profile in ("economy", "balanced", "quality", "privacy", "offline")
        }

        self.assertEqual(
            routes,
            {
                "economy": "local",
                "balanced": "local_then_verify",
                "quality": "premium",
                "privacy": "local_then_verify",
                "offline": "local",
            },
        )

    def test_privacy_and_offline_hard_block_remote_without_authority(self) -> None:
        privacy = plan_assistant_route(
            build_assistant_task(
                "Research current evidence.",
                profile="privacy",
                required_capabilities=("web",),
            ),
            self.config,
            workspace=ROOT,
        )
        offline = plan_assistant_route(
            build_assistant_task(
                "Use large context.",
                profile="offline",
                required_capabilities=("large_context",),
                allow_remote=True,
            ),
            self.config,
            workspace=ROOT,
        )

        self.assertEqual(privacy.route, "blocked")
        self.assertFalse(privacy.remote_allowed)
        self.assertEqual(offline.route, "blocked")
        self.assertFalse(offline.remote_allowed)
        self.assertFalse(offline.local_runtime["agent_tool_network_access"])

    def test_write_authority_is_enforced_not_just_ranked(self) -> None:
        external = plan_assistant_route(
            build_assistant_task(
                "Perform an external write.",
                risk_class="write_external",
                allow_remote=True,
            ),
            self.config,
            workspace=ROOT,
        )
        no_remote_workspace = plan_assistant_route(
            build_assistant_task(
                "Edit code.",
                required_capabilities=("code",),
                risk_class="write_local",
                allow_remote=True,
            ),
            self.config,
            workspace=ROOT,
        )

        self.assertEqual(external.route, "blocked")
        self.assertIn("authority:write_external", external.local_gaps)
        self.assertEqual(no_remote_workspace.route, "local")
        self.assertIn(
            "authority:remote_workspace_opt_in",
            no_remote_workspace.premium_gaps,
        )

    def test_web_capability_is_materialized_in_argv(self) -> None:
        task = build_assistant_task(
            "Research a public source.",
            profile="quality",
            required_capabilities=("web",),
            required_tools=("web",),
            allow_remote=True,
        )
        prompt = json.dumps({"objective": "bounded"})
        plan = build_codex_command_plan(
            self.config.premium,
            prompt=prompt,
            workspace=ROOT,
            demand=task.capability_demand,
        )

        self.assertIn("--search", plan.argv)
        self.assertTrue(plan.network_access)

    def test_command_plan_materializes_sandbox_and_keeps_prompt_out_of_argv(
        self,
    ) -> None:
        objective = "Run tests; $(touch should-never-run) && echo unsafe"
        task = build_assistant_task(objective, allow_remote=True)
        prompt = json.dumps({"objective": task.objective})
        plan = build_codex_command_plan(
            self.config.local,
            prompt=prompt,
            workspace=ROOT,
            demand=task.capability_demand,
        )

        self.assertNotIn(objective, plan.argv)
        self.assertEqual(plan.argv[-1], "-")
        self.assertIn("--ignore-user-config", plan.argv)
        self.assertIn("--ignore-rules", plan.argv)
        self.assertEqual(plan.sandbox, "read-only")
        self.assertIn("sandbox_workspace_write.network_access=false", plan.argv)

    def test_command_plan_binds_adapter_and_ephemeral_environment_contract(
        self,
    ) -> None:
        baseline = build_codex_command_plan(
            self.config.local,
            prompt="bounded",
            workspace=ROOT,
        )
        with (
            patch.object(
                assistant_bridge_module,
                "_CODEX_ADAPTER_ID",
                "alternate_cli",
            ),
            patch.object(
                assistant_bridge_module,
                "_CODEX_EPHEMERAL_ENVIRONMENT_KEYS",
                ("ALTERNATE_HOME",),
            ),
        ):
            alternate = build_codex_command_plan(
                replace(self.config.local, adapter="alternate_cli"),
                prompt="bounded",
                workspace=ROOT,
            )

        self.assertEqual(baseline.adapter_id, "codex_cli")
        self.assertEqual(
            baseline.ephemeral_environment_keys,
            ("CODEX_HOME", "HOME"),
        )
        self.assertNotEqual(baseline.command_sha256, alternate.command_sha256)
        self.assertNotEqual(
            assistant_bridge_module._command_authority_sha256(baseline),
            assistant_bridge_module._command_authority_sha256(alternate),
        )

    def test_command_plan_payload_hides_executable_path_and_version_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_marker = "private-version-material"
            executable = Path(tmp) / (
                f"tool-{version_marker}.exe"
                if os.name == "nt"
                else f"tool-{version_marker}"
            )
            if os.name == "nt":
                shutil.copy2(sys.executable, executable)
            else:
                executable.write_text(
                    f"#!/bin/sh\necho '{version_marker}'\n",
                    encoding="utf-8",
                )
                executable.chmod(0o700)
            provider = replace(
                self.config.local,
                executable=str(executable),
                launcher_args=(),
            )
            plan = build_codex_command_plan(
                provider,
                prompt="safe",
                workspace=ROOT,
                runtime_policy=self.config.runtime,
            )

        rendered = json.dumps(plan.payload(), sort_keys=True)
        self.assertNotIn(str(executable), rendered)
        self.assertNotIn(version_marker, rendered)
        self.assertIn(plan.executable_identity.sha256, rendered)

    def test_command_plan_binds_private_executable_identity(self) -> None:
        environment = assistant_bridge_module._sanitized_environment(
            self.config.local.environment_allowlist
        )
        identity = assistant_bridge_module.resolve_executable(
            self.config.local.executable,
            env=environment,
        )
        alternate = replace(identity, requested="alternate-request-name")

        with patch.object(
            assistant_bridge_module.ExecutableIdentity,
            "payload",
            return_value={"public": "constant"},
        ):
            with patch.object(
                assistant_bridge_module,
                "resolve_executable",
                return_value=identity,
            ):
                first = build_codex_command_plan(
                    self.config.local,
                    prompt="bounded",
                    workspace=ROOT,
                )
            with patch.object(
                assistant_bridge_module,
                "resolve_executable",
                return_value=alternate,
            ):
                second = build_codex_command_plan(
                    self.config.local,
                    prompt="bounded",
                    workspace=ROOT,
                )

        self.assertNotEqual(first.command_sha256, second.command_sha256)

    @unittest.skipIf(os.name == "nt", "POSIX env shebang fixture")
    def test_direct_codex_shebang_chain_is_private_and_passed_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            launcher = root / "direct-codex-secret"
            launcher.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "argv = sys.argv[1:]\n"
                "output = Path(argv[argv.index('--output-last-message') + 1])\n"
                "sys.stdin.read()\n"
                "output.write_text('VERIFIED', encoding='utf-8')\n",
                encoding="utf-8",
            )
            launcher.chmod(0o700)
            provider = replace(
                self.config.local,
                executable=str(launcher),
                launcher_args=(),
                launcher_entrypoint="",
                launcher_companions=(),
            )
            output = root / "final.txt"
            plan = build_codex_command_plan(
                provider,
                prompt="private prompt",
                workspace=root,
                output_path=output,
                runtime_policy=self.config.runtime,
            )
            rendered_chain = json.dumps(
                plan.launcher_chain.payload(), sort_keys=True
            )

            with patch.object(
                assistant_bridge_module,
                "execute_process",
                wraps=assistant_bridge_runtime.execute_process,
            ) as execute:
                result = execute_codex_command(
                    plan,
                    prompt="private prompt",
                    output_path=output,
                    timeout_seconds=10,
                )

        passed_chain = execute.call_args.kwargs["launcher_chain"]
        self.assertEqual(result.status, "completed")
        self.assertTrue(execute.call_args.kwargs["policy"].require_launcher_chain)
        self.assertEqual(passed_chain, plan.launcher_chain)
        self.assertEqual(passed_chain.argv, plan.argv[1:])
        self.assertIsNotNone(passed_chain.env_launcher)
        self.assertIsNotNone(passed_chain.interpreter)
        self.assertNotIn(str(root), rendered_chain)
        self.assertNotIn(str(launcher), rendered_chain)
        self.assertNotIn("#!/usr/bin/env python3", rendered_chain)
        self.assertNotIn(os.environ.get("HOME", "<missing-home>"), rendered_chain)
        self.assertNotIn(provider.model, rendered_chain)

    def test_launcher_authority_is_semantic_across_ephemeral_roots_and_homes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            roots = (Path(first_tmp), Path(second_tmp))
            for root in roots:
                _initialize_git(root)
                (root / "launcher.py").write_text(
                    "print('same launcher')\n", encoding="utf-8"
                )
            provider = replace(
                self.config.local,
                executable=sys.executable,
                launcher_args=("launcher.py",),
                launcher_entrypoint="launcher.py",
                launcher_companions=(),
            )
            with patch.dict(os.environ, {"HOME": str(roots[0] / "home-a")}):
                first = build_codex_command_plan(
                    provider,
                    prompt="semantic",
                    workspace=roots[0],
                    output_path=roots[0] / "out.txt",
                    runtime_policy=self.config.runtime,
                    ephemeral_workspace=True,
                )
            with patch.dict(os.environ, {"HOME": str(roots[1] / "home-b")}):
                second = build_codex_command_plan(
                    provider,
                    prompt="semantic",
                    workspace=roots[1],
                    output_path=roots[1] / "out.txt",
                    runtime_policy=self.config.runtime,
                    ephemeral_workspace=True,
                )

            self.assertEqual(
                first.launcher_authority_sha256,
                second.launcher_authority_sha256,
            )
            self.assertNotEqual(
                first.launcher_chain.fingerprint,
                second.launcher_chain.fingerprint,
            )
            (roots[1] / "launcher.py").write_text(
                "print('content drift')\n", encoding="utf-8"
            )
            drifted = build_codex_command_plan(
                provider,
                prompt="semantic",
                workspace=roots[1],
                output_path=roots[1] / "out.txt",
                runtime_policy=self.config.runtime,
                ephemeral_workspace=True,
            )

        self.assertNotEqual(
            first.launcher_authority_sha256,
            drifted.launcher_authority_sha256,
        )

    def test_launcher_chain_policy_and_declarations_fail_closed(self) -> None:
        self.assertTrue(self.config.runtime.require_launcher_chain)
        self.assertTrue(
            self.config.runtime.process_policy().require_launcher_chain
        )
        with self.assertRaisesRegex(AssistantBridgeError, "launcher chains"):
            replace(self.config.runtime, require_launcher_chain=False)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            script = root / "wrapper.py"
            script.write_text("print('not launched')\n", encoding="utf-8")
            undeclared = replace(
                self.config.local,
                executable=sys.executable,
                launcher_args=("wrapper.py",),
                launcher_entrypoint="",
                launcher_companions=(),
            )
            with self.assertRaisesRegex(AssistantBridgeError, "attestation"):
                build_codex_command_plan(
                    undeclared,
                    prompt="strict",
                    workspace=root,
                    runtime_policy=self.config.runtime,
                )

            link = root / "wrapper-link.py"
            link.symlink_to(script.name)
            linked = replace(
                undeclared,
                launcher_args=("wrapper-link.py",),
                launcher_entrypoint="wrapper-link.py",
            )
            with self.assertRaisesRegex(AssistantBridgeError, "attestation"):
                build_codex_command_plan(
                    linked,
                    prompt="strict",
                    workspace=root,
                    runtime_policy=self.config.runtime,
                )

    def test_entrypoint_drift_blocks_before_process_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            script = root / "wrapper.py"
            script.write_text("print('planned')\n", encoding="utf-8")
            provider = replace(
                self.config.local,
                executable=sys.executable,
                launcher_args=("wrapper.py",),
                launcher_entrypoint="wrapper.py",
                launcher_companions=(),
            )
            output = root / "out.txt"
            plan = build_codex_command_plan(
                provider,
                prompt="strict",
                workspace=root,
                output_path=output,
                runtime_policy=self.config.runtime,
            )
            script.write_text("print('drifted')\n", encoding="utf-8")
            reservations = 0

            def reserve():
                nonlocal reservations
                reservations += 1
                return None

            result = execute_codex_command(
                plan,
                prompt="strict",
                output_path=output,
                timeout_seconds=10,
                reserve_launch=reserve,
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "runtime_attestation_failed")
        self.assertEqual(reservations, 0)

    def test_premium_auth_staging_is_exclusive_bounded_and_fd_hardened(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "auth.json"
            content = b'{"fixture":"credential-secret"}'
            expected = assistant_bridge_module.PremiumAuthAttestation(
                source_path=str(root / "source-auth.json"),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                content=content,
            )
            real_open = os.open
            real_fchmod = getattr(os, "fchmod", None)
            with patch.object(
                assistant_bridge_module.os,
                "open",
                wraps=real_open,
            ) as opened:
                if real_fchmod is None:
                    staged = assistant_bridge_module._stage_premium_auth(
                        destination,
                        content,
                        expected=expected,
                    )
                    hardened = None
                else:
                    with patch.object(
                        assistant_bridge_module.os,
                        "fchmod",
                        wraps=real_fchmod,
                    ) as hardened:
                        staged = assistant_bridge_module._stage_premium_auth(
                            destination,
                            content,
                            expected=expected,
                        )

            creation = next(
                call
                for call in opened.call_args_list
                if len(call.args) >= 3
                and Path(call.args[0]) == destination
            )
            creation_flags = int(creation.args[1])
            assistant_bridge_module._verify_staged_premium_auth(
                staged,
                expected=expected,
            )

        self.assertTrue(creation_flags & os.O_EXCL)
        if getattr(os, "O_NOFOLLOW", 0):
            self.assertTrue(creation_flags & os.O_NOFOLLOW)
        if hardened is not None:
            hardened.assert_any_call(ANY, 0o600)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(staged.mode), 0o600)
        else:
            self.assertTrue(stat.S_ISREG(staged.mode))
        self.assertFalse(staged.metadata_payload()["hard_containment"])
        self.assertEqual(
            staged.metadata_payload()["verification_scope"],
            "pre_reservation_same_user_change_detection",
        )
        self.assertNotIn("credential-secret", repr(staged))

    @unittest.skipIf(os.name == "nt", "POSIX symlink staging semantics")
    def test_premium_auth_staging_rejects_preexisting_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b'{"fixture":"credential"}'
            source = root / "source.json"
            source.write_bytes(content)
            destination = root / "auth.json"
            destination.symlink_to(source.name)
            expected = assistant_bridge_module.PremiumAuthAttestation(
                source_path=str(source),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                content=content,
            )

            with self.assertRaisesRegex(AssistantBridgeError, "stage"):
                assistant_bridge_module._stage_premium_auth(
                    destination,
                    content,
                    expected=expected,
                )

    def test_provider_config_rejects_pre_confirmation_version_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads(
                (ROOT / "configs" / "assistant-bridge.json").read_text(
                    encoding="utf-8"
                )
            )
            raw["providers"]["local"]["version_args"] = ["--version"]
            path = Path(tmp) / "assistant-bridge.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(AssistantBridgeError, "Unknown provider"):
                load_assistant_bridge_config(path)

    def test_codex_adapter_rejects_unreviewed_extra_args(self) -> None:
        for value in (
            "--dangerously-bypass-approvals-and-sandbox",
            "-moverride",
            "--harmless-looking=value",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(AssistantBridgeError, "authority"),
            ):
                build_codex_command_plan(
                    replace(self.config.local, extra_args=(value,)),
                    prompt="bounded",
                    workspace=ROOT,
                )

    def test_check_contract_rejects_vacuous_or_unknown_shapes(self) -> None:
        for check in (
            {"id": "empty", "type": "contains_all", "values": []},
            {"id": "groups", "type": "contains_all_groups", "groups": []},
            {"id": "unknown", "type": "nonempty", "raw": "leak"},
        ):
            with self.subTest(check=check), tempfile.TemporaryDirectory() as tmp:
                raw = json.loads(
                    (ROOT / "configs" / "assistant-bridge.json").read_text(
                        encoding="utf-8"
                    )
                )
                raw["verification"]["output_checks"] = [check]
                raw["state"]["budget_ledger_path"] = str(Path(tmp) / "ledger.json")
                path = Path(tmp) / "bridge.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(AssistantBridgeError):
                    load_assistant_bridge_config(path)

    def test_contracts_are_deeply_immutable(self) -> None:
        task = build_assistant_task("Read status.")
        receipt = plan_assistant_route(task, self.config, workspace=ROOT)

        with self.assertRaises(TypeError):
            self.config.profiles["offline"] = self.config.profiles["balanced"]  # type: ignore[index]
        with self.assertRaises(TypeError):
            self.config.verification_checks[0]["type"] = "contains_all"  # type: ignore[index]
        with self.assertRaises(TypeError):
            receipt.task["profile"] = "quality"  # type: ignore[index]

    def test_verifier_executes_the_exact_environment_bound_to_its_plan(self) -> None:
        capability = assistant_bridge_module.verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            verifier = root / "environment_verifier.py"
            verifier.write_text(
                "import os\n"
                "raise SystemExit(\n"
                "    os.environ.get('BRIDGE_TEST_MARKER') != 'first'\n"
                ")\n",
                encoding="utf-8",
            )
            spec = assistant_bridge_module.CommandVerifierSpec(
                id="environment-binding",
                argv=(sys.executable, "{workspace}/environment_verifier.py"),
                launcher_entrypoint="{workspace}/environment_verifier.py",
                timeout_seconds=10,
                environment_allowlist=("BRIDGE_TEST_MARKER",),
            )
            task = build_assistant_task("Verify a bounded environment.")
            workspace = attest_workspace(root)
            real_build = assistant_bridge_module._build_verifier_plan

            with patch.dict(os.environ, {"BRIDGE_TEST_MARKER": "first"}):
                plan = real_build(
                    spec,
                    workspace=root,
                    runtime_policy=self.config.runtime,
                )

                def drift_after_binding(*args, **kwargs):
                    current = real_build(*args, **kwargs)
                    os.environ["BRIDGE_TEST_MARKER"] = "second"
                    return current

                with patch.object(
                    assistant_bridge_module,
                    "_build_verifier_plan",
                    side_effect=drift_after_binding,
                ), patch.object(
                    assistant_bridge_module,
                    "execute_process",
                    wraps=assistant_bridge_runtime.execute_process,
                ) as execute:
                    evidence = assistant_bridge_module._run_bound_verifier(
                        plan,
                        task=task,
                        workspace=workspace,
                        verifier_workspace=root,
                    )

        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.code, "command_passed")
        self.assertTrue(execute.call_args.kwargs["policy"].require_launcher_chain)
        self.assertIsNotNone(execute.call_args.kwargs["launcher_chain"])

    def test_verifier_rebuilds_exact_resource_chain_in_disposable_workspace(self) -> None:
        capability = assistant_bridge_module.verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            roots = (Path(first_tmp), Path(second_tmp))
            for root in roots:
                _initialize_git(root)
                (root / "verifier.py").write_text(
                    "print('verified')\n", encoding="utf-8"
                )
            spec = assistant_bridge_module.CommandVerifierSpec(
                id="disposable-chain",
                argv=(sys.executable, "{workspace}/verifier.py"),
                launcher_entrypoint="{workspace}/verifier.py",
                timeout_seconds=10,
            )
            plan = assistant_bridge_module._build_verifier_plan(
                spec,
                workspace=roots[0],
                runtime_policy=self.config.runtime,
            )
            with patch.object(
                assistant_bridge_module,
                "execute_process",
                wraps=assistant_bridge_runtime.execute_process,
            ) as execute:
                evidence = assistant_bridge_module._run_bound_verifier(
                    plan,
                    task=build_assistant_task("Verify disposable copy."),
                    workspace=attest_workspace(roots[0]),
                    verifier_workspace=roots[1],
                )

        current = execute.call_args.kwargs["launcher_chain"]
        self.assertTrue(evidence.passed)
        self.assertEqual(current.cwd, str(roots[1].resolve()))
        self.assertIsNotNone(plan.resources)
        assert plan.resources is not None
        self.assertIsNotNone(plan.resources.launcher_chain)
        assert plan.resources.launcher_chain is not None
        self.assertNotEqual(
            current.fingerprint,
            plan.resources.launcher_chain.fingerprint,
        )
        self.assertEqual(
            current.argv,
            execute.call_args.args[1],
        )
        self.assertEqual(
            current.environment.sha256,
            assistant_bridge_runtime.fingerprint_environment(
                execute.call_args.kwargs["env"]
            ).sha256,
        )
        self.assertEqual(plan.resources.wall_time_milliseconds, 10_000)

    @unittest.skipIf(
        os.name == "nt",
        "Windows has no runnable hard-sandbox verifier backend",
    )
    def test_verifier_propagates_unverified_process_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            (root / "verifier.py").write_text(
                "print('must be reaped')\n", encoding="utf-8"
            )
            spec = assistant_bridge_module.CommandVerifierSpec(
                id="cleanup-boundary",
                argv=(sys.executable, "{workspace}/verifier.py"),
                launcher_entrypoint="{workspace}/verifier.py",
                timeout_seconds=10,
            )
            plan = assistant_bridge_module._build_verifier_plan(
                spec,
                workspace=root,
                runtime_policy=self.config.runtime,
            )
            cleanup = assistant_bridge_module.ProcessCleanupError(
                "cleanup could not be verified",
                details={"verified": False},
            )

            with patch.object(
                assistant_bridge_module,
                "execute_process",
                side_effect=cleanup,
            ), self.assertRaises(assistant_bridge_module.ProcessCleanupError) as raised:
                assistant_bridge_module._run_bound_verifier(
                    plan,
                    task=build_assistant_task("Verify cleanup propagation."),
                    workspace=attest_workspace(root),
                    verifier_workspace=root,
                )

        self.assertIs(raised.exception, cleanup)

    def test_provider_and_verifier_deny_environment_injection_variables(
        self,
    ) -> None:
        for name in (
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "GIT_CONFIG_COUNT",
            "NODE_OPTIONS",
            "PYTHONPATH",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AssistantBridgeError, "denied injection"):
                    replace(self.config.local, environment_allowlist=(name,))
                with self.assertRaisesRegex(AssistantBridgeError, "denied injection"):
                    replace(
                        self.config.command_verifiers[0],
                        environment_allowlist=(name,),
                    )

    def test_capsule_redacts_multiform_secrets_and_repr_is_safe(self) -> None:
        task = build_assistant_task(
            'Fix {"api_key":"json-secret"} password: two secret words',
            constraints=(
                "Bearer bearer-secret",
                "AWS_SECRET_ACCESS_KEY=aws secret material",
                "https://user:url-password@example.test/path?token=query-secret",
            ),
            allow_remote=True,
        )
        receipt = plan_assistant_route(task, self.config, workspace=ROOT)
        capsule = build_escalation_capsule(
            task,
            receipt,
            (),
            self.config.capsule,
            failure_codes=("verification-failed",),
            diff_text="+Authorization: Basic dXNlcjpwYXNz\n+safe=true",
        )
        rendered = json.dumps(capsule.payload(), sort_keys=True)

        for secret in (
            "json-secret",
            "two secret words",
            "bearer-secret",
            "aws secret material",
            "url-password",
            "query-secret",
            "dXNlcjpwYXNz",
        ):
            self.assertNotIn(secret, rendered)
            self.assertNotIn(secret, repr(task))
            self.assertNotIn(secret, repr(capsule))
        self.assertIn("[redacted]", rendered)

    def test_capsule_recursively_redacts_capabilities_and_mapping_keys(self) -> None:
        credential = "AKIAIOSFODNN7EXAMPLE"
        task = build_assistant_task(
            "Review the requested capability.",
            required_capabilities=(credential,),
            allow_remote=True,
        )
        receipt = plan_assistant_route(task, self.config, workspace=ROOT)
        evidence = VerificationEvidence(
            id="fixture-verifier",
            verifier="fixture-verifier",
            kind="external",
            passed=False,
            code=credential,
            artifact_sha256="a" * 64,
            task_fingerprint=task.task_fingerprint,
            workspace_fingerprint=receipt.workspace.fingerprint,
            verifier_spec_sha256="b" * 64,
        )
        capsule = build_escalation_capsule(
            task,
            receipt,
            (evidence,),
            self.config.capsule,
            failure_codes=("capability_gap",),
        )
        rendered = json.dumps(
            {
                "capsule": capsule.payload(),
                "metadata": capsule.metadata_payload(),
            },
            sort_keys=True,
        )

        self.assertNotIn(credential, rendered)
        self.assertNotIn(credential, repr(capsule))
        self.assertIn("[redacted]", rendered)
        with self.assertRaisesRegex(AssistantBridgeError, "mapping key"):
            assistant_bridge_module._redact_public_capsule_payload(
                {"nested": {credential: "safe"}},
                self.config.capsule.secret_redaction,
            )

    def test_capsule_refuses_to_drop_critical_objective_or_constraints(self) -> None:
        task = build_assistant_task(
            "x" * 3100,
            allow_remote=True,
        )
        receipt = plan_assistant_route(task, self.config, workspace=ROOT)
        with self.assertRaisesRegex(AssistantBridgeError, "represented safely"):
            build_escalation_capsule(
                task,
                receipt,
                (),
                self.config.capsule,
                failure_codes=("failed",),
            )

    def test_capsule_output_uses_exclusive_atomic_peer_and_syncs_directory(
        self,
    ) -> None:
        task = build_assistant_task("Persist bounded evidence.", allow_remote=True)
        receipt = plan_assistant_route(task, self.config, workspace=ROOT)
        capsule = build_escalation_capsule(
            task,
            receipt,
            (),
            self.config.capsule,
            failure_codes=("verification_failed",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "capsule.json"
            predictable = Path(tmp) / "capsule.json.tmp"
            predictable.write_text("must-survive", encoding="utf-8")
            with patch.object(
                assistant_bridge_module.os,
                "fsync",
                wraps=assistant_bridge_module.os.fsync,
            ) as fsync:
                AssistantBridgeRunner._write_capsule(capsule, target)
            rendered = json.loads(target.read_text(encoding="utf-8"))
            leftovers = sorted(
                item.name
                for item in Path(tmp).iterdir()
                if item.name not in {"capsule.json", "capsule.json.tmp"}
            )
            predictable_content = predictable.read_text(encoding="utf-8")

        self.assertEqual(rendered["capsule_id"], capsule.capsule_id)
        self.assertEqual(predictable_content, "must-survive")
        self.assertEqual(leftovers, [])
        self.assertGreaterEqual(fsync.call_count, 1 if os.name == "nt" else 2)

    def test_capsule_output_rejects_target_and_parent_links(self) -> None:
        task = build_assistant_task("Persist bounded evidence.", allow_remote=True)
        receipt = plan_assistant_route(task, self.config, workspace=ROOT)
        capsule = build_escalation_capsule(
            task,
            receipt,
            (),
            self.config.capsule,
            failure_codes=("verification_failed",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_target = root / "real.json"
            real_target.write_text("do-not-overwrite", encoding="utf-8")
            target_link = root / "target-link.json"
            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_link = root / "parent-link"
            try:
                target_link.symlink_to(real_target)
                parent_link.symlink_to(real_parent, target_is_directory=True)
            except OSError as exc:  # pragma: no cover - host policy dependent.
                self.skipTest(f"symbolic links unavailable: {exc}")

            with self.assertRaisesRegex(AssistantBridgeError, "link|reparse"):
                AssistantBridgeRunner._write_capsule(capsule, target_link)
            with self.assertRaisesRegex(AssistantBridgeError, "link|reparse"):
                AssistantBridgeRunner._write_capsule(
                    capsule,
                    parent_link / "capsule.json",
                )

            preserved = real_target.read_text(encoding="utf-8")
            parent_contents = list(real_parent.iterdir())

        self.assertEqual(preserved, "do-not-overwrite")
        self.assertEqual(parent_contents, [])

    def test_evidence_file_rejects_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "checks": [
                            {
                                "id": "test",
                                "passed": False,
                                "code": "failed",
                                "raw_output": "must not enter metadata",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssistantBridgeError, "Unknown verification"):
                load_verification_evidence(path)

    def test_workspace_attestation_reports_exact_snapshot_telemetry(
        self,
    ) -> None:
        with _git_workspace() as workspace:
            tracked = workspace / "tracked.txt"
            tracked.write_text("staged\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=workspace, check=True)
            tracked.write_text("staged\nunstaged\n", encoding="utf-8")
            untracked = workspace / "untracked.txt"
            untracked.write_text("first\n", encoding="utf-8")
            first = attest_workspace(workspace)
            first_snapshot = snapshot_workspace(workspace, WorkspaceScopePolicy())
            untracked.write_text("second\n", encoding="utf-8")
            second = attest_workspace(workspace)
            second_snapshot = snapshot_workspace(workspace, WorkspaceScopePolicy())

        self.assertEqual(
            first.fingerprint,
            assistant_bridge_module._receipt_workspace_attestation(
                first_snapshot
            ).fingerprint,
        )
        self.assertEqual(first.index_sha256, first_snapshot.index_sha256)
        self.assertEqual(first.status_sha256, first_snapshot.status_sha256)
        self.assertEqual(first.manifest_sha256, first_snapshot.manifest_sha256)
        self.assertEqual(first.file_count, len(first_snapshot.files))
        self.assertEqual(first.total_bytes, first_snapshot.total_bytes)
        self.assertEqual(
            second.fingerprint,
            assistant_bridge_module._receipt_workspace_attestation(
                second_snapshot
            ).fingerprint,
        )
        self.assertNotEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_git_evidence_ignores_ambient_and_repository_execution_hooks(
        self,
    ) -> None:
        if os.name == "nt":
            self.skipTest("executable marker fixture uses POSIX scripts")
        with _git_workspace() as workspace, tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper_marker = root / "helper-ran"
            helper = root / "hostile-helper"
            helper.write_text(
                f"#!/bin/sh\n: > '{helper_marker}'\ncat\n",
                encoding="utf-8",
            )
            helper.chmod(0o700)
            (workspace / ".gitattributes").write_text(
                "tracked.txt diff=hostile\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "config", "core.fsmonitor", str(helper)],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "config", "diff.hostile.textconv", str(helper)],
                cwd=workspace,
                check=True,
            )
            (workspace / "tracked.txt").write_text("changed\n", encoding="utf-8")

            ambient_marker = root / "ambient-git-ran"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_git = fake_bin / "git"
            fake_git.write_text(
                f"#!/bin/sh\n: > '{ambient_marker}'\nexit 99\n",
                encoding="utf-8",
            )
            fake_git.chmod(0o700)
            poisoned = {
                "PATH": str(fake_bin),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.fsmonitor",
                "GIT_CONFIG_VALUE_0": str(helper),
            }
            with patch.dict(os.environ, poisoned, clear=False):
                evidence = assistant_bridge_module.collect_git_evidence(
                    workspace,
                    self.config.capsule,
                    include_excerpt=True,
                )
            self.assertTrue(evidence.sha256)
            self.assertFalse(helper_marker.exists())
            self.assertFalse(ambient_marker.exists())

    def test_git_evidence_fails_closed_when_workspace_changes_mid_collection(
        self,
    ) -> None:
        with _git_workspace() as workspace:
            expected = snapshot_workspace(workspace, WorkspaceScopePolicy())
            real_factory = assistant_bridge_module.trusted_git_session

            def drifting_factory(root: Path):
                session = real_factory(root)
                calls = 0

                class DriftingSession:
                    def run(self, method, *, max_output_bytes):
                        nonlocal calls
                        result = method(max_output_bytes=max_output_bytes)
                        calls += 1
                        if calls == 2:
                            (workspace / "tracked.txt").write_text(
                                "concurrent\n",
                                encoding="utf-8",
                            )
                        return result

                    def staged_diff(self, *, max_output_bytes):
                        return self.run(
                            session.staged_diff,
                            max_output_bytes=max_output_bytes,
                        )

                    def unstaged_diff(self, *, max_output_bytes):
                        return self.run(
                            session.unstaged_diff,
                            max_output_bytes=max_output_bytes,
                        )

                    def untracked_paths(self, *, max_output_bytes):
                        return self.run(
                            session.untracked_paths,
                            max_output_bytes=max_output_bytes,
                        )

                return DriftingSession()

            with (
                patch.object(
                    assistant_bridge_module,
                    "trusted_git_session",
                    side_effect=drifting_factory,
                ),
                self.assertRaisesRegex(AssistantBridgeError, "changed during"),
            ):
                assistant_bridge_module.collect_git_evidence(
                    workspace,
                    self.config.capsule,
                    include_excerpt=False,
                    expected_snapshot=expected,
                )


class AssistantBridgeExecutionTests(unittest.TestCase):
    def test_plan_and_route_inspection_never_execute_bound_binaries(self) -> None:
        with _fake_bridge(process_proxy=True) as fixture:
            task = build_assistant_task(
                "Inspect the confirmed process boundary.",
                profile="quality",
            )

            fixture.runner.plan(task, workspace=fixture.root)
            fixture.runner.inspect_route(task, workspace=fixture.root)
            process_started = (fixture.root / "process-probe.log").exists()

        self.assertFalse(process_started)

    def test_write_capability_preflight_blocks_before_any_bound_process(self) -> None:
        unavailable = WorkspaceWriteCapability(
            supported=False,
            backend="unsupported",
            reason="native atomic no-replace primitive unavailable",
        )
        with (
            _fake_bridge(process_proxy=True) as fixture,
            _fake_environment(fixture),
            patch.object(
                assistant_bridge_module,
                "workspace_write_capability",
                return_value=unavailable,
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
            )
            plan = fixture.runner.plan(task, workspace=fixture.root)
            result = fixture.runner.run(
                task,
                workspace=fixture.root,
                confirmation=str(plan["confirmation_id"]),
            )
            invocations = _read_jsonl(fixture.log)
            process_started = (fixture.root / "process-probe.log").exists()

        receipt = plan["route_receipt"]
        self.assertEqual(receipt["route"], "blocked")
        self.assertIn(
            "workspace_write_capability_unavailable",
            receipt["rationale_codes"],
        )
        self.assertEqual(
            plan["authority"]["workspace_write_capability"],
            unavailable.payload(),
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertEqual(invocations, [])
        self.assertFalse(process_started)

    def test_unsupported_write_capability_does_not_block_read_only_route(self) -> None:
        unavailable = WorkspaceWriteCapability(
            supported=False,
            backend="unsupported",
            reason="native atomic no-replace primitive unavailable",
        )
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture),
            patch.object(
                assistant_bridge_module,
                "workspace_write_capability",
                return_value=unavailable,
            ),
        ):
            task = build_assistant_task("Inspect only.", profile="offline")
            result = fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(result.status, "completed")
        self.assertEqual([item["mode"] for item in invocations], ["local"])

    def test_write_capability_descriptor_is_bound_to_confirmation(self) -> None:
        planned = WorkspaceWriteCapability(True, "test-backend-a")
        changed = WorkspaceWriteCapability(True, "test-backend-b")
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture),
            patch.object(
                assistant_bridge_module,
                "workspace_write_capability",
                side_effect=(planned, changed),
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            plan = fixture.runner.plan(task, workspace=fixture.root)
            with self.assertRaisesRegex(AssistantBridgeError, "confirmation"):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                )
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(
            plan["authority"]["workspace_write_capability"],
            planned.payload(),
        )
        self.assertEqual(invocations, [])

    def test_legacy_run_blocks_independent_write_before_provider(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="verified-candidate\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
            )
            plan = fixture.runner.plan(task, workspace=fixture.root)
            result = fixture.runner.run(
                task,
                workspace=fixture.root,
                confirmation=str(plan["confirmation_id"]),
            )
            source = (fixture.root / "tracked.txt").read_text(encoding="utf-8")
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(plan["route_receipt"]["route"], "blocked")
        self.assertIn(
            "two_phase_required",
            plan["route_receipt"]["rationale_codes"],
        )
        self.assertEqual(result.status, "blocked")
        self.assertEqual(source, "initial\n")
        self.assertEqual(invocations, [])

    def test_candidate_command_task_has_no_apply_authority(self) -> None:
        with _fake_bridge() as fixture:
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
            )
            workspace = attest_workspace(fixture.root)
            candidate_evidence = VerificationEvidence(
                id="fixture-task-verifier",
                verifier="command-task",
                kind="command",
                passed=True,
                code="command_passed",
                artifact_sha256="a" * 64,
                task_fingerprint=task.task_fingerprint,
                workspace_fingerprint=workspace.fingerprint,
                verifier_spec_sha256="b" * 64,
            )

            evidence = fixture.runner._enforce_completion_contract(
                task,
                (candidate_evidence,),
                workspace,
                change_count=1,
            )

        self.assertTrue(
            any(item.code == "two_phase_required" for item in evidence)
        )

    def test_two_phase_write_never_launches_candidate_verifier(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                verifier_exit=1,
                write_relative="tracked.txt",
                write_content="must-not-apply\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
            )
            result = fixture.run(task)
            source = (fixture.root / "tracked.txt").read_text(encoding="utf-8")
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertEqual(source, "initial\n")
        self.assertEqual(invocations, [])

    @unittest.skipIf(
        os.name == "nt",
        "Windows has no runnable hard-sandbox verifier backend",
    )
    def test_verifier_cleanup_failure_prevents_escalation_and_budget(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task(
                "Verify locally before any escalation.",
                required_verifier_ids=("fixture-task-verifier",),
                allow_remote=True,
            )
            real_execute = assistant_bridge_runtime.execute_process

            def fail_verifier_cleanup(executable, args=(), **kwargs):
                if any("fixture_task_verifier.py" in item for item in args):
                    raise assistant_bridge_module.ProcessCleanupError(
                        "verifier cleanup failed",
                        details={"verified": False},
                    )
                return real_execute(executable, args, **kwargs)

            with (
                patch.object(
                    assistant_bridge_module,
                    "execute_process",
                    side_effect=fail_verifier_cleanup,
                ),
                patch.object(
                    fixture.runner,
                    "_reserve_budget",
                    wraps=fixture.runner._reserve_budget,
                ) as reserve,
                self.assertRaises(assistant_bridge_module.ProcessCleanupError),
            ):
                fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(reserve.call_count, 0)
        self.assertEqual([item["mode"] for item in invocations], ["local"])

    def test_two_phase_write_does_not_use_candidate_delta_as_authority(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="candidate-visible\n",
                verifier_expected_content="different-candidate\n",
            ),
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
            )
            result = fixture.run(task)
            source = (fixture.root / "tracked.txt").read_text(encoding="utf-8")
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertEqual(source, "initial\n")
        self.assertEqual(invocations, [])

    def test_read_only_mutation_is_rejected_and_discarded(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture, write_relative="forbidden.txt"),
        ):
            task = build_assistant_task("Inspect only.", profile="offline")
            result = fixture.run(task)
            leaked = (fixture.root / "forbidden.txt").exists()

        self.assertEqual(result.code, "workspace_authority_violated")
        self.assertFalse(leaked)

    def test_legacy_write_never_reaches_the_apply_hook(self) -> None:
        real_apply = assistant_bridge_module.apply_changeset

        def drift_then_apply(**kwargs):
            source = Path(kwargs["source_snapshot"].root)
            (source / "tracked.txt").write_text(
                "concurrent-source-change\n", encoding="utf-8"
            )
            return real_apply(**kwargs)

        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="candidate-must-not-win\n",
            ),
            patch.object(
                assistant_bridge_module,
                "apply_changeset",
                side_effect=drift_then_apply,
            ) as apply,
        ):
            task = build_assistant_task(
                "Change the tracked file.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
            )
            result = fixture.run(task)
            source = (fixture.root / "tracked.txt").read_text(encoding="utf-8")

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertEqual(source, "initial\n")
        apply.assert_not_called()

    def test_confirmation_ticket_is_one_shot(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Inspect only.", profile="offline")
            plan = fixture.runner.plan(task, workspace=fixture.root)
            token = str(plan["confirmation_id"])
            first = fixture.runner.run(
                task, workspace=fixture.root, confirmation=token
            )
            with self.assertRaisesRegex(AssistantBridgeError, "consumed"):
                fixture.runner.run(task, workspace=fixture.root, confirmation=token)

        self.assertEqual(first.status, "completed")

    def test_auth_unavailable_downgrades_balanced_but_blocks_quality(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            auth = Path(os.environ["CODEX_HOME"]) / "auth.json"
            auth.unlink()
            balanced = fixture.runner.plan(
                build_assistant_task("Use local first."), workspace=fixture.root
            )
            quality = fixture.runner.plan(
                build_assistant_task("Use premium.", profile="quality"),
                workspace=fixture.root,
            )

        self.assertEqual(balanced["route_receipt"]["route"], "local")
        self.assertIn(
            "premium_auth_unavailable",
            balanced["route_receipt"]["rationale_codes"],
        )
        self.assertEqual(quality["route_receipt"]["route"], "blocked")

    def test_injected_registry_dispatches_the_complete_provider_lifecycle(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            delegate = default_provider_adapter_registry().require("codex_cli")
            recording = _RecordingProviderAdapter(delegate)
            fixture.runner = AssistantBridgeRunner.with_provider_adapters(
                fixture.config,
                adapter_registry=ProviderAdapterRegistry((recording,)),
                state_ledger=fixture.runner.state_ledger,
            )

            local = fixture.run(build_assistant_task("Stay local.", profile="offline"))
            premium = fixture.run(
                build_assistant_task("Use premium.", profile="quality")
            )

        self.assertEqual(local.status, "completed")
        self.assertEqual(premium.status, "completed")
        self.assertGreaterEqual(recording.calls["validate_provider"], 2)
        self.assertGreaterEqual(recording.calls["runtime_descriptor"], 4)
        self.assertEqual(recording.calls["build_command_plan"], 6)
        self.assertEqual(recording.calls["attest_remote_binding"], 2)
        self.assertEqual(recording.calls["execute_command"], 2)

    def test_unknown_task_verifier_is_rejected_before_launch(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task(
                "Change code.",
                profile="offline",
                risk_class="write_local",
                required_verifier_ids=("unknown-verifier",),
            )
            with self.assertRaisesRegex(AssistantBridgeError, "Unknown required"):
                fixture.runner.plan(task, workspace=fixture.root)

        self.assertEqual(_read_jsonl(fixture.log), [])

    def test_node_like_write_without_independent_evidence_is_blocked(
        self,
    ) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture, write_relative="src/index.js"),
        ):
            (fixture.root / "package.json").write_text(
                '{"scripts":{"test":"node --test"}}', encoding="utf-8"
            )
            subprocess.run(
                ["git", "add", "package.json"], cwd=fixture.root, check=True
            )
            subprocess.run(
                ["git", "commit", "-q", "-m", "node fixture"],
                cwd=fixture.root,
                check=True,
            )
            task = build_assistant_task(
                "Change the Node source.",
                profile="offline",
                required_capabilities=("code",),
                risk_class="write_local",
            )
            result = fixture.run(task)
            candidate_exists = (fixture.root / "src" / "index.js").exists()
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertFalse(candidate_exists)
        self.assertEqual(invocations, [])

    def test_quality_profile_executes_the_exact_confirmed_premium_plan(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            result = fixture.run(
                build_assistant_task("Use the quality tier.", profile="quality")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_provider, "codex-premium")
        self.assertEqual(result.premium_calls_used, 1)

    def test_premium_preview_preserves_external_launcher_under_temp_root(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task(
                "Use the quality tier.",
                profile="quality",
            )
            receipt = fixture.runner.inspect_route(task, workspace=fixture.root)
            launcher = Path(
                fixture.config.premium.launcher_args[0]
            ).resolve(strict=True)
            provider = replace(
                fixture.config.premium,
                launcher_args=(str(launcher),),
                launcher_entrypoint=str(launcher),
            )
            adapter = fixture.runner._provider_adapter(provider)
            prompt = "semantic premium preview"
            preview = assistant_bridge_module._premium_preview_plan(
                adapter,
                provider,
                task,
                prompt,
                receipt,
                workspace=fixture.root,
                runtime_policy=fixture.config.runtime,
            )
            with tempfile.TemporaryDirectory(
                prefix="mymoe-capsule-regression-"
            ) as tmp:
                actual = adapter.build_command_plan(
                    provider,
                    prompt=prompt,
                    workspace=tmp,
                    demand=task.capability_demand,
                    output_path=Path(tmp) / "premium-final.txt",
                    workspace_access=str(
                        receipt.premium_runtime["workspace_access"]
                    ),
                    runtime_policy=fixture.config.runtime,
                    ephemeral_workspace=True,
                )

        self.assertEqual(
            preview.launcher_authority_sha256,
            actual.launcher_authority_sha256,
        )
        self.assertEqual(preview.command_sha256, actual.command_sha256)

    def test_direct_premium_failure_codes_match_the_confirmed_preview(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task(
                "Use premium with prior evidence.",
                profile="quality",
            )
            external = (_external_evidence(task, fixture.root, passed=False),)
            capsule_path = fixture.root / "capsule.json"

            result = fixture.run(
                task,
                external_evidence=external,
                capsule_out=capsule_path,
            )
            capsule = json.loads(capsule_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            capsule["failure_codes"],
            ["policy_selected_premium", "tests-failed"],
        )

    def test_direct_premium_write_requires_the_two_phase_path(
        self,
    ) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                write_relative="tracked.txt",
                write_content="premium-candidate\n",
            ),
        ):
            task = build_assistant_task(
                "Apply a verified premium change.",
                profile="quality",
                required_capabilities=("code",),
                risk_class="write_local",
                required_verifier_ids=("fixture-task-verifier",),
                allow_remote=True,
                allow_remote_workspace=True,
            )
            prior = (_external_evidence(task, fixture.root, passed=False),)
            result = fixture.run(task, external_evidence=prior)
            source = (fixture.root / "tracked.txt").read_text(encoding="utf-8")
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertEqual(source, "initial\n")
        self.assertEqual(invocations, [])
        self.assertEqual(
            [item.code for item in result.prior_verification],
            ["tests-failed"],
        )

    def test_premium_budget_is_reserved_once_and_never_for_local(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            with patch.object(
                fixture.runner,
                "_reserve_budget",
                wraps=fixture.runner._reserve_budget,
            ) as reserve:
                premium = fixture.run(
                    build_assistant_task("Use premium.", profile="quality")
                )
            premium_reservations = reserve.call_count

        with _fake_bridge() as fixture, _fake_environment(fixture):
            with patch.object(
                fixture.runner,
                "_reserve_budget",
                wraps=fixture.runner._reserve_budget,
            ) as reserve:
                local = fixture.run(
                    build_assistant_task("Stay local.", profile="offline")
                )
            local_reservations = reserve.call_count

        self.assertEqual(premium.status, "completed")
        self.assertEqual(premium_reservations, 1)
        self.assertEqual(local.status, "completed")
        self.assertEqual(local_reservations, 0)

    def test_premium_plan_mismatch_does_not_consume_budget(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            plan = fixture.runner.plan(task, workspace=fixture.root)
            real_build = assistant_bridge_module.build_codex_command_plan
            calls = 0

            def mismatched_runtime_plan(*args, **kwargs):
                nonlocal calls
                calls += 1
                command = real_build(*args, **kwargs)
                if calls == 2:
                    return replace(command, command_sha256="0" * 64)
                return command

            with (
                patch.object(
                    assistant_bridge_module,
                    "build_codex_command_plan",
                    side_effect=mismatched_runtime_plan,
                ),
                self.assertRaisesRegex(AssistantBridgeError, "confirmed plan"),
            ):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                )

            recovered = fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(recovered.status, "completed")
        self.assertEqual([item["mode"] for item in invocations], ["premium"])

    def test_premium_launcher_drift_is_rejected_before_budget_reservation(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            plan = fixture.runner.plan(task, workspace=fixture.root)
            launcher = Path(fixture.config.premium.launcher_entrypoint)
            launcher.write_text(
                launcher.read_text(encoding="utf-8") + "\n# content drift\n",
                encoding="utf-8",
            )

            with patch.object(
                fixture.runner,
                "_reserve_budget",
                wraps=fixture.runner._reserve_budget,
            ) as reserve, self.assertRaises(AssistantBridgeError):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                )

        self.assertEqual(reserve.call_count, 0)

    def test_premium_auth_mismatch_does_not_consume_budget(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            plan = fixture.runner.plan(task, workspace=fixture.root)
            auth = Path(os.environ["CODEX_HOME"]) / "auth.json"
            original_auth = auth.read_text(encoding="utf-8")
            real_workspace = assistant_bridge_module._premium_workspace

            @contextmanager
            def workspace_with_auth_drift(*args, **kwargs):
                with real_workspace(*args, **kwargs) as premium_workspace:
                    auth.write_text('{"fixture":"changed"}', encoding="utf-8")
                    try:
                        yield premium_workspace
                    finally:
                        auth.write_text(original_auth, encoding="utf-8")

            with (
                patch.object(
                    assistant_bridge_module,
                    "_premium_workspace",
                    workspace_with_auth_drift,
                ),
                self.assertRaisesRegex(AssistantBridgeError, "authentication"),
            ):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                )

            recovered = fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(recovered.status, "completed")
        self.assertEqual([item["mode"] for item in invocations], ["premium"])

    def test_staged_premium_auth_drift_blocks_before_budget_reservation(
        self,
    ) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            real_verify = assistant_bridge_module._verify_staged_premium_auth

            def tamper_before_reservation(staged, *, expected):
                Path(staged.path).write_text(
                    '{"fixture":"staged-drift"}',
                    encoding="utf-8",
                )
                return real_verify(staged, expected=expected)

            with (
                patch.object(
                    assistant_bridge_module,
                    "_verify_staged_premium_auth",
                    side_effect=tamper_before_reservation,
                ),
                patch.object(
                    fixture.runner,
                    "_reserve_budget",
                    wraps=fixture.runner._reserve_budget,
                ) as reserve,
            ):
                result = fixture.run(task)
            launched = fixture.log.exists()

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "premium_runtime_unavailable")
        self.assertEqual(result.premium_calls_used, 0)
        self.assertEqual(reserve.call_count, 0)
        self.assertFalse(launched)

    def test_premium_runtime_preflight_failure_does_not_consume_budget(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            plan = fixture.runner.plan(task, workspace=fixture.root)
            with patch.object(
                assistant_bridge_module,
                "_preflight_process_runtime",
                side_effect=assistant_bridge_module.AssistantBridgeRuntimeError(
                    "runtime unavailable"
                ),
            ):
                blocked = fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                )

            recovered = fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(blocked.code, "premium_runtime_unavailable")
        self.assertEqual(blocked.premium_calls_used, 0)
        self.assertEqual(recovered.status, "completed")
        self.assertEqual([item["mode"] for item in invocations], ["premium"])

    def test_premium_popen_failure_releases_budget_for_a_retry(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            real_popen = assistant_bridge_runtime.subprocess.Popen

            def fail_premium_popen(argv, *args, **kwargs):
                if isinstance(argv, (list, tuple)) and "--strict-config" in argv:
                    raise OSError("fixed launch failure")
                return real_popen(argv, *args, **kwargs)

            with patch.object(
                assistant_bridge_runtime.subprocess,
                "Popen",
                side_effect=fail_premium_popen,
            ):
                blocked = fixture.run(task)

            recovered = fixture.run(task)

        self.assertEqual(blocked.code, "premium_runtime_unavailable")
        self.assertEqual(blocked.premium_calls_used, 0)
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.premium_calls_used, 1)

    def test_premium_commit_failure_is_accounted_and_never_released(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            with (
                patch.object(
                    fixture.runner.state_ledger,
                    "commit_budget",
                    side_effect=assistant_bridge_module.BridgeLedgerError(
                        "sensitive ledger failure"
                    ),
                ),
                patch.object(
                    fixture.runner.state_ledger,
                    "release_budget_after_popen_failure",
                    wraps=fixture.runner.state_ledger.release_budget_after_popen_failure,
                ) as release,
            ):
                blocked = fixture.run(task)

        self.assertEqual(blocked.code, "premium_runtime_unavailable")
        self.assertEqual(blocked.premium_calls_used, 1)
        self.assertEqual(release.call_count, 0)
        self.assertNotIn("sensitive ledger failure", json.dumps(blocked.user_payload()))

    def test_premium_release_failure_keeps_pending_budget_accounted(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Use premium.", profile="quality")
            real_popen = assistant_bridge_runtime.subprocess.Popen

            def fail_premium_popen(argv, *args, **kwargs):
                if isinstance(argv, (list, tuple)) and "--strict-config" in argv:
                    raise OSError("fixed launch failure")
                return real_popen(argv, *args, **kwargs)

            with (
                patch.object(
                    assistant_bridge_runtime.subprocess,
                    "Popen",
                    side_effect=fail_premium_popen,
                ),
                patch.object(
                    fixture.runner.state_ledger,
                    "release_budget_after_popen_failure",
                    side_effect=assistant_bridge_module.BridgeLedgerError(
                        "sensitive release failure"
                    ),
                ),
            ):
                blocked = fixture.run(task)

        self.assertEqual(blocked.code, "premium_runtime_unavailable")
        self.assertEqual(blocked.premium_calls_used, 1)
        self.assertNotIn("sensitive release failure", json.dumps(blocked.user_payload()))

    def test_declared_write_uses_two_phase_even_when_no_delta_is_expected(
        self,
    ) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED local user result",
                premium_output="VERIFIED premium result",
            ),
        ):
            task = build_assistant_task(
                "Make a bounded local change.",
                required_capabilities=("code",),
                risk_class="write_local",
                no_change_expected=True,
                required_verifier_ids=("fixture-task-verifier",),
                allow_remote=True,
                allow_remote_workspace=True,
            )
            result = fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        telemetry = json.dumps(result.metadata_payload(), sort_keys=True)
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertEqual(invocations, [])
        self.assertNotIn("local user result", telemetry)

    def test_failure_language_cannot_false_positive(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="I could not complete this task; tests are failing.",
                premium_output="VERIFIED premium recovery",
            ),
        ):
            task = build_assistant_task(
                "Analyze this locally.",
                profile="offline",
            )
            result = fixture.run(task)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.code, "local_verification_failed_remote_forbidden")
        self.assertTrue(
            any(
                item.id == "assistant-output-honest" and not item.passed
                for item in result.verification
            )
        )

    def test_failed_bound_evidence_escalates_only_a_redacted_capsule(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED LOCAL_RAW_TRANSCRIPT",
                premium_output="VERIFIED premium result",
            ),
        ):
            task = build_assistant_task(
                "Fix api_key=supersecret without loading the original chat.",
                allow_remote=True,
            )
            external = (_external_evidence(task, fixture.root, passed=False),)
            capsule_path = fixture.root / "capsule.json"
            result = fixture.run(
                task,
                external_evidence=external,
                capsule_out=capsule_path,
            )
            invocations = _read_jsonl(fixture.log)
            premium_prompt = str(invocations[1]["stdin"])
            capsule_text = capsule_path.read_text(encoding="utf-8")

        self.assertEqual(result.status, "completed")
        self.assertEqual([item["mode"] for item in invocations], ["local", "premium"])
        self.assertNotIn("LOCAL_RAW_TRANSCRIPT", premium_prompt)
        self.assertNotIn("supersecret", premium_prompt)
        self.assertIn("[redacted]", premium_prompt)
        self.assertNotIn("supersecret", capsule_text)
        premium_cd = _argv_value(invocations[1]["argv"], "--cd")
        self.assertNotEqual(Path(premium_cd).resolve(), fixture.root.resolve())

    def test_external_evidence_is_not_replayable_across_task_or_workspace(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            first_task = build_assistant_task("First task.", allow_remote=True)
            evidence = _external_evidence(first_task, fixture.root, passed=True)
            second_task = build_assistant_task("Second task.", allow_remote=True)

            with self.assertRaisesRegex(AssistantBridgeError, "different task"):
                fixture.run(second_task, external_evidence=(evidence,))

            changed = fixture.root / "changed.txt"
            changed.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(AssistantBridgeError, "workspace state"):
                fixture.run(first_task, external_evidence=(evidence,))

    def test_confirmation_is_bound_to_current_workspace_and_prompt(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            task = build_assistant_task("Inspect the workspace.")
            plan = fixture.runner.plan(task, workspace=fixture.root)

            with self.assertRaisesRegex(AssistantBridgeError, "confirmation"):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                    include_diff=True,
                )

            (fixture.root / "changed.txt").write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(AssistantBridgeError, "confirmation"):
                fixture.runner.run(
                    task,
                    workspace=fixture.root,
                    confirmation=str(plan["confirmation_id"]),
                )

            output = fixture.root / "final.txt"
            command = build_codex_command_plan(
                fixture.config.local,
                prompt="first prompt",
                workspace=fixture.root,
                output_path=output,
            )
            with self.assertRaisesRegex(AssistantBridgeError, "prompt"):
                execute_codex_command(
                    command,
                    prompt="different prompt",
                    output_path=output,
                    timeout_seconds=10,
                )

    def test_missing_or_os_invalid_launcher_fails_closed_without_premium(self) -> None:
        for executable in ("/missing/mymoe-codex", str(ROOT)):
            with (
                self.subTest(executable=executable),
                _fake_bridge(
                    local_executable=executable,
                    local_launcher_args=(),
                ) as fixture,
                _fake_environment(fixture),
                self.assertRaisesRegex(AssistantBridgeError, "attestation"),
            ):
                task = build_assistant_task("Keep this local-first.", allow_remote=True)
                fixture.run(task)

    def test_shell_shaped_objective_is_stdin_only_and_environment_is_sanitized(
        self,
    ) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(fixture),
            patch.dict(
                os.environ,
                {"LEAK_ME": "environment-secret"},
            ),
        ):
            sentinel = fixture.root / "must-not-exist"
            task = build_assistant_task(
                f"Check syntax; touch {sentinel}; $(touch {sentinel})",
                profile="offline",
            )
            result = fixture.run(task)
            invocation = _read_jsonl(fixture.log)[0]

        self.assertEqual(result.status, "completed")
        self.assertFalse(sentinel.exists())
        self.assertNotIn(str(sentinel), invocation["argv"])
        prompt_payload = json.loads(invocation["stdin"].rsplit("\n\n", 1)[-1])
        self.assertIn(str(sentinel), prompt_payload["objective"])
        self.assertIsNone(invocation["leaked_env"])

    def test_durable_budget_survives_runner_recreation(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="I could not complete this task.",
                premium_output="VERIFIED premium recovery",
            ),
        ):
            task = build_assistant_task("Recover if local fails.", allow_remote=True)
            first = fixture.run(task)
            recreated = _FakeBridgeFixture(
                fixture.root,
                fixture.config_path,
                fixture.app_config_path,
                fixture.log,
            )
            second = recreated.run(task)
            renamed = recreated.run(replace(task, task_id="same-work-new-label"))
            invocations = _read_jsonl(fixture.log)

        self.assertEqual(first.premium_calls_used, 1)
        self.assertEqual(second.status, "blocked")
        self.assertEqual(second.code, "durable_premium_budget_exhausted")
        self.assertEqual(renamed.status, "blocked")
        self.assertEqual(renamed.code, "durable_premium_budget_exhausted")
        self.assertEqual([item["mode"] for item in invocations].count("premium"), 1)

    def test_stdout_and_final_output_limits_fail_closed(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED",
                stdout_bytes=2 * 1024 * 1024 + 100_000,
            ),
        ):
            result = fixture.run(
                build_assistant_task("Bound stdout.", profile="offline")
            )
        self.assertEqual(result.status, "failed")
        self.assertTrue(
            any(
                item.code == "launcher_output_limit_exceeded"
                for item in result.verification
            )
        )

        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED",
                output_bytes=1024 * 1024 + 1,
            ),
        ):
            result = fixture.run(
                build_assistant_task("Bound output.", profile="offline")
            )
        self.assertEqual(result.status, "failed")
        self.assertTrue(
            any(
                item.code == "final_output_limit_exceeded"
                for item in result.verification
            )
        )


class AssistantBridgeCliTests(unittest.TestCase):
    def test_app_config_requires_explicit_bridge_execution_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads(
                (ROOT / "configs" / "app.json").read_text(encoding="utf-8")
            )
            raw["permissions"].pop("assistant_bridge_execution_policy")
            path = Path(tmp) / "app.json"
            path.write_text(json.dumps(raw), encoding="utf-8")

            config = load_app_config(path)

        self.assertEqual(
            config.permissions.assistant_bridge_execution_policy,
            "disabled",
        )

    def test_cli_plan_is_metadata_only_and_emits_bound_confirmation(self) -> None:
        with _fake_bridge() as fixture:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--assistant-task-file",
                    "tests/fixtures/assistant-bridge.task.json",
                    "--assistant-bridge-config",
                    str(fixture.config_path),
                    "--assistant-workspace",
                    str(fixture.root),
                    "--app-config",
                    str(fixture.app_config_path),
                    "--json",
                ],
                cwd=ROOT,
                env=_python_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["route_receipt"]["route"], "blocked")
        self.assertIn(
            "two_phase_required",
            payload["route_receipt"]["rationale_codes"],
        )
        self.assertTrue(str(payload["confirmation_id"]).startswith("confirm-v2-"))
        self.assertNotIn("Refactor the parser", completed.stdout)

    def test_cli_execute_requires_exact_receipt_not_boolean(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--assistant-task",
                "Run a task.",
                "--assistant-bridge-execute",
            ],
            cwd=ROOT,
            env=_python_env(),
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires --assistant-confirm-receipt", completed.stderr)

    def test_cli_returns_user_output_and_records_control_plane_metadata(self) -> None:
        with (
            _fake_bridge() as fixture,
            _fake_environment(
                fixture,
                local_output="VERIFIED result visible to user",
            ),
        ):
            plan = _run_cli_plan(fixture, "Run a safe local task.", profile="offline")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--assistant-task",
                    "Run a safe local task.",
                    "--assistant-profile",
                    "offline",
                    "--assistant-bridge-config",
                    str(fixture.config_path),
                    "--assistant-workspace",
                    str(fixture.root),
                    "--app-config",
                    str(fixture.app_config_path),
                    "--assistant-bridge-execute",
                    "--assistant-confirm-receipt",
                    str(plan["confirmation_id"]),
                    "--json",
                ],
                cwd=ROOT,
                env=_python_env(),
                text=True,
                capture_output=True,
                check=True,
            )

            audit = _read_jsonl(fixture.root / "runtime" / "audit.jsonl")
            runs = _read_jsonl(fixture.root / "runtime" / "runs.jsonl")

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["telemetry"]["status"], "completed")
        self.assertEqual(
            payload["result"]["content"], "VERIFIED result visible to user"
        )
        self.assertEqual(audit[-1]["action"], "assistant_bridge.execute")
        self.assertNotIn("Run a safe local task", json.dumps(audit))
        self.assertNotIn("result visible to user", json.dumps(runs))

    def test_cli_app_policy_can_disable_bridge_execution(self) -> None:
        with _fake_bridge(process_proxy=True) as fixture:
            raw = json.loads(fixture.app_config_path.read_text(encoding="utf-8"))
            raw["permissions"]["assistant_bridge_execution_policy"] = "disabled"
            fixture.app_config_path.write_text(json.dumps(raw), encoding="utf-8")
            plan = _run_cli_plan(fixture, "Run a safe task.", profile="offline")
            completed = _run_cli_execution(
                fixture,
                "Run a safe task.",
                plan,
                profile="offline",
            )
            audit = _read_jsonl(fixture.root / "runtime" / "audit.jsonl")
            process_started = (fixture.root / "process-probe.log").exists()

        self.assertEqual(completed.returncode, 2)
        self.assertIn("policy disables", completed.stderr)
        self.assertEqual([event["status"] for event in audit], ["denied"])
        self.assertFalse(process_started)

    def test_cli_local_only_denial_never_executes_bound_binaries(self) -> None:
        with _fake_bridge(process_proxy=True) as fixture:
            raw = json.loads(fixture.app_config_path.read_text(encoding="utf-8"))
            raw["permissions"]["assistant_bridge_execution_policy"] = "local_only"
            fixture.app_config_path.write_text(json.dumps(raw), encoding="utf-8")
            plan = _run_cli_plan(fixture, "Use the quality tier.", profile="quality")

            completed = _run_cli_execution(
                fixture,
                "Use the quality tier.",
                plan,
                profile="quality",
            )
            process_started = (fixture.root / "process-probe.log").exists()

        self.assertEqual(completed.returncode, 2)
        self.assertIn("local_only", completed.stderr)
        self.assertFalse(process_started)

    def test_cli_records_failed_terminal_audit_when_receipt_is_rejected(self) -> None:
        with _fake_bridge() as fixture:
            plan = _run_cli_plan(fixture, "Run a safe task.", profile="offline")
            plan["confirmation_id"] = f"confirm-v2-{'A' * 43}"
            completed = _run_cli_execution(
                fixture,
                "Run a safe task.",
                plan,
                profile="offline",
            )
            audit = _read_jsonl(fixture.root / "runtime" / "audit.jsonl")

        self.assertEqual(completed.returncode, 2)
        self.assertEqual([event["status"] for event in audit], ["started", "failed"])
        self.assertNotIn(str(plan["confirmation_id"]), json.dumps(audit))

    def test_cli_rejects_unrelated_options_instead_of_ignoring_them(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--assistant-task",
                "Run a task.",
                "--chat-title",
                "silently ignored before",
            ],
            cwd=ROOT,
            env=_python_env(),
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot be combined", completed.stderr)


class _FakeBridgeFixture:
    def __init__(
        self,
        root: Path,
        config_path: Path,
        app_config_path: Path,
        log: Path,
    ) -> None:
        self.root = root
        self.config_path = config_path
        self.app_config_path = app_config_path
        self.log = log
        self.config = load_assistant_bridge_config(config_path)
        self.runner = AssistantBridgeRunner(self.config)

    def run(
        self,
        task,
        *,
        external_evidence=(),
        include_diff: bool = False,
        capsule_out: str | Path | None = None,
    ):
        plan = self.runner.plan(
            task,
            workspace=self.root,
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        )
        return self.runner.run(
            task,
            workspace=self.root,
            confirmation=str(plan["confirmation_id"]),
            external_evidence=external_evidence,
            include_diff=include_diff,
            capsule_out=capsule_out,
        )


class _RecordingProviderAdapter:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.adapter_id = delegate.adapter_id
        self.configuration_sha256 = delegate.configuration_sha256
        self.ephemeral_environment_keys = delegate.ephemeral_environment_keys
        self.calls = {
            "validate_provider": 0,
            "runtime_descriptor": 0,
            "build_command_plan": 0,
            "attest_remote_binding": 0,
            "execute_command": 0,
        }

    def validate_provider(self, provider) -> None:
        self.calls["validate_provider"] += 1
        self.delegate.validate_provider(provider)

    def runtime_descriptor(self, provider, task, **kwargs):
        self.calls["runtime_descriptor"] += 1
        return self.delegate.runtime_descriptor(provider, task, **kwargs)

    def build_command_plan(self, provider, **kwargs):
        self.calls["build_command_plan"] += 1
        return self.delegate.build_command_plan(provider, **kwargs)

    def attest_remote_binding(self, provider):
        self.calls["attest_remote_binding"] += 1
        return self.delegate.attest_remote_binding(provider)

    def execute_command(self, provider, plan, **kwargs):
        self.calls["execute_command"] += 1
        return self.delegate.execute_command(provider, plan, **kwargs)


@contextmanager
def _fake_bridge(
    *,
    local_executable: str | None = None,
    local_launcher_args: tuple[str, ...] | None = None,
    process_proxy: bool = False,
):
    with tempfile.TemporaryDirectory() as tmp:
        fixture_root = Path(tmp)
        root = fixture_root / "workspace"
        root.mkdir()
        _initialize_git(root)
        launcher = fixture_root / "fake_codex.py"
        launcher.write_text(
            """from __future__ import annotations
import json
import os
from pathlib import Path
import sys

argv = sys.argv[1:]
prompt = sys.stdin.read()
mode = "local" if "--oss" in argv else "premium"
output_flag = argv.index("--output-last-message")
output_path = Path(argv[output_flag + 1])
content = os.environ.get(
    "FAKE_LOCAL_OUTPUT" if mode == "local" else "FAKE_PREMIUM_OUTPUT",
    "VERIFIED",
)
output_bytes = int(os.environ.get("FAKE_OUTPUT_BYTES", "0"))
if output_bytes:
    content = "V" * output_bytes
output_path.write_text(content, encoding="utf-8")
write_relative = os.environ.get("FAKE_WRITE_RELATIVE")
if write_relative:
    workspace = Path(argv[argv.index("--cd") + 1])
    target = workspace / write_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(os.environ.get("FAKE_WRITE_CONTENT", "candidate-change\\n"), encoding="utf-8")
log_path = Path(os.environ["FAKE_CODEX_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(
        json.dumps(
            {
                "mode": mode,
                "argv": argv,
                "stdin": prompt,
                "leaked_env": os.environ.get("LEAK_ME"),
            }
        )
        + "\\n"
    )
stdout_bytes = int(os.environ.get("FAKE_STDOUT_BYTES", "0"))
if stdout_bytes:
    sys.stdout.write("x" * stdout_bytes)
else:
    print("fake diagnostic output")
raise SystemExit(int(os.environ.get("FAKE_EXIT_CODE", "0")))
""",
            encoding="utf-8",
        )
        process_executable = sys.executable
        proxy_launcher_args: list[str] | None = None
        proxy_launcher_entrypoint = ""
        proxy_launcher_companions: list[str] = []
        if process_proxy:
            process_log = root / "process-probe.log"
            proxy = fixture_root / "python_proxy.py"
            proxy.write_text(
                f"#!{sys.executable}\n"
                "from pathlib import Path\n"
                "import os\n"
                "import sys\n"
                f"with Path({str(process_log)!r}).open('a', encoding='utf-8') as handle:\n"
                "    handle.write('launched\\n')\n"
                "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n",
                encoding="utf-8",
            )
            proxy.chmod(0o700)
            if os.name == "nt":
                proxy_launcher_args = [str(proxy), str(launcher)]
                proxy_launcher_entrypoint = str(proxy)
                proxy_launcher_companions = [str(launcher)]
            else:
                process_executable = str(proxy)
                proxy_launcher_args = [str(launcher)]
                proxy_launcher_companions = [str(launcher)]
        raw = json.loads(
            (ROOT / "configs" / "assistant-bridge.json").read_text(encoding="utf-8")
        )
        for command_verifier in raw["verification"]["command_verifiers"]:
            if command_verifier["id"] == "project-test-suite":
                command_verifier["workspace_python_paths"] = []
        executable = local_executable or process_executable
        local_uses_process_proxy = (
            process_proxy
            and local_executable is None
            and local_launcher_args is None
        )
        launcher_args = (
            list(local_launcher_args)
            if local_launcher_args is not None
            else (
                list(proxy_launcher_args)
                if local_uses_process_proxy and proxy_launcher_args is not None
                else [str(launcher)]
            )
        )
        raw["providers"]["local"]["executable"] = executable
        raw["providers"]["local"]["launcher_args"] = launcher_args
        raw["providers"]["premium"]["executable"] = process_executable
        local_uses_fixture_launcher = local_launcher_args is None
        raw["providers"]["local"]["launcher_entrypoint"] = (
            proxy_launcher_entrypoint
            if local_uses_process_proxy
            else str(launcher)
            if local_uses_fixture_launcher
            else ""
        )
        raw["providers"]["local"]["launcher_companions"] = (
            list(proxy_launcher_companions)
            if local_uses_process_proxy
            else []
        )
        raw["providers"]["premium"]["launcher_args"] = (
            list(proxy_launcher_args)
            if proxy_launcher_args is not None
            else [str(launcher)]
        )
        raw["providers"]["premium"]["launcher_entrypoint"] = (
            proxy_launcher_entrypoint if process_proxy else str(launcher)
        )
        raw["providers"]["premium"]["launcher_companions"] = (
            list(proxy_launcher_companions) if process_proxy else []
        )
        verifier = root / "fixture_task_verifier.py"
        verifier.write_text(
            "from pathlib import Path\n"
            "import os\n"
            "relative = os.environ.get('FAKE_WRITE_RELATIVE', '')\n"
            "expected = os.environ.get('FAKE_VERIFIER_EXPECT_CONTENT', '')\n"
            "delta_visible = (\n"
            "    not relative\n"
            "    or Path(relative).read_text(encoding='utf-8') == expected\n"
            ")\n"
            "raise SystemExit(\n"
            "    int(os.environ.get('FAKE_VERIFIER_EXIT', '0'))\n"
            "    if delta_visible\n"
            "    else 91\n"
            ")\n",
            encoding="utf-8",
        )
        allowed_env = [
            "FAKE_CODEX_LOG",
            "FAKE_EXIT_CODE",
            "FAKE_LOCAL_OUTPUT",
            "FAKE_OUTPUT_BYTES",
            "FAKE_PREMIUM_OUTPUT",
            "FAKE_STDOUT_BYTES",
            "FAKE_VERIFIER_EXIT",
            "FAKE_WRITE_CONTENT",
            "FAKE_WRITE_RELATIVE",
        ]
        raw["providers"]["local"]["environment_allowlist"] = allowed_env
        raw["providers"]["premium"]["environment_allowlist"] = allowed_env
        raw["verification"]["output_checks"] = [
            {
                "id": "verified-marker",
                "type": "contains_all",
                "values": ["VERIFIED"],
            },
            *raw["verification"]["output_checks"][1:],
        ]
        raw["state"]["ledger_path"] = str(root / "runtime" / "bridge-state.json")
        raw["state"]["namespace"] = "fixture-assistant-bridge"
        raw["workspace"]["transaction_state_dir"] = str(
            root / "runtime" / "transactions"
        )
        raw["verification"]["command_verifiers"] = [
            *raw["verification"]["command_verifiers"],
            {
                "id": "fixture-task-verifier",
                "kind": "command",
                "argv": [
                    process_executable,
                    "{workspace}/fixture_task_verifier.py",
                ],
                "launcher_entrypoint": (
                    "" if process_proxy else "{workspace}/fixture_task_verifier.py"
                ),
                "launcher_companions": (
                    ["{workspace}/fixture_task_verifier.py"]
                    if process_proxy
                    else []
                ),
                "runtime_read_roots": ["{python_runtime}"],
                "timeout_seconds": 30,
                "purpose": "task",
                "execution_boundary": "hard_sandbox",
                "network_policy": "denied",
                "environment_allowlist": [
                    "FAKE_VERIFIER_EXIT",
                    "FAKE_VERIFIER_EXPECT_CONTENT",
                    "FAKE_WRITE_CONTENT",
                    "FAKE_WRITE_RELATIVE",
                ],
                "required_for_capabilities": [],
                "required_for_tools": [],
                "required_for_risks": [],
            },
        ]
        config_path = root / "assistant-bridge.json"
        config_path.write_text(json.dumps(raw), encoding="utf-8")

        app_raw = json.loads(
            (ROOT / "configs" / "app.json").read_text(encoding="utf-8")
        )
        app_raw["runtime"]["work_dir"] = str(root / "runtime")
        app_config_path = root / "app.json"
        app_config_path.write_text(json.dumps(app_raw), encoding="utf-8")
        auth_home = root / "runtime" / "codex-home"
        auth_home.mkdir(parents=True)
        (auth_home / "auth.json").write_text(
            '{"fixture":"credential"}', encoding="utf-8"
        )
        with patch.dict(os.environ, {"CODEX_HOME": str(auth_home)}):
            yield _FakeBridgeFixture(
                root,
                config_path,
                app_config_path,
                root / "invocations.jsonl",
            )


@contextmanager
def _fake_environment(
    fixture: _FakeBridgeFixture,
    *,
    local_output: str = "VERIFIED locally",
    premium_output: str = "VERIFIED remotely",
    output_bytes: int = 0,
    stdout_bytes: int = 0,
    verifier_exit: int = 0,
    write_relative: str = "",
    write_content: str = "candidate-change\n",
    verifier_expected_content: str | None = None,
):
    with patch.dict(
        os.environ,
        {
            "FAKE_CODEX_LOG": str(fixture.log),
            "FAKE_LOCAL_OUTPUT": local_output,
            "FAKE_PREMIUM_OUTPUT": premium_output,
            "FAKE_OUTPUT_BYTES": str(output_bytes),
            "FAKE_STDOUT_BYTES": str(stdout_bytes),
            "FAKE_VERIFIER_EXIT": str(verifier_exit),
            "FAKE_VERIFIER_EXPECT_CONTENT": (
                write_content
                if verifier_expected_content is None
                else verifier_expected_content
            ),
            "FAKE_WRITE_RELATIVE": write_relative,
            "FAKE_WRITE_CONTENT": write_content,
            "FAKE_EXIT_CODE": "0",
        },
    ):
        yield


@contextmanager
def _git_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _initialize_git(root)
        yield root


def _initialize_git(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Antonio Antenore"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "ant_ant95@hotmail.it"],
        cwd=root,
        check=True,
    )
    (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "invocations.jsonl\nprocess-probe.log\nruntime/\ncapsule.json\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "tracked.txt", ".gitignore"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test fixture"],
        cwd=root,
        check=True,
    )


def _load_test_assistant_bridge_config():
    config = load_assistant_bridge_config(
        ROOT / "configs" / "assistant-bridge.json"
    )
    executable = str(Path(sys.executable).resolve(strict=True))
    return replace(
        config,
        local=replace(config.local, executable=executable),
        premium=replace(config.premium, executable=executable),
    )


def _external_evidence(task, workspace: Path, *, passed: bool) -> VerificationEvidence:
    workspace_fingerprint = attest_workspace(workspace).fingerprint
    code = "tests-passed" if passed else "tests-failed"
    return VerificationEvidence(
        id="focused-tests",
        verifier="test-runner",
        kind="external",
        passed=passed,
        code=code,
        artifact_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        observed_chars=len(code),
        evidence_ref="artifact://focused-tests",
        task_fingerprint=task.task_fingerprint,
        workspace_fingerprint=workspace_fingerprint,
        verifier_spec_sha256=EXTERNAL_SPEC_SHA256,
    )


def _run_cli_plan(
    fixture: _FakeBridgeFixture,
    objective: str,
    *,
    profile: str,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "local_moe.cli",
            "--assistant-task",
            objective,
            "--assistant-profile",
            profile,
            "--assistant-bridge-config",
            str(fixture.config_path),
            "--assistant-workspace",
            str(fixture.root),
            "--app-config",
            str(fixture.app_config_path),
            "--json",
        ],
        cwd=ROOT,
        env=_python_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _run_cli_execution(
    fixture: _FakeBridgeFixture,
    objective: str,
    plan: dict[str, object],
    *,
    profile: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "local_moe.cli",
            "--assistant-task",
            objective,
            "--assistant-profile",
            profile,
            "--assistant-bridge-config",
            str(fixture.config_path),
            "--assistant-workspace",
            str(fixture.root),
            "--app-config",
            str(fixture.app_config_path),
            "--assistant-bridge-execute",
            "--assistant-confirm-receipt",
            str(plan["confirmation_id"]),
            "--json",
        ],
        cwd=ROOT,
        env=_python_env(),
        text=True,
        capture_output=True,
    )


def _argv_value(raw: object, flag: str) -> str:
    assert isinstance(raw, list)
    index = raw.index(flag)
    return str(raw[index + 1])


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


if __name__ == "__main__":
    unittest.main()
