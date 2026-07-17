from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_moe.assistant_bridge import (
    AssistantBridgeError,
    AssistantBridgeRunner,
    VerificationEvidence,
    attest_workspace,
    build_assistant_task,
    build_codex_command_plan,
    build_escalation_capsule,
    execute_codex_command,
    load_assistant_bridge_config,
    load_assistant_task,
    load_verification_evidence,
    plan_assistant_route,
)
from local_moe.app_config import load_app_config


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SPEC_SHA256 = (
    "bbd7d15a562a8eb4ffc6b52cf6f89855589c5047e89d5d22c23645e912ff1e30"
)


class AssistantBridgeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_assistant_bridge_config(
            ROOT / "configs" / "assistant-bridge.json"
        )

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

    def test_dangerous_extra_args_are_rejected(self) -> None:
        for value in (
            "--dangerously-bypass-approvals-and-sandbox",
            "-moverride",
            "--harmless-looking=value",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(AssistantBridgeError, "authority"),
            ):
                replace(self.config.local, extra_args=(value,))

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

    def test_evidence_file_rejects_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
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

    def test_complete_git_attestation_covers_staged_unstaged_and_untracked(
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
            untracked.write_text("second\n", encoding="utf-8")
            second = attest_workspace(workspace)

        self.assertNotEqual(first.staged_diff_sha256, hashlib.sha256(b"").hexdigest())
        self.assertNotEqual(first.unstaged_diff_sha256, hashlib.sha256(b"").hexdigest())
        self.assertEqual(first.untracked_count, 1)
        self.assertNotEqual(
            first.untracked_manifest_sha256, second.untracked_manifest_sha256
        )
        self.assertNotEqual(first.fingerprint, second.fingerprint)


class AssistantBridgeExecutionTests(unittest.TestCase):
    def test_quality_profile_executes_the_exact_confirmed_premium_plan(self) -> None:
        with _fake_bridge() as fixture, _fake_environment(fixture):
            result = fixture.run(
                build_assistant_task("Use the quality tier.", profile="quality")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_provider, "codex-premium")
        self.assertEqual(result.premium_calls_used, 1)

    def test_verified_local_result_is_returned_to_the_user_without_premium(
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
                allow_remote=True,
                allow_remote_workspace=True,
            )
            result = fixture.run(task)
            invocations = _read_jsonl(fixture.log)

        telemetry = json.dumps(result.metadata_payload(), sort_keys=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.code, "local_verification_passed")
        self.assertEqual([item["mode"] for item in invocations], ["local"])
        self.assertEqual(
            result.user_payload()["result"]["content"], "VERIFIED local user result"
        )
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
            ):
                task = build_assistant_task("Keep this local-first.", allow_remote=True)
                result = fixture.run(task)

            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.code, "local_launcher_unavailable")

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
        self.assertIn(str(sentinel), invocation["stdin"])
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
        self.assertEqual(payload["route_receipt"]["route"], "local_then_verify")
        self.assertTrue(str(payload["confirmation_id"]).startswith("confirm-"))
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
        with _fake_bridge() as fixture:
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

        self.assertEqual(completed.returncode, 2)
        self.assertIn("policy disables", completed.stderr)

    def test_cli_records_failed_terminal_audit_when_receipt_is_rejected(self) -> None:
        with _fake_bridge() as fixture:
            plan = _run_cli_plan(fixture, "Run a safe task.", profile="offline")
            plan["confirmation_id"] = f"confirm-{'0' * 64}"
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


@contextmanager
def _fake_bridge(
    *,
    local_executable: str | None = None,
    local_launcher_args: tuple[str, ...] | None = None,
):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _initialize_git(root)
        launcher = root / "fake_codex.py"
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
        raw = json.loads(
            (ROOT / "configs" / "assistant-bridge.json").read_text(encoding="utf-8")
        )
        executable = local_executable or sys.executable
        launcher_args = (
            list(local_launcher_args)
            if local_launcher_args is not None
            else [str(launcher)]
        )
        raw["providers"]["local"]["executable"] = executable
        raw["providers"]["local"]["launcher_args"] = launcher_args
        raw["providers"]["premium"]["executable"] = sys.executable
        raw["providers"]["premium"]["launcher_args"] = [str(launcher)]
        allowed_env = [
            "FAKE_CODEX_LOG",
            "FAKE_EXIT_CODE",
            "FAKE_LOCAL_OUTPUT",
            "FAKE_OUTPUT_BYTES",
            "FAKE_PREMIUM_OUTPUT",
            "FAKE_STDOUT_BYTES",
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
        raw["state"]["budget_ledger_path"] = str(root / "runtime" / "budget.json")
        config_path = root / "assistant-bridge.json"
        config_path.write_text(json.dumps(raw), encoding="utf-8")

        app_raw = json.loads(
            (ROOT / "configs" / "app.json").read_text(encoding="utf-8")
        )
        app_raw["runtime"]["work_dir"] = str(root / "runtime")
        app_config_path = root / "app.json"
        app_config_path.write_text(json.dumps(app_raw), encoding="utf-8")
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
):
    with patch.dict(
        os.environ,
        {
            "FAKE_CODEX_LOG": str(fixture.log),
            "FAKE_LOCAL_OUTPUT": local_output,
            "FAKE_PREMIUM_OUTPUT": premium_output,
            "FAKE_OUTPUT_BYTES": str(output_bytes),
            "FAKE_STDOUT_BYTES": str(stdout_bytes),
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
    (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    (root / ".gitignore").write_text(
        "invocations.jsonl\nruntime/\ncapsule.json\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "tracked.txt", ".gitignore"], cwd=root, check=True)
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Antonio Antenore",
            "GIT_AUTHOR_EMAIL": "ant_ant95@hotmail.it",
            "GIT_COMMITTER_NAME": "Antonio Antenore",
            "GIT_COMMITTER_EMAIL": "ant_ant95@hotmail.it",
        }
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "test fixture"],
        cwd=root,
        env=env,
        check=True,
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
