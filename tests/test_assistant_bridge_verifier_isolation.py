from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import local_moe.assistant_bridge as assistant_bridge
import local_moe.assistant_bridge_verifier_isolation as verifier_isolation
from local_moe.assistant_bridge_runtime import ProcessCleanupError
from local_moe.assistant_bridge_verifier_isolation import (
    VerifierIsolationCapability,
    build_verifier_isolation_plan,
    verifier_isolation_capability,
)
from local_moe.assistant_bridge_workspace import (
    WorkspaceScopePolicy,
    build_changeset,
    materialize_workspace,
    snapshot_workspace,
)

from tests.test_assistant_bridge import (
    ROOT,
    _fake_bridge,
    _fake_environment,
    _initialize_git,
)


class VerifierIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = assistant_bridge.load_assistant_bridge_config(
            ROOT / "configs" / "assistant-bridge.json"
        )

    def test_live_python_command_runs_only_through_supported_sandbox(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            evidence = self._run_command(
                root,
                assistant_bridge.CommandVerifierSpec(
                    id="live-python-probe",
                    argv=("{python}", "-c", "print('sandbox-ok')"),
                    timeout_seconds=10,
                ),
            )

        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.code, "command_passed")

    def test_declared_read_grant_changes_profile_and_argv_binding(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            workspace = fixture / "workspace"
            workspace.mkdir()
            artifact = fixture / "declared-runtime-artifact.txt"
            artifact.write_text("bound\n", encoding="utf-8")
            baseline = build_verifier_isolation_plan(
                self.config.verifier_isolation,
                capability,
                workspace=workspace,
                command_argv=("/usr/bin/true",),
                runtime_read_roots=(),
                temp_namespace="a" * 24,
            )
            granted = build_verifier_isolation_plan(
                self.config.verifier_isolation,
                capability,
                workspace=workspace,
                command_argv=("/usr/bin/true",),
                runtime_read_roots=(),
                temp_namespace="a" * 24,
                attested_read_artifacts=(artifact,),
            )

        self.assertNotEqual(baseline.profile_sha256, granted.profile_sha256)
        self.assertNotEqual(baseline.argv_sha256, granted.argv_sha256)
        self.assertNotEqual(baseline.binding_sha256, granted.binding_sha256)

    def test_live_sandbox_denies_network_and_host_reads_and_mutations(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            root = fixture / "workspace"
            root.mkdir()
            _initialize_git(root)
            outside = fixture / "outside-secret.txt"
            outside.write_text("host-secret\n", encoding="utf-8")
            probe = root / "isolation_probe.py"
            probe.write_text(
                "from pathlib import Path\n"
                "import errno\n"
                "import socket\n"
                "import sys\n"
                "outside = Path(sys.argv[1])\n"
                "try:\n"
                "    outside.read_text(encoding='utf-8')\n"
                "except OSError:\n"
                "    read_blocked = True\n"
                "else:\n"
                "    read_blocked = False\n"
                "created = outside.with_name('outside-created.txt')\n"
                "try:\n"
                "    created.write_text('escape', encoding='utf-8')\n"
                "except OSError:\n"
                "    pass\n"
                "sock = None\n"
                "try:\n"
                "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "    sock.settimeout(0.2)\n"
                "    sock.connect(('198.51.100.1', 9))\n"
                "except OSError as exc:\n"
                "    network_blocked = exc.errno in {\n"
                "        errno.EPERM, errno.EACCES, errno.ENETUNREACH,\n"
                "        errno.EHOSTUNREACH, errno.EADDRNOTAVAIL,\n"
                "    }\n"
                "else:\n"
                "    network_blocked = False\n"
                "finally:\n"
                "    if sock is not None:\n"
                "        sock.close()\n"
                "raise SystemExit(\n"
                "    0 if all((read_blocked, network_blocked)) else 97\n"
                ")\n",
                encoding="utf-8",
            )
            evidence = self._run_command(
                root,
                assistant_bridge.CommandVerifierSpec(
                    id="containment-probe",
                    argv=(
                        "{python}",
                        "{workspace}/isolation_probe.py",
                        str(outside),
                    ),
                    launcher_entrypoint="{workspace}/isolation_probe.py",
                    timeout_seconds=10,
                ),
            )

            self.assertFalse((fixture / "outside-created.txt").exists())
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.code, "command_passed")

    def test_unsupported_backend_blocks_without_any_bound_launch(self) -> None:
        unavailable = VerifierIsolationCapability(
            supported=False,
            backend="unsupported",
            reason="test backend unavailable",
        )
        with (
            _fake_bridge(process_proxy=True) as fixture,
            _fake_environment(fixture),
            patch.object(
                assistant_bridge,
                "verifier_isolation_capability",
                return_value=unavailable,
            ),
            patch.object(assistant_bridge, "execute_process") as execute,
        ):
            task = assistant_bridge.build_assistant_task(
                "Run a required isolated verifier.",
                profile="quality",
                required_verifier_ids=("fixture-task-verifier",),
                allow_remote=True,
            )
            result = fixture.run(task)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.code, "route_blocked")
        self.assertIn(
            "verifier_isolation_unavailable",
            result.receipt.rationale_codes,
        )
        self.assertEqual(result.premium_calls_used, 0)
        execute.assert_not_called()
        self.assertFalse((fixture.root / "process-probe.log").exists())

    def test_unsupported_plan_payload_claims_no_active_isolation(self) -> None:
        capability = VerifierIsolationCapability(
            supported=False,
            backend="bwrap",
            reason="test backend unavailable",
        )
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_verifier_isolation_plan(
                self.config.verifier_isolation,
                capability,
                workspace=tmp,
                command_argv=("/usr/bin/true",),
                runtime_read_roots=(),
                temp_namespace="a" * 24,
            )
            payload = plan.payload()

        self.assertFalse(payload["capability"]["supported"])
        self.assertIsNone(payload["profile_sha256"])
        self.assertIsNone(payload["sandbox_argv_sha256"])
        self.assertIsNone(payload["workspace"])
        self.assertIsNone(payload["runtime_roots_access"])
        self.assertIsNone(payload["system_roots_access"])
        self.assertIsNone(payload["temporary_storage"])
        self.assertEqual(payload["network"], "unavailable")

    def test_candidate_git_config_is_never_copied_or_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            root = fixture / "workspace"
            root.mkdir()
            _initialize_git(root)
            marker = fixture / "candidate-git-config-ran"
            helper = fixture / "hostile-helper.sh"
            helper.write_text(
                f"#!/bin/sh\n: > {str(marker)!r}\nexit 99\n",
                encoding="utf-8",
            )
            helper.chmod(0o700)
            subprocess.run(
                ["git", "config", "core.fsmonitor", str(helper)],
                cwd=root,
                check=True,
            )
            policy = WorkspaceScopePolicy()
            spec = assistant_bridge.CommandVerifierSpec(
                id="trusted-git-probe",
                kind="trusted_git_diff_check",
                argv=(),
                timeout_seconds=10,
                purpose="hygiene",
                execution_boundary="trusted_git_session",
                network_policy="denied",
                runtime_read_roots=(),
            )
            source_snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(source_snapshot, policy) as candidate:
                (candidate.root / ".gitattributes").write_text(
                    "*.txt diff=hostile\n",
                    encoding="utf-8",
                )
                subprocess.run(
                    ["git", "config", "core.fsmonitor", str(helper)],
                    cwd=candidate.root,
                    check=True,
                )
                subprocess.run(
                    ["git", "config", "diff.hostile.command", str(helper)],
                    cwd=candidate.root,
                    check=True,
                )
                expected = snapshot_workspace(candidate.root, policy)
                candidate_files = candidate.snapshot()
                changes = build_changeset(candidate.baseline_files, candidate_files)
                plan = assistant_bridge._build_verifier_plan(
                    spec,
                    workspace=candidate.root,
                    runtime_policy=self.config.runtime,
                    isolation_policy=self.config.verifier_isolation,
                )
                with assistant_bridge._disposable_verifier_workspace(
                    candidate.root,
                    source_snapshot=candidate.source_snapshot,
                    baseline_files=candidate.baseline_files,
                    expected_snapshot=expected,
                    candidate_files=candidate_files,
                    changes=changes,
                    policy=policy,
                ) as disposable:
                    copied_config = (disposable / ".git" / "config").read_text(
                        encoding="utf-8"
                    )
                    self.assertNotIn(str(helper), copied_config)
                    evidence = assistant_bridge._run_bound_verifier(
                        plan,
                        task=assistant_bridge.build_assistant_task(
                            "Check Git hygiene."
                        ),
                        workspace=assistant_bridge.attest_workspace(candidate.root),
                        verifier_workspace=disposable,
                    )

            self.assertFalse(marker.exists())
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.code, "builtin_passed")

    def test_trusted_git_builtin_checks_delta_against_baseline_head(self) -> None:
        cases = (
            ("tracked-clean", "tracked.txt", "changed cleanly\n", True),
            (
                "tracked-trailing-whitespace",
                "tracked.txt",
                "changed with trailing whitespace \n",
                False,
            ),
            ("added-clean", "added.txt", "added cleanly\n", True),
            (
                "added-trailing-whitespace",
                "added.txt",
                "added with trailing whitespace \n",
                False,
            ),
        )
        for label, relative, content, should_pass in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _initialize_git(root)
                policy = WorkspaceScopePolicy()
                source_snapshot = snapshot_workspace(root, policy)
                with materialize_workspace(source_snapshot, policy) as candidate:
                    (candidate.root / relative).write_text(
                        content,
                        encoding="utf-8",
                    )
                    final_snapshot = snapshot_workspace(candidate.root, policy)
                    candidate_files = candidate.snapshot()
                    changes = build_changeset(
                        candidate.baseline_files,
                        candidate_files,
                    )
                    spec = assistant_bridge.CommandVerifierSpec(
                        id=f"trusted-git-{label}",
                        kind="trusted_git_diff_check",
                        argv=(),
                        timeout_seconds=10,
                        purpose="hygiene",
                        execution_boundary="trusted_git_session",
                        network_policy="denied",
                        runtime_read_roots=(),
                    )
                    plan = assistant_bridge._build_verifier_plan(
                        spec,
                        workspace=candidate.root,
                        runtime_policy=self.config.runtime,
                        isolation_policy=self.config.verifier_isolation,
                    )
                    with assistant_bridge._disposable_verifier_workspace(
                        candidate.root,
                        source_snapshot=candidate.source_snapshot,
                        baseline_files=candidate.baseline_files,
                        expected_snapshot=final_snapshot,
                        candidate_files=candidate_files,
                        changes=changes,
                        policy=policy,
                    ) as disposable:
                        git = assistant_bridge.trusted_git_executable().resolved_path
                        head_content = subprocess.run(
                            [git, "show", "HEAD:tracked.txt"],
                            cwd=disposable,
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout
                        added_in_head = subprocess.run(
                            [git, "cat-file", "-e", "HEAD:added.txt"],
                            cwd=disposable,
                            capture_output=True,
                        ).returncode == 0
                        worktree_content = (disposable / relative).read_text(
                            encoding="utf-8"
                        )
                        evidence = assistant_bridge._run_bound_verifier(
                            plan,
                            task=assistant_bridge.build_assistant_task(
                                "Check candidate Git hygiene."
                            ),
                            workspace=assistant_bridge.attest_workspace(
                                candidate.root
                            ),
                            verifier_workspace=disposable,
                        )

                self.assertEqual(head_content, "initial\n")
                self.assertFalse(added_in_head)
                self.assertEqual(worktree_content, content)
                self.assertEqual(evidence.passed, should_pass)
                self.assertEqual(
                    evidence.code,
                    "builtin_passed" if should_pass else "builtin_check_failed",
                )

    def test_trusted_git_builtin_preserves_tracked_deletion_against_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            policy = WorkspaceScopePolicy()
            source_snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(source_snapshot, policy) as candidate:
                (candidate.root / "tracked.txt").unlink()
                final_snapshot = snapshot_workspace(candidate.root, policy)
                candidate_files = candidate.snapshot()
                changes = build_changeset(
                    candidate.baseline_files,
                    candidate_files,
                )
                spec = assistant_bridge.CommandVerifierSpec(
                    id="trusted-git-tracked-deletion",
                    kind="trusted_git_diff_check",
                    argv=(),
                    timeout_seconds=10,
                    purpose="hygiene",
                    execution_boundary="trusted_git_session",
                    network_policy="denied",
                    runtime_read_roots=(),
                )
                plan = assistant_bridge._build_verifier_plan(
                    spec,
                    workspace=candidate.root,
                    runtime_policy=self.config.runtime,
                    isolation_policy=self.config.verifier_isolation,
                )
                with assistant_bridge._disposable_verifier_workspace(
                    candidate.root,
                    source_snapshot=candidate.source_snapshot,
                    baseline_files=candidate.baseline_files,
                    expected_snapshot=final_snapshot,
                    candidate_files=candidate_files,
                    changes=changes,
                    policy=policy,
                ) as disposable:
                    git = assistant_bridge.trusted_git_executable().resolved_path
                    head_content = subprocess.run(
                        [git, "show", "HEAD:tracked.txt"],
                        cwd=disposable,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout
                    deleted_from_worktree = not (disposable / "tracked.txt").exists()
                    evidence = assistant_bridge._run_bound_verifier(
                        plan,
                        task=assistant_bridge.build_assistant_task(
                            "Check tracked deletion hygiene."
                        ),
                        workspace=assistant_bridge.attest_workspace(candidate.root),
                        verifier_workspace=disposable,
                    )

        self.assertEqual(head_content, "initial\n")
        self.assertTrue(deleted_from_worktree)
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.code, "builtin_passed")

    def test_trusted_git_builtin_rejects_candidate_attribute_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            policy = WorkspaceScopePolicy()
            source_snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(source_snapshot, policy) as candidate:
                (candidate.root / ".gitattributes").write_text(
                    "tracked.txt -diff whitespace=-trailing-space\n",
                    encoding="utf-8",
                )
                (candidate.root / "tracked.txt").write_text(
                    "candidate bypass attempt \n",
                    encoding="utf-8",
                )
                final_snapshot = snapshot_workspace(candidate.root, policy)
                candidate_files = candidate.snapshot()
                changes = build_changeset(
                    candidate.baseline_files,
                    candidate_files,
                )
                spec = assistant_bridge.CommandVerifierSpec(
                    id="trusted-git-attribute-bypass",
                    kind="trusted_git_diff_check",
                    argv=(),
                    timeout_seconds=10,
                    purpose="hygiene",
                    execution_boundary="trusted_git_session",
                    network_policy="denied",
                    runtime_read_roots=(),
                )
                plan = assistant_bridge._build_verifier_plan(
                    spec,
                    workspace=candidate.root,
                    runtime_policy=self.config.runtime,
                    isolation_policy=self.config.verifier_isolation,
                )
                with assistant_bridge._disposable_verifier_workspace(
                    candidate.root,
                    source_snapshot=candidate.source_snapshot,
                    baseline_files=candidate.baseline_files,
                    expected_snapshot=final_snapshot,
                    candidate_files=candidate_files,
                    changes=changes,
                    policy=policy,
                ) as disposable:
                    evidence = assistant_bridge._run_bound_verifier(
                        plan,
                        task=assistant_bridge.build_assistant_task(
                            "Reject candidate Git attribute bypass."
                        ),
                        workspace=assistant_bridge.attest_workspace(candidate.root),
                        verifier_workspace=disposable,
                    )

        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.code, "builtin_check_failed")

    def test_linux_bubblewrap_plan_drops_capabilities_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            argv = verifier_isolation._bubblewrap_argv(
                workspace=workspace,
                runtime_roots=(),
                attested_read_artifacts=(),
                command_argv=("/usr/bin/true",),
            )

        separator = argv.index("--")
        self.assertLess(argv.index("--unshare-all"), separator)
        self.assertLess(argv.index("--unshare-net"), separator)
        self.assertLess(argv.index("--cap-drop"), separator)
        self.assertNotIn("--new-session", argv[:separator])
        self.assertEqual(argv[argv.index("--cap-drop") + 1], "ALL")
        self.assertEqual(argv[separator + 1 :], ("/usr/bin/true",))

    def test_linux_capability_probe_requires_a_successful_namespace_launch(
        self,
    ) -> None:
        outcomes = (
            (subprocess.CompletedProcess((), 0), True),
            (subprocess.CompletedProcess((), 1), False),
        )
        for completed, expected in outcomes:
            with self.subTest(returncode=completed.returncode), patch.object(
                verifier_isolation.subprocess,
                "run",
                return_value=completed,
            ) as run:
                supported = verifier_isolation._probe_bubblewrap_backend(
                    "/usr/bin/bwrap"
                )

            self.assertIs(supported, expected)
            argv = run.call_args.args[0]
            separator = argv.index("--")
            self.assertEqual(argv[0], "/usr/bin/bwrap")
            self.assertIn("--unshare-all", argv[:separator])
            self.assertIn("--unshare-net", argv[:separator])
            self.assertEqual(argv[separator + 1 :], ["/usr/bin/true"])
            self.assertIs(run.call_args.kwargs["start_new_session"], True)

        with patch.object(
            verifier_isolation.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("/usr/bin/bwrap", 5),
        ):
            self.assertFalse(
                verifier_isolation._probe_bubblewrap_backend("/usr/bin/bwrap")
            )

    def test_linux_backend_file_is_not_a_capability_without_namespace_probe(
        self,
    ) -> None:
        executable = SimpleNamespace(
            resolved_path="/usr/bin/bwrap",
            launch_path="/usr/bin/bwrap",
        )
        metadata = SimpleNamespace(st_mode=0o100755, st_uid=0)
        with (
            patch.object(Path, "lstat", return_value=metadata),
            patch.object(Path, "resolve", return_value=Path("/usr/bin/bwrap")),
            patch.object(verifier_isolation.os, "access", return_value=True),
            patch.object(
                verifier_isolation,
                "resolve_executable",
                return_value=executable,
            ),
            patch.object(
                verifier_isolation,
                "_probe_bubblewrap_backend",
                return_value=False,
            ) as probe,
        ):
            capability = verifier_isolation._attest_backend(
                Path("/usr/bin/bwrap"),
                "bwrap",
            )

        self.assertFalse(capability.supported)
        self.assertIn("unusable", capability.reason)
        probe.assert_called_once_with(
            "/usr/bin/bwrap",
            environment={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )

    def test_linux_bubblewrap_omits_artifact_covered_by_venv_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp).resolve()
            workspace = fixture / "workspace"
            runtime = fixture / "venv"
            workspace.mkdir()
            (runtime / "bin").mkdir(parents=True)
            launcher = runtime / "bin" / "python"
            launcher.symlink_to("/usr/bin/true")
            argv = verifier_isolation._bubblewrap_argv(
                workspace=workspace,
                runtime_roots=(runtime,),
                attested_read_artifacts=(launcher,),
                command_argv=(str(launcher),),
            )

        separator = argv.index("--")
        wrapper = argv[:separator]
        bind_sources = tuple(
            wrapper[index + 1]
            for index, item in enumerate(wrapper)
            if item == "--ro-bind"
        )
        self.assertIn(str(runtime), bind_sources)
        self.assertNotIn(str(launcher), bind_sources)
        self.assertEqual(argv[separator + 1], str(launcher))

    def test_cleanup_failure_propagates_from_sandboxed_verifier(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            spec = assistant_bridge.CommandVerifierSpec(
                id="cleanup-probe",
                argv=("{python}", "-c", "print('never')"),
                timeout_seconds=10,
            )
            plan = assistant_bridge._build_verifier_plan(
                spec,
                workspace=root,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
            )
            with (
                patch.object(
                    assistant_bridge,
                    "execute_process",
                    side_effect=ProcessCleanupError(
                        "cleanup could not be verified",
                        details={"root_reaped": False},
                    ),
                ),
                self.assertRaises(ProcessCleanupError),
            ):
                assistant_bridge._run_bound_verifier(
                    plan,
                    task=assistant_bridge.build_assistant_task("Check cleanup."),
                    workspace=assistant_bridge.attest_workspace(root),
                    verifier_workspace=root,
                )

    def test_sequential_and_repeated_verifiers_use_clean_temp_roots(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            evidence = []
            for verifier_id in ("first-temp-probe", "second-temp-probe", "first-temp-probe"):
                evidence.append(
                    self._run_command(
                        root,
                        assistant_bridge.CommandVerifierSpec(
                            id=verifier_id,
                            argv=(
                                "{python}",
                                "-c",
                                "import os,tempfile;fd,_=tempfile.mkstemp();os.close(fd)",
                            ),
                            timeout_seconds=10,
                        ),
                    )
                )
            leftovers = tuple(root.glob(".mymoe-verifier-tmp-*"))

        self.assertTrue(all(item.passed for item in evidence))
        self.assertEqual(leftovers, ())

    def test_verifier_temp_cleanup_failure_fails_closed(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            plan = assistant_bridge._build_verifier_plan(
                assistant_bridge.CommandVerifierSpec(
                    id="temp-cleanup-probe",
                    argv=("{python}", "-c", "print('cleanup')"),
                    timeout_seconds=10,
                ),
                workspace=root,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
            )
            with (
                patch.object(
                    assistant_bridge,
                    "_cleanup_verifier_internal_temp",
                    side_effect=assistant_bridge.AssistantBridgeError(
                        "Verifier temp-cleanup-probe internal temporary cleanup "
                        "could not be verified."
                    ),
                ),
                self.assertRaisesRegex(
                    assistant_bridge.AssistantBridgeError,
                    "cleanup could not be verified",
                ),
            ):
                assistant_bridge._run_bound_verifier(
                    plan,
                    task=assistant_bridge.build_assistant_task(
                        "Verify temporary cleanup."
                    ),
                    workspace=assistant_bridge.attest_workspace(root),
                    verifier_workspace=root,
                )

    def test_process_cleanup_error_survives_temp_cleanup_failure(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            plan = assistant_bridge._build_verifier_plan(
                assistant_bridge.CommandVerifierSpec(
                    id="combined-cleanup-probe",
                    argv=("{python}", "-c", "print('never')"),
                    timeout_seconds=10,
                ),
                workspace=root,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
            )
            process_error = ProcessCleanupError(
                "process cleanup failed",
                details={"root_reaped": False},
            )
            with (
                patch.object(
                    assistant_bridge,
                    "execute_process",
                    side_effect=process_error,
                ),
                patch.object(
                    assistant_bridge,
                    "_cleanup_verifier_internal_temp",
                    side_effect=assistant_bridge.AssistantBridgeError(
                        "Verifier combined-cleanup-probe internal temporary cleanup "
                        "could not be verified."
                    ),
                ),
                self.assertRaises(ProcessCleanupError) as caught,
            ):
                assistant_bridge._run_bound_verifier(
                    plan,
                    task=assistant_bridge.build_assistant_task(
                        "Preserve process cleanup failure."
                    ),
                    workspace=assistant_bridge.attest_workspace(root),
                    verifier_workspace=root,
                )

        self.assertIs(caught.exception, process_error)
        self.assertIsInstance(
            caught.exception.__cause__, assistant_bridge.AssistantBridgeError
        )

    def test_typed_unittest_runner_cannot_be_shadowed_by_candidate(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "app_under_test.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_guard.py").write_text(
                "import unittest\n"
                "from app_under_test import VALUE\n"
                "class GuardTest(unittest.TestCase):\n"
                "    def test_failure_cannot_be_hidden(self):\n"
                "        self.assertEqual(VALUE, 2)\n",
                encoding="utf-8",
            )
            policy = WorkspaceScopePolicy()
            spec = assistant_bridge.CommandVerifierSpec(
                id="typed-unittest-shadow-probe",
                argv=(
                    "{python}",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-v",
                ),
                timeout_seconds=20,
                python_runner="unittest",
                workspace_python_paths=("src", "."),
            )
            source_snapshot = snapshot_workspace(root, policy)
            source_plan = assistant_bridge._build_verifier_plan(
                spec,
                workspace=root,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
            )
            with materialize_workspace(source_snapshot, policy) as candidate:
                fake_runner = (
                    "from pathlib import Path\n"
                    "Path('candidate-runner-marker').write_text('executed')\n"
                    "raise SystemExit(0)\n"
                )
                (candidate.root / "src" / "unittest.py").write_text(
                    fake_runner,
                    encoding="utf-8",
                )
                (candidate.root / "unittest.py").write_text(
                    fake_runner,
                    encoding="utf-8",
                )
                expected = snapshot_workspace(candidate.root, policy)
                candidate_files = candidate.snapshot()
                changes = build_changeset(
                    candidate.baseline_files,
                    candidate_files,
                )
                with assistant_bridge._disposable_verifier_workspace(
                    candidate.root,
                    source_snapshot=candidate.source_snapshot,
                    baseline_files=candidate.baseline_files,
                    expected_snapshot=expected,
                    candidate_files=candidate_files,
                    changes=changes,
                    policy=policy,
                ) as disposable:
                    copied_plan = assistant_bridge._build_verifier_plan(
                        spec,
                        workspace=disposable,
                        runtime_policy=self.config.runtime,
                        isolation_policy=self.config.verifier_isolation,
                    )
                    evidence = assistant_bridge._run_bound_verifier(
                        source_plan,
                        task=assistant_bridge.build_assistant_task(
                            "Run a typed unittest verifier."
                        ),
                        workspace=assistant_bridge.attest_workspace(candidate.root),
                        verifier_workspace=disposable,
                    )
                    marker_exists = (
                        disposable / "candidate-runner-marker"
                    ).exists()

        self.assertIsNotNone(source_plan.python_runner_identity)
        assert source_plan.python_runner_identity is not None
        self.assertEqual(source_plan.python_runner_identity.name, "unittest")
        self.assertEqual(
            source_plan.python_runner_identity.manifest_sha256,
            copied_plan.python_runner_identity.manifest_sha256,
        )
        self.assertEqual(source_plan.plan_sha256, copied_plan.plan_sha256)
        self.assertEqual(
            source_plan.payload()["python_runner"]["manifest_sha256"],
            source_plan.python_runner_identity.manifest_sha256,
        )
        self.assertFalse(marker_exists)
        self.assertFalse(evidence.passed)
        self.assertNotEqual(evidence.code, "command_passed")

    def test_typed_unittest_runner_manifest_drift_blocks_prelaunch(self) -> None:
        capability = verifier_isolation_capability(
            self.config.verifier_isolation
        )
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_git(root)
            (root / "src").mkdir()
            spec = assistant_bridge.CommandVerifierSpec(
                id="typed-unittest-drift-probe",
                argv=("{python}", "-m", "unittest", "discover"),
                timeout_seconds=20,
                python_runner="unittest",
                workspace_python_paths=("src", "."),
            )
            plan = assistant_bridge._build_verifier_plan(
                spec,
                workspace=root,
                runtime_policy=self.config.runtime,
                isolation_policy=self.config.verifier_isolation,
            )
            assert plan.python_runner_identity is not None
            drifted_manifest = (
                "0" * 64,
                plan.python_runner_identity.file_count,
                plan.python_runner_identity.total_bytes,
            )
            with (
                patch.object(
                    assistant_bridge,
                    "_python_runner_manifest",
                    return_value=drifted_manifest,
                ),
                patch.object(assistant_bridge, "execute_process") as execute,
                self.assertRaisesRegex(
                    assistant_bridge.AssistantBridgeError,
                    "no longer matches",
                ),
            ):
                assistant_bridge._run_bound_verifier(
                    plan,
                    task=assistant_bridge.build_assistant_task(
                        "Reject typed runner drift."
                    ),
                    workspace=assistant_bridge.attest_workspace(root),
                    verifier_workspace=root,
                )

        execute.assert_not_called()

    def test_workspace_python_paths_reject_untyped_module_runner(self) -> None:
        with self.assertRaisesRegex(
            assistant_bridge.AssistantBridgeError,
            "typed Python runner",
        ):
            assistant_bridge.CommandVerifierSpec(
                id="untyped-module-probe",
                argv=("{python}", "-m", "candidate_runner"),
                timeout_seconds=20,
                workspace_python_paths=("src",),
            )

    def _run_command(
        self,
        root: Path,
        spec: assistant_bridge.CommandVerifierSpec,
    ) -> assistant_bridge.VerificationEvidence:
        plan = assistant_bridge._build_verifier_plan(
            spec,
            workspace=root,
            runtime_policy=self.config.runtime,
            isolation_policy=self.config.verifier_isolation,
        )
        self.assertIsNotNone(plan.isolation)
        assert plan.isolation is not None
        self.assertTrue(plan.isolation.capability.supported)
        self.assertIn(
            plan.isolation.capability.backend,
            {"sandbox-exec", "bwrap"},
        )
        return assistant_bridge._run_bound_verifier(
            plan,
            task=assistant_bridge.build_assistant_task("Run isolation probe."),
            workspace=assistant_bridge.attest_workspace(root),
            verifier_workspace=root,
        )


if __name__ == "__main__":
    unittest.main()
