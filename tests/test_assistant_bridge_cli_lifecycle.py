from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from local_moe.assistant_bridge import AssistantBridgeConfirmationError
from local_moe.assistant_bridge_attestation import create_ed25519_dsse_envelope
from local_moe.assistant_lifecycle_cli import (
    AssistantBridgeCliError,
    AssistantBridgeCliOutcome,
    _load_attestation_envelopes,
    _status_error_is_not_ready,
    _workflow_status_payload,
    canonical_json,
    run_lifecycle_cli,
)
from local_moe.assistant_bridge_integrity import sha256_bytes
from local_moe.assistant_bridge_lifecycle import GeneratedCandidate
from local_moe.assistant_bridge_two_phase import (
    TwoPhaseConfirmationNotReadyError,
    TwoPhaseWorkflowError,
    TwoPhaseWorkflowConflictError,
    candidate_workspace_snapshot_fingerprint,
)
from local_moe.assistant_bridge_two_phase_contracts import AttestationCheck
from local_moe.assistant_bridge_workspace import (
    WorkspaceScopePolicy,
    apply_changeset as real_apply_changeset,
)
from tests.test_assistant_bridge_lifecycle import _CandidateGenerator, _Fixture


CONFIG_SHA256 = "c" * 64
SOURCE_SHA256 = "s" * 64
TASK_SHA256 = "t" * 64
OPERATION_SHA256 = "o" * 64
STAGE_KEY = "stage-cli-idempotency-00000001"


class AssistantBridgeLifecycleCliTests(unittest.TestCase):
    def test_lifecycle_invocation_errors_are_canonical_and_redacted(self) -> None:
        private_task = "private lifecycle task"
        private_workspace = "/private/lifecycle/workspace"
        cases = (
            [
                "mymoe",
                "--assistant-bridge-stage",
                "--assistant-task",
                private_task,
                "--assistant-workspace",
                private_workspace,
                "--assistant-idempotency-key",
                STAGE_KEY,
            ],
            [
                "mymoe",
                "--assistant-bridge-stage",
                "--assistant-task",
                private_task,
                "--assistant-workspace",
                private_workspace,
                "--assistant-workflow-config",
                "/private/workflow.json",
                "--assistant-idempotency-key",
                STAGE_KEY,
                "--prompt",
                "private conflicting prompt",
            ],
            [
                "mymoe",
                "--assistant-bridge-stage",
                "--assistant-bridge-status",
                "wf-private",
            ],
            [
                "mymoe",
                "--assistant-bridge-status",
                "wf-private",
                "--assistant-workflow-config",
                "workflow.json",
                "--app-config",
                "configs/app.json",
            ],
            [
                "mymoe",
                "--assistant-bridge-status",
                "wf-private",
                "--assistant-workflow-config",
                "workflow.json",
                "--runtime-optimizer-runs-limit",
                "100",
            ],
            [
                "mymoe",
                "--assistant-bridge-stat",
                "wf-private",
                "--assistant-workflow-config",
                "workflow.json",
            ],
        )
        from local_moe.cli import main

        for argv in cases:
            with self.subTest(argv=argv):
                error_output = StringIO()
                with patch.object(sys, "argv", argv), redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as raised:
                        main()

                self.assertEqual(raised.exception.code, 2)
                payload = json.loads(error_output.getvalue())
                self.assertEqual(payload["error"]["code"], "invocation_invalid")
                self.assertNotIn("usage:", error_output.getvalue())
                self.assertNotIn(private_task, error_output.getvalue())
                self.assertNotIn(private_workspace, error_output.getvalue())

    def test_lifecycle_workflow_id_validation_is_canonical(self) -> None:
        from local_moe.cli import main

        for workflow_id in ("../x", "x/y", "x" * 129):
            with self.subTest(workflow_id=workflow_id):
                error_output = StringIO()
                argv = [
                    "mymoe",
                    "--assistant-bridge-status",
                    workflow_id,
                    "--assistant-workflow-config",
                    "workflow.json",
                ]
                with patch.object(sys, "argv", argv), redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as raised:
                        main()

                self.assertEqual(raised.exception.code, 2)
                payload = json.loads(error_output.getvalue())
                self.assertEqual(payload["error"]["code"], "invocation_invalid")

    def test_stage_confirmation_not_ready_is_three_but_conflict_is_four(self) -> None:
        task = SimpleNamespace(
            task_fingerprint=TASK_SHA256,
            capability_demand=SimpleNamespace(risk_class="write_local"),
        )
        lifecycle = _lifecycle()
        lifecycle.stage_operation_sha256.return_value = OPERATION_SHA256
        lifecycle.find_stage_replay.return_value = None
        generator = Mock()
        generator.inspect_candidate.return_value = SimpleNamespace(route="local")
        generator.request.return_value = object()
        context = (lifecycle, generator, "/workspace", Mock())
        args = _stage_args(confirmation="candidate-confirmation")

        with (
            patch(
                "local_moe.assistant_lifecycle_cli.load_cli_assistant_task",
                return_value=task,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_lifecycle_context",
                return_value=context,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._source_fingerprint",
                return_value=SOURCE_SHA256,
            ),
        ):
            lifecycle.stage.side_effect = AssistantBridgeConfirmationError(
                "new confirmation required"
            )
            with self.assertRaises(AssistantBridgeCliError) as not_ready:
                run_lifecycle_cli(args, app_config=_app_config())
            lifecycle.stage.side_effect = TwoPhaseWorkflowError("binding conflict")
            with self.assertRaises(AssistantBridgeCliError) as conflict:
                run_lifecycle_cli(args, app_config=_app_config())

        self.assertEqual(not_ready.exception.exit_code, 3)
        self.assertEqual(not_ready.exception.code, "confirmation_not_ready")
        self.assertEqual(conflict.exception.exit_code, 4)

    def test_resume_confirmation_not_ready_is_three_but_conflict_is_four(self) -> None:
        lifecycle = _lifecycle()
        record = SimpleNamespace(
            status="ready",
            binding=SimpleNamespace(source_fingerprint="d" * 64),
        )
        context = (lifecycle, Mock(), "/workspace", Mock(), record)
        args = _resume_args()

        with (
            patch(
                "local_moe.assistant_lifecycle_cli._recover_applying_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._replay_applied_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                return_value=context,
            ),
        ):
            lifecycle.apply_resume.side_effect = TwoPhaseConfirmationNotReadyError(
                "new confirmation required"
            )
            with self.assertRaises(AssistantBridgeCliError) as not_ready:
                run_lifecycle_cli(args, app_config=_app_config())
            lifecycle.apply_resume.side_effect = TwoPhaseWorkflowError(
                "binding conflict"
            )
            with self.assertRaises(AssistantBridgeCliError) as conflict:
                run_lifecycle_cli(args, app_config=_app_config())

        self.assertEqual(not_ready.exception.exit_code, 3)
        self.assertEqual(not_ready.exception.code, "confirmation_not_ready")
        self.assertEqual(conflict.exception.exit_code, 4)

    def test_invalid_stage_app_config_is_redacted_exit_two(self) -> None:
        error_output = StringIO()
        private_task = "private task text"
        private_workspace = "/private/workspace/path"
        argv = [
            "mymoe",
            "--assistant-bridge-stage",
            "--assistant-task",
            private_task,
            "--assistant-workspace",
            private_workspace,
            "--assistant-workflow-config",
            "/private/workflow.json",
            "--assistant-idempotency-key",
            STAGE_KEY,
            "--app-config",
            "/missing/private-app.json",
        ]

        with patch.object(sys, "argv", argv), redirect_stderr(error_output):
            from local_moe.cli import main

            with self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 2)
        payload = json.loads(error_output.getvalue())
        self.assertEqual(payload["error"]["code"], "application_config_invalid")
        self.assertNotIn(private_task, error_output.getvalue())
        self.assertNotIn(private_workspace, error_output.getvalue())

    def test_resume_plan_idempotency_conflict_is_runtime_exit_four(self) -> None:
        lifecycle = _lifecycle()
        lifecycle.plan_resume.side_effect = TwoPhaseWorkflowConflictError(
            "redacted conflict"
        )
        record = SimpleNamespace(
            workflow_id="wf-ready",
            binding=SimpleNamespace(source_fingerprint="d" * 64),
        )
        args = _resume_plan_args(
            workflow_id="wf-ready",
            attestation_file="unused",
        )
        args.assistant_attestation_file = []

        with patch(
            "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
            return_value=(lifecycle, Mock(), "/workspace", Mock(), record),
        ):
            with self.assertRaises(AssistantBridgeCliError) as raised:
                run_lifecycle_cli(args)

        self.assertEqual(raised.exception.exit_code, 4)
        self.assertEqual(raised.exception.code, "resume_plan_conflict")

    def test_stage_status_resume_plan_and_resume_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            generator = _CliCandidateGenerator(fixture.source)
            lifecycle = fixture.lifecycle(generator)
            context = (
                lifecycle,
                generator,
                str(fixture.source),
                WorkspaceScopePolicy(),
            )
            task = SimpleNamespace(
                task_fingerprint="1" * 64,
                capability_demand=SimpleNamespace(risk_class="write_local"),
            )
            stage_args = _stage_args()
            with (
                patch(
                    "local_moe.assistant_lifecycle_cli.load_cli_assistant_task",
                    return_value=task,
                ),
                patch(
                    "local_moe.assistant_lifecycle_cli._build_lifecycle_context",
                    return_value=context,
                ),
            ):
                candidate_plan = run_lifecycle_cli(stage_args)
                stage_args.assistant_confirm_receipt = str(
                    candidate_plan.payload["confirmation_id"]
                )
                staged = run_lifecycle_cli(
                    stage_args,
                    app_config=_app_config(),
                )

            workflow_id = str(staged.payload["workflowId"])
            status_args = _status_args(
                workflow_id=workflow_id,
                workflow_config=str(fixture.config_path),
            )
            status = run_lifecycle_cli(status_args)
            self.assertEqual(status.payload["status"], "staged")

            requirement = lifecycle.config.trust.policy.verifiers[0]
            now = time.time()
            envelope = create_ed25519_dsse_envelope(
                lifecycle.status(workflow_id).binding,
                requirement,
                fixture.private_key,
                attestation_id="cli-independent-attestation",
                issued_at=now,
                expires_at=now + 60,
                checks=(
                    AttestationCheck(
                        check_id="project-tests",
                        passed=True,
                        evidence_sha256=sha256_bytes(b"passed"),
                    ),
                ),
            )
            envelope_path = fixture.root / "attestation.dsse.json"
            envelope_path.write_bytes(envelope)
            durable_record = lifecycle.status(workflow_id)
            resume_context = (*context, durable_record)
            resume_plan_args = _resume_plan_args(
                workflow_id=workflow_id,
                attestation_file=str(envelope_path),
            )
            with patch(
                "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                return_value=resume_context,
            ):
                resume_plan = run_lifecycle_cli(resume_plan_args)

            resume_args = _resume_args(
                workflow_id=workflow_id,
                plan_id=str(resume_plan.payload["planId"]),
                confirmation=str(resume_plan.payload["confirmationId"]),
            )
            ready_record = lifecycle.status(workflow_id)
            with (
                patch(
                    "local_moe.assistant_lifecycle_cli._recover_applying_if_needed",
                    return_value=None,
                ),
                patch(
                    "local_moe.assistant_lifecycle_cli._replay_applied_if_needed",
                    return_value=None,
                ),
                patch(
                    "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                    return_value=(*context, ready_record),
                ),
            ):
                applied = run_lifecycle_cli(
                    resume_args,
                    app_config=_app_config(),
                )

            source_content = (fixture.source / "app.txt").read_text(encoding="utf-8")

        self.assertEqual(applied.exit_code, 0)
        self.assertEqual(applied.payload["status"], "applied")
        self.assertEqual(source_content, "candidate\n")

    def test_main_recovers_applying_before_config_policy_and_cas_loading(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            now = time.time() - 10
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint="1" * 64,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=now,
            )
            requirement = lifecycle.config.trust.policy.verifiers[0]
            envelope = create_ed25519_dsse_envelope(
                receipt.binding,
                requirement,
                fixture.private_key,
                attestation_id="cli-recovery-attestation",
                issued_at=now + 1,
                expires_at=now + 120,
                checks=(
                    AttestationCheck(
                        check_id="project-tests",
                        passed=True,
                        evidence_sha256=sha256_bytes(b"passed"),
                    ),
                ),
            )
            plan = lifecycle.plan_resume(
                receipt.workflow_id,
                workspace=fixture.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key="resume-cli-recovery-operation-0001",
                attestation_envelopes=(envelope,),
                now=now + 2,
            )

            def crash(**kwargs: object) -> object:
                return real_apply_changeset(
                    **kwargs,
                    _fault_after_mutation=0,
                )

            with patch(
                "local_moe.assistant_bridge_two_phase.apply_changeset",
                crash,
            ):
                with self.assertRaises(RuntimeError):
                    lifecycle.apply_resume(
                        receipt.workflow_id,
                        workspace=fixture.source,
                        expected_source_fingerprint=(
                            receipt.binding.source_fingerprint
                        ),
                        expected_config_sha256=lifecycle.effective_config_sha256,
                        plan_id=plan.plan_id,
                        confirmation_id=plan.confirmation_id,
                        now=now + 3,
                    )
            applying = lifecycle.status(receipt.workflow_id, now=now + 4)
            transaction_id = applying.apply_transaction_id
            transaction = (
                lifecycle.config.state.transaction_state_dir
                / f"transaction-{transaction_id}"
            )
            shutil.rmtree(lifecycle.config.state.cas_path / "objects")
            fixture.public_key_path.unlink()
            output = StringIO()
            argv = [
                "mymoe",
                "--assistant-bridge-resume",
                receipt.workflow_id,
                "--assistant-workspace",
                str(fixture.source),
                "--assistant-workflow-config",
                str(fixture.config_path),
                "--assistant-resume-plan-id",
                plan.plan_id,
                f"--assistant-confirm-receipt={plan.confirmation_id}",
                "--assistant-bridge-config",
                str(fixture.root / "missing-provider.json"),
                "--app-config",
                str(fixture.root / "missing-app.json"),
            ]

            with (
                patch.object(sys, "argv", argv),
                patch("local_moe.cli.load_app_config") as load_app_config,
                patch(
                    "local_moe.assistant_lifecycle_cli._read_resume_record",
                    side_effect=AssertionError("strict status must be skipped"),
                ),
                patch(
                    "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_runtime",
                    side_effect=AssertionError("provider must be skipped"),
                ),
                redirect_stdout(output),
            ):
                from local_moe.cli import main

                with self.assertRaises(SystemExit) as raised:
                    main()

            payload = json.loads(output.getvalue())
            self.assertEqual(raised.exception.code, 3)
            load_app_config.assert_not_called()
            self.assertEqual(payload["code"], "recovered_confirmation_required")
            self.assertEqual(payload["status"], "ready")
            self.assertTrue(payload["idempotentReplay"])
            self.assertEqual(
                (fixture.source / "app.txt").read_text(encoding="utf-8"),
                "source\n",
            )
            self.assertFalse(transaction.exists())

    def test_applying_recovery_rejects_mismatched_journal_source(self) -> None:
        from local_moe.assistant_bridge_two_phase_recovery import (
            TwoPhaseApplyingRecoveryError,
            build_two_phase_applying_recovery,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture, lifecycle, receipt, _, _, transaction = _prepare_applying_workflow(
                Path(temporary)
            )
            journal = transaction / "journal.json"
            payload = json.loads(journal.read_text(encoding="utf-8"))
            payload["source_fingerprint"] = "f" * 64
            journal.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(TwoPhaseApplyingRecoveryError):
                build_two_phase_applying_recovery(
                    lifecycle.config.state
                ).recover_if_applying(
                    receipt.workflow_id,
                    workspace=fixture.source,
                )

            self.assertEqual(
                lifecycle.status(receipt.workflow_id).status,
                "applying",
            )
            self.assertEqual(
                (fixture.source / "app.txt").read_text(encoding="utf-8"),
                "candidate\n",
            )
            self.assertTrue(transaction.exists())

    def test_applying_recovery_falls_back_for_journal_without_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture, lifecycle, receipt, plan, _, transaction = (
                _prepare_applying_workflow(Path(temporary))
            )
            journal = transaction / "journal.json"
            payload = json.loads(journal.read_text(encoding="utf-8"))
            payload.pop("workspace_policy")
            journal.write_text(json.dumps(payload), encoding="utf-8")
            applying = lifecycle.status(receipt.workflow_id)
            args = _resume_args(
                workflow_id=receipt.workflow_id,
                plan_id=plan.plan_id,
                confirmation=plan.confirmation_id,
            )
            args.assistant_workflow_config = str(fixture.config_path)
            args.assistant_workspace = str(fixture.source)

            with (
                patch(
                    "local_moe.assistant_lifecycle_cli._replay_applied_if_needed",
                    return_value=None,
                ),
                patch(
                    "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                    return_value=(
                        lifecycle,
                        lifecycle.candidate_generator,
                        str(fixture.source),
                        lifecycle.workflow_service.config.workspace_policy,
                        applying,
                    ),
                ),
            ):
                result = run_lifecycle_cli(
                    args,
                    app_config=_app_config(execution_policy="disabled"),
                )

            self.assertEqual(
                result.payload["code"],
                "recovered_confirmation_required",
            )
            self.assertFalse(transaction.exists())

    def test_applying_recovery_rejects_unrelated_workspace_drift(self) -> None:
        from local_moe.assistant_bridge_two_phase_recovery import (
            TwoPhaseApplyingRecoveryError,
            build_two_phase_applying_recovery,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture, lifecycle, receipt, _, _, transaction = _prepare_applying_workflow(
                Path(temporary)
            )
            (fixture.source / "unrelated.txt").write_text(
                "drift\n",
                encoding="utf-8",
            )

            with self.assertRaises(TwoPhaseApplyingRecoveryError):
                build_two_phase_applying_recovery(
                    lifecycle.config.state
                ).recover_if_applying(
                    receipt.workflow_id,
                    workspace=fixture.source,
                )

            self.assertEqual(
                lifecycle.status(receipt.workflow_id).status,
                "applying",
            )
            self.assertEqual(
                (fixture.source / "app.txt").read_text(encoding="utf-8"),
                "source\n",
            )
            self.assertTrue((fixture.source / "unrelated.txt").exists())
            self.assertTrue(transaction.exists())

    def test_applying_recovery_retries_after_db_reset_interruption(self) -> None:
        from local_moe.assistant_bridge_two_phase_recovery import (
            build_two_phase_applying_recovery,
        )

        with tempfile.TemporaryDirectory() as temporary:
            fixture, lifecycle, receipt, _, _, transaction = _prepare_applying_workflow(
                Path(temporary)
            )
            recovery = build_two_phase_applying_recovery(lifecycle.config.state)
            with patch.object(
                recovery.store,
                "reset_applying_after_recovery",
                side_effect=RuntimeError("injected reset interruption"),
            ):
                with self.assertRaises(RuntimeError):
                    recovery.recover_if_applying(
                        receipt.workflow_id,
                        workspace=fixture.source,
                    )

            journal = transaction / "journal.json"
            self.assertEqual(
                json.loads(journal.read_text(encoding="utf-8"))["status"],
                "recovered",
            )
            self.assertEqual(
                lifecycle.status(receipt.workflow_id).status,
                "applying",
            )
            self.assertEqual(
                (fixture.source / "app.txt").read_text(encoding="utf-8"),
                "source\n",
            )
            shutil.rmtree(lifecycle.config.state.cas_path / "objects")
            fixture.public_key_path.unlink()

            result = build_two_phase_applying_recovery(
                lifecycle.config.state
            ).recover_if_applying(
                receipt.workflow_id,
                workspace=fixture.source,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.code, "recovered_confirmation_required")
            self.assertFalse(transaction.exists())

    def test_applying_recovery_defers_cleanup_without_ambiguous_failure(self) -> None:
        from local_moe.assistant_bridge_two_phase_recovery import (
            build_two_phase_applying_recovery,
        )
        from local_moe.assistant_bridge_workspace import WorkspaceSecurityError

        with tempfile.TemporaryDirectory() as temporary:
            fixture, lifecycle, receipt, _, _, transaction = _prepare_applying_workflow(
                Path(temporary)
            )
            recovery = build_two_phase_applying_recovery(lifecycle.config.state)
            with patch(
                "local_moe.assistant_bridge_two_phase_recovery.finalize_recovered_workspace_transaction",
                side_effect=WorkspaceSecurityError("injected cleanup fault"),
            ):
                result = recovery.recover_if_applying(
                    receipt.workflow_id,
                    workspace=fixture.source,
                )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.code, "recovered_confirmation_required")
            self.assertTrue(transaction.exists())
            shutil.rmtree(lifecycle.config.state.cas_path / "objects")
            fixture.public_key_path.unlink()

            replay = build_two_phase_applying_recovery(
                lifecycle.config.state
            ).recover_if_applying(
                receipt.workflow_id,
                workspace=fixture.source,
            )

            self.assertEqual(replay, result)
            self.assertFalse(transaction.exists())

    def test_applied_replay_validates_exact_consumed_authority_without_loaders(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
            now = time.time() - 10
            receipt = lifecycle.stage(
                "change-app",
                source_workspace=fixture.source,
                task_fingerprint="1" * 64,
                expected_source_fingerprint=fixture.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key=STAGE_KEY,
                now=now,
            )
            requirement = lifecycle.config.trust.policy.verifiers[0]
            envelope = create_ed25519_dsse_envelope(
                receipt.binding,
                requirement,
                fixture.private_key,
                attestation_id="cli-applied-replay-attestation",
                issued_at=now + 1,
                expires_at=now + 120,
                checks=(
                    AttestationCheck(
                        check_id="project-tests",
                        passed=True,
                        evidence_sha256=sha256_bytes(b"passed"),
                    ),
                ),
            )
            plan = lifecycle.plan_resume(
                receipt.workflow_id,
                workspace=fixture.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                idempotency_key="resume-cli-applied-operation-0001",
                attestation_envelopes=(envelope,),
                now=now + 2,
            )
            lifecycle.apply_resume(
                receipt.workflow_id,
                workspace=fixture.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=now + 3,
            )
            shutil.rmtree(lifecycle.config.state.cas_path)
            fixture.public_key_path.unlink()

            base_argv = [
                "mymoe",
                "--assistant-bridge-resume",
                receipt.workflow_id,
                "--assistant-workspace",
                str(fixture.source),
                "--assistant-workflow-config",
                str(fixture.config_path),
                "--assistant-bridge-config",
                str(fixture.root / "missing-provider.json"),
                "--app-config",
                str(fixture.root / "missing-app.json"),
            ]
            cases = (
                (
                    plan.plan_id,
                    plan.confirmation_id,
                    0,
                    "already_applied",
                ),
                ("f" * 64, plan.confirmation_id, 4, "workflow_conflict"),
                (plan.plan_id, "wrong-confirmation", 3, "confirmation_not_ready"),
            )
            from local_moe.cli import main

            for plan_id, confirmation, exit_code, code in cases:
                with self.subTest(code=code):
                    output = StringIO()
                    error_output = StringIO()
                    argv = [
                        *base_argv,
                        "--assistant-resume-plan-id",
                        plan_id,
                        f"--assistant-confirm-receipt={confirmation}",
                    ]
                    with (
                        patch.object(sys, "argv", argv),
                        patch("local_moe.cli.load_app_config") as load_app_config,
                        patch(
                            "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_runtime",
                            side_effect=AssertionError("provider must be skipped"),
                        ),
                        redirect_stdout(output),
                        redirect_stderr(error_output),
                    ):
                        if exit_code:
                            with self.assertRaises(SystemExit) as raised:
                                main()
                            self.assertEqual(raised.exception.code, exit_code)
                            payload = json.loads(error_output.getvalue())
                        else:
                            main()
                            payload = json.loads(output.getvalue())
                    load_app_config.assert_not_called()
                    self.assertEqual(
                        payload.get("code") or payload["error"]["code"], code
                    )

    def test_resume_unknown_and_uninitialized_fall_through_to_typed_status(
        self,
    ) -> None:
        from local_moe.cli import main

        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _Fixture(Path(temporary))
                if initialized:
                    fixture.lifecycle(_CandidateGenerator(fixture.source))
                error_output = StringIO()
                argv = [
                    "mymoe",
                    "--assistant-bridge-resume",
                    "wf-unknown",
                    "--assistant-workspace",
                    str(fixture.source),
                    "--assistant-workflow-config",
                    str(fixture.config_path),
                    "--assistant-resume-plan-id",
                    "f" * 64,
                    "--assistant-confirm-receipt=unknown-confirmation",
                    "--app-config",
                    str(fixture.root / "missing-app.json"),
                ]
                with (
                    patch.object(sys, "argv", argv),
                    patch("local_moe.cli.load_app_config") as load_app_config,
                    redirect_stderr(error_output),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        main()

                load_app_config.assert_not_called()
                self.assertEqual(raised.exception.code, 3)
                payload = json.loads(error_output.getvalue())
                self.assertEqual(payload["error"]["code"], "workflow_not_ready")

    def test_status_exit_classification_uses_typed_code_only(self) -> None:
        self.assertTrue(
            _status_error_is_not_ready(SimpleNamespace(code="workflow_not_found"))
        )
        self.assertTrue(
            _status_error_is_not_ready(SimpleNamespace(code="state_uninitialized"))
        )
        self.assertFalse(
            _status_error_is_not_ready(
                SimpleNamespace(
                    code="state_invalid",
                    message="workflow not found text must not affect routing",
                )
            )
        )

    def test_cli_import_does_not_eagerly_load_bridge_provider_or_trust_modules(
        self,
    ) -> None:
        program = """
import sys
import local_moe.cli

blocked = {
    "local_moe.assistant_bridge",
    "local_moe.assistant_bridge_attestation",
    "local_moe.assistant_bridge_provider_registry",
    "local_moe.assistant_bridge_two_phase_config",
}
loaded = sorted(name for name in blocked if name in sys.modules)
if loaded:
    raise SystemExit(",".join(loaded))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

        completed = subprocess.run(
            [sys.executable, "-c", program],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_status_dispatch_precedes_app_config_loading(self) -> None:
        output = StringIO()
        outcome = AssistantBridgeCliOutcome(
            {
                "schemaVersion": "1.0",
                "mode": "assistant_bridge_status",
                "status": "staged",
            }
        )
        argv = [
            "mymoe",
            "--assistant-bridge-status",
            "wf-status",
            "--assistant-workflow-config",
            "workflow.json",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("local_moe.cli.run_lifecycle_cli", return_value=outcome),
            patch("local_moe.cli.load_app_config") as load_app_config,
            redirect_stdout(output),
        ):
            from local_moe.cli import main

            main()

        load_app_config.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), outcome.payload)

    def test_resume_plan_dispatch_does_not_load_app_config(self) -> None:
        output = StringIO()
        outcome = AssistantBridgeCliOutcome(
            {
                "schemaVersion": "1.0",
                "mode": "assistant_bridge_resume_plan",
            }
        )
        argv = [
            "mymoe",
            "--assistant-bridge-resume-plan",
            "wf-ready",
            "--assistant-workflow-config",
            "workflow.json",
            "--assistant-workspace",
            "/workspace",
            "--assistant-idempotency-key",
            "resume-cli-idempotency-0000001",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("local_moe.cli.run_lifecycle_cli", return_value=outcome),
            patch("local_moe.cli.load_app_config") as load_app_config,
            redirect_stdout(output),
        ):
            from local_moe.cli import main

            main()

        load_app_config.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), outcome.payload)

    def test_stage_replay_returns_before_candidate_plan_or_confirmation(self) -> None:
        args = _stage_args(confirmation="stale-but-unused")
        task = SimpleNamespace(task_fingerprint=TASK_SHA256)
        replay = Mock()
        replay.payload.return_value = {
            "mode": "assistant_bridge_stage",
            "idempotentReplay": True,
        }
        lifecycle = _lifecycle()
        lifecycle.stage_operation_sha256.return_value = OPERATION_SHA256
        lifecycle.find_stage_replay.return_value = replay
        generator = Mock()

        with (
            patch(
                "local_moe.assistant_lifecycle_cli.load_cli_assistant_task",
                return_value=task,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_lifecycle_context",
                return_value=(lifecycle, generator, "/workspace", Mock()),
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._source_fingerprint",
                return_value=SOURCE_SHA256,
            ),
        ):
            outcome = run_lifecycle_cli(args)

        self.assertTrue(outcome.payload["idempotentReplay"])
        generator.plan_candidate.assert_not_called()
        generator.inspect_candidate.assert_not_called()
        generator.request.assert_not_called()
        lifecycle.stage.assert_not_called()

    def test_stage_plan_and_confirmed_run_share_operation_binding(self) -> None:
        task = SimpleNamespace(
            task_fingerprint=TASK_SHA256,
            capability_demand=SimpleNamespace(risk_class="write_local"),
        )
        lifecycle = _lifecycle()
        lifecycle.stage_operation_sha256.return_value = OPERATION_SHA256
        lifecycle.find_stage_replay.return_value = None
        staged = Mock()
        staged.payload.return_value = {"mode": "assistant_bridge_stage"}
        lifecycle.stage.return_value = staged
        generator = Mock()
        generator.plan_candidate.return_value = {
            "mode": "assistant_bridge_candidate_plan",
            "confirmation_id": "candidate-confirmation",
            "operation_sha256": OPERATION_SHA256,
        }
        generator.inspect_candidate.return_value = SimpleNamespace(route="local")
        request = object()
        generator.request.return_value = request
        app_config = _app_config()

        patches = (
            patch(
                "local_moe.assistant_lifecycle_cli.load_cli_assistant_task",
                return_value=task,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_lifecycle_context",
                return_value=(lifecycle, generator, "/workspace", Mock()),
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._source_fingerprint",
                return_value=SOURCE_SHA256,
            ),
        )
        with patches[0], patches[1], patches[2]:
            plan = run_lifecycle_cli(_stage_args())
            staged_outcome = run_lifecycle_cli(
                _stage_args(confirmation="candidate-confirmation"),
                app_config=app_config,
            )

        self.assertEqual(plan.payload["operation_sha256"], OPERATION_SHA256)
        self.assertEqual(staged_outcome.payload["mode"], "assistant_bridge_stage")
        self.assertEqual(
            generator.plan_candidate.call_args.kwargs["operation_sha256"],
            OPERATION_SHA256,
        )
        self.assertEqual(
            generator.inspect_candidate.call_args.kwargs["operation_sha256"],
            OPERATION_SHA256,
        )
        self.assertEqual(
            generator.request.call_args.kwargs["operation_sha256"],
            OPERATION_SHA256,
        )
        self.assertIs(lifecycle.stage.call_args.args[0], request)

    def test_resume_uses_durable_source_binding_and_enabled_write_policy(self) -> None:
        lifecycle = _lifecycle()
        record = SimpleNamespace(
            status="ready",
            binding=SimpleNamespace(source_fingerprint="d" * 64),
        )
        result = Mock(status="applied", code="applied")
        result.payload.return_value = {
            "mode": "assistant_bridge_resume",
            "status": "applied",
        }
        lifecycle.apply_resume.return_value = result
        args = _resume_args()

        with (
            patch(
                "local_moe.assistant_lifecycle_cli._recover_applying_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._replay_applied_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                return_value=(lifecycle, Mock(), "/workspace", Mock(), record),
            ),
        ):
            outcome = run_lifecycle_cli(args, app_config=_app_config())

        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(
            lifecycle.apply_resume.call_args.kwargs["expected_source_fingerprint"],
            "d" * 64,
        )

        with (
            patch(
                "local_moe.assistant_lifecycle_cli._recover_applying_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._replay_applied_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                return_value=(lifecycle, Mock(), "/workspace", Mock(), record),
            ),
        ):
            with self.assertRaises(AssistantBridgeCliError) as raised:
                run_lifecycle_cli(
                    args,
                    app_config=_app_config(execution_policy="disabled"),
                )
        self.assertEqual(raised.exception.exit_code, 3)

    def test_applied_resume_replay_needs_no_current_app_or_provider(self) -> None:
        transaction_id = "a" * 64
        result_sha256 = "b" * 64
        root_sha256 = "c" * 64
        record = SimpleNamespace(
            workflow_id="wf-applied",
            status="applied",
            workspace_root_sha256=root_sha256,
            binding=SimpleNamespace(candidate_fingerprint="d" * 64),
            apply_transaction_id=transaction_id,
            result_sha256=result_sha256,
        )
        args = _resume_args(workflow_id="wf-applied")

        with (
            patch(
                "local_moe.assistant_lifecycle_cli._recover_applying_if_needed",
                return_value=None,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._replay_applied_if_needed",
                return_value=record,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_context",
                return_value=(None, None, "/workspace", None, record),
            ),
            patch(
                "local_moe.assistant_bridge_two_phase_recovery.recovery_workspace_root_sha256",
                return_value=root_sha256,
            ),
            patch(
                "local_moe.assistant_lifecycle_cli._load_resume_app_config"
            ) as load_app_config,
            patch(
                "local_moe.assistant_lifecycle_cli._build_resume_lifecycle_runtime"
            ) as build_runtime,
        ):
            outcome = run_lifecycle_cli(
                args,
                app_config=_app_config(execution_policy="disabled"),
            )

        load_app_config.assert_not_called()
        build_runtime.assert_not_called()
        self.assertEqual(outcome.payload["code"], "already_applied")
        self.assertTrue(outcome.payload["idempotentReplay"])
        self.assertEqual(outcome.payload["transactionId"], transaction_id)

    def test_status_projection_and_json_are_metadata_only_and_deterministic(
        self,
    ) -> None:
        binding = SimpleNamespace(
            binding_sha256="b" * 64,
            task_fingerprint=TASK_SHA256,
            source_fingerprint=SOURCE_SHA256,
            candidate_fingerprint="c" * 64,
            config_sha256=CONFIG_SHA256,
            expires_at=200.0,
            verification_policy=SimpleNamespace(quorum=1),
        )
        attestation = SimpleNamespace(
            verifier_id="verifier",
            adapter_id="adapter",
            key_id="key",
            attestation_id="attestation",
            evidence_sha256="e" * 64,
            statement_sha256="f" * 64,
            issued_at=100.0,
            expires_at=180.0,
            recorded_at=110.0,
            envelope=b"private-envelope",
            statement=b"private-statement",
        )
        record = SimpleNamespace(
            workflow_id="wf-status",
            status="ready",
            binding=binding,
            active_attestation_count=1,
            attestations=(attestation,),
            quorum_satisfied=True,
            apply_transaction_id="",
            recovered_transaction_id="",
            result_sha256="",
            created_at=100.0,
            updated_at=110.0,
        )

        payload = _workflow_status_payload(record)
        encoded = canonical_json(payload)

        self.assertEqual(encoded, canonical_json(payload))
        self.assertNotIn("private-envelope", encoded)
        self.assertNotIn("private-statement", encoded)
        self.assertNotIn("/workspace", encoded)
        self.assertEqual(payload["privacy"], "metadata_only")

    def test_attestation_reader_rejects_links_without_echoing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            envelope = root / "attestation.json"
            envelope.write_bytes(b"{}")
            linked = root / "linked.json"
            try:
                linked.symlink_to(envelope)
            except OSError as exc:
                self.skipTest(f"links unavailable: {exc}")

            with self.assertRaises(AssistantBridgeCliError) as raised:
                _load_attestation_envelopes((str(linked),))

        self.assertEqual(raised.exception.exit_code, 2)
        self.assertNotIn(str(linked), canonical_json(raised.exception.payload()))


def _lifecycle() -> Mock:
    lifecycle = Mock()
    lifecycle.effective_config_sha256 = CONFIG_SHA256
    return lifecycle


def _app_config(
    *,
    execution_policy: str = "hybrid_receipt_confirmation",
) -> SimpleNamespace:
    return SimpleNamespace(
        permissions=SimpleNamespace(
            assistant_bridge_execution_policy=execution_policy,
            default_write_policy="allow",
        )
    )


def _stage_args(*, confirmation: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        assistant_bridge_stage=True,
        assistant_bridge_status=None,
        assistant_bridge_resume_plan=None,
        assistant_bridge_resume=None,
        assistant_verification=None,
        assistant_idempotency_key=STAGE_KEY,
        assistant_local_provider=None,
        assistant_include_diff=False,
        assistant_confirm_receipt=confirmation,
    )


def _status_args(*, workflow_id: str, workflow_config: str) -> SimpleNamespace:
    return SimpleNamespace(
        assistant_bridge_stage=False,
        assistant_bridge_status=workflow_id,
        assistant_bridge_resume_plan=None,
        assistant_bridge_resume=None,
        assistant_workflow_config=workflow_config,
    )


def _resume_plan_args(
    *,
    workflow_id: str,
    attestation_file: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        assistant_bridge_stage=False,
        assistant_bridge_status=None,
        assistant_bridge_resume_plan=workflow_id,
        assistant_bridge_resume=None,
        assistant_attestation_file=[attestation_file],
        assistant_idempotency_key="resume-cli-idempotency-0000001",
    )


def _resume_args(
    *,
    workflow_id: str = "wf-resume",
    plan_id: str = "p" * 64,
    confirmation: str = "resume-confirmation",
) -> SimpleNamespace:
    return SimpleNamespace(
        assistant_bridge_stage=False,
        assistant_bridge_status=None,
        assistant_bridge_resume_plan=None,
        assistant_bridge_resume=workflow_id,
        assistant_resume_plan_id=plan_id,
        assistant_confirm_receipt=confirmation,
        assistant_workspace="/workspace",
    )


def _prepare_applying_workflow(root: Path) -> tuple[object, ...]:
    fixture = _Fixture(root)
    lifecycle = fixture.lifecycle(_CandidateGenerator(fixture.source))
    now = time.time() - 10
    receipt = lifecycle.stage(
        "change-app",
        source_workspace=fixture.source,
        task_fingerprint="1" * 64,
        expected_source_fingerprint=fixture.source_fingerprint,
        expected_config_sha256=lifecycle.effective_config_sha256,
        idempotency_key=STAGE_KEY,
        now=now,
    )
    requirement = lifecycle.config.trust.policy.verifiers[0]
    envelope = create_ed25519_dsse_envelope(
        receipt.binding,
        requirement,
        fixture.private_key,
        attestation_id="cli-applying-recovery-attestation",
        issued_at=now + 1,
        expires_at=now + 120,
        checks=(
            AttestationCheck(
                check_id="project-tests",
                passed=True,
                evidence_sha256=sha256_bytes(b"passed"),
            ),
        ),
    )
    plan = lifecycle.plan_resume(
        receipt.workflow_id,
        workspace=fixture.source,
        expected_source_fingerprint=receipt.binding.source_fingerprint,
        expected_config_sha256=lifecycle.effective_config_sha256,
        idempotency_key="resume-cli-applying-helper-0001",
        attestation_envelopes=(envelope,),
        now=now + 2,
    )

    def crash(**kwargs: object) -> object:
        return real_apply_changeset(**kwargs, _fault_after_mutation=0)

    with patch("local_moe.assistant_bridge_two_phase.apply_changeset", crash):
        try:
            lifecycle.apply_resume(
                receipt.workflow_id,
                workspace=fixture.source,
                expected_source_fingerprint=receipt.binding.source_fingerprint,
                expected_config_sha256=lifecycle.effective_config_sha256,
                plan_id=plan.plan_id,
                confirmation_id=plan.confirmation_id,
                now=now + 3,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("transaction crash was not injected")
    applying = lifecycle.status(receipt.workflow_id, now=now + 4)
    transaction_id = applying.apply_transaction_id
    transaction = (
        lifecycle.config.state.transaction_state_dir / f"transaction-{transaction_id}"
    )
    return fixture, lifecycle, receipt, plan, transaction_id, transaction


class _CliCandidateGenerator:
    configuration_sha256 = "2" * 64

    def __init__(self, source: Path) -> None:
        self.source = source
        self.operation_sha256 = ""

    def plan_candidate(self, _task: object, **kwargs: object) -> dict[str, object]:
        self.operation_sha256 = str(kwargs["operation_sha256"])
        return {
            "mode": "assistant_bridge_candidate_plan",
            "confirmation_id": "candidate-confirmation",
            "operation_sha256": self.operation_sha256,
        }

    def inspect_candidate(self, _task: object, **kwargs: object) -> object:
        if kwargs["operation_sha256"] != self.operation_sha256:
            raise AssertionError("candidate operation changed")
        return SimpleNamespace(route="local")

    def request(
        self,
        _task: object,
        *,
        confirmation: str,
        operation_sha256: str,
        **_kwargs: object,
    ) -> str:
        if (
            confirmation != "candidate-confirmation"
            or operation_sha256 != self.operation_sha256
        ):
            raise AssertionError("candidate confirmation binding changed")
        return "change-app"

    @contextmanager
    def generate(
        self,
        request: str,
        *,
        source_workspace: str | Path,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        expected_operation_sha256: str,
    ):
        if (
            request != "change-app"
            or Path(source_workspace) != self.source
            or expected_config_sha256 == ""
            or expected_operation_sha256 != self.operation_sha256
        ):
            raise AssertionError("candidate stage binding changed")
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "candidate"
            shutil.copytree(self.source, candidate)
            (candidate / "app.txt").write_text("candidate\n", encoding="utf-8")
            yield GeneratedCandidate(
                workspace=candidate,
                source_fingerprint=expected_source_fingerprint,
                candidate_snapshot_fingerprint=(
                    candidate_workspace_snapshot_fingerprint(
                        candidate,
                        WorkspaceScopePolicy(),
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
