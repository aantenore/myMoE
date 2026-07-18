"""Deterministic CLI composition for Assistant Bridge two-phase workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence


EXIT_SUCCESS = 0
EXIT_INVOCATION = 2
EXIT_NOT_READY = 3
EXIT_RUNTIME = 4

_MAX_ATTESTATION_FILES = 64
_MAX_ATTESTATION_BYTES = 8 * 1024 * 1024
_MAX_ATTESTATION_AGGREGATE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class AssistantBridgeCliOutcome:
    payload: dict[str, object]
    exit_code: int = EXIT_SUCCESS


class AssistantBridgeCliError(RuntimeError):
    """A redacted, stable lifecycle CLI failure."""

    def __init__(self, *, mode: str, code: str, exit_code: int) -> None:
        super().__init__(code)
        self.mode = mode
        self.code = code
        self.exit_code = exit_code

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": "1.0",
            "mode": f"assistant_bridge_{self.mode}",
            "status": "error",
            "error": {"code": self.code},
        }


def canonical_json(value: dict[str, object]) -> str:
    """Serialize one public CLI payload without representation-dependent output."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def lifecycle_mode(args: Any) -> str | None:
    if bool(getattr(args, "assistant_bridge_stage", False)):
        return "stage"
    if getattr(args, "assistant_bridge_status", None) is not None:
        return "status"
    if getattr(args, "assistant_bridge_resume_plan", None) is not None:
        return "resume_plan"
    if getattr(args, "assistant_bridge_resume", None) is not None:
        return "resume"
    return None


def load_cli_assistant_task(args: Any) -> Any:
    """Build or load the same task contract used by the legacy bridge CLI."""

    from .assistant_bridge import build_assistant_task, load_assistant_task

    if args.assistant_task_file is not None:
        return load_assistant_task(args.assistant_task_file)
    return build_assistant_task(
        args.assistant_task,
        profile=args.assistant_profile or "balanced",
        required_capabilities=args.assistant_capability or (),
        required_tools=args.assistant_required_tool or (),
        risk_class=args.assistant_risk or "read_only",
        constraints=args.assistant_constraint or (),
        allow_remote=(
            True
            if args.assistant_allow_remote
            else False
            if args.assistant_deny_remote
            else None
        ),
        allow_remote_workspace=args.assistant_allow_remote_workspace,
        max_premium_calls=args.assistant_max_premium_calls,
    )


def run_lifecycle_cli(
    args: Any,
    *,
    app_config: Any | None = None,
) -> AssistantBridgeCliOutcome:
    mode = lifecycle_mode(args)
    if mode is None:
        raise AssistantBridgeCliError(
            mode="lifecycle",
            code="lifecycle_mode_required",
            exit_code=EXIT_INVOCATION,
        )
    if mode == "status":
        return _run_status(args)
    if mode == "stage":
        return _run_stage(args, app_config=app_config)
    if mode == "resume_plan":
        return _run_resume_plan(args)
    return _run_resume(args, app_config=app_config)


def _run_status(args: Any) -> AssistantBridgeCliOutcome:
    """Read durable state without loading trust, providers, or app configuration."""

    from .assistant_bridge_two_phase_state import (
        load_two_phase_state_config,
    )
    from .assistant_bridge_two_phase_status import (
        TwoPhaseStatusError,
        build_two_phase_status_reader,
    )

    try:
        state = load_two_phase_state_config(args.assistant_workflow_config)
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="status",
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc
    try:
        record = build_two_phase_status_reader(state).status(
            args.assistant_bridge_status
        )
    except TwoPhaseStatusError as exc:
        not_ready = _status_error_is_not_ready(exc)
        raise AssistantBridgeCliError(
            mode="status",
            code=("workflow_not_ready" if not_ready else "status_runtime_failed"),
            exit_code=(EXIT_NOT_READY if not_ready else EXIT_RUNTIME),
        ) from exc
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="status",
            code="status_runtime_failed",
            exit_code=EXIT_RUNTIME,
        ) from exc
    return AssistantBridgeCliOutcome(_workflow_status_payload(record))


def _run_stage(
    args: Any,
    *,
    app_config: Any | None,
) -> AssistantBridgeCliOutcome:
    from .assistant_bridge import AssistantBridgeConfirmationError

    try:
        task = load_cli_assistant_task(args)
        context = _build_lifecycle_context(args)
        external_evidence = _load_quality_evidence(args.assistant_verification)
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="stage",
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc

    lifecycle, generator, workspace, workspace_policy = context
    try:
        source_fingerprint = _source_fingerprint(workspace, workspace_policy)
        operation_sha256 = lifecycle.stage_operation_sha256(
            source_workspace=workspace,
            task_fingerprint=task.task_fingerprint,
            expected_source_fingerprint=source_fingerprint,
            expected_config_sha256=lifecycle.effective_config_sha256,
            idempotency_key=args.assistant_idempotency_key,
        )
        replay = lifecycle.find_stage_replay(
            source_workspace=workspace,
            task_fingerprint=task.task_fingerprint,
            expected_source_fingerprint=source_fingerprint,
            expected_config_sha256=lifecycle.effective_config_sha256,
            idempotency_key=args.assistant_idempotency_key,
        )
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="stage",
            code="stage_conflict",
            exit_code=EXIT_RUNTIME,
        ) from exc
    if replay is not None:
        return AssistantBridgeCliOutcome(replay.payload())

    common = {
        "workspace": workspace,
        "expected_config_sha256": lifecycle.effective_config_sha256,
        "operation_sha256": operation_sha256,
        "local_provider_override": args.assistant_local_provider,
        "external_evidence": external_evidence,
        "include_diff": args.assistant_include_diff,
    }
    confirmation = args.assistant_confirm_receipt
    if confirmation is None:
        try:
            plan = generator.plan_candidate(task, **common)
        except (OSError, ValueError) as exc:
            raise AssistantBridgeCliError(
                mode="stage",
                code="candidate_plan_failed",
                exit_code=EXIT_RUNTIME,
            ) from exc
        exit_code = (
            EXIT_SUCCESS if plan.get("confirmation_id") is not None else EXIT_NOT_READY
        )
        return AssistantBridgeCliOutcome(plan, exit_code)

    try:
        receipt = generator.inspect_candidate(task, **common)
        _require_stage_app_authority(app_config, task, receipt.route)
        request = generator.request(
            task,
            confirmation=confirmation,
            operation_sha256=operation_sha256,
            local_provider_override=args.assistant_local_provider,
            external_evidence=external_evidence,
            include_diff=args.assistant_include_diff,
        )
        staged = lifecycle.stage(
            request,
            source_workspace=workspace,
            task_fingerprint=task.task_fingerprint,
            expected_source_fingerprint=source_fingerprint,
            expected_config_sha256=lifecycle.effective_config_sha256,
            idempotency_key=args.assistant_idempotency_key,
        )
    except AssistantBridgeCliError:
        raise
    except AssistantBridgeConfirmationError as exc:
        raise AssistantBridgeCliError(
            mode="stage",
            code="confirmation_not_ready",
            exit_code=EXIT_NOT_READY,
        ) from exc
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="stage",
            code="stage_runtime_failed",
            exit_code=EXIT_RUNTIME,
        ) from exc
    return AssistantBridgeCliOutcome(staged.payload())


def _run_resume_plan(args: Any) -> AssistantBridgeCliOutcome:
    from .assistant_bridge_two_phase import TwoPhaseWorkflowConflictError

    try:
        lifecycle, _, workspace, _, record = _build_resume_lifecycle_context(
            args,
            workflow_id=args.assistant_bridge_resume_plan,
            mode="resume_plan",
        )
        envelopes = _load_attestation_envelopes(args.assistant_attestation_file or ())
    except AssistantBridgeCliError:
        raise
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="resume_plan",
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc
    try:
        plan = lifecycle.plan_resume(
            args.assistant_bridge_resume_plan,
            workspace=workspace,
            expected_source_fingerprint=record.binding.source_fingerprint,
            expected_config_sha256=lifecycle.effective_config_sha256,
            idempotency_key=args.assistant_idempotency_key,
            attestation_envelopes=envelopes,
        )
    except TwoPhaseWorkflowConflictError as exc:
        raise AssistantBridgeCliError(
            mode="resume_plan",
            code="resume_plan_conflict",
            exit_code=EXIT_RUNTIME,
        ) from exc
    except (OSError, ValueError) as exc:
        exit_code = _resume_plan_failure_exit(lifecycle, record.workflow_id)
        raise AssistantBridgeCliError(
            mode="resume_plan",
            code=(
                "workflow_not_ready"
                if exit_code == EXIT_NOT_READY
                else "workflow_conflict"
            ),
            exit_code=exit_code,
        ) from exc
    return AssistantBridgeCliOutcome(plan.payload())


def _run_resume(
    args: Any,
    *,
    app_config: Any | None,
) -> AssistantBridgeCliOutcome:
    from .assistant_bridge_two_phase import TwoPhaseConfirmationNotReadyError
    from .assistant_bridge_two_phase_recovery import (
        TwoPhaseApplyingRecoveryError,
        TwoPhaseApplyingRecoveryUnavailable,
    )
    from .assistant_bridge_two_phase_status import (
        TwoPhaseAppliedReplayError,
        TwoPhaseAppliedReplayNotReadyError,
        TwoPhaseAppliedReplayUnavailable,
    )

    try:
        recovered = _recover_applying_if_needed(args)
    except TwoPhaseApplyingRecoveryUnavailable:
        recovered = None
    except TwoPhaseApplyingRecoveryError as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="resume_recovery_failed",
            exit_code=EXIT_RUNTIME,
        ) from exc
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc
    if recovered is not None:
        return AssistantBridgeCliOutcome(
            recovered.payload(),
            _resume_result_exit(recovered),
        )

    try:
        replayed = _replay_applied_if_needed(args)
    except TwoPhaseAppliedReplayUnavailable:
        replayed = None
    except TwoPhaseAppliedReplayNotReadyError as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="confirmation_not_ready",
            exit_code=EXIT_NOT_READY,
        ) from exc
    except TwoPhaseAppliedReplayError as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="workflow_conflict",
            exit_code=EXIT_RUNTIME,
        ) from exc
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc
    if replayed is not None:
        return _applied_replay_outcome(replayed, args.assistant_workspace)

    try:
        lifecycle, _, workspace, _, record = _build_resume_lifecycle_context(
            args,
            workflow_id=args.assistant_bridge_resume,
            mode="resume",
            state_only=True,
        )
    except AssistantBridgeCliError:
        raise
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc
    if record.status == "applied":
        return _applied_replay_outcome(record, workspace)
    if record.status in {"staged", "attested", "expired"}:
        raise AssistantBridgeCliError(
            mode="resume",
            code="confirmation_not_ready",
            exit_code=EXIT_NOT_READY,
        )
    if record.status in {"conflicted", "failed"}:
        raise AssistantBridgeCliError(
            mode="resume",
            code="workflow_conflict",
            exit_code=EXIT_RUNTIME,
        )
    if record.status == "ready":
        app_config = _load_resume_app_config(args, app_config)
        _require_resume_app_authority(app_config)
    try:
        if lifecycle is None:
            lifecycle, _, _, _ = _build_resume_lifecycle_runtime(args, workspace)
        result = lifecycle.apply_resume(
            args.assistant_bridge_resume,
            workspace=workspace,
            expected_source_fingerprint=record.binding.source_fingerprint,
            expected_config_sha256=lifecycle.effective_config_sha256,
            plan_id=args.assistant_resume_plan_id,
            confirmation_id=args.assistant_confirm_receipt,
        )
    except TwoPhaseConfirmationNotReadyError as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="confirmation_not_ready",
            exit_code=EXIT_NOT_READY,
        ) from exc
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="resume_runtime_failed",
            exit_code=EXIT_RUNTIME,
        ) from exc
    return AssistantBridgeCliOutcome(result.payload(), _resume_result_exit(result))


def _build_lifecycle_context(args: Any) -> tuple[Any, Any, str, Any]:
    from .assistant_bridge import (
        AssistantBridgeRunner,
        load_assistant_bridge_config,
    )
    from .assistant_bridge_lifecycle import build_two_phase_lifecycle
    from .assistant_bridge_two_phase_config import (
        load_two_phase_lifecycle_config,
    )

    workspace = str(args.assistant_workspace)
    runner = AssistantBridgeRunner(
        load_assistant_bridge_config(args.assistant_bridge_config)
    )
    generator = runner.candidate_generator()
    config = load_two_phase_lifecycle_config(args.assistant_workflow_config)
    lifecycle = build_two_phase_lifecycle(
        config,
        governed_workspace=workspace,
        workspace_policy=runner.config.workspace.scope,
        candidate_generator=generator,
    )
    return lifecycle, generator, workspace, runner.config.workspace.scope


def _build_resume_lifecycle_context(
    args: Any,
    *,
    workflow_id: str,
    mode: str,
    state_only: bool = False,
) -> tuple[Any, Any, str, Any, Any]:
    workspace, _, record = _read_resume_record(
        args,
        workflow_id=workflow_id,
        mode=mode,
    )
    if state_only:
        return None, None, workspace, None, record
    lifecycle, generator, workspace, workspace_policy = _build_resume_lifecycle_runtime(
        args, workspace
    )
    return lifecycle, generator, workspace, workspace_policy, record


def _read_resume_record(
    args: Any,
    *,
    workflow_id: str,
    mode: str,
) -> tuple[str, Any, Any]:
    from .assistant_bridge_two_phase_state import load_two_phase_state_config
    from .assistant_bridge_two_phase_status import (
        TwoPhaseAppliedReplayError,
        TwoPhaseAppliedReplayNotReadyError,
        TwoPhaseStatusError,
        build_two_phase_status_reader,
    )

    workspace = str(args.assistant_workspace)
    try:
        state = load_two_phase_state_config(args.assistant_workflow_config)
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode=mode,
            code="workflow_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc
    try:
        reader = build_two_phase_status_reader(state)
        record = reader.status(workflow_id)
        if mode == "resume" and record.status == "applied":
            record = reader.replay_applied(
                workflow_id,
                plan_id=args.assistant_resume_plan_id,
                confirmation_id=args.assistant_confirm_receipt,
            )
    except TwoPhaseAppliedReplayNotReadyError as exc:
        raise AssistantBridgeCliError(
            mode=mode,
            code="confirmation_not_ready",
            exit_code=EXIT_NOT_READY,
        ) from exc
    except TwoPhaseAppliedReplayError as exc:
        raise AssistantBridgeCliError(
            mode=mode,
            code="workflow_conflict",
            exit_code=EXIT_RUNTIME,
        ) from exc
    except TwoPhaseStatusError as exc:
        not_ready = _status_error_is_not_ready(exc)
        raise AssistantBridgeCliError(
            mode=mode,
            code=("workflow_not_ready" if not_ready else "status_runtime_failed"),
            exit_code=(EXIT_NOT_READY if not_ready else EXIT_RUNTIME),
        ) from exc
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode=mode,
            code="status_runtime_failed",
            exit_code=EXIT_RUNTIME,
        ) from exc
    return workspace, state, record


def _recover_applying_if_needed(args: Any) -> Any | None:
    from .assistant_bridge_two_phase_recovery import (
        build_two_phase_applying_recovery,
    )
    from .assistant_bridge_two_phase_state import load_two_phase_state_config

    state = load_two_phase_state_config(args.assistant_workflow_config)
    return build_two_phase_applying_recovery(state).recover_if_applying(
        args.assistant_bridge_resume,
        workspace=args.assistant_workspace,
    )


def _replay_applied_if_needed(args: Any) -> Any | None:
    from .assistant_bridge_two_phase_state import load_two_phase_state_config
    from .assistant_bridge_two_phase_status import (
        build_two_phase_applied_replay_reader,
    )

    state = load_two_phase_state_config(args.assistant_workflow_config)
    return build_two_phase_applied_replay_reader(state).replay(
        args.assistant_bridge_resume,
        plan_id=args.assistant_resume_plan_id,
        confirmation_id=args.assistant_confirm_receipt,
    )


def _applied_replay_outcome(record: Any, workspace: str) -> AssistantBridgeCliOutcome:
    from .assistant_bridge_two_phase_contracts import ResumeResult
    from .assistant_bridge_two_phase_recovery import (
        TwoPhaseApplyingRecoveryError,
        recovery_workspace_root_sha256,
    )

    try:
        root_sha256 = recovery_workspace_root_sha256(workspace)
    except TwoPhaseApplyingRecoveryError as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="resume_runtime_failed",
            exit_code=EXIT_RUNTIME,
        ) from exc
    if root_sha256 != record.workspace_root_sha256:
        raise AssistantBridgeCliError(
            mode="resume",
            code="workflow_conflict",
            exit_code=EXIT_RUNTIME,
        )
    replay = ResumeResult(
        workflow_id=record.workflow_id,
        status="applied",
        code="already_applied",
        candidate_fingerprint=record.binding.candidate_fingerprint,
        transaction_id=record.apply_transaction_id,
        result_sha256=record.result_sha256,
        idempotent_replay=True,
    )
    return AssistantBridgeCliOutcome(replay.payload())


def _build_resume_lifecycle_runtime(
    args: Any,
    workspace: str,
) -> tuple[Any, Any, str, Any]:
    from .assistant_bridge import (
        AssistantBridgeRunner,
        load_assistant_bridge_config,
    )
    from .assistant_bridge_lifecycle import build_two_phase_lifecycle
    from .assistant_bridge_two_phase_config import (
        load_two_phase_lifecycle_config,
    )

    runner = AssistantBridgeRunner(
        load_assistant_bridge_config(args.assistant_bridge_config)
    )
    generator = runner.candidate_generator()
    config = load_two_phase_lifecycle_config(args.assistant_workflow_config)
    lifecycle = build_two_phase_lifecycle(
        config,
        governed_workspace=workspace,
        workspace_policy=runner.config.workspace.scope,
        candidate_generator=generator,
    )
    return lifecycle, generator, workspace, runner.config.workspace.scope


def _load_resume_app_config(args: Any, app_config: Any | None) -> Any:
    if app_config is not None:
        return app_config
    from .app_config import load_app_config

    try:
        return load_app_config(args.app_config)
    except (OSError, ValueError) as exc:
        raise AssistantBridgeCliError(
            mode="resume",
            code="application_config_invalid",
            exit_code=EXIT_INVOCATION,
        ) from exc


def _source_fingerprint(workspace: str, workspace_policy: Any) -> str:
    from .assistant_bridge_workspace import snapshot_workspace

    return snapshot_workspace(
        workspace,
        workspace_policy,
    ).fingerprint


def _load_quality_evidence(path: str | None) -> tuple[Any, ...]:
    if path is None:
        return ()
    from .assistant_bridge import load_verification_evidence

    return tuple(load_verification_evidence(path))


def _status_error_is_not_ready(error: Exception) -> bool:
    return getattr(error, "code", "") in {
        "state_uninitialized",
        "workflow_not_found",
    }


def _load_attestation_envelopes(paths: Sequence[str]) -> tuple[bytes, ...]:
    from .assistant_bridge_two_phase_state import read_bounded_regular_file

    if len(paths) > _MAX_ATTESTATION_FILES:
        raise AssistantBridgeCliError(
            mode="resume_plan",
            code="attestation_input_invalid",
            exit_code=EXIT_INVOCATION,
        )
    total = 0
    result: list[bytes] = []
    for value in paths:
        try:
            envelope = read_bounded_regular_file(
                Path(value),
                max_bytes=_MAX_ATTESTATION_BYTES,
                label="independent attestation",
            )
        except (OSError, ValueError) as exc:
            raise AssistantBridgeCliError(
                mode="resume_plan",
                code="attestation_input_invalid",
                exit_code=EXIT_INVOCATION,
            ) from exc
        total += len(envelope)
        if total > _MAX_ATTESTATION_AGGREGATE_BYTES:
            raise AssistantBridgeCliError(
                mode="resume_plan",
                code="attestation_input_invalid",
                exit_code=EXIT_INVOCATION,
            )
        result.append(envelope)
    return tuple(result)


def _workflow_status_payload(record: Any) -> dict[str, object]:
    binding = record.binding
    return {
        "schemaVersion": "1.0",
        "mode": "assistant_bridge_status",
        "workflowId": record.workflow_id,
        "status": record.status,
        "bindingSha256": binding.binding_sha256,
        "taskFingerprint": binding.task_fingerprint,
        "sourceFingerprint": binding.source_fingerprint,
        "candidateFingerprint": binding.candidate_fingerprint,
        "configSha256": binding.config_sha256,
        "expiresAt": binding.expires_at,
        "quorum": {
            "required": binding.verification_policy.quorum,
            "verified": record.active_attestation_count,
            "recorded": len(record.attestations),
            "satisfied": record.quorum_satisfied,
        },
        "attestations": [
            {
                "verifierId": item.verifier_id,
                "adapterId": item.adapter_id,
                "keyId": item.key_id,
                "attestationId": item.attestation_id,
                "evidenceSha256": item.evidence_sha256,
                "statementSha256": item.statement_sha256,
                "issuedAt": item.issued_at,
                "expiresAt": item.expires_at,
                "recordedAt": item.recorded_at,
            }
            for item in record.attestations
        ],
        "applyTransactionId": record.apply_transaction_id or None,
        "recoveredTransactionId": record.recovered_transaction_id or None,
        "resultSha256": record.result_sha256 or None,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
        "privacy": "metadata_only",
    }


def _require_stage_app_authority(app_config: Any | None, task: Any, route: str) -> None:
    if app_config is None:
        raise AssistantBridgeCliError(
            mode="stage",
            code="application_policy_required",
            exit_code=EXIT_INVOCATION,
        )
    policy = (
        str(
            getattr(
                app_config.permissions,
                "assistant_bridge_execution_policy",
                "disabled",
            )
        )
        .strip()
        .lower()
    )
    if policy == "disabled" or (
        policy == "local_only" and route in {"local_then_verify", "premium"}
    ):
        raise AssistantBridgeCliError(
            mode="stage",
            code="stage_not_permitted",
            exit_code=EXIT_NOT_READY,
        )
    if policy not in {"local_only", "hybrid_receipt_confirmation"}:
        raise AssistantBridgeCliError(
            mode="stage",
            code="application_policy_invalid",
            exit_code=EXIT_INVOCATION,
        )
    if route == "blocked":
        raise AssistantBridgeCliError(
            mode="stage",
            code="candidate_not_ready",
            exit_code=EXIT_NOT_READY,
        )
    if task.capability_demand.risk_class == "write_local":
        write_policy = str(app_config.permissions.default_write_policy).strip().lower()
        if write_policy in {"deny", "denied", "disabled", "forbidden"}:
            raise AssistantBridgeCliError(
                mode="stage",
                code="stage_not_permitted",
                exit_code=EXIT_NOT_READY,
            )


def _require_resume_app_authority(app_config: Any | None) -> None:
    if app_config is None:
        raise AssistantBridgeCliError(
            mode="resume",
            code="application_policy_required",
            exit_code=EXIT_INVOCATION,
        )
    execution_policy = (
        str(
            getattr(
                app_config.permissions,
                "assistant_bridge_execution_policy",
                "disabled",
            )
        )
        .strip()
        .lower()
    )
    write_policy = str(app_config.permissions.default_write_policy).strip().lower()
    if execution_policy not in {
        "local_only",
        "hybrid_receipt_confirmation",
    } or write_policy in {"deny", "denied", "disabled", "forbidden"}:
        raise AssistantBridgeCliError(
            mode="resume",
            code="resume_not_permitted",
            exit_code=EXIT_NOT_READY,
        )


def _resume_plan_failure_exit(lifecycle: Any, workflow_id: str) -> int:
    try:
        status = lifecycle.status(workflow_id).status
    except (OSError, ValueError):
        return EXIT_RUNTIME
    return EXIT_RUNTIME if status in {"conflicted", "failed"} else EXIT_NOT_READY


def _resume_result_exit(result: Any) -> int:
    if result.status == "applied":
        return EXIT_SUCCESS
    if result.code in {
        "recovered_confirmation_required",
        "recovered_expired",
    } or result.status in {"staged", "attested", "ready", "expired"}:
        return EXIT_NOT_READY
    return EXIT_RUNTIME
