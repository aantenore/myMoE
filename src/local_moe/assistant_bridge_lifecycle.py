from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import stat
from typing import ContextManager, Generic, Protocol, Sequence, TypeVar

from .assistant_bridge_cas import ContentAddressedStore
from .assistant_bridge_integrity import canonical_sha256
from .assistant_bridge_two_phase import (
    TwoPhaseWorkflowConfig,
    TwoPhaseWorkflowService,
    candidate_workspace_snapshot_fingerprint,
)
from .assistant_bridge_two_phase_config import TwoPhaseLifecycleConfig
from .assistant_bridge_two_phase_contracts import (
    ResumePlan,
    ResumeResult,
    StageReceipt,
    require_sha256,
)
from .assistant_bridge_workflow_store import (
    SQLiteWorkflowStore,
    WorkflowRecord,
)
from .assistant_bridge_workspace import WorkspaceScopePolicy


TRequest = TypeVar("TRequest")


class TwoPhaseLifecycleError(ValueError):
    """Raised when deterministic lifecycle composition cannot proceed."""


@dataclass(frozen=True)
class _ResolvedStatePaths:
    database: Path
    cas: Path
    transactions: Path

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(
            str(value) for value in (self.database, self.cas, self.transactions)
        )


@dataclass(frozen=True)
class GeneratedCandidate:
    workspace: Path = field(repr=False)
    source_fingerprint: str
    candidate_snapshot_fingerprint: str

    def __post_init__(self) -> None:
        require_sha256(self.source_fingerprint, "generated source_fingerprint")
        require_sha256(
            self.candidate_snapshot_fingerprint,
            "generated candidate_snapshot_fingerprint",
        )
        raw = Path(os.path.abspath(os.fspath(self.workspace)))
        try:
            details = raw.lstat()
        except OSError as exc:
            raise TwoPhaseLifecycleError(
                "Generated candidate workspace is unavailable."
            ) from exc
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise TwoPhaseLifecycleError(
                "Generated candidate workspace must be a non-link directory."
            )
        object.__setattr__(self, "workspace", raw)


class CandidateGenerator(Protocol[TRequest]):
    """Provider-neutral port that yields one disposable candidate workspace."""

    @property
    def configuration_sha256(self) -> str: ...

    def generate(
        self,
        request: TRequest,
        *,
        source_workspace: str | Path,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        expected_operation_sha256: str,
    ) -> ContextManager[GeneratedCandidate]: ...


class TwoPhaseLifecycle(Generic[TRequest]):
    """Compose candidate generation with durable stage/resume services."""

    def __init__(
        self,
        *,
        workflow_service: TwoPhaseWorkflowService,
        candidate_generator: CandidateGenerator[TRequest],
        config: TwoPhaseLifecycleConfig,
        governed_workspace: Path,
    ) -> None:
        generator_sha256 = require_sha256(
            candidate_generator.configuration_sha256,
            "candidate generator configuration_sha256",
        )
        self.workflow_service = workflow_service
        self.candidate_generator = candidate_generator
        self.config = config
        self._governed_workspace = governed_workspace
        self._generator_configuration_sha256 = generator_sha256
        self.effective_config_sha256 = canonical_sha256(
            {
                "candidateGeneratorConfigSha256": generator_sha256,
                "twoPhaseConfigSha256": config.effective_sha256,
            }
        )

    def stage(
        self,
        request: TRequest,
        *,
        source_workspace: str | Path,
        task_fingerprint: str,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        idempotency_key: str,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> StageReceipt:
        replay = self.find_stage_replay(
            source_workspace=source_workspace,
            task_fingerprint=task_fingerprint,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            idempotency_key=idempotency_key,
            ttl_seconds=ttl_seconds,
            now=now,
        )
        if replay is not None:
            return replay
        self._require_generator_config_unchanged()
        operation_sha256 = self.stage_operation_sha256(
            source_workspace=source_workspace,
            task_fingerprint=task_fingerprint,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            idempotency_key=idempotency_key,
            ttl_seconds=ttl_seconds,
        )
        with self.candidate_generator.generate(
            request,
            source_workspace=source_workspace,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            expected_operation_sha256=operation_sha256,
        ) as candidate:
            if candidate.source_fingerprint != expected_source_fingerprint:
                raise TwoPhaseLifecycleError(
                    "Candidate generator used another source fingerprint."
                )
            if (
                candidate_workspace_snapshot_fingerprint(
                    candidate.workspace,
                    self.workflow_service.config.workspace_policy,
                )
                != candidate.candidate_snapshot_fingerprint
            ):
                raise TwoPhaseLifecycleError(
                    "Candidate workspace no longer matches the evaluated snapshot."
                )
            return self.workflow_service.stage_candidate(
                source_workspace=source_workspace,
                candidate_workspace=candidate.workspace,
                task_fingerprint=task_fingerprint,
                expected_source_fingerprint=expected_source_fingerprint,
                expected_config_sha256=expected_config_sha256,
                expected_candidate_snapshot_fingerprint=(
                    candidate.candidate_snapshot_fingerprint
                ),
                verification_policy=self.config.trust.policy,
                idempotency_key=idempotency_key,
                ttl_seconds=ttl_seconds,
                now=now,
            )

    def stage_operation_sha256(
        self,
        *,
        source_workspace: str | Path,
        task_fingerprint: str,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        idempotency_key: str,
        ttl_seconds: float | None = None,
    ) -> str:
        """Bind candidate confirmation to the exact durable stage operation."""

        self._require_current_config(expected_config_sha256)
        self._require_governed_workspace(source_workspace)
        require_sha256(task_fingerprint, "task_fingerprint")
        require_sha256(expected_source_fingerprint, "expected_source_fingerprint")
        return canonical_sha256(
            {
                "contract": "assistant-bridge-stage-operation/v1",
                "taskFingerprint": task_fingerprint,
                "sourceFingerprint": expected_source_fingerprint,
                "configSha256": expected_config_sha256,
                "verificationPolicy": self.config.trust.policy.payload(),
                "stageIdentity": self.workflow_service.stage_operation_identity(
                    idempotency_key,
                    ttl_seconds=ttl_seconds,
                ),
            }
        )

    def find_stage_replay(
        self,
        *,
        source_workspace: str | Path,
        task_fingerprint: str,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        idempotency_key: str,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> StageReceipt | None:
        """Return an exact replay before candidate planning or confirmation use."""

        self._require_current_config(expected_config_sha256)
        self._require_governed_workspace(source_workspace)
        return self.workflow_service.find_stage_replay(
            source_workspace=source_workspace,
            task_fingerprint=task_fingerprint,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            verification_policy=self.config.trust.policy,
            idempotency_key=idempotency_key,
            ttl_seconds=ttl_seconds,
            now=now,
        )

    def status(self, workflow_id: str, *, now: float | None = None) -> WorkflowRecord:
        return self.workflow_service.status(workflow_id, now=now)

    def plan_resume(
        self,
        workflow_id: str,
        *,
        workspace: str | Path,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        idempotency_key: str,
        attestation_envelopes: Sequence[bytes] = (),
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> ResumePlan:
        self._require_current_config(expected_config_sha256)
        self._require_governed_workspace(workspace)
        return self.workflow_service.plan_resume(
            workflow_id,
            workspace=workspace,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            idempotency_key=idempotency_key,
            attestation_envelopes=attestation_envelopes,
            ttl_seconds=ttl_seconds,
            now=now,
        )

    def apply_resume(
        self,
        workflow_id: str,
        *,
        workspace: str | Path,
        expected_source_fingerprint: str,
        expected_config_sha256: str,
        plan_id: str,
        confirmation_id: str,
        now: float | None = None,
    ) -> ResumeResult:
        self._require_current_config(expected_config_sha256)
        self._require_governed_workspace(workspace)
        return self.workflow_service.apply_resume(
            workflow_id,
            workspace=workspace,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            plan_id=plan_id,
            confirmation_id=confirmation_id,
            now=now,
        )

    def _require_current_config(self, expected_config_sha256: str) -> None:
        require_sha256(expected_config_sha256, "expected_config_sha256")
        if expected_config_sha256 != self.effective_config_sha256:
            raise TwoPhaseLifecycleError(
                "Lifecycle configuration no longer matches the expected digest."
            )

    def _require_generator_config_unchanged(self) -> None:
        current_generator_sha256 = require_sha256(
            self.candidate_generator.configuration_sha256,
            "candidate generator configuration_sha256",
        )
        if current_generator_sha256 != self._generator_configuration_sha256:
            raise TwoPhaseLifecycleError(
                "Lifecycle configuration no longer matches the expected digest."
            )

    def _require_governed_workspace(self, workspace: str | Path) -> None:
        if _resolve_existing_directory(workspace, "Governed workspace") != (
            self._governed_workspace
        ):
            raise TwoPhaseLifecycleError(
                "Lifecycle operation targets another governed workspace."
            )


def build_two_phase_lifecycle(
    config: TwoPhaseLifecycleConfig,
    *,
    governed_workspace: str | Path,
    workspace_policy: WorkspaceScopePolicy,
    candidate_generator: CandidateGenerator[TRequest],
) -> TwoPhaseLifecycle[TRequest]:
    workspace = _resolve_existing_directory(governed_workspace, "Governed workspace")
    state_paths = _resolve_disjoint_state_paths(
        workspace,
        database=config.state.database_path,
        cas=config.state.cas_path,
        transactions=config.state.transaction_state_dir,
    )
    cas = ContentAddressedStore(state_paths.cas)
    store = SQLiteWorkflowStore(
        state_paths.database,
        evidence_cas=cas,
        timeout=config.state.sqlite_timeout_seconds,
    )
    workflow = TwoPhaseWorkflowService(
        store=store,
        cas=cas,
        config=TwoPhaseWorkflowConfig(
            workspace_policy=workspace_policy,
            transaction_state_dir=str(state_paths.transactions),
            durable_state_paths=state_paths.values,
            candidate_ttl_seconds=config.state.candidate_ttl_seconds,
            confirmation_ttl_seconds=config.state.confirmation_ttl_seconds,
            transaction_lock_ttl_seconds=(config.state.transaction_lock_ttl_seconds),
        ),
        trust_store=config.trust.build_trust_store(),
    )
    return TwoPhaseLifecycle(
        workflow_service=workflow,
        candidate_generator=candidate_generator,
        config=config,
        governed_workspace=workspace,
    )


def _resolve_disjoint_state_paths(
    workspace: Path,
    *,
    database: Path,
    cas: Path,
    transactions: Path,
) -> _ResolvedStatePaths:
    resolved = _ResolvedStatePaths(
        database=_resolve_configured_path(database),
        cas=_resolve_configured_path(cas),
        transactions=_resolve_configured_path(transactions),
    )
    if any(
        _paths_overlap(workspace, value)
        for value in (
            resolved.database,
            resolved.cas,
            resolved.transactions,
        )
    ):
        raise TwoPhaseLifecycleError(
            "Durable state paths must be outside the governed workspace."
        )
    return resolved


def _resolve_existing_directory(value: str | Path, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise TwoPhaseLifecycleError(f"{label} must be a non-link directory.")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TwoPhaseLifecycleError(f"{label} is unavailable.") from exc
    if not resolved.is_dir():
        raise TwoPhaseLifecycleError(f"{label} must be a directory.")
    return resolved


def _resolve_configured_path(value: Path) -> Path:
    raw = Path(value).expanduser()
    missing: list[str] = []
    current = raw
    while not current.exists():
        if current.is_symlink() or current.parent == current:
            raise TwoPhaseLifecycleError("Durable state path is unavailable.")
        missing.append(current.name)
        current = current.parent
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TwoPhaseLifecycleError("Durable state path is unavailable.") from exc
    for component in reversed(missing):
        resolved /= component
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
    except ValueError:
        pass
    else:
        return True
    try:
        right.relative_to(left)
    except ValueError:
        return False
    return True
