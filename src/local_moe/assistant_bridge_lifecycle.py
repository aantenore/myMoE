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
    ) -> ContextManager[GeneratedCandidate]: ...


class TwoPhaseLifecycle(Generic[TRequest]):
    """Compose candidate generation with durable stage/resume services."""

    def __init__(
        self,
        *,
        workflow_service: TwoPhaseWorkflowService,
        candidate_generator: CandidateGenerator[TRequest],
        config: TwoPhaseLifecycleConfig,
    ) -> None:
        generator_sha256 = require_sha256(
            candidate_generator.configuration_sha256,
            "candidate generator configuration_sha256",
        )
        self.workflow_service = workflow_service
        self.candidate_generator = candidate_generator
        self.config = config
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
        self._require_current_config(expected_config_sha256)
        replay = self.workflow_service.find_stage_replay(
            source_workspace=source_workspace,
            task_fingerprint=task_fingerprint,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
            verification_policy=self.config.trust.policy,
            idempotency_key=idempotency_key,
            now=now,
        )
        if replay is not None:
            return replay
        self._require_generator_config_unchanged()
        with self.candidate_generator.generate(
            request,
            source_workspace=source_workspace,
            expected_source_fingerprint=expected_source_fingerprint,
            expected_config_sha256=expected_config_sha256,
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


def build_two_phase_lifecycle(
    config: TwoPhaseLifecycleConfig,
    *,
    workspace_policy: WorkspaceScopePolicy,
    candidate_generator: CandidateGenerator[TRequest],
) -> TwoPhaseLifecycle[TRequest]:
    cas = ContentAddressedStore(config.state.cas_path)
    store = SQLiteWorkflowStore(
        config.state.database_path,
        evidence_cas=cas,
        timeout=config.state.sqlite_timeout_seconds,
    )
    workflow = TwoPhaseWorkflowService(
        store=store,
        cas=cas,
        config=TwoPhaseWorkflowConfig(
            workspace_policy=workspace_policy,
            transaction_state_dir=str(config.state.transaction_state_dir),
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
    )
