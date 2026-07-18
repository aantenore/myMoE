from __future__ import annotations

from typing import Literal

from .assistant_bridge_cas import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    ContentAddressedStoreUninitializedError,
)
from .assistant_bridge_two_phase_state import TwoPhaseStateConfig
from .assistant_bridge_two_phase_contracts import TwoPhaseContractError
from .assistant_bridge_workflow_store import (
    SQLiteWorkflowStore,
    WorkflowArtifactError,
    WorkflowClockConflictError,
    WorkflowConfirmationNotReadyError,
    WorkflowNotFoundError,
    WorkflowRecord,
    WorkflowStoreError,
    WorkflowStoreUninitializedError,
)


TwoPhaseStatusErrorCode = Literal[
    "state_uninitialized",
    "workflow_not_found",
    "state_invalid",
    "clock_conflict",
    "artifact_invalid",
    "status_runtime_failed",
]
_STATUS_ERROR_MESSAGES: dict[TwoPhaseStatusErrorCode, str] = {
    "state_uninitialized": "Two-phase workflow state is not initialized.",
    "workflow_not_found": "Two-phase workflow was not found.",
    "state_invalid": "Two-phase workflow state is invalid.",
    "clock_conflict": "Two-phase workflow clock is inconsistent.",
    "artifact_invalid": "Two-phase workflow artifacts are invalid.",
    "status_runtime_failed": "Two-phase workflow status could not be read.",
}
_CANDIDATE_REQUIRED_STATES = frozenset({"staged", "attested", "ready", "applying"})


class TwoPhaseStatusError(ValueError):
    """Raised when durable workflow status cannot be read safely."""

    def __init__(self, code: TwoPhaseStatusErrorCode) -> None:
        self.code = code
        super().__init__(_STATUS_ERROR_MESSAGES[code])


class TwoPhaseAppliedReplayNotReadyError(ValueError):
    """Raised when an applied replay lacks its original confirmation."""


class TwoPhaseAppliedReplayError(ValueError):
    """Raised when applied replay authority cannot be trusted."""


class TwoPhaseAppliedReplayUnavailable(TwoPhaseAppliedReplayError):
    """Signals that initialized applied state is not available."""


class TwoPhaseStatusReader:
    """Read workflow state and validate artifacts required by live workflows.

    Candidate artifacts are authoritative for staged, attested, ready, and
    applying workflows, so their complete CAS closure is read and hashed before
    those states are reported. Applied, conflicted, failed, and expired states
    are historical outcomes and remain readable after candidate retention ends.
    """

    def __init__(
        self,
        store: SQLiteWorkflowStore,
        candidate_cas: ContentAddressedStore,
    ) -> None:
        self.store = store
        self.candidate_cas = candidate_cas

    def status(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord:
        try:
            record = self.store.read_workflow(workflow_id, now=now)
        except WorkflowNotFoundError as exc:
            raise TwoPhaseStatusError("workflow_not_found") from exc
        except WorkflowClockConflictError as exc:
            raise TwoPhaseStatusError("clock_conflict") from exc
        except WorkflowArtifactError as exc:
            raise TwoPhaseStatusError("artifact_invalid") from exc
        except (TwoPhaseContractError, WorkflowStoreError) as exc:
            raise TwoPhaseStatusError("state_invalid") from exc
        except Exception as exc:
            raise TwoPhaseStatusError("status_runtime_failed") from exc
        if record.status not in _CANDIDATE_REQUIRED_STATES:
            return record
        try:
            manifest, _ = self.candidate_cas.validate_candidate_closure(
                record.binding.manifest,
                record.binding.changeset,
            )
            source = manifest.get("source")
            if (
                manifest.get("sourceFingerprint") != record.binding.source_fingerprint
                or not isinstance(source, dict)
                or source.get("rootSha256") != record.workspace_root_sha256
            ):
                raise ContentAddressedStoreError(
                    "Candidate artifact binding is invalid."
                )
        except (ContentAddressedStoreError, TwoPhaseContractError) as exc:
            raise TwoPhaseStatusError("artifact_invalid") from exc
        except Exception as exc:
            raise TwoPhaseStatusError("status_runtime_failed") from exc
        return record

    def replay_applied(
        self,
        workflow_id: str,
        *,
        plan_id: str,
        confirmation_id: str,
        now: float | None = None,
    ) -> WorkflowRecord:
        try:
            record = self.store.read_applied_replay(
                workflow_id,
                plan_id=plan_id,
                confirmation_id=confirmation_id,
                now=now,
            )
            if record is None:
                raise TwoPhaseAppliedReplayError(
                    "Applied replay state changed while being read."
                )
            return record
        except WorkflowConfirmationNotReadyError as exc:
            raise TwoPhaseAppliedReplayNotReadyError(
                "Applied replay confirmation is unavailable."
            ) from exc
        except WorkflowStoreError as exc:
            raise TwoPhaseAppliedReplayError(
                "Applied replay authority is invalid."
            ) from exc
        except Exception as exc:
            raise TwoPhaseAppliedReplayError(
                "Applied replay authority could not be read."
            ) from exc


class TwoPhaseAppliedReplayReader:
    """Validate historical apply authority from durable metadata only."""

    def __init__(self, store: SQLiteWorkflowStore) -> None:
        self.store = store

    def replay(
        self,
        workflow_id: str,
        *,
        plan_id: str,
        confirmation_id: str,
        now: float | None = None,
    ) -> WorkflowRecord | None:
        try:
            return self.store.read_applied_replay(
                workflow_id,
                plan_id=plan_id,
                confirmation_id=confirmation_id,
                now=now,
            )
        except WorkflowNotFoundError:
            return None
        except WorkflowConfirmationNotReadyError as exc:
            raise TwoPhaseAppliedReplayNotReadyError(
                "Applied replay confirmation is unavailable."
            ) from exc
        except WorkflowStoreError as exc:
            raise TwoPhaseAppliedReplayError(
                "Applied replay authority is invalid."
            ) from exc
        except Exception as exc:
            raise TwoPhaseAppliedReplayError(
                "Applied replay authority could not be read."
            ) from exc


def build_two_phase_status_reader(
    config: TwoPhaseStateConfig,
) -> TwoPhaseStatusReader:
    try:
        cas = ContentAddressedStore(
            config.cas_path,
            create_if_missing=False,
        )
        store = SQLiteWorkflowStore(
            config.database_path,
            evidence_cas=cas,
            timeout=config.sqlite_timeout_seconds,
            read_only=True,
        )
    except (
        ContentAddressedStoreUninitializedError,
        WorkflowStoreUninitializedError,
    ) as exc:
        raise TwoPhaseStatusError("state_uninitialized") from exc
    except (ContentAddressedStoreError, WorkflowStoreError) as exc:
        raise TwoPhaseStatusError("state_invalid") from exc
    except Exception as exc:
        raise TwoPhaseStatusError("status_runtime_failed") from exc
    return TwoPhaseStatusReader(store, cas)


def build_two_phase_applied_replay_reader(
    config: TwoPhaseStateConfig,
) -> TwoPhaseAppliedReplayReader:
    try:
        store = SQLiteWorkflowStore(
            config.database_path,
            timeout=config.sqlite_timeout_seconds,
            replay_only=True,
        )
    except WorkflowStoreUninitializedError as exc:
        raise TwoPhaseAppliedReplayUnavailable(
            "Applied replay state is not initialized."
        ) from exc
    except WorkflowStoreError as exc:
        raise TwoPhaseAppliedReplayError("Applied replay state is invalid.") from exc
    except Exception as exc:
        raise TwoPhaseAppliedReplayError(
            "Applied replay state could not be opened."
        ) from exc
    return TwoPhaseAppliedReplayReader(store)
