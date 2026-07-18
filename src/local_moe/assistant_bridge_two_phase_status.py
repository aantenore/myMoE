from __future__ import annotations

from .assistant_bridge_cas import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from .assistant_bridge_two_phase_state import TwoPhaseStateConfig
from .assistant_bridge_workflow_store import (
    SQLiteWorkflowStore,
    WorkflowRecord,
    WorkflowStoreError,
)


class TwoPhaseStatusError(ValueError):
    """Raised when durable workflow status cannot be read safely."""


class TwoPhaseStatusReader:
    """Read workflow state without importing providers or trust adapters."""

    def __init__(self, store: SQLiteWorkflowStore) -> None:
        self.store = store

    def status(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord:
        try:
            return self.store.read_workflow(workflow_id, now=now)
        except (ValueError, WorkflowStoreError) as exc:
            raise TwoPhaseStatusError(str(exc)) from exc


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
    except (ContentAddressedStoreError, WorkflowStoreError) as exc:
        raise TwoPhaseStatusError(str(exc)) from exc
    return TwoPhaseStatusReader(store)
