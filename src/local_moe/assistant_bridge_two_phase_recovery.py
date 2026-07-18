from __future__ import annotations

import os
from pathlib import Path
import stat

from .assistant_bridge_integrity import sha256_bytes
from .assistant_bridge_two_phase_contracts import ResumeResult
from .assistant_bridge_two_phase_state import TwoPhaseStateConfig
from .assistant_bridge_workflow_store import (
    SQLiteWorkflowStore,
    WorkflowNotFoundError,
    WorkflowRecord,
    WorkflowStoreError,
    WorkflowStoreUninitializedError,
)
from .assistant_bridge_workspace import (
    WorkspaceRecoveryPolicyUnavailable,
    WorkspaceSecurityError,
    finalize_recovered_workspace_transaction,
    recover_workspace_transaction,
)


class TwoPhaseApplyingRecoveryError(ValueError):
    """Raised when an in-flight workspace transaction cannot be recovered."""


class TwoPhaseApplyingRecoveryUnavailable(TwoPhaseApplyingRecoveryError):
    """Signals that no initialized applying state exists to recover."""


class TwoPhaseApplyingRecovery:
    """Rollback-only service for a durably authorized applying workflow.

    The service deliberately has no candidate store, trust adapter, provider,
    or application-policy dependency. It can only roll back a journal whose
    transaction is already bound to an ``applying`` database record.
    """

    def __init__(
        self,
        store: SQLiteWorkflowStore,
        *,
        transaction_state_dir: Path,
        transaction_lock_ttl_seconds: float,
    ) -> None:
        self.store = store
        self.transaction_state_dir = transaction_state_dir
        self.transaction_lock_ttl_seconds = transaction_lock_ttl_seconds

    def recover_if_applying(
        self,
        workflow_id: str,
        *,
        workspace: str | Path,
        now: float | None = None,
    ) -> ResumeResult | None:
        try:
            record = self.store.read_applying_recovery(workflow_id, now=now)
            if record is None:
                cleanup = self.store.read_recovered_cleanup(
                    workflow_id,
                    now=now,
                )
                if cleanup is None:
                    return None
                recovered, transaction_id = cleanup
                if not _has_recovery_journal(
                    self.transaction_state_dir,
                    transaction_id,
                ):
                    return None
                root = _recovery_workspace_root(workspace)
                if recovery_workspace_root_sha256(root) != (
                    recovered.workspace_root_sha256
                ):
                    raise TwoPhaseApplyingRecoveryError(
                        "Applying recovery targets another workspace root."
                    )
                finalize_recovered_workspace_transaction(
                    state_dir=self.transaction_state_dir,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=(recovered.binding.source_fingerprint),
                    lock_ttl_seconds=self.transaction_lock_ttl_seconds,
                )
                return _recovery_result(recovered, transaction_id)
            transaction_id = record.apply_transaction_id
            if not _has_recovery_journal(
                self.transaction_state_dir,
                transaction_id,
            ):
                return None
            root = _recovery_workspace_root(workspace)
            if recovery_workspace_root_sha256(root) != record.workspace_root_sha256:
                raise TwoPhaseApplyingRecoveryError(
                    "Applying recovery targets another workspace root."
                )
            recover_workspace_transaction(
                state_dir=self.transaction_state_dir,
                transaction_id=transaction_id,
                source_root=root,
                lock_ttl_seconds=self.transaction_lock_ttl_seconds,
                expected_source_fingerprint=record.binding.source_fingerprint,
                retain_recovered_journal=True,
            )
            recovered, _ = self.store.reset_applying_after_recovery(
                workflow_id,
                transaction_id=transaction_id,
                now=now,
            )
            try:
                finalize_recovered_workspace_transaction(
                    state_dir=self.transaction_state_dir,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=(record.binding.source_fingerprint),
                    lock_ttl_seconds=self.transaction_lock_ttl_seconds,
                )
            except (OSError, ValueError, WorkspaceSecurityError):
                pass
        except WorkflowNotFoundError:
            return None
        except WorkspaceRecoveryPolicyUnavailable as exc:
            raise TwoPhaseApplyingRecoveryUnavailable(
                "Applying recovery needs the current lifecycle configuration."
            ) from exc
        except TwoPhaseApplyingRecoveryError:
            raise
        except (OSError, ValueError, WorkspaceSecurityError, WorkflowStoreError) as exc:
            raise TwoPhaseApplyingRecoveryError(
                "Applying workflow recovery failed."
            ) from exc
        return _recovery_result(recovered, transaction_id)


def build_two_phase_applying_recovery(
    config: TwoPhaseStateConfig,
) -> TwoPhaseApplyingRecovery:
    try:
        store = SQLiteWorkflowStore(
            config.database_path,
            timeout=config.sqlite_timeout_seconds,
            recovery_only=True,
        )
    except WorkflowStoreUninitializedError as exc:
        raise TwoPhaseApplyingRecoveryUnavailable(
            "Applying workflow recovery state is not initialized."
        ) from exc
    except WorkflowStoreError as exc:
        raise TwoPhaseApplyingRecoveryError(
            "Applying workflow recovery state is invalid."
        ) from exc
    return TwoPhaseApplyingRecovery(
        store,
        transaction_state_dir=config.transaction_state_dir,
        transaction_lock_ttl_seconds=config.transaction_lock_ttl_seconds,
    )


def recovery_workspace_root_sha256(workspace: str | Path) -> str:
    """Return the same canonical root binding used by workspace snapshots."""

    root = _recovery_workspace_root(workspace)
    return sha256_bytes(str(root).encode("utf-8"))


def _recovery_result(record: WorkflowRecord, transaction_id: str) -> ResumeResult:
    status = record.status
    return ResumeResult(
        workflow_id=record.workflow_id,
        status=status,
        code=(
            "recovered_expired"
            if status == "expired"
            else "recovered_confirmation_required"
        ),
        candidate_fingerprint=record.binding.candidate_fingerprint,
        transaction_id=transaction_id,
        result_sha256=None,
        idempotent_replay=True,
    )


def _has_recovery_journal(state_dir: Path, transaction_id: str) -> bool:
    transaction = state_dir / f"transaction-{transaction_id}"
    journal = transaction / "journal.json"
    try:
        transaction_state = transaction.lstat()
        journal_state = journal.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TwoPhaseApplyingRecoveryError(
            "Applying recovery journal is unavailable."
        ) from exc
    if (
        stat.S_ISLNK(transaction_state.st_mode)
        or not stat.S_ISDIR(transaction_state.st_mode)
        or stat.S_ISLNK(journal_state.st_mode)
        or not stat.S_ISREG(journal_state.st_mode)
    ):
        raise TwoPhaseApplyingRecoveryError("Applying recovery journal is invalid.")
    return True


def _recovery_workspace_root(workspace: str | Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(workspace).expanduser())))
    try:
        state = raw.lstat()
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TwoPhaseApplyingRecoveryError(
            "Applying recovery workspace is unavailable."
        ) from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise TwoPhaseApplyingRecoveryError("Applying recovery workspace is invalid.")
    return resolved
