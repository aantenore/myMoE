from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .assistant_bridge_attestation import AttestationTrustStore
from .assistant_bridge_cas import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from .assistant_bridge_integrity import canonical_sha256, sha256_bytes
from .assistant_bridge_two_phase_contracts import (
    CandidateBinding,
    ResumePlan,
    ResumeResult,
    StageReceipt,
    VerificationPolicy,
    require_sha256,
)
from .assistant_bridge_workflow_store import (
    SQLiteWorkflowStore,
    WorkflowRecord,
    WorkflowStoreError,
)
from .assistant_bridge_workspace import (
    WorkspaceChange,
    WorkspaceFile,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    WorkspaceSnapshot,
    apply_changeset,
    build_changeset,
    recover_workspace_transaction,
    snapshot_materialized,
    snapshot_workspace,
)


class TwoPhaseWorkflowError(ValueError):
    """Raised when a two-phase workflow cannot progress safely."""


@dataclass(frozen=True)
class TwoPhaseWorkflowConfig:
    workspace_policy: WorkspaceScopePolicy
    transaction_state_dir: str
    candidate_ttl_seconds: float = 24 * 60 * 60
    confirmation_ttl_seconds: float = 300
    transaction_lock_ttl_seconds: float = 120

    def __post_init__(self) -> None:
        if not self.transaction_state_dir:
            raise TwoPhaseWorkflowError("Transaction state directory is required.")
        for value, minimum, maximum, label in (
            (self.candidate_ttl_seconds, 1, 7 * 24 * 60 * 60, "candidate TTL"),
            (self.confirmation_ttl_seconds, 1, 3600, "confirmation TTL"),
            (self.transaction_lock_ttl_seconds, 1, 86_400, "transaction lock TTL"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise TwoPhaseWorkflowError(f"{label} is outside safe bounds.")


class TwoPhaseWorkflowService:
    """The only coordinator allowed to move staged candidates into a workspace."""

    def __init__(
        self,
        *,
        store: SQLiteWorkflowStore,
        cas: ContentAddressedStore,
        config: TwoPhaseWorkflowConfig,
        trust_store: AttestationTrustStore | None = None,
    ) -> None:
        self.store = store
        self.cas = cas
        self.config = config
        self.trust_store = trust_store

    def stage_candidate(
        self,
        *,
        source_workspace: str | Path,
        candidate_workspace: str | Path,
        task_fingerprint: str,
        config_sha256: str,
        verification_policy: VerificationPolicy,
        idempotency_key: str,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> StageReceipt:
        require_sha256(task_fingerprint, "task_fingerprint")
        require_sha256(config_sha256, "config_sha256")
        lifetime = (
            self.config.candidate_ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )
        if isinstance(lifetime, bool) or not 1 <= lifetime <= 7 * 24 * 60 * 60:
            raise TwoPhaseWorkflowError("Candidate TTL is outside safe bounds.")
        current = _now(now)
        try:
            source = snapshot_workspace(
                source_workspace, self.config.workspace_policy
            )
            candidate_files = _candidate_files(
                source,
                snapshot_materialized(
                    candidate_workspace, self.config.workspace_policy
                ),
            )
            changes = build_changeset(source.files, candidate_files)
            source_identity = _source_identity(source)
            manifest, changeset = self.cas.store_candidate(
                candidate_workspace,
                tuple(item.payload() for item in candidate_files),
                tuple(_change_payload(item) for item in changes),
                source_fingerprint=source.fingerprint,
                source_identity=source_identity,
            )
            workflow_id, challenge, stage_sha256 = self.store.stage_identity(
                idempotency_key
            )
            binding = CandidateBinding(
                workflow_id=workflow_id,
                stage_idempotency_sha256=stage_sha256,
                task_fingerprint=task_fingerprint,
                config_sha256=config_sha256,
                source_fingerprint=source.fingerprint,
                challenge_sha256=sha256_bytes(challenge.encode("utf-8")),
                manifest=manifest,
                changeset=changeset,
                verification_policy=verification_policy,
                created_at=current,
                expires_at=current + float(lifetime),
            )
            record, replay = self.store.create_workflow(
                binding,
                challenge=challenge,
                stage_idempotency_key=idempotency_key,
                workspace_root_sha256=str(source_identity["rootSha256"]),
                now=current,
            )
        except (
            ContentAddressedStoreError,
            OSError,
            ValueError,
            WorkspaceSecurityError,
            WorkflowStoreError,
        ) as exc:
            if isinstance(exc, TwoPhaseWorkflowError):
                raise
            raise TwoPhaseWorkflowError(str(exc)) from exc
        if record.status not in {"staged", "attested", "ready"}:
            raise TwoPhaseWorkflowError(
                "Stage idempotency replay targets a terminal workflow."
            )
        return StageReceipt(
            workflow_id=record.workflow_id,
            status=record.status,
            binding=record.binding,
            challenge=challenge,
            idempotent_replay=replay,
        )

    def status(
        self, workflow_id: str, *, now: float | None = None
    ) -> WorkflowRecord:
        try:
            return self.store.get_workflow(workflow_id, now=now)
        except (ValueError, WorkflowStoreError) as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc

    def record_attestation(
        self,
        workflow_id: str,
        envelope: bytes,
        *,
        now: float | None = None,
    ) -> WorkflowRecord:
        if self.trust_store is None:
            raise TwoPhaseWorkflowError(
                "Independent attestation trust is not configured."
            )
        try:
            record, _ = self.store.record_attestation_envelope(
                workflow_id,
                envelope,
                trust_store=self.trust_store,
                now=now,
            )
            return record
        except (ValueError, WorkflowStoreError) as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc

    def plan_resume(
        self,
        workflow_id: str,
        *,
        workspace: str | Path,
        idempotency_key: str,
        attestation_envelopes: Sequence[bytes] = (),
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> ResumePlan:
        current = _now(now)
        record = self.status(workflow_id, now=current)
        source = self._attest_original_workspace(record, workspace)
        if source is None:
            self._conflict(record, "source-drift-before-resume-plan", current)
            raise TwoPhaseWorkflowError(
                "Workspace source drifted after candidate staging."
            )
        for envelope in attestation_envelopes:
            record = self.record_attestation(workflow_id, envelope, now=current)
        record = self._reverify_attestations(workflow_id, now=current)
        try:
            return self.store.issue_resume_plan(
                workflow_id,
                idempotency_key=idempotency_key,
                ttl_seconds=(
                    self.config.confirmation_ttl_seconds
                    if ttl_seconds is None
                    else ttl_seconds
                ),
                now=current,
            )
        except (ValueError, WorkflowStoreError) as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc

    def apply_resume(
        self,
        workflow_id: str,
        *,
        workspace: str | Path,
        plan_id: str,
        confirmation_id: str,
        now: float | None = None,
    ) -> ResumeResult:
        current_time = _now(now)
        record = self.status(workflow_id, now=current_time)
        if record.status in {"staged", "attested", "ready"}:
            record = self._reverify_attestations(
                workflow_id, now=current_time
            )
        try:
            manifest, changeset = self.cas.load_candidate(
                record.binding.manifest, record.binding.changeset
            )
            source_files, candidate_files, changes = _artifact_workspace_state(
                manifest, changeset
            )
            self._validate_artifact_binding(record, manifest)
            current = snapshot_workspace(workspace, self.config.workspace_policy)
            self._validate_workspace_root(record, current)
        except (ContentAddressedStoreError, ValueError, WorkspaceSecurityError) as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc

        if record.status not in {"applying", "applied"}:
            if (
                current.fingerprint != record.binding.source_fingerprint
                or tuple(current.files) != source_files
            ):
                conflicted = self._conflict(
                    record, "source-drift-before-apply", current_time
                )
                return _resume_result(
                    conflicted,
                    code="source_drift",
                    idempotent_replay=False,
                )
        try:
            applying, confirmation_replay = self.store.consume_resume_confirmation(
                workflow_id,
                plan_id=plan_id,
                confirmation_id=confirmation_id,
                binding_sha256=record.binding.binding_sha256,
                now=current_time,
            )
        except (ValueError, WorkflowStoreError) as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc
        if applying.status == "applied":
            return _resume_result(
                applying,
                code="already_applied",
                idempotent_replay=True,
            )
        if applying.status != "applying" and confirmation_replay:
            return _resume_result(
                applying,
                code=_recovery_code(applying),
                idempotent_replay=True,
            )
        transaction_id = applying.apply_transaction_id
        if not transaction_id:
            raise TwoPhaseWorkflowError("Apply transaction identity is unavailable.")

        journal_exists = self._journal_exists(transaction_id)
        if confirmation_replay and journal_exists:
            return self._recover_and_require_confirmation(
                applying,
                workspace=workspace,
                transaction_id=transaction_id,
                now=current_time,
            )
        if confirmation_replay and _candidate_state_matches(
            current,
            candidate_files,
            source_identity=manifest["source"],
        ):
            return self._finalize_applied(
                applying,
                current,
                code="applied_recovered",
                now=current_time,
                idempotent_replay=True,
            )
        if (
            confirmation_replay
            and current.fingerprint == applying.binding.source_fingerprint
            and tuple(current.files) == source_files
        ):
            try:
                recovered, recovery_replay = self.store.reset_after_recovery(
                    workflow_id,
                    transaction_id=transaction_id,
                    now=current_time,
                )
            except WorkflowStoreError as exc:
                raise TwoPhaseWorkflowError(str(exc)) from exc
            return _resume_result(
                recovered,
                code=_recovery_code(recovered),
                idempotent_replay=(confirmation_replay or recovery_replay),
            )
        if current.fingerprint != applying.binding.source_fingerprint:
            conflicted = self._conflict(
                applying, "source-drift-while-applying", current_time
            )
            return _resume_result(
                conflicted,
                code="source_drift",
                idempotent_replay=confirmation_replay,
            )
        if not changes:
            return self._finalize_applied(
                applying,
                current,
                code="verified_no_change",
                now=current_time,
                idempotent_replay=confirmation_replay,
            )

        try:
            with self.cas.materialize_candidate(record.binding.manifest) as candidate:
                result = apply_changeset(
                    source_snapshot=current,
                    candidate_root=candidate,
                    candidate_files=candidate_files,
                    changes=changes,
                    policy=self.config.workspace_policy,
                    state_dir=self.config.transaction_state_dir,
                    transaction_id=transaction_id,
                    lock_ttl_seconds=self.config.transaction_lock_ttl_seconds,
                )
        except (ContentAddressedStoreError, WorkspaceSecurityError):
            if self._journal_exists(transaction_id):
                return self._recover_and_require_confirmation(
                    applying,
                    workspace=workspace,
                    transaction_id=transaction_id,
                    now=current_time,
                )
            try:
                after_error = snapshot_workspace(
                    workspace, self.config.workspace_policy
                )
            except WorkspaceSecurityError as exc:
                raise TwoPhaseWorkflowError(str(exc)) from exc
            if after_error.fingerprint == applying.binding.source_fingerprint:
                recovered, recovery_replay = self.store.reset_after_recovery(
                    workflow_id,
                    transaction_id=transaction_id,
                    now=current_time,
                )
                return _resume_result(
                    recovered,
                    code=_recovery_code(recovered),
                    idempotent_replay=(confirmation_replay or recovery_replay),
                )
            conflicted = self._conflict(
                applying, "unrecoverable-apply-drift", current_time
            )
            return _resume_result(
                conflicted,
                code="source_drift",
                idempotent_replay=confirmation_replay,
            )
        return self._finalize_applied(
            applying,
            result,
            code="applied",
            now=current_time,
            idempotent_replay=confirmation_replay,
        )

    def _attest_original_workspace(
        self, record: WorkflowRecord, workspace: str | Path
    ) -> WorkspaceSnapshot | None:
        try:
            current = snapshot_workspace(workspace, self.config.workspace_policy)
            self._validate_workspace_root(record, current)
        except WorkspaceSecurityError as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc
        return (
            current
            if current.fingerprint == record.binding.source_fingerprint
            else None
        )

    @staticmethod
    def _validate_workspace_root(
        record: WorkflowRecord, snapshot: WorkspaceSnapshot
    ) -> None:
        root_sha256 = snapshot.payload().get("root_sha256")
        if root_sha256 != record.workspace_root_sha256:
            raise TwoPhaseWorkflowError("Workflow targets another workspace root.")

    @staticmethod
    def _validate_artifact_binding(
        record: WorkflowRecord, manifest: Mapping[str, Any]
    ) -> None:
        if manifest.get("sourceFingerprint") != record.binding.source_fingerprint:
            raise TwoPhaseWorkflowError("Candidate source artifact binding is invalid.")
        source = manifest.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("rootSha256") != record.workspace_root_sha256
        ):
            raise TwoPhaseWorkflowError("Candidate workspace root binding is invalid.")

    def _recover_and_require_confirmation(
        self,
        record: WorkflowRecord,
        *,
        workspace: str | Path,
        transaction_id: str,
        now: float,
    ) -> ResumeResult:
        try:
            recover_workspace_transaction(
                state_dir=self.config.transaction_state_dir,
                transaction_id=transaction_id,
                source_root=workspace,
                lock_ttl_seconds=self.config.transaction_lock_ttl_seconds,
            )
            recovered_snapshot = snapshot_workspace(
                workspace, self.config.workspace_policy
            )
        except WorkspaceSecurityError:
            conflicted = self._conflict(
                record, "workspace-recovery-failed", now
            )
            return _resume_result(
                conflicted,
                code="recovery_failed",
                idempotent_replay=True,
            )
        if recovered_snapshot.fingerprint != record.binding.source_fingerprint:
            conflicted = self._conflict(
                record, "workspace-recovery-source-drift", now
            )
            return _resume_result(
                conflicted,
                code="recovery_failed",
                idempotent_replay=True,
            )
        try:
            ready, _ = self.store.reset_after_recovery(
                record.workflow_id,
                transaction_id=transaction_id,
                now=now,
            )
        except WorkflowStoreError as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc
        return _resume_result(
            ready,
            code=_recovery_code(ready),
            idempotent_replay=True,
        )

    def _reverify_attestations(
        self,
        workflow_id: str,
        *,
        now: float,
    ) -> WorkflowRecord:
        if self.trust_store is None:
            raise TwoPhaseWorkflowError(
                "Independent attestation trust is not configured."
            )
        try:
            return self.store.reverify_attestations(
                workflow_id,
                trust_store=self.trust_store,
                now=now,
            )
        except (ValueError, WorkflowStoreError) as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc

    def _finalize_applied(
        self,
        record: WorkflowRecord,
        snapshot: WorkspaceSnapshot,
        *,
        code: str,
        now: float,
        idempotent_replay: bool,
    ) -> ResumeResult:
        result_sha256 = canonical_sha256(
            {
                "workflowId": record.workflow_id,
                "bindingSha256": record.binding.binding_sha256,
                "candidateContentSha256": record.binding.candidate_content_sha256,
                "transactionId": record.apply_transaction_id,
                "workspace": snapshot.payload(),
            }
        )
        try:
            applied, store_replay = self.store.mark_applied(
                record.workflow_id,
                transaction_id=record.apply_transaction_id,
                result_sha256=result_sha256,
                now=now,
            )
        except WorkflowStoreError as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc
        return _resume_result(
            applied,
            code=code,
            idempotent_replay=idempotent_replay or store_replay,
        )

    def _conflict(
        self, record: WorkflowRecord, reason: str, now: float
    ) -> WorkflowRecord:
        try:
            return self.store.mark_conflicted(
                record.workflow_id, reason=reason, now=now
            )
        except WorkflowStoreError as exc:
            raise TwoPhaseWorkflowError(str(exc)) from exc

    def _journal_exists(self, transaction_id: str) -> bool:
        require_sha256(transaction_id, "transaction_id")
        raw = Path(self.config.transaction_state_dir).expanduser()
        try:
            raw_state = raw.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(raw_state.st_mode) or not stat.S_ISDIR(raw_state.st_mode):
            raise TwoPhaseWorkflowError("Workspace transaction state root is unsafe.")
        try:
            state = raw.resolve(strict=True)
        except OSError as exc:
            raise TwoPhaseWorkflowError(
                "Workspace transaction state root is unavailable."
            ) from exc
        transaction = state / f"transaction-{transaction_id}"
        journal = transaction / "journal.json"
        try:
            metadata = journal.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise TwoPhaseWorkflowError("Workspace transaction journal is unsafe.")
        try:
            journal.resolve(strict=True).relative_to(state)
        except (OSError, ValueError) as exc:
            raise TwoPhaseWorkflowError(
                "Workspace transaction journal escaped its state root."
            ) from exc
        return True


def _source_identity(snapshot: WorkspaceSnapshot) -> dict[str, object]:
    return {
        "rootSha256": snapshot.payload()["root_sha256"],
        "fingerprint": snapshot.fingerprint,
        "gitRepository": snapshot.git_repository,
        "headSha": snapshot.head_sha if snapshot.git_repository else None,
        "indexSha256": snapshot.index_sha256,
    }


def _candidate_files(
    source: WorkspaceSnapshot,
    observed: Sequence[WorkspaceFile],
) -> tuple[WorkspaceFile, ...]:
    result = {item.path: item for item in observed}
    source_files = {item.path: item for item in source.files}
    empty_sha256 = sha256_bytes(b"")
    for path in source.tracked_paths:
        if path in result:
            continue
        previous = source_files.get(path)
        result[path] = WorkspaceFile(
            path=path,
            kind="missing",
            sha256=empty_sha256,
            size=0,
            mode=0,
            direction=(previous.direction if previous else "round_trip"),
        )
    return tuple(sorted(result.values()))


def _change_payload(change: WorkspaceChange) -> dict[str, object]:
    return {
        "path": change.path,
        "before": None if change.before is None else change.before.payload(),
        "after": None if change.after is None else change.after.payload(),
    }


def _artifact_workspace_state(
    manifest: Mapping[str, Any], changeset: Mapping[str, Any]
) -> tuple[
    tuple[WorkspaceFile, ...],
    tuple[WorkspaceFile, ...],
    tuple[WorkspaceChange, ...],
]:
    raw_files = manifest.get("files")
    raw_changes = changeset.get("changes")
    if not isinstance(raw_files, list) or not isinstance(raw_changes, list):
        raise TwoPhaseWorkflowError("Candidate artifact records are invalid.")
    candidate_files = tuple(
        _workspace_file(item, content_field=True) for item in raw_files
    )
    changes: list[WorkspaceChange] = []
    source = {item.path: item for item in candidate_files}
    for raw in raw_changes:
        if not isinstance(raw, Mapping):
            raise TwoPhaseWorkflowError("Candidate change record is invalid.")
        path = raw.get("path")
        before_raw = raw.get("before")
        after_raw = raw.get("after")
        before = (
            None
            if before_raw is None
            else _workspace_file(before_raw, content_field=False)
        )
        after = (
            None
            if after_raw is None
            else _workspace_file(after_raw, content_field=False)
        )
        if not isinstance(path, str) or any(
            item is not None and item.path != path for item in (before, after)
        ):
            raise TwoPhaseWorkflowError("Candidate change path binding is invalid.")
        changes.append(WorkspaceChange(path=path, before=before, after=after))
        if before is None:
            source.pop(path, None)
        else:
            source[path] = before
    source_files = tuple(sorted(source.values()))
    expected = build_changeset(source_files, candidate_files)
    if tuple(changes) != expected:
        raise TwoPhaseWorkflowError("Candidate changeset reconstruction is invalid.")
    return source_files, candidate_files, tuple(changes)


def _workspace_file(value: Any, *, content_field: bool) -> WorkspaceFile:
    if not isinstance(value, Mapping):
        raise TwoPhaseWorkflowError("Candidate file record is invalid.")
    expected = {"path", "kind", "sha256", "size", "mode", "direction"}
    if content_field:
        expected.add("content")
    if set(value) != expected:
        raise TwoPhaseWorkflowError("Candidate file record shape is invalid.")
    return WorkspaceFile(
        path=value["path"],
        kind=value["kind"],
        sha256=value["sha256"],
        size=value["size"],
        mode=value["mode"],
        direction=value["direction"],
    )


def _candidate_state_matches(
    snapshot: WorkspaceSnapshot,
    candidate_files: Sequence[WorkspaceFile],
    *,
    source_identity: Mapping[str, Any],
) -> bool:
    return (
        tuple(snapshot.files) == tuple(candidate_files)
        and snapshot.git_repository == source_identity.get("gitRepository")
        and (snapshot.head_sha if snapshot.git_repository else None)
        == source_identity.get("headSha")
        and snapshot.index_sha256 == source_identity.get("indexSha256")
        and snapshot.payload().get("root_sha256")
        == source_identity.get("rootSha256")
    )


def _resume_result(
    record: WorkflowRecord,
    *,
    code: str,
    idempotent_replay: bool,
) -> ResumeResult:
    return ResumeResult(
        workflow_id=record.workflow_id,
        status=record.status,
        code=code,
        candidate_fingerprint=record.binding.candidate_fingerprint,
        transaction_id=(
            record.apply_transaction_id
            or record.recovered_transaction_id
            or None
        ),
        result_sha256=record.result_sha256 or None,
        idempotent_replay=idempotent_replay,
    )


def _recovery_code(record: WorkflowRecord) -> str:
    return (
        "recovered_expired"
        if record.status == "expired"
        else "recovered_confirmation_required"
    )


def _now(value: float | None) -> float:
    import time

    current = time.time() if value is None else value
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise TwoPhaseWorkflowError("Workflow timestamp is invalid.")
    result = float(current)
    if result < 0 or not math.isfinite(result):
        raise TwoPhaseWorkflowError("Workflow timestamp is invalid.")
    return result
