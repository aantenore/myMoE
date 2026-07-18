from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import base64
import binascii
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import time
from typing import Any, Iterator, Mapping

from platformdirs import user_state_path

from .assistant_bridge_cas import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from .assistant_bridge_integrity import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from .assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    CandidateBinding,
    IndependentAttestation,
    ResumePlan,
    TwoPhaseContractError,
    WORKFLOW_STATES,
    require_safe_id,
    require_sha256,
)
from .assistant_bridge_two_phase_ports import EvidenceStore


WORKFLOW_STORE_SCHEMA_VERSION = "2.0"
_ENVELOPE_MEDIA_TYPE = "application/vnd.dsse.envelope+json"
_STATEMENT_MEDIA_TYPE = "application/vnd.in-toto+json"
_SECRET_BYTES = 32
_MAX_ATTESTATION_ARTIFACT_BYTES = 8 * 1024 * 1024
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class WorkflowStoreError(ValueError):
    """Raised when durable two-phase workflow state cannot be trusted."""


class WorkflowStoreUninitializedError(WorkflowStoreError):
    """Raised when durable workflow state has not been initialized yet."""


class WorkflowNotFoundError(WorkflowStoreError):
    """Raised when a requested workflow does not exist."""


class WorkflowClockConflictError(WorkflowStoreError):
    """Raised when workflow time moves behind its durable watermark."""


class WorkflowArtifactError(WorkflowStoreError):
    """Raised when durable workflow evidence cannot be validated."""


class WorkflowOperationConflictError(WorkflowStoreError):
    """Raised when an idempotency key is bound to another operation."""

    code = "operation_conflict"


class WorkflowConfirmationNotReadyError(WorkflowStoreError):
    """Raised when resume requires a fresh confirmation plan."""

    code = "confirmation_not_ready"


@dataclass(frozen=True)
class WorkflowStatePaths:
    database: Path
    cas_root: Path


@dataclass(frozen=True)
class RecordedAttestation:
    verifier_id: str
    adapter_id: str
    key_id: str
    attestation_id: str
    evidence_sha256: str
    statement_sha256: str
    envelope: ArtifactDescriptor
    statement: ArtifactDescriptor
    issued_at: float
    expires_at: float
    recorded_at: float

    def __post_init__(self) -> None:
        try:
            for value, label in (
                (self.verifier_id, "recorded verifier_id"),
                (self.adapter_id, "recorded adapter_id"),
                (self.key_id, "recorded key_id"),
                (self.attestation_id, "recorded attestation_id"),
            ):
                require_safe_id(value, label)
            require_sha256(self.evidence_sha256, "recorded evidence_sha256")
            require_sha256(self.statement_sha256, "recorded statement_sha256")
        except TwoPhaseContractError as exc:
            raise WorkflowStoreError(
                "Recorded attestation identity is invalid."
            ) from exc
        if (
            self.envelope.media_type != _ENVELOPE_MEDIA_TYPE
            or self.envelope.sha256 != self.evidence_sha256
            or self.statement.media_type != _STATEMENT_MEDIA_TYPE
            or self.statement.sha256 != self.statement_sha256
        ):
            raise WorkflowStoreError("Recorded evidence descriptors are incoherent.")
        issued = _wall_time(self.issued_at)
        expires = _wall_time(self.expires_at)
        recorded = _wall_time(self.recorded_at)
        if not issued < expires or not issued <= recorded <= expires:
            raise WorkflowStoreError("Recorded attestation timeline is invalid.")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "recorded_at", recorded)

    def payload(self) -> dict[str, object]:
        return {
            "verifierId": self.verifier_id,
            "adapterId": self.adapter_id,
            "keyId": self.key_id,
            "attestationId": self.attestation_id,
            "evidenceSha256": self.evidence_sha256,
            "statementSha256": self.statement_sha256,
            "envelope": self.envelope.payload(),
            "statement": self.statement.payload(),
            "issuedAt": self.issued_at,
            "expiresAt": self.expires_at,
            "recordedAt": self.recorded_at,
        }


@dataclass(frozen=True)
class WorkflowRecord:
    workflow_id: str
    status: str
    binding: CandidateBinding
    workspace_root_sha256: str
    attestations: tuple[RecordedAttestation, ...]
    active_attestation_count: int
    apply_transaction_id: str
    recovered_transaction_id: str
    result_sha256: str
    created_at: float
    updated_at: float
    last_wall_time: float

    def __post_init__(self) -> None:
        try:
            require_safe_id(self.workflow_id, "workflow_id")
        except TwoPhaseContractError as exc:
            raise WorkflowStoreError("Workflow identity is invalid.") from exc
        if self.status not in WORKFLOW_STATES:
            raise WorkflowStoreError("Workflow status is invalid.")
        try:
            require_sha256(self.workspace_root_sha256, "workspace_root_sha256")
            if self.apply_transaction_id:
                require_sha256(self.apply_transaction_id, "apply_transaction_id")
            if self.recovered_transaction_id:
                require_sha256(
                    self.recovered_transaction_id, "recovered_transaction_id"
                )
            if self.result_sha256:
                require_sha256(self.result_sha256, "result_sha256")
        except TwoPhaseContractError as exc:
            raise WorkflowStoreError("Workflow digest state is invalid.") from exc
        if self.status == "applying":
            if not self.apply_transaction_id or self.result_sha256:
                raise WorkflowStoreError("Applying workflow state is incoherent.")
        elif self.status == "applied":
            if not self.apply_transaction_id or not self.result_sha256:
                raise WorkflowStoreError("Applied workflow state is incoherent.")
        elif self.apply_transaction_id or self.result_sha256:
            raise WorkflowStoreError("Non-apply workflow carries active apply state.")
        ordered = tuple(sorted(self.attestations, key=lambda item: item.verifier_id))
        if len({item.verifier_id for item in ordered}) != len(ordered):
            raise WorkflowStoreError("Workflow repeats an attestation verifier.")
        object.__setattr__(self, "attestations", ordered)
        if (
            isinstance(self.active_attestation_count, bool)
            or not isinstance(self.active_attestation_count, int)
            or not 0 <= self.active_attestation_count <= len(ordered)
        ):
            raise WorkflowStoreError("Active attestation count is invalid.")
        created = _wall_time(self.created_at)
        updated = _wall_time(self.updated_at)
        last_wall = _wall_time(self.last_wall_time)
        if not created <= updated <= last_wall:
            raise WorkflowStoreError("Workflow timestamps are out of order.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "last_wall_time", last_wall)

    @property
    def quorum_satisfied(self) -> bool:
        return self.active_attestation_count >= self.binding.verification_policy.quorum

    def quorum_satisfied_at(self, now: float) -> bool:
        current = _wall_time(now)
        return (
            sum(
                item.issued_at <= current <= item.expires_at
                for item in self.attestations
            )
            >= self.binding.verification_policy.quorum
        )

    def quorum_expires_at(self, now: float) -> float:
        current = _wall_time(now)
        expiries = sorted(
            (
                item.expires_at
                for item in self.attestations
                if item.issued_at <= current <= item.expires_at
            ),
            reverse=True,
        )
        quorum = self.binding.verification_policy.quorum
        if len(expiries) < quorum:
            raise WorkflowStoreError("Workflow has no currently valid quorum.")
        return expiries[quorum - 1]

    def payload(self) -> dict[str, object]:
        return {
            "schemaVersion": WORKFLOW_STORE_SCHEMA_VERSION,
            "workflowId": self.workflow_id,
            "status": self.status,
            "binding": self.binding.payload(),
            "bindingSha256": self.binding.binding_sha256,
            "workspaceRootSha256": self.workspace_root_sha256,
            "attestations": [item.payload() for item in self.attestations],
            "quorum": {
                "required": self.binding.verification_policy.quorum,
                "verified": self.active_attestation_count,
                "recorded": len(self.attestations),
                "satisfied": self.quorum_satisfied,
            },
            "applyTransactionId": self.apply_transaction_id or None,
            "recoveredTransactionId": self.recovered_transaction_id or None,
            "resultSha256": self.result_sha256 or None,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


@dataclass(frozen=True)
class WorkflowEvent:
    sequence: int
    workflow_id: str
    event_type: str
    event_key: str
    payload_sha256: str
    occurred_at: float


def default_workflow_state_paths() -> WorkflowStatePaths:
    base = user_state_path("myMoE", appauthor=False, ensure_exists=False)
    root = Path(base) / "assistant-bridge" / "v2"
    return WorkflowStatePaths(
        database=root / "workflows.sqlite3",
        cas_root=root / "cas",
    )


class SQLiteWorkflowStore:
    """Durable, quorum-aware, replay-safe workflow state and event journal."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        evidence_cas: EvidenceStore | None = None,
        timeout: float = 5.0,
        read_only: bool = False,
        recovery_only: bool = False,
        replay_only: bool = False,
    ) -> None:
        if (
            not isinstance(read_only, bool)
            or not isinstance(recovery_only, bool)
            or not isinstance(replay_only, bool)
        ):
            raise WorkflowStoreError("Workflow store access mode is invalid.")
        if sum((read_only, recovery_only, replay_only)) > 1:
            raise WorkflowStoreError("Workflow store access modes are incompatible.")
        if (read_only or recovery_only or replay_only) and path is None:
            raise WorkflowStoreError(
                "Read-only workflow status requires an explicit database path."
            )
        if (recovery_only or replay_only) and evidence_cas is not None:
            raise WorkflowStoreError(
                "Metadata-only workflow access cannot use artifact storage."
            )
        self._read_only = read_only or replay_only
        self._status_read_only = read_only
        self._recovery_only = recovery_only
        self._replay_only = replay_only
        if read_only or recovery_only or replay_only:
            assert path is not None
            self.path = _prepare_existing_database_path(Path(path))
            cas_root = self.path.parent / "cas"
        elif path is None:
            defaults = default_workflow_state_paths()
            self.path = _prepare_database_path(defaults.database)
            cas_root = defaults.cas_root
        else:
            self.path = _prepare_database_path(Path(path))
            cas_root = self.path.parent / "cas"
        if recovery_only or replay_only:
            self.evidence_cas: EvidenceStore | None = None
        else:
            try:
                self.evidence_cas = (
                    ContentAddressedStore(
                        cas_root,
                        create_if_missing=not read_only,
                    )
                    if evidence_cas is None
                    else evidence_cas
                )
            except ContentAddressedStoreError as exc:
                raise WorkflowStoreError(str(exc)) from exc
        if not 0.1 <= timeout <= 60:
            raise WorkflowStoreError("SQLite workflow timeout is outside safe bounds.")
        self.timeout = timeout
        self._database_identity: tuple[int, int] | None = None
        self._secret = (
            _load_existing_secret(self.path.with_suffix(".key"))
            if read_only or recovery_only or replay_only
            else _load_or_create_secret(self.path.with_suffix(".key"))
        )
        if read_only or recovery_only or replay_only:
            self._database_identity = _database_identity(self.path)
            with self._connect() as connection:
                self._validate_existing_schema(connection)
            return
        with self._connect() as connection:
            self._initialize(connection)
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise WorkflowStoreError(
                "Workflow database permissions are unavailable."
            ) from exc
        self._database_identity = _database_identity(self.path)

    def stage_identity(self, idempotency_key: str) -> tuple[str, str, str]:
        key = _idempotency_key(idempotency_key)
        digest = sha256_bytes(key)
        workflow_mac = self._mac(b"workflow\x00" + key).hex()
        challenge_mac = self._mac(b"challenge\x00" + key)
        workflow_id = f"wf-{workflow_mac[:32]}"
        challenge = base64.urlsafe_b64encode(challenge_mac).decode("ascii").rstrip("=")
        return workflow_id, challenge, digest

    def create_workflow(
        self,
        binding: CandidateBinding,
        *,
        challenge: str,
        stage_idempotency_key: str,
        workspace_root_sha256: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        current = _wall_time(now)
        require_sha256(workspace_root_sha256, "workspace_root_sha256")
        workflow_id, expected_challenge, idempotency_sha256 = self.stage_identity(
            stage_idempotency_key
        )
        if (
            binding.workflow_id != workflow_id
            or challenge != expected_challenge
            or binding.stage_idempotency_sha256 != idempotency_sha256
            or sha256_bytes(challenge.encode("utf-8")) != binding.challenge_sha256
        ):
            raise WorkflowStoreError("Stage identity or challenge binding is invalid.")
        if not binding.created_at <= current <= binding.expires_at:
            raise WorkflowStoreError("Candidate binding is not currently valid.")
        binding_json = canonical_json_bytes(binding.payload()).decode("utf-8")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ? "
                "OR stage_idempotency_sha256 = ?",
                (binding.workflow_id, idempotency_sha256),
            ).fetchone()
            if row is not None:
                record = self._record(connection, row, active_at=current)
                if (
                    record.workflow_id != binding.workflow_id
                    or not _same_stage_operation(record.binding, binding)
                    or record.workspace_root_sha256 != workspace_root_sha256
                ):
                    raise WorkflowStoreError(
                        "Stage idempotency key is already bound to another candidate."
                    )
                return record, True
            connection.execute(
                """
                INSERT INTO workflows (
                    workflow_id, stage_idempotency_sha256, status,
                    binding_json, binding_sha256, workspace_root_sha256,
                    created_at, expires_at, updated_at, last_wall_time
                ) VALUES (?, ?, 'staged', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    binding.workflow_id,
                    idempotency_sha256,
                    binding_json,
                    binding.binding_sha256,
                    workspace_root_sha256,
                    binding.created_at,
                    binding.expires_at,
                    current,
                    current,
                ),
            )
            self._append_event(
                connection,
                binding.workflow_id,
                event_type="candidate_staged",
                event_key=f"stage:{idempotency_sha256}",
                payload={
                    "bindingSha256": binding.binding_sha256,
                    "candidateContentSha256": binding.candidate_content_sha256,
                },
                now=current,
            )
            return self._selected_record(
                connection, binding.workflow_id, active_at=current
            ), False

    def get_workflow(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord:
        record = self.find_workflow(workflow_id, now=now)
        if record is None:
            raise WorkflowStoreError("Workflow was not found.")
        return record

    def read_workflow(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord:
        """Return a current status view without persisting derived transitions."""

        if not self._status_read_only:
            raise WorkflowStoreError("Read-only workflow access was not configured.")
        require_safe_id(workflow_id, "workflow_id")
        current = _wall_time(now)
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                row = connection.execute(
                    "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
                ).fetchone()
                if row is None:
                    raise WorkflowNotFoundError("Workflow was not found.")
                record = self._record(connection, row, active_at=current)
                self._check_clock(record, current)
            finally:
                connection.rollback()
        if record.status in {"applying", "applied", "conflicted", "failed", "expired"}:
            return record
        if current > record.binding.expires_at:
            return replace(record, status="expired")
        if record.quorum_satisfied:
            status = "ready"
        elif record.active_attestation_count:
            status = "attested"
        else:
            status = "staged"
        return record if status == record.status else replace(record, status=status)

    def read_applying_recovery(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord | None:
        """Read only the durable authority needed to roll back an applying write.

        This path deliberately validates binding, transaction, database, and
        attestation metadata without dereferencing candidate or evidence CAS
        objects. It never reports any state except ``applying`` and cannot be
        used to grant fresh write authority.
        """

        if not self._recovery_only:
            raise WorkflowStoreError("Applying recovery access was not configured.")
        require_safe_id(workflow_id, "workflow_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise WorkflowNotFoundError("Workflow was not found.")
            record = self._record(
                connection,
                row,
                active_at=current,
                validate_artifacts=False,
            )
            self._check_clock(record, current)
            if record.status != "applying":
                return None
            self._validate_applying_recovery_binding(connection, record)
            return record

    def read_recovered_cleanup(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, str] | None:
        """Read a committed recovery whose retained journal needs cleanup."""

        if not self._recovery_only:
            raise WorkflowStoreError("Applying recovery access was not configured.")
        require_safe_id(workflow_id, "workflow_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise WorkflowNotFoundError("Workflow was not found.")
            record = self._record(
                connection,
                row,
                active_at=current,
                validate_artifacts=False,
            )
            self._check_clock(record, current)
            transaction_id = record.recovered_transaction_id
            if record.status == "applying" or not transaction_id:
                return None
            recovery = connection.execute(
                "SELECT workflow_id FROM apply_recoveries WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
            if recovery is None or str(recovery["workflow_id"]) != workflow_id:
                raise WorkflowStoreError(
                    "Workflow recovery transaction binding is invalid."
                )
            self._validate_recovery_transaction_binding(
                connection,
                record,
                transaction_id,
            )
            if current > record.binding.expires_at and record.status != "expired":
                record = replace(record, status="expired")
            return record, transaction_id

    def read_applied_cleanup(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord | None:
        """Read metadata needed only to clean an already-applied transaction."""

        if not self._recovery_only:
            raise WorkflowStoreError(
                "Committed apply cleanup access was not configured."
            )
        require_safe_id(workflow_id, "workflow_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise WorkflowNotFoundError("Workflow was not found.")
            record = self._record(
                connection,
                row,
                active_at=current,
                validate_artifacts=False,
            )
            self._check_clock(record, current)
            if record.status != "applied" or not record.apply_transaction_id:
                return None
            self._validate_recovery_transaction_binding(
                connection,
                record,
                record.apply_transaction_id,
            )
            self._validate_applied_event_binding(
                connection,
                record,
                transaction_id=record.apply_transaction_id,
                result_sha256=record.result_sha256,
            )
            return record

    def read_applied_replay(
        self,
        workflow_id: str,
        *,
        plan_id: str,
        confirmation_id: str,
        now: float | None = None,
    ) -> WorkflowRecord | None:
        """Validate the consumed authority bound to one historical apply."""

        if not self._read_only:
            raise WorkflowStoreError("Applied replay requires read-only access.")
        require_safe_id(workflow_id, "workflow_id")
        require_sha256(plan_id, "plan_id")
        if not isinstance(confirmation_id, str) or not confirmation_id:
            raise WorkflowConfirmationNotReadyError("Resume confirmation is invalid.")
        current = _wall_time(now)
        token_sha256 = sha256_bytes(confirmation_id.encode("utf-8"))
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                record = self._selected_record(
                    connection,
                    workflow_id,
                    active_at=current,
                    validate_artifacts=not self._replay_only,
                )
                self._check_clock(record, current)
                if record.status != "applied":
                    return None
                confirmation = connection.execute(
                    "SELECT * FROM resume_confirmations WHERE token_sha256 = ?",
                    (token_sha256,),
                ).fetchone()
                if confirmation is None:
                    raise WorkflowConfirmationNotReadyError(
                        "Resume confirmation is invalid."
                    )
                expected_token = self._confirmation_token(
                    workflow_id,
                    str(confirmation["plan_id"]),
                    str(confirmation["idempotency_sha256"]),
                )
                if not hmac.compare_digest(expected_token, confirmation_id):
                    raise WorkflowStoreError(
                        "Resume confirmation integrity is invalid."
                    )
                if (
                    str(confirmation["workflow_id"]) != workflow_id
                    or str(confirmation["plan_id"]) != plan_id
                    or str(confirmation["binding_sha256"])
                    != record.binding.binding_sha256
                    or confirmation["consumed_at"] is None
                    or confirmation["revoked_at"] is not None
                    or _resume_transaction_id(
                        workflow_id,
                        plan_id,
                        record.binding.binding_sha256,
                    )
                    != record.apply_transaction_id
                ):
                    raise WorkflowStoreError(
                        "Applied replay authority binding is invalid."
                    )
                return record
            finally:
                connection.rollback()

    def find_workflow(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord | None:
        """Return a workflow when present without weakening normal validation.

        Stage composition uses this lookup before invoking an expensive candidate
        generator. Existing records still receive the same clock, expiry, and
        attestation-status checks as ``get_workflow``.
        """

        require_safe_id(workflow_id, "workflow_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if row is None:
                return None
            record = self._record(connection, row, active_at=current)
            self._check_clock(record, current)
            if record.status == "applying":
                return record
            if record.status not in {"applied", "conflicted", "failed", "expired"}:
                if current > record.binding.expires_at:
                    self._transition(
                        connection,
                        record,
                        status="expired",
                        event_type="workflow_expired",
                        event_key="expiry",
                        payload={"expiresAt": record.binding.expires_at},
                        now=current,
                    )
                    return self._selected_record(
                        connection, workflow_id, active_at=current
                    )
                if record.status != "applying":
                    record = self._refresh_attestation_status(
                        connection, record, now=current
                    )
            return record

    def list_workflows(self, *, limit: int = 100) -> tuple[WorkflowRecord, ...]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise WorkflowStoreError("Workflow list limit is outside safe bounds.")
        with self._transaction() as connection:
            current = _wall_time(None)
            rows = connection.execute(
                "SELECT * FROM workflows ORDER BY updated_at DESC, workflow_id LIMIT ?",
                (limit,),
            ).fetchall()
            records: list[WorkflowRecord] = []
            for row in rows:
                record = self._record(connection, row, active_at=current)
                self._check_clock(record, current)
                if record.status not in {
                    "applying",
                    "applied",
                    "conflicted",
                    "failed",
                    "expired",
                }:
                    if current > record.binding.expires_at:
                        self._transition(
                            connection,
                            record,
                            status="expired",
                            event_type="workflow_expired",
                            event_key="expiry",
                            payload={"expiresAt": record.binding.expires_at},
                            now=current,
                        )
                        record = self._selected_record(
                            connection, record.workflow_id, active_at=current
                        )
                    elif record.status != "applying":
                        record = self._refresh_attestation_status(
                            connection, record, now=current
                        )
                records.append(record)
            return tuple(records)

    def record_verified_attestation(
        self,
        workflow_id: str,
        attestation: IndependentAttestation,
        *,
        binding_sha256: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        """Persist evidence already verified by the trusted workflow service."""

        require_safe_id(workflow_id, "workflow_id")
        require_sha256(binding_sha256, "binding_sha256")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id, active_at=current)
            record = self._refresh_attestation_status(connection, record, now=current)
            self._check_live(record, current)
            if record.status not in {"staged", "attested", "ready"}:
                raise WorkflowStoreError(
                    "Independent attestation cannot be recorded in this state."
                )
            try:
                requirement = record.binding.verification_policy.requirement(
                    attestation.verifier_id
                )
            except ValueError as exc:
                raise WorkflowStoreError(str(exc)) from exc
            if binding_sha256 != record.binding.binding_sha256:
                raise WorkflowStoreError(
                    "Verified attestation targets another workflow binding."
                )
            if (
                attestation.adapter_id != requirement.adapter_id
                or attestation.key_id != requirement.key_id
                or not attestation.issued_at <= current <= attestation.expires_at
            ):
                raise WorkflowStoreError(
                    "Verified attestation metadata does not match workflow policy."
                )
            existing_evidence = connection.execute(
                "SELECT workflow_id, verifier_id, expires_at, superseded_at "
                "FROM workflow_attestations "
                "WHERE evidence_sha256 = ?",
                (attestation.evidence_sha256,),
            ).fetchone()
            if existing_evidence is not None:
                if (
                    str(existing_evidence["workflow_id"]) == workflow_id
                    and str(existing_evidence["verifier_id"]) == attestation.verifier_id
                    and existing_evidence["superseded_at"] is None
                    and current <= float(existing_evidence["expires_at"])
                ):
                    return self._selected_record(
                        connection, workflow_id, active_at=current
                    ), True
                raise WorkflowStoreError("Independent attestation was replayed.")
            existing_verifier = connection.execute(
                "SELECT evidence_sha256, expires_at FROM workflow_attestations "
                "WHERE workflow_id = ? AND verifier_id = ? "
                "AND superseded_at IS NULL",
                (workflow_id, attestation.verifier_id),
            ).fetchone()
            if existing_verifier is not None:
                if current <= float(existing_verifier["expires_at"]):
                    raise WorkflowStoreError(
                        "Workflow verifier is already bound to a live attestation."
                    )
                connection.execute(
                    "UPDATE workflow_attestations SET superseded_at = ? "
                    "WHERE evidence_sha256 = ? AND superseded_at IS NULL",
                    (current, str(existing_verifier["evidence_sha256"])),
                )
                self._append_event(
                    connection,
                    workflow_id,
                    event_type="independent_attestation_superseded",
                    event_key=(
                        "attestation-expired:"
                        f"{str(existing_verifier['evidence_sha256'])}"
                    ),
                    payload={
                        "evidenceSha256": str(existing_verifier["evidence_sha256"]),
                        "verifierId": attestation.verifier_id,
                    },
                    now=current,
                )
            existing_id = connection.execute(
                "SELECT workflow_id FROM workflow_attestations "
                "WHERE verifier_id = ? AND attestation_id = ?",
                (attestation.verifier_id, attestation.attestation_id),
            ).fetchone()
            if existing_id is not None:
                raise WorkflowStoreError("Signed attestation id was replayed.")
            metadata = attestation.metadata_payload()
            metadata_json = canonical_json_bytes(metadata).decode("utf-8")
            statement_sha256 = sha256_bytes(attestation.statement_bytes)
            try:
                envelope_descriptor = self.evidence_cas.put_bytes(
                    attestation.envelope_bytes,
                    media_type=_ENVELOPE_MEDIA_TYPE,
                )
                statement_descriptor = self.evidence_cas.put_bytes(
                    attestation.statement_bytes,
                    media_type=_STATEMENT_MEDIA_TYPE,
                )
            except ContentAddressedStoreError as exc:
                raise WorkflowStoreError(str(exc)) from exc
            envelope_descriptor_json = canonical_json_bytes(
                envelope_descriptor.payload()
            ).decode("utf-8")
            statement_descriptor_json = canonical_json_bytes(
                statement_descriptor.payload()
            ).decode("utf-8")
            connection.execute(
                """
                INSERT INTO workflow_attestations (
                    evidence_sha256, workflow_id, verifier_id, adapter_id, key_id,
                    attestation_id, statement_sha256, envelope_descriptor_json,
                    statement_descriptor_json, metadata_json, metadata_sha256,
                    issued_at, expires_at, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attestation.evidence_sha256,
                    workflow_id,
                    attestation.verifier_id,
                    attestation.adapter_id,
                    attestation.key_id,
                    attestation.attestation_id,
                    statement_sha256,
                    envelope_descriptor_json,
                    statement_descriptor_json,
                    metadata_json,
                    sha256_bytes(metadata_json.encode("utf-8")),
                    attestation.issued_at,
                    attestation.expires_at,
                    current,
                ),
            )
            verified = int(
                connection.execute(
                    "SELECT COUNT(*) FROM workflow_attestations "
                    "WHERE workflow_id = ? AND superseded_at IS NULL "
                    "AND issued_at <= ? AND expires_at >= ?",
                    (workflow_id, current, current),
                ).fetchone()[0]
            )
            status = (
                "ready"
                if verified >= record.binding.verification_policy.quorum
                else "attested"
            )
            connection.execute(
                "UPDATE workflows SET status = ?, updated_at = ?, last_wall_time = ? "
                "WHERE workflow_id = ?",
                (status, current, current, workflow_id),
            )
            self._append_event(
                connection,
                workflow_id,
                event_type="independent_attestation_verified",
                event_key=f"attestation:{attestation.evidence_sha256}",
                payload={
                    **metadata,
                    "verifiedCount": verified,
                    "requiredQuorum": record.binding.verification_policy.quorum,
                },
                now=current,
            )
            return self._selected_record(
                connection, workflow_id, active_at=current
            ), False

    def issue_resume_plan(
        self,
        workflow_id: str,
        *,
        idempotency_key: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> ResumePlan:
        require_safe_id(workflow_id, "workflow_id")
        if not 1 <= ttl_seconds <= 3600:
            raise WorkflowStoreError("Resume confirmation TTL is outside safe bounds.")
        current = _wall_time(now)
        key = _idempotency_key(idempotency_key)
        idempotency_sha256 = sha256_bytes(key)
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id, active_at=current)
            record = self._refresh_attestation_status(connection, record, now=current)
            self._check_live(record, current)
            if record.status != "ready" or not record.quorum_satisfied_at(current):
                raise WorkflowStoreError(
                    "A resume plan requires a verified ready workflow."
                )
            existing = connection.execute(
                "SELECT * FROM resume_confirmations WHERE idempotency_sha256 = ?",
                (idempotency_sha256,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["workflow_id"]) != workflow_id
                    or str(existing["binding_sha256"]) != record.binding.binding_sha256
                ):
                    raise WorkflowOperationConflictError(
                        "Resume idempotency key is bound to another operation."
                    )
                return self._resume_plan(record, existing, idempotent_replay=True)
            expires_at = min(
                current + ttl_seconds,
                record.binding.expires_at,
                record.quorum_expires_at(current),
            )
            evidence = [
                item.evidence_sha256
                for item in record.attestations
                if item.issued_at <= current <= item.expires_at
            ]
            plan_id = canonical_sha256(
                {
                    "workflowId": workflow_id,
                    "bindingSha256": record.binding.binding_sha256,
                    "attestationEvidenceSha256": evidence,
                    "idempotencySha256": idempotency_sha256,
                    "issuedAt": current,
                    "expiresAt": expires_at,
                    "authority": "single_write_local_resume",
                }
            )
            token = self._confirmation_token(workflow_id, plan_id, idempotency_sha256)
            token_sha256 = sha256_bytes(token.encode("utf-8"))
            connection.execute(
                "UPDATE resume_confirmations SET revoked_at = ? "
                "WHERE workflow_id = ? AND consumed_at IS NULL AND revoked_at IS NULL",
                (current, workflow_id),
            )
            connection.execute(
                """
                INSERT INTO resume_confirmations (
                    token_sha256, workflow_id, idempotency_sha256, plan_id,
                    binding_sha256, issued_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token_sha256,
                    workflow_id,
                    idempotency_sha256,
                    plan_id,
                    record.binding.binding_sha256,
                    current,
                    expires_at,
                ),
            )
            connection.execute(
                "UPDATE workflows SET updated_at = ?, last_wall_time = ? "
                "WHERE workflow_id = ?",
                (current, current, workflow_id),
            )
            self._append_event(
                connection,
                workflow_id,
                event_type="resume_confirmation_issued",
                event_key=f"resume-plan:{plan_id}",
                payload={
                    "planId": plan_id,
                    "idempotencySha256": idempotency_sha256,
                    "expiresAt": expires_at,
                },
                now=current,
            )
            row = connection.execute(
                "SELECT * FROM resume_confirmations WHERE token_sha256 = ?",
                (token_sha256,),
            ).fetchone()
            assert row is not None
            return self._resume_plan(record, row, idempotent_replay=False)

    def load_attestation_envelope(self, attestation: RecordedAttestation) -> bytes:
        """Load one persisted envelope for verification by the workflow service."""

        if attestation.envelope.size_bytes > _MAX_ATTESTATION_ARTIFACT_BYTES:
            raise WorkflowArtifactError(
                "Persisted attestation envelope exceeds its read bound."
            )
        try:
            envelope = self.evidence_cas.get_bytes(attestation.envelope)
        except (OSError, ValueError) as exc:
            raise WorkflowArtifactError(str(exc)) from exc
        if sha256_bytes(envelope) != attestation.evidence_sha256:
            raise WorkflowStoreError(
                "Persisted attestation envelope binding is invalid."
            )
        return envelope

    def consume_resume_confirmation(
        self,
        workflow_id: str,
        *,
        plan_id: str,
        confirmation_id: str,
        binding_sha256: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        require_safe_id(workflow_id, "workflow_id")
        require_sha256(plan_id, "plan_id")
        require_sha256(binding_sha256, "binding_sha256")
        current = _wall_time(now)
        if not isinstance(confirmation_id, str) or not confirmation_id:
            raise WorkflowConfirmationNotReadyError("Resume confirmation is invalid.")
        token_sha256 = sha256_bytes(confirmation_id.encode("utf-8"))
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id, active_at=current)
            if record.status not in {"applying", "applied"}:
                record = self._refresh_attestation_status(
                    connection, record, now=current
                )
            self._check_clock(record, current)
            confirmation = connection.execute(
                "SELECT * FROM resume_confirmations WHERE token_sha256 = ?",
                (token_sha256,),
            ).fetchone()
            if confirmation is None:
                raise WorkflowConfirmationNotReadyError(
                    "Resume confirmation is invalid."
                )
            expected_token = self._confirmation_token(
                workflow_id,
                str(confirmation["plan_id"]),
                str(confirmation["idempotency_sha256"]),
            )
            if not hmac.compare_digest(expected_token, confirmation_id):
                raise WorkflowStoreError("Resume confirmation integrity is invalid.")
            if (
                str(confirmation["workflow_id"]) != workflow_id
                or str(confirmation["plan_id"]) != plan_id
                or str(confirmation["binding_sha256"]) != binding_sha256
                or binding_sha256 != record.binding.binding_sha256
            ):
                raise WorkflowStoreError("Resume confirmation binding is invalid.")
            transaction_id = _resume_transaction_id(
                workflow_id, plan_id, binding_sha256
            )
            if confirmation["consumed_at"] is not None and record.status in {
                "applying",
                "applied",
            }:
                if record.apply_transaction_id != transaction_id:
                    raise WorkflowStoreError(
                        "Consumed confirmation lost its apply transaction binding."
                    )
                return record, True
            if confirmation["consumed_at"] is not None:
                recovery = connection.execute(
                    "SELECT workflow_id FROM apply_recoveries WHERE transaction_id = ?",
                    (transaction_id,),
                ).fetchone()
                if recovery is not None:
                    if str(recovery["workflow_id"]) != workflow_id:
                        raise WorkflowStoreError(
                            "Recovered confirmation binding is invalid."
                        )
                    return record, True
            self._check_clock(record, current)
            if record.status == "expired" or current > record.binding.expires_at:
                raise WorkflowConfirmationNotReadyError("Workflow expired.")
            if record.status != "ready" or not record.quorum_satisfied_at(current):
                raise WorkflowConfirmationNotReadyError(
                    "Workflow is not ready to resume."
                )
            if confirmation["revoked_at"] is not None:
                raise WorkflowConfirmationNotReadyError(
                    "Resume confirmation was revoked."
                )
            if confirmation["consumed_at"] is not None:
                raise WorkflowConfirmationNotReadyError(
                    "Resume confirmation was already consumed."
                )
            if current < float(confirmation["issued_at"]):
                raise WorkflowStoreError("Clock rollback detected for confirmation.")
            if current > float(confirmation["expires_at"]):
                raise WorkflowConfirmationNotReadyError("Resume confirmation expired.")
            connection.execute(
                "UPDATE resume_confirmations SET consumed_at = ? "
                "WHERE token_sha256 = ?",
                (current, token_sha256),
            )
            connection.execute(
                """
                UPDATE workflows
                SET status = 'applying', apply_transaction_id = ?,
                    updated_at = ?, last_wall_time = ?
                WHERE workflow_id = ?
                """,
                (transaction_id, current, current, workflow_id),
            )
            self._append_event(
                connection,
                workflow_id,
                event_type="resume_authorized",
                event_key=f"resume:{plan_id}",
                payload={"planId": plan_id, "transactionId": transaction_id},
                now=current,
            )
            return self._selected_record(
                connection, workflow_id, active_at=current
            ), False

    def mark_applied(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        result_sha256: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        return self._mark_applied(
            workflow_id,
            transaction_id=transaction_id,
            result_sha256=result_sha256,
            now=now,
            validate_artifacts=True,
            recovery_binding=False,
        )

    def mark_applied_after_committed_recovery(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        result_sha256: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        if not self._recovery_only:
            raise WorkflowStoreError(
                "Committed apply recovery access was not configured."
            )
        return self._mark_applied(
            workflow_id,
            transaction_id=transaction_id,
            result_sha256=result_sha256,
            now=now,
            validate_artifacts=False,
            recovery_binding=True,
        )

    def _mark_applied(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        result_sha256: str,
        now: float | None,
        validate_artifacts: bool,
        recovery_binding: bool,
    ) -> tuple[WorkflowRecord, bool]:
        require_safe_id(workflow_id, "workflow_id")
        require_sha256(transaction_id, "transaction_id")
        require_sha256(result_sha256, "result_sha256")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(
                connection,
                workflow_id,
                active_at=current,
                validate_artifacts=validate_artifacts,
            )
            self._check_clock(record, current)
            if recovery_binding:
                if record.status == "applying":
                    self._validate_applying_recovery_binding(connection, record)
                elif record.status == "applied":
                    self._validate_recovery_transaction_binding(
                        connection,
                        record,
                        transaction_id,
                    )
                    self._validate_applied_event_binding(
                        connection,
                        record,
                        transaction_id=transaction_id,
                        result_sha256=result_sha256,
                    )
                else:
                    raise WorkflowStoreError(
                        "Committed apply recovery state is invalid."
                    )
            if record.status == "applied":
                if (
                    record.apply_transaction_id == transaction_id
                    and record.result_sha256 == result_sha256
                ):
                    return record, True
                raise WorkflowStoreError("Applied workflow result binding changed.")
            if (
                record.status != "applying"
                or record.apply_transaction_id != transaction_id
            ):
                raise WorkflowStoreError("Workflow apply transaction is invalid.")
            connection.execute(
                """
                UPDATE workflows
                SET status = 'applied', result_sha256 = ?,
                    updated_at = ?, last_wall_time = ?
                WHERE workflow_id = ?
                """,
                (result_sha256, current, current, workflow_id),
            )
            self._append_event(
                connection,
                workflow_id,
                event_type="candidate_applied",
                event_key=f"applied:{transaction_id}",
                payload={
                    "transactionId": transaction_id,
                    "resultSha256": result_sha256,
                },
                now=current,
            )
            return self._selected_record(
                connection,
                workflow_id,
                active_at=current,
                validate_artifacts=validate_artifacts,
            ), False

    def reset_after_recovery(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        return self._reset_after_recovery(
            workflow_id,
            transaction_id=transaction_id,
            now=now,
            validate_artifacts=True,
        )

    def reset_applying_after_recovery(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        """Persist an applying-only journal rollback without CAS authority."""

        if not self._recovery_only:
            raise WorkflowStoreError("Applying recovery access was not configured.")
        return self._reset_after_recovery(
            workflow_id,
            transaction_id=transaction_id,
            now=now,
            validate_artifacts=False,
        )

    def _reset_after_recovery(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        now: float | None,
        validate_artifacts: bool,
    ) -> tuple[WorkflowRecord, bool]:
        require_safe_id(workflow_id, "workflow_id")
        require_sha256(transaction_id, "transaction_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(
                connection,
                workflow_id,
                active_at=current,
                validate_artifacts=validate_artifacts,
            )
            self._check_clock(record, current)
            if not validate_artifacts:
                self._validate_applying_recovery_binding(connection, record)
            existing = connection.execute(
                "SELECT workflow_id FROM apply_recoveries WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["workflow_id"]) != workflow_id:
                    raise WorkflowStoreError(
                        "Workflow recovery transaction binding is invalid."
                    )
                if record.status not in {"applying", "applied"}:
                    record = self._refresh_attestation_status(
                        connection, record, now=current
                    )
                return record, True
            if (
                record.status != "applying"
                or record.apply_transaction_id != transaction_id
            ):
                raise WorkflowStoreError("Workflow recovery transaction is invalid.")
            connection.execute(
                "INSERT INTO apply_recoveries "
                "(transaction_id, workflow_id, recovered_at) VALUES (?, ?, ?)",
                (transaction_id, workflow_id, current),
            )
            recovered_status = _recovered_status(record, current)
            connection.execute(
                """
                UPDATE workflows
                SET status = ?, apply_transaction_id = NULL,
                    updated_at = ?, last_wall_time = ?
                WHERE workflow_id = ?
                """,
                (
                    recovered_status,
                    current,
                    current,
                    workflow_id,
                ),
            )
            self._append_event(
                connection,
                workflow_id,
                event_type="apply_recovered",
                event_key=f"recovered:{transaction_id}",
                payload={
                    "transactionId": transaction_id,
                    "status": recovered_status,
                },
                now=current,
            )
            return self._selected_record(
                connection,
                workflow_id,
                active_at=current,
                validate_artifacts=validate_artifacts,
            ), False

    def mark_conflicted(
        self,
        workflow_id: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> WorkflowRecord:
        require_safe_id(workflow_id, "workflow_id")
        if not isinstance(reason, str) or not reason:
            raise WorkflowStoreError("Conflict reason is invalid.")
        current = _wall_time(now)
        reason_sha256 = sha256_bytes(reason.encode("utf-8"))
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id, active_at=current)
            self._check_clock(record, current)
            if record.status == "applied":
                raise WorkflowStoreError("Applied workflow cannot become conflicted.")
            if record.status != "conflicted":
                if record.apply_transaction_id:
                    connection.execute(
                        "UPDATE workflows SET apply_transaction_id = NULL "
                        "WHERE workflow_id = ?",
                        (workflow_id,),
                    )
                self._transition(
                    connection,
                    record,
                    status="conflicted",
                    event_type="source_drift_detected",
                    event_key=f"conflict:{reason_sha256}",
                    payload={"reasonSha256": reason_sha256},
                    now=current,
                )
            return self._selected_record(connection, workflow_id, active_at=current)

    def events(self, workflow_id: str) -> tuple[WorkflowEvent, ...]:
        require_safe_id(workflow_id, "workflow_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workflow_events WHERE workflow_id = ? ORDER BY sequence",
                (workflow_id,),
            ).fetchall()
        return tuple(
            WorkflowEvent(
                sequence=int(row["sequence"]),
                workflow_id=str(row["workflow_id"]),
                event_type=str(row["event_type"]),
                event_key=str(row["event_key"]),
                payload_sha256=str(row["payload_sha256"]),
                occurred_at=float(row["occurred_at"]),
            )
            for row in rows
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        expected = self._database_identity
        if (
            not self._read_only
            and expected is not None
            and _database_identity(self.path) != expected
        ):
            raise WorkflowStoreError("Workflow database identity changed.")
        target_context = (
            _read_only_database_target(self.path, expected_identity=expected)
            if self._read_only
            else nullcontext(self.path)
        )
        with target_context as target:
            try:
                connection = sqlite3.connect(
                    target,
                    timeout=self.timeout,
                    isolation_level=None,
                    uri=self._read_only,
                )
            except sqlite3.Error as exc:
                raise WorkflowStoreError("Workflow database is unavailable.") from exc
            connection.row_factory = sqlite3.Row
            try:
                if (
                    not self._read_only
                    and expected is not None
                    and _database_identity(self.path) != expected
                ):
                    raise WorkflowStoreError(
                        "Workflow database identity changed while opened."
                    )
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA trusted_schema = OFF")
                connection.execute("PRAGMA temp_store = MEMORY")
                connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
                if self._read_only:
                    connection.execute("PRAGMA query_only = ON")
                    query_only = int(
                        connection.execute("PRAGMA query_only").fetchone()[0]
                    )
                    journal = ""
                    synchronous = 0
                else:
                    query_only = 0
                    journal = str(
                        connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                    )
                    connection.execute("PRAGMA synchronous = FULL")
                    synchronous = int(
                        connection.execute("PRAGMA synchronous").fetchone()[0]
                    )
                foreign_keys = int(
                    connection.execute("PRAGMA foreign_keys").fetchone()[0]
                )
                invalid_read = self._read_only and (
                    query_only != 1 or foreign_keys != 1
                )
                invalid_write = not self._read_only and (
                    journal.lower() != "wal" or foreign_keys != 1 or synchronous != 2
                )
                if invalid_read or invalid_write:
                    raise WorkflowStoreError(
                        "SQLite durability or referential-integrity mode is unavailable."
                    )
                yield connection
            except sqlite3.Error as exc:
                raise WorkflowStoreError("Workflow database operation failed.") from exc
            finally:
                connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self._read_only:
            raise WorkflowStoreError("Read-only workflow status cannot mutate state.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                stage_idempotency_sha256 TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'staged', 'attested', 'ready', 'applying', 'applied',
                        'conflicted', 'expired', 'failed'
                    )
                ),
                binding_json TEXT NOT NULL,
                binding_sha256 TEXT NOT NULL,
                workspace_root_sha256 TEXT NOT NULL,
                apply_transaction_id TEXT,
                result_sha256 TEXT,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_wall_time REAL NOT NULL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS workflow_attestations (
                evidence_sha256 TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                verifier_id TEXT NOT NULL,
                adapter_id TEXT NOT NULL,
                key_id TEXT NOT NULL,
                attestation_id TEXT NOT NULL,
                statement_sha256 TEXT NOT NULL,
                envelope_descriptor_json TEXT NOT NULL,
                statement_descriptor_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                metadata_sha256 TEXT NOT NULL,
                issued_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                recorded_at REAL NOT NULL,
                superseded_at REAL,
                UNIQUE(verifier_id, attestation_id)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS workflow_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                event_type TEXT NOT NULL,
                event_key TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                UNIQUE(workflow_id, event_key)
            ) STRICT;
            CREATE TABLE IF NOT EXISTS resume_confirmations (
                token_sha256 TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                idempotency_sha256 TEXT NOT NULL UNIQUE,
                plan_id TEXT NOT NULL,
                binding_sha256 TEXT NOT NULL,
                issued_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                consumed_at REAL,
                revoked_at REAL
            ) STRICT;
            CREATE TABLE IF NOT EXISTS apply_recoveries (
                transaction_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                recovered_at REAL NOT NULL
            ) STRICT;
            CREATE INDEX IF NOT EXISTS workflow_events_workflow_idx
                ON workflow_events(workflow_id, sequence);
            CREATE INDEX IF NOT EXISTS attestations_workflow_idx
                ON workflow_attestations(workflow_id, verifier_id);
            CREATE UNIQUE INDEX IF NOT EXISTS active_attestations_verifier_idx
                ON workflow_attestations(workflow_id, verifier_id)
                WHERE superseded_at IS NULL;
            CREATE INDEX IF NOT EXISTS resume_confirmations_workflow_idx
                ON resume_confirmations(workflow_id, issued_at);
            INSERT INTO store_meta(key, value)
                VALUES ('schema_version', '2.0')
                ON CONFLICT(key) DO NOTHING;
            COMMIT;
            """
        )
        row = connection.execute(
            "SELECT value FROM store_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or str(row["value"]) != WORKFLOW_STORE_SCHEMA_VERSION:
            raise WorkflowStoreError("Workflow store schema version is unsupported.")
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise WorkflowStoreError(
                "Workflow database permissions are unavailable."
            ) from exc
        _validate_database_file(self.path)

    def _validate_existing_schema(self, connection: sqlite3.Connection) -> None:
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            ).fetchall()
        except sqlite3.Error as exc:
            raise WorkflowStoreError("Workflow store schema is unavailable.") from exc
        required = {
            "apply_recoveries",
            "resume_confirmations",
            "store_meta",
            "workflow_attestations",
            "workflow_events",
            "workflows",
        }
        available = {str(item["name"]) for item in rows}
        if not required.intersection(available):
            raise WorkflowStoreUninitializedError(
                "Workflow store schema is unavailable."
            )
        if not required.issubset(available):
            raise WorkflowStoreError("Workflow store schema is unavailable.")
        try:
            row = connection.execute(
                "SELECT value FROM store_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise WorkflowStoreError("Workflow store schema is unavailable.") from exc
        if row is None:
            raise WorkflowStoreError("Workflow store schema is unavailable.")
        if str(row["value"]) != WORKFLOW_STORE_SCHEMA_VERSION:
            raise WorkflowStoreError("Workflow store schema is unavailable.")

    def _selected_record(
        self,
        connection: sqlite3.Connection,
        workflow_id: str,
        *,
        active_at: float | None = None,
        validate_artifacts: bool = True,
    ) -> WorkflowRecord:
        row = connection.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise WorkflowNotFoundError("Workflow was not found.")
        return self._record(
            connection,
            row,
            active_at=active_at,
            validate_artifacts=validate_artifacts,
        )

    def _record(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        active_at: float | None = None,
        validate_artifacts: bool = True,
    ) -> WorkflowRecord:
        try:
            binding_raw = json.loads(str(row["binding_json"]))
            if not isinstance(binding_raw, Mapping):
                raise TypeError("binding must be an object")
            binding = CandidateBinding.from_payload(binding_raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise WorkflowStoreError("Persisted workflow record is invalid.") from exc
        if (
            binding.binding_sha256 != str(row["binding_sha256"])
            or binding.workflow_id != str(row["workflow_id"])
            or binding.stage_idempotency_sha256 != str(row["stage_idempotency_sha256"])
            or binding.created_at != float(row["created_at"])
            or binding.expires_at != float(row["expires_at"])
        ):
            raise WorkflowStoreError("Persisted workflow binding was tampered with.")
        attestation_rows = connection.execute(
            "SELECT * FROM workflow_attestations WHERE workflow_id = ? "
            "AND superseded_at IS NULL ORDER BY verifier_id",
            (binding.workflow_id,),
        ).fetchall()
        attestations = tuple(
            self._attestation(
                attestation_row,
                binding=binding,
                validate_artifacts=validate_artifacts,
            )
            for attestation_row in attestation_rows
        )
        recovery = connection.execute(
            "SELECT transaction_id FROM apply_recoveries WHERE workflow_id = ? "
            "ORDER BY recovered_at DESC, transaction_id DESC LIMIT 1",
            (binding.workflow_id,),
        ).fetchone()
        evaluation_time = (
            float(row["last_wall_time"]) if active_at is None else _wall_time(active_at)
        )
        active_count = sum(
            item.issued_at <= evaluation_time <= item.expires_at
            for item in attestations
        )
        record = WorkflowRecord(
            workflow_id=str(row["workflow_id"]),
            status=str(row["status"]),
            binding=binding,
            workspace_root_sha256=str(row["workspace_root_sha256"]),
            attestations=attestations,
            active_attestation_count=active_count,
            apply_transaction_id=str(row["apply_transaction_id"] or ""),
            recovered_transaction_id=(
                "" if recovery is None else str(recovery["transaction_id"])
            ),
            result_sha256=str(row["result_sha256"] or ""),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_wall_time=float(row["last_wall_time"]),
        )
        return record

    def _attestation(
        self,
        row: sqlite3.Row,
        *,
        binding: CandidateBinding,
        validate_artifacts: bool = True,
    ) -> RecordedAttestation:
        metadata_json = str(row["metadata_json"])
        if sha256_bytes(metadata_json.encode("utf-8")) != str(row["metadata_sha256"]):
            raise WorkflowStoreError(
                "Persisted attestation metadata was tampered with."
            )
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise WorkflowStoreError(
                "Persisted attestation metadata is invalid."
            ) from exc
        if (
            not isinstance(metadata, dict)
            or canonical_json_bytes(metadata).decode("utf-8") != metadata_json
        ):
            raise WorkflowStoreError("Persisted attestation metadata is not canonical.")
        expected = {
            "adapterId": str(row["adapter_id"]),
            "verifierId": str(row["verifier_id"]),
            "keyId": str(row["key_id"]),
            "attestationId": str(row["attestation_id"]),
            "evidenceSha256": str(row["evidence_sha256"]),
            "statementSha256": str(row["statement_sha256"]),
            "issuedAt": float(row["issued_at"]),
            "expiresAt": float(row["expires_at"]),
        }
        if metadata != expected:
            raise WorkflowStoreError(
                "Persisted attestation metadata binding is invalid."
            )
        envelope_descriptor = _descriptor_from_json(
            str(row["envelope_descriptor_json"]),
            media_type=_ENVELOPE_MEDIA_TYPE,
            label="attestation envelope",
        )
        statement_descriptor = _descriptor_from_json(
            str(row["statement_descriptor_json"]),
            media_type=_STATEMENT_MEDIA_TYPE,
            label="attestation statement",
        )
        if (
            envelope_descriptor.sha256 != expected["evidenceSha256"]
            or statement_descriptor.sha256 != expected["statementSha256"]
        ):
            raise WorkflowStoreError(
                "Persisted attestation descriptor binding is invalid."
            )
        if (
            envelope_descriptor.size_bytes > _MAX_ATTESTATION_ARTIFACT_BYTES
            or statement_descriptor.size_bytes > _MAX_ATTESTATION_ARTIFACT_BYTES
        ):
            raise WorkflowArtifactError(
                "Persisted attestation artifact exceeds its read bound."
            )
        if validate_artifacts:
            evidence_cas = self.evidence_cas
            if evidence_cas is None:
                raise WorkflowArtifactError(
                    "Persisted attestation storage is unavailable."
                )
            try:
                envelope_bytes = evidence_cas.get_bytes(envelope_descriptor)
                statement_bytes = evidence_cas.get_bytes(statement_descriptor)
            except ContentAddressedStoreError as exc:
                raise WorkflowArtifactError(str(exc)) from exc
            try:
                _validate_persisted_attestation_artifacts(
                    envelope_bytes,
                    statement_bytes,
                    binding=binding,
                    expected=expected,
                )
            except WorkflowStoreError as exc:
                raise WorkflowArtifactError(str(exc)) from exc
        attestation = RecordedAttestation(
            verifier_id=expected["verifierId"],
            adapter_id=expected["adapterId"],
            key_id=expected["keyId"],
            attestation_id=expected["attestationId"],
            evidence_sha256=expected["evidenceSha256"],
            statement_sha256=expected["statementSha256"],
            envelope=envelope_descriptor,
            statement=statement_descriptor,
            issued_at=expected["issuedAt"],
            expires_at=expected["expiresAt"],
            recorded_at=float(row["recorded_at"]),
        )
        _validate_recorded_attestation_lifetime(attestation, binding)
        return attestation

    def _resume_plan(
        self,
        record: WorkflowRecord,
        row: sqlite3.Row,
        *,
        idempotent_replay: bool,
    ) -> ResumePlan:
        plan_id = str(row["plan_id"])
        idempotency_sha256 = str(row["idempotency_sha256"])
        token = self._confirmation_token(
            record.workflow_id, plan_id, idempotency_sha256
        )
        if sha256_bytes(token.encode("utf-8")) != str(row["token_sha256"]):
            raise WorkflowStoreError("Persisted resume confirmation was tampered with.")
        return ResumePlan(
            workflow_id=record.workflow_id,
            plan_id=plan_id,
            confirmation_id=token,
            confirmation_expires_at=float(row["expires_at"]),
            candidate_fingerprint=record.binding.candidate_fingerprint,
            source_fingerprint=record.binding.source_fingerprint,
            binding_sha256=record.binding.binding_sha256,
            idempotent_replay=idempotent_replay,
        )

    def _validate_applying_recovery_binding(
        self,
        connection: sqlite3.Connection,
        record: WorkflowRecord,
    ) -> None:
        if record.status != "applying" or not record.apply_transaction_id:
            raise WorkflowStoreError(
                "Applying recovery requires an in-flight transaction."
            )
        self._validate_recovery_transaction_binding(
            connection,
            record,
            record.apply_transaction_id,
        )

    def _validate_recovery_transaction_binding(
        self,
        connection: sqlite3.Connection,
        record: WorkflowRecord,
        transaction_id: str,
    ) -> None:
        require_sha256(transaction_id, "recovery transaction_id")
        rows = connection.execute(
            "SELECT * FROM resume_confirmations "
            "WHERE workflow_id = ? AND consumed_at IS NOT NULL",
            (record.workflow_id,),
        ).fetchall()
        matches = 0
        for row in rows:
            plan_id = str(row["plan_id"])
            idempotency_sha256 = str(row["idempotency_sha256"])
            binding_sha256 = str(row["binding_sha256"])
            token_sha256 = str(row["token_sha256"])
            try:
                require_sha256(plan_id, "persisted recovery plan_id")
                require_sha256(
                    idempotency_sha256,
                    "persisted recovery idempotency_sha256",
                )
                require_sha256(
                    binding_sha256,
                    "persisted recovery binding_sha256",
                )
                require_sha256(token_sha256, "persisted recovery token_sha256")
                issued_at = _wall_time(float(row["issued_at"]))
                expires_at = _wall_time(float(row["expires_at"]))
                consumed_at = _wall_time(float(row["consumed_at"]))
            except (TypeError, ValueError) as exc:
                raise WorkflowStoreError(
                    "Applying recovery confirmation is invalid."
                ) from exc
            persisted_transaction_id = _resume_transaction_id(
                record.workflow_id,
                plan_id,
                binding_sha256,
            )
            if persisted_transaction_id != transaction_id:
                continue
            expected_token_sha256 = sha256_bytes(
                self._confirmation_token(
                    record.workflow_id,
                    plan_id,
                    idempotency_sha256,
                ).encode("utf-8")
            )
            if (
                binding_sha256 != record.binding.binding_sha256
                or not hmac.compare_digest(token_sha256, expected_token_sha256)
                or not issued_at <= consumed_at <= expires_at
                or row["revoked_at"] is not None
            ):
                raise WorkflowStoreError(
                    "Applying recovery confirmation binding is invalid."
                )
            matches += 1
        if matches != 1:
            raise WorkflowStoreError(
                "Applying recovery transaction binding is invalid."
            )

    @staticmethod
    def _validate_applied_event_binding(
        connection: sqlite3.Connection,
        record: WorkflowRecord,
        *,
        transaction_id: str,
        result_sha256: str,
    ) -> None:
        require_sha256(result_sha256, "applied result_sha256")
        event = connection.execute(
            "SELECT event_type, payload_sha256 FROM workflow_events "
            "WHERE workflow_id = ? AND event_key = ?",
            (record.workflow_id, f"applied:{transaction_id}"),
        ).fetchone()
        expected_payload = canonical_sha256(
            {
                "transactionId": transaction_id,
                "resultSha256": result_sha256,
            }
        )
        if (
            event is None
            or str(event["event_type"]) != "candidate_applied"
            or str(event["payload_sha256"]) != expected_payload
            or record.result_sha256 != result_sha256
        ):
            raise WorkflowStoreError("Applied workflow event binding is invalid.")

    def _check_clock(self, record: WorkflowRecord, now: float) -> None:
        if now < record.last_wall_time:
            raise WorkflowClockConflictError("Clock rollback detected for workflow.")

    def _check_live(self, record: WorkflowRecord, now: float) -> None:
        self._check_clock(record, now)
        if record.status == "expired" or now > record.binding.expires_at:
            raise WorkflowStoreError("Workflow expired.")

    def _refresh_attestation_status(
        self,
        connection: sqlite3.Connection,
        record: WorkflowRecord,
        *,
        now: float,
    ) -> WorkflowRecord:
        if record.status in {"applying", "applied", "conflicted", "failed", "expired"}:
            return record
        active = tuple(
            item
            for item in record.attestations
            if item.issued_at <= now <= item.expires_at
        )
        if len(active) >= record.binding.verification_policy.quorum:
            desired = "ready"
        elif active:
            desired = "attested"
        else:
            desired = "staged"
        if desired == record.status:
            return self._selected_record(connection, record.workflow_id, active_at=now)
        evidence = sorted(item.evidence_sha256 for item in active)
        state_sha256 = canonical_sha256(
            {
                "status": desired,
                "activeEvidenceSha256": evidence,
                "quorum": record.binding.verification_policy.quorum,
            }
        )
        self._transition(
            connection,
            record,
            status=desired,
            event_type="attestation_quorum_changed",
            event_key=f"quorum:{state_sha256}",
            payload={
                "status": desired,
                "activeEvidenceSha256": evidence,
                "quorum": record.binding.verification_policy.quorum,
            },
            now=now,
        )
        return self._selected_record(connection, record.workflow_id, active_at=now)

    def _transition(
        self,
        connection: sqlite3.Connection,
        record: WorkflowRecord,
        *,
        status: str,
        event_type: str,
        event_key: str,
        payload: Mapping[str, Any],
        now: float,
    ) -> None:
        if status not in WORKFLOW_STATES:
            raise WorkflowStoreError("Workflow transition state is invalid.")
        connection.execute(
            "UPDATE workflows SET status = ?, updated_at = ?, last_wall_time = ? "
            "WHERE workflow_id = ?",
            (status, now, now, record.workflow_id),
        )
        self._append_event(
            connection,
            record.workflow_id,
            event_type=event_type,
            event_key=event_key,
            payload=payload,
            now=now,
        )

    def _append_event(
        self,
        connection: sqlite3.Connection,
        workflow_id: str,
        *,
        event_type: str,
        event_key: str,
        payload: Mapping[str, Any],
        now: float,
    ) -> None:
        require_safe_id(event_type, "event_type")
        if not isinstance(event_key, str) or not 1 <= len(event_key) <= 256:
            raise WorkflowStoreError("Workflow event key is invalid.")
        payload_sha256 = canonical_sha256(dict(payload))
        existing = connection.execute(
            "SELECT payload_sha256 FROM workflow_events "
            "WHERE workflow_id = ? AND event_key = ?",
            (workflow_id, event_key),
        ).fetchone()
        if existing is not None:
            if str(existing["payload_sha256"]) != payload_sha256:
                raise WorkflowStoreError("Workflow event replay changed its payload.")
            return
        connection.execute(
            """
            INSERT INTO workflow_events (
                workflow_id, event_type, event_key, payload_sha256, occurred_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (workflow_id, event_type, event_key, payload_sha256, now),
        )

    def _mac(self, value: bytes) -> bytes:
        return hmac.new(self._secret, value, hashlib.sha256).digest()

    def _confirmation_token(
        self, workflow_id: str, plan_id: str, idempotency_sha256: str
    ) -> str:
        value = canonical_json_bytes(
            {
                "workflowId": workflow_id,
                "planId": plan_id,
                "idempotencySha256": idempotency_sha256,
                "purpose": "single-write-local-resume/v1",
            }
        )
        return base64.urlsafe_b64encode(self._mac(value)).decode("ascii").rstrip("=")


def _descriptor_from_json(
    value: str,
    *,
    media_type: str,
    label: str,
) -> ArtifactDescriptor:
    try:
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise TypeError("descriptor must be an object")
        if canonical_json_bytes(payload).decode("utf-8") != value:
            raise ValueError("descriptor must be canonical")
        descriptor = ArtifactDescriptor.from_payload(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise WorkflowStoreError(f"Persisted {label} descriptor is invalid.") from exc
    if descriptor.media_type != media_type:
        raise WorkflowStoreError(f"Persisted {label} media type is invalid.")
    return descriptor


def _resume_transaction_id(
    workflow_id: str,
    plan_id: str,
    binding_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "workflowId": workflow_id,
            "planId": plan_id,
            "bindingSha256": binding_sha256,
            "authority": "workspace-transaction/v1",
        }
    )


def _recovered_status(record: WorkflowRecord, now: float) -> str:
    if now > record.binding.expires_at:
        return "expired"
    if record.quorum_satisfied_at(now):
        return "ready"
    if any(item.issued_at <= now <= item.expires_at for item in record.attestations):
        return "attested"
    return "staged"


def _validate_recorded_attestation_lifetime(
    attestation: RecordedAttestation,
    binding: CandidateBinding,
) -> None:
    if (
        attestation.issued_at < binding.created_at
        or attestation.expires_at > binding.expires_at
    ):
        raise WorkflowStoreError(
            "Recorded attestation lifetime is outside the workflow."
        )


def _validate_persisted_attestation_artifacts(
    envelope_bytes: bytes,
    statement_bytes: bytes,
    *,
    binding: CandidateBinding,
    expected: Mapping[str, object],
) -> None:
    try:
        envelope = json.loads(envelope_bytes)
        statement = json.loads(statement_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowStoreError("Persisted attestation evidence is invalid.") from exc
    if (
        not isinstance(envelope, Mapping)
        or canonical_json_bytes(envelope) != envelope_bytes
        or set(envelope) != {"payloadType", "payload", "signatures"}
        or not isinstance(statement, Mapping)
        or canonical_json_bytes(statement) != statement_bytes
    ):
        raise WorkflowStoreError("Persisted attestation evidence is not canonical.")
    payload = envelope.get("payload")
    signatures = envelope.get("signatures")
    if (
        not isinstance(payload, str)
        or not isinstance(signatures, list)
        or len(signatures) != 1
        or not isinstance(signatures[0], Mapping)
        or signatures[0].get("keyid") != expected["keyId"]
    ):
        raise WorkflowStoreError("Persisted attestation envelope binding is invalid.")
    try:
        decoded_statement = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WorkflowStoreError("Persisted attestation payload is invalid.") from exc
    if decoded_statement != statement_bytes:
        raise WorkflowStoreError("Persisted envelope and statement do not match.")
    predicate = statement.get("predicate")
    attestation = (
        predicate.get("attestation") if isinstance(predicate, Mapping) else None
    )
    outcome = predicate.get("outcome") if isinstance(predicate, Mapping) else None
    if (
        not isinstance(predicate, Mapping)
        or predicate.get("binding") != binding.payload()
        or predicate.get("bindingSha256") != binding.binding_sha256
        or not isinstance(attestation, Mapping)
        or attestation.get("verifierId") != expected["verifierId"]
        or attestation.get("adapterId") != expected["adapterId"]
        or attestation.get("keyId") != expected["keyId"]
        or attestation.get("attestationId") != expected["attestationId"]
        or attestation.get("issuedAt") != expected["issuedAt"]
        or attestation.get("expiresAt") != expected["expiresAt"]
        or not isinstance(outcome, Mapping)
        or outcome.get("passed") is not True
    ):
        raise WorkflowStoreError("Persisted attestation statement binding is invalid.")


def _prepare_database_path(value: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(value.expanduser())))
    if raw.name in {"", ".", ".."}:
        raise WorkflowStoreError("Workflow database path is invalid.")
    if raw.exists() and raw.is_symlink():
        raise WorkflowStoreError("Workflow database cannot be a symbolic link.")
    try:
        raw.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise WorkflowStoreError("Workflow database parent is unavailable.") from exc
    state = parent.lstat()
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise WorkflowStoreError("Workflow database parent must be a real directory.")
    return parent / raw.name


def _prepare_existing_database_path(value: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(value.expanduser())))
    if raw.is_symlink():
        raise WorkflowStoreError("Workflow database is unavailable.")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise WorkflowStoreUninitializedError(
            "Workflow database is unavailable."
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise WorkflowStoreError("Workflow database is unavailable.") from exc
    _validate_database_file(resolved)
    return resolved


def _database_identity(path: Path) -> tuple[int, int]:
    state = _validate_database_file(path)
    return int(state.st_dev), int(state.st_ino)


def _database_read_snapshot(path: Path) -> tuple[int, int, int, int, int]:
    state = _validate_database_file(path)
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )


def _require_no_database_sidecars(path: Path) -> None:
    sidecars = tuple(Path(f"{path}{suffix}") for suffix in ("-journal", "-shm", "-wal"))
    if any(sidecar.exists() or sidecar.is_symlink() for sidecar in sidecars):
        raise WorkflowStoreError(
            "Workflow database has an active journal and is unavailable for read-only status."
        )


def _validate_database_file(path: Path) -> os.stat_result:
    try:
        state = path.lstat()
    except OSError as exc:
        raise WorkflowStoreError("Workflow database is unavailable.") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise WorkflowStoreError("Workflow database must be a regular file.")
    if os.name == "posix" and stat.S_IMODE(state.st_mode) & 0o077:
        raise WorkflowStoreError("Workflow database permissions are too broad.")
    return state


def _database_descriptor_snapshot(descriptor: int) -> tuple[int, int, int, int, int]:
    try:
        state = os.fstat(descriptor)
    except OSError as exc:
        raise WorkflowStoreError("Workflow database is unavailable.") from exc
    if not stat.S_ISREG(state.st_mode):
        raise WorkflowStoreError("Workflow database must be a regular file.")
    if os.name == "posix" and stat.S_IMODE(state.st_mode) & 0o077:
        raise WorkflowStoreError("Workflow database permissions are too broad.")
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )


def _database_descriptor_uri(
    descriptor: int,
    snapshot: tuple[int, int, int, int, int],
) -> str | None:
    if os.name != "posix":
        return None
    alias = Path("/dev/fd") / str(descriptor)
    alias_descriptor = -1
    try:
        alias_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        alias_descriptor = os.open(alias, alias_flags)
        alias_snapshot = _database_descriptor_snapshot(alias_descriptor)
    except OSError:
        return None
    except WorkflowStoreError:
        return None
    finally:
        if alias_descriptor >= 0:
            os.close(alias_descriptor)
    if alias_snapshot != snapshot:
        return None
    return f"file:/dev/fd/{descriptor}?mode=ro&immutable=1"


def _copy_database_snapshot(
    source_descriptor: int,
    source_snapshot: tuple[int, int, int, int, int],
    *,
    configured_state_root: Path,
) -> Path:
    temporary_descriptor = -1
    temporary_path: Path | None = None
    try:
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        configured_root = configured_state_root.resolve(strict=True)
        if temporary_root == configured_root or temporary_root.is_relative_to(
            configured_root
        ):
            raise WorkflowStoreError(
                "Workflow database snapshot location is not isolated."
            )
        temporary_descriptor, raw_path = tempfile.mkstemp(
            prefix="mymoe-workflow-status-",
            suffix=".sqlite3",
            dir=temporary_root,
        )
        temporary_path = Path(raw_path)
        if os.name == "posix":
            os.fchmod(temporary_descriptor, 0o600)
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        remaining = source_snapshot[2]
        source_digest = hashlib.sha256()
        while remaining:
            chunk = os.read(source_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise WorkflowStoreError("Workflow database snapshot is truncated.")
            source_digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(temporary_descriptor, chunk[offset:])
                if written <= 0:
                    raise OSError("database snapshot write made no progress")
                offset += written
            remaining -= len(chunk)
        if os.read(source_descriptor, 1):
            raise WorkflowStoreError(
                "Workflow database snapshot exceeds its size binding."
            )
        if _database_descriptor_snapshot(source_descriptor) != source_snapshot:
            raise WorkflowStoreError("Workflow database changed while snapshotted.")
        os.fsync(temporary_descriptor)
        temporary_state = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(temporary_state.st_mode)
            or int(temporary_state.st_size) != source_snapshot[2]
            or (os.name == "posix" and stat.S_IMODE(temporary_state.st_mode) & 0o077)
        ):
            raise WorkflowStoreError("Workflow database snapshot is invalid.")
        os.lseek(temporary_descriptor, 0, os.SEEK_SET)
        copied_digest = hashlib.sha256()
        remaining = source_snapshot[2]
        while remaining:
            chunk = os.read(temporary_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise WorkflowStoreError("Workflow database snapshot is truncated.")
            copied_digest.update(chunk)
            remaining -= len(chunk)
        if os.read(temporary_descriptor, 1):
            raise WorkflowStoreError(
                "Workflow database snapshot exceeds its size binding."
            )
        if copied_digest.digest() != source_digest.digest():
            raise WorkflowStoreError("Workflow database snapshot digest is invalid.")
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        return temporary_path
    except OSError as exc:
        raise WorkflowStoreError("Workflow database snapshot is unavailable.") from exc
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_path is not None and temporary_descriptor >= 0:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def _read_only_database_target(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None,
) -> Iterator[str]:
    _require_no_database_sidecars(path)
    path_snapshot = _database_read_snapshot(path)
    if expected_identity is not None and path_snapshot[:2] != expected_identity:
        raise WorkflowStoreError("Workflow database identity changed.")
    descriptor = -1
    snapshot_path: Path | None = None
    try:
        descriptor = os.open(path, _READ_FLAGS)
        descriptor_snapshot = _database_descriptor_snapshot(descriptor)
        if descriptor_snapshot != path_snapshot:
            raise WorkflowStoreError("Workflow database identity changed while opened.")
        target = _database_descriptor_uri(descriptor, descriptor_snapshot)
        if target is None:
            snapshot_path = _copy_database_snapshot(
                descriptor,
                descriptor_snapshot,
                configured_state_root=path.parent,
            )
            target = f"{snapshot_path.as_uri()}?mode=ro&immutable=1"
        yield target
    except OSError as exc:
        raise WorkflowStoreError("Workflow database is unavailable.") from exc
    finally:
        verification_error: WorkflowStoreError | None = None
        if descriptor >= 0:
            try:
                if _database_descriptor_snapshot(descriptor) != path_snapshot:
                    raise WorkflowStoreError(
                        "Workflow database changed during read-only status."
                    )
            except WorkflowStoreError as exc:
                verification_error = exc
            finally:
                os.close(descriptor)
        if snapshot_path is not None:
            try:
                snapshot_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if verification_error is None:
                    verification_error = WorkflowStoreError(
                        "Workflow database snapshot could not be removed."
                    )
                    verification_error.__cause__ = exc
        try:
            _require_no_database_sidecars(path)
            if _database_read_snapshot(path) != path_snapshot:
                raise WorkflowStoreError(
                    "Workflow database changed during read-only status."
                )
        except WorkflowStoreError as exc:
            if verification_error is None:
                verification_error = exc
        if verification_error is not None:
            raise verification_error


def _same_stage_operation(left: CandidateBinding, right: CandidateBinding) -> bool:
    """Compare stage intent while excluding absolute freshness timestamps."""

    return (
        left.workflow_id == right.workflow_id
        and left.stage_idempotency_sha256 == right.stage_idempotency_sha256
        and left.task_fingerprint == right.task_fingerprint
        and left.config_sha256 == right.config_sha256
        and left.source_fingerprint == right.source_fingerprint
        and left.challenge_sha256 == right.challenge_sha256
        and left.verification_policy == right.verification_policy
        and left.lifetime_matches(right.lifetime_seconds)
        and right.lifetime_matches(left.lifetime_seconds)
    )


def _load_or_create_secret(path: Path) -> bytes:
    flags = _READ_FLAGS
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        create_flags |= getattr(os, "O_BINARY", 0)
        create_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        value = os.urandom(_SECRET_BYTES)
        try:
            descriptor = os.open(path, create_flags, 0o600)
            offset = 0
            while offset < len(value):
                written = os.write(descriptor, value[offset:])
                if written <= 0:
                    raise OSError("state secret write made no progress")
                offset += written
            os.fsync(descriptor)
        except OSError as exc:
            raise WorkflowStoreError(
                "Workflow state secret could not be created."
            ) from exc
        finally:
            if "descriptor" in locals():
                os.close(descriptor)
        _fsync_directory(path.parent)
        try:
            created = path.lstat()
        except OSError as exc:
            raise WorkflowStoreError(
                "Workflow state secret identity is unavailable."
            ) from exc
        if (
            not stat.S_ISREG(created.st_mode)
            or stat.S_ISLNK(created.st_mode)
            or (os.name == "posix" and stat.S_IMODE(created.st_mode) & 0o077)
            or created.st_size != _SECRET_BYTES
        ):
            raise WorkflowStoreError("Workflow state secret identity is invalid.")
        return value
    except OSError as exc:
        raise WorkflowStoreError("Workflow state secret is unavailable.") from exc
    return _read_secret_descriptor(descriptor)


def _load_existing_secret(path: Path) -> bytes:
    raw = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        before = raw.lstat()
    except FileNotFoundError as exc:
        raise WorkflowStoreUninitializedError(
            "Workflow state secret is unavailable."
        ) from exc
    except OSError as exc:
        raise WorkflowStoreError("Workflow state secret is unavailable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise WorkflowStoreError("Workflow state secret is not a regular file.")
    try:
        descriptor = os.open(raw, _READ_FLAGS)
    except FileNotFoundError as exc:
        raise WorkflowStoreUninitializedError(
            "Workflow state secret is unavailable."
        ) from exc
    except OSError as exc:
        raise WorkflowStoreError("Workflow state secret is unavailable.") from exc
    return _read_secret_descriptor(descriptor, expected=before)


def _read_secret_descriptor(
    descriptor: int,
    *,
    expected: os.stat_result | None = None,
) -> bytes:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (
                expected is not None
                and (before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino)
            )
            or (os.name == "posix" and stat.S_IMODE(before.st_mode) & 0o077)
        ):
            raise WorkflowStoreError("Workflow state secret is not private.")
        value = os.read(descriptor, _SECRET_BYTES + 1)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise WorkflowStoreError("Workflow state secret is unavailable.") from exc
    finally:
        os.close(descriptor)
    if (
        len(value) != _SECRET_BYTES
        or (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_size,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_size,
        )
        or (
            expected is not None
            and (after.st_dev, after.st_ino) != (expected.st_dev, expected.st_ino)
        )
    ):
        raise WorkflowStoreError("Workflow state secret identity is invalid.")
    return value


def _idempotency_key(value: str) -> bytes:
    if not isinstance(value, str):
        raise WorkflowStoreError("Idempotency key must be text.")
    encoded = value.encode("utf-8")
    if not 16 <= len(encoded) <= 1024:
        raise WorkflowStoreError("Idempotency key length is outside safe bounds.")
    return encoded


def _wall_time(value: float | None) -> float:
    current = time.time() if value is None else value
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or not math.isfinite(float(current))
        or float(current) < 0
    ):
        raise WorkflowStoreError("Workflow timestamp is invalid.")
    return float(current)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except PermissionError as exc:
        # The Windows CRT cannot open directory descriptors for fsync. The
        # secret itself is fsynced before this point and is created O_EXCL, so
        # suppress only the platform's specific unsupported EACCES result.
        if os.name == "nt" and exc.errno == errno.EACCES:
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
