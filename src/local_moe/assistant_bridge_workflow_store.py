from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any, Iterator, Mapping, Sequence

from platformdirs import user_state_path

from .assistant_bridge_attestation import (
    AttestationTrustStore,
    AttestationVerificationError,
)
from .assistant_bridge_integrity import canonical_json_bytes, canonical_sha256, sha256_bytes
from .assistant_bridge_two_phase_contracts import (
    CandidateBinding,
    IndependentAttestation,
    ResumePlan,
    WORKFLOW_STATES,
    require_safe_id,
    require_sha256,
)


WORKFLOW_STORE_SCHEMA_VERSION = "1.0"
_SECRET_BYTES = 32
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
    os, "O_NOFOLLOW", 0
)


class WorkflowStoreError(ValueError):
    """Raised when durable two-phase workflow state cannot be trusted."""


@dataclass(frozen=True)
class WorkflowStatePaths:
    database: Path
    cas_root: Path


@dataclass(frozen=True, order=True)
class RecordedAttestation:
    verifier_id: str
    adapter_id: str
    key_id: str
    attestation_id: str
    evidence_sha256: str
    statement_sha256: str
    issued_at: float
    expires_at: float
    recorded_at: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.verifier_id, "recorded verifier_id"),
            (self.adapter_id, "recorded adapter_id"),
            (self.key_id, "recorded key_id"),
            (self.attestation_id, "recorded attestation_id"),
        ):
            require_safe_id(value, label)
        require_sha256(self.evidence_sha256, "recorded evidence_sha256")
        require_sha256(self.statement_sha256, "recorded statement_sha256")
        for value in (self.issued_at, self.expires_at, self.recorded_at):
            _wall_time(value)

    def payload(self) -> dict[str, object]:
        return {
            "verifierId": self.verifier_id,
            "adapterId": self.adapter_id,
            "keyId": self.key_id,
            "attestationId": self.attestation_id,
            "evidenceSha256": self.evidence_sha256,
            "statementSha256": self.statement_sha256,
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
    apply_transaction_id: str
    result_sha256: str
    created_at: float
    updated_at: float
    last_wall_time: float

    def __post_init__(self) -> None:
        require_safe_id(self.workflow_id, "workflow_id")
        if self.status not in WORKFLOW_STATES:
            raise WorkflowStoreError("Workflow status is invalid.")
        require_sha256(self.workspace_root_sha256, "workspace_root_sha256")
        if self.apply_transaction_id:
            require_sha256(self.apply_transaction_id, "apply_transaction_id")
        if self.result_sha256:
            require_sha256(self.result_sha256, "result_sha256")
        ordered = tuple(sorted(self.attestations, key=lambda item: item.verifier_id))
        if len({item.verifier_id for item in ordered}) != len(ordered):
            raise WorkflowStoreError("Workflow repeats an attestation verifier.")
        object.__setattr__(self, "attestations", ordered)

    @property
    def quorum_satisfied(self) -> bool:
        return len(self.attestations) >= self.binding.verification_policy.quorum

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
                "verified": len(self.attestations),
                "satisfied": self.quorum_satisfied,
            },
            "applyTransactionId": self.apply_transaction_id or None,
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
    base = user_state_path("myMoE", appauthor=False, ensure_exists=True)
    root = Path(base) / "assistant-bridge" / "v1"
    return WorkflowStatePaths(
        database=root / "workflows.sqlite3",
        cas_root=root / "cas",
    )


class SQLiteWorkflowStore:
    """Durable, quorum-aware, replay-safe workflow state and event journal."""

    def __init__(self, path: str | Path | None = None, *, timeout: float = 5.0) -> None:
        selected = default_workflow_state_paths().database if path is None else Path(path)
        self.path = _prepare_database_path(selected)
        if not 0.1 <= timeout <= 60:
            raise WorkflowStoreError("SQLite workflow timeout is outside safe bounds.")
        self.timeout = timeout
        self._database_identity: tuple[int, int] | None = None
        self._secret = _load_or_create_secret(self.path.with_suffix(".key"))
        with self._connect() as connection:
            self._initialize(connection)
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise WorkflowStoreError("Workflow database permissions are unavailable.") from exc
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
                record = self._record(connection, row)
                self._check_clock(record, current)
                if (
                    record.workflow_id != binding.workflow_id
                    or record.binding.binding_sha256 != binding.binding_sha256
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
            return self._selected_record(connection, binding.workflow_id), False

    def get_workflow(
        self,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> WorkflowRecord:
        require_safe_id(workflow_id, "workflow_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id)
            self._check_clock(record, current)
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
                    return self._selected_record(connection, workflow_id)
            return record

    def list_workflows(self, *, limit: int = 100) -> tuple[WorkflowRecord, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise WorkflowStoreError("Workflow list limit is outside safe bounds.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workflows ORDER BY updated_at DESC, workflow_id LIMIT ?",
                (limit,),
            ).fetchall()
            return tuple(self._record(connection, row) for row in rows)

    def record_attestation_envelope(
        self,
        workflow_id: str,
        envelope: bytes,
        *,
        trust_store: AttestationTrustStore,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        require_safe_id(workflow_id, "workflow_id")
        if type(trust_store) is not AttestationTrustStore:
            raise WorkflowStoreError("Attestation trust store type is unsupported.")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id)
            self._check_live(record, current)
            if record.status not in {"staged", "attested", "ready"}:
                raise WorkflowStoreError(
                    "Independent attestation cannot be recorded in this state."
                )
            try:
                attestation = trust_store.verify(record.binding, envelope, now=current)
            except AttestationVerificationError as exc:
                raise WorkflowStoreError(str(exc)) from exc
            existing_evidence = connection.execute(
                "SELECT workflow_id, verifier_id FROM workflow_attestations "
                "WHERE evidence_sha256 = ?",
                (attestation.evidence_sha256,),
            ).fetchone()
            if existing_evidence is not None:
                if (
                    str(existing_evidence["workflow_id"]) == workflow_id
                    and str(existing_evidence["verifier_id"])
                    == attestation.verifier_id
                ):
                    return self._selected_record(connection, workflow_id), True
                raise WorkflowStoreError(
                    "Independent attestation replayed across workflows."
                )
            existing_verifier = connection.execute(
                "SELECT evidence_sha256 FROM workflow_attestations "
                "WHERE workflow_id = ? AND verifier_id = ?",
                (workflow_id, attestation.verifier_id),
            ).fetchone()
            if existing_verifier is not None:
                raise WorkflowStoreError(
                    "Workflow verifier is already bound to another attestation."
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
            connection.execute(
                """
                INSERT INTO workflow_attestations (
                    evidence_sha256, workflow_id, verifier_id, adapter_id, key_id,
                    attestation_id, statement_sha256, metadata_json,
                    metadata_sha256, issued_at, expires_at, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attestation.evidence_sha256,
                    workflow_id,
                    attestation.verifier_id,
                    attestation.adapter_id,
                    attestation.key_id,
                    attestation.attestation_id,
                    statement_sha256,
                    metadata_json,
                    sha256_bytes(metadata_json.encode("utf-8")),
                    attestation.issued_at,
                    attestation.expires_at,
                    current,
                ),
            )
            verified = int(
                connection.execute(
                    "SELECT COUNT(*) FROM workflow_attestations WHERE workflow_id = ?",
                    (workflow_id,),
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
            return self._selected_record(connection, workflow_id), False

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
            record = self._selected_record(connection, workflow_id)
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
                    or str(existing["binding_sha256"])
                    != record.binding.binding_sha256
                ):
                    raise WorkflowStoreError(
                        "Resume idempotency key is bound to another operation."
                    )
                return self._resume_plan(record, existing, idempotent_replay=True)
            expires_at = min(
                current + ttl_seconds,
                record.binding.expires_at,
                record.quorum_expires_at(current),
            )
            evidence = [item.evidence_sha256 for item in record.attestations]
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
            token = self._confirmation_token(
                workflow_id, plan_id, idempotency_sha256
            )
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
            raise WorkflowStoreError("Resume confirmation is invalid.")
        token_sha256 = sha256_bytes(confirmation_id.encode("utf-8"))
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id)
            self._check_clock(record, current)
            confirmation = connection.execute(
                "SELECT * FROM resume_confirmations WHERE token_sha256 = ?",
                (token_sha256,),
            ).fetchone()
            if confirmation is None:
                raise WorkflowStoreError("Resume confirmation is invalid.")
            expected_token = self._confirmation_token(
                workflow_id,
                str(confirmation["plan_id"]),
                str(confirmation["idempotency_sha256"]),
            )
            if not hmac.compare_digest(expected_token, confirmation_id):
                raise WorkflowStoreError("Resume confirmation is invalid.")
            if (
                str(confirmation["workflow_id"]) != workflow_id
                or str(confirmation["plan_id"]) != plan_id
                or str(confirmation["binding_sha256"]) != binding_sha256
                or binding_sha256 != record.binding.binding_sha256
            ):
                raise WorkflowStoreError("Resume confirmation binding is invalid.")
            if confirmation["consumed_at"] is not None and record.status in {
                "applying",
                "applied",
            }:
                if not record.apply_transaction_id:
                    raise WorkflowStoreError(
                        "Consumed confirmation lost its apply transaction binding."
                    )
                return record, True
            self._check_live(record, current)
            if record.status != "ready" or not record.quorum_satisfied_at(current):
                raise WorkflowStoreError("Workflow is not ready to resume.")
            if confirmation["revoked_at"] is not None:
                raise WorkflowStoreError("Resume confirmation was revoked.")
            if confirmation["consumed_at"] is not None:
                raise WorkflowStoreError("Resume confirmation was already consumed.")
            if current < float(confirmation["issued_at"]):
                raise WorkflowStoreError("Clock rollback detected for confirmation.")
            if current > float(confirmation["expires_at"]):
                raise WorkflowStoreError("Resume confirmation expired.")
            transaction_id = canonical_sha256(
                {
                    "workflowId": workflow_id,
                    "planId": plan_id,
                    "bindingSha256": binding_sha256,
                    "authority": "workspace-transaction/v1",
                }
            )
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
            return self._selected_record(connection, workflow_id), False

    def mark_applied(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        result_sha256: str,
        now: float | None = None,
    ) -> tuple[WorkflowRecord, bool]:
        require_safe_id(workflow_id, "workflow_id")
        require_sha256(transaction_id, "transaction_id")
        require_sha256(result_sha256, "result_sha256")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id)
            self._check_clock(record, current)
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
            return self._selected_record(connection, workflow_id), False

    def reset_after_recovery(
        self,
        workflow_id: str,
        *,
        transaction_id: str,
        now: float | None = None,
    ) -> WorkflowRecord:
        require_safe_id(workflow_id, "workflow_id")
        require_sha256(transaction_id, "transaction_id")
        current = _wall_time(now)
        with self._transaction() as connection:
            record = self._selected_record(connection, workflow_id)
            self._check_clock(record, current)
            if (
                record.status != "applying"
                or record.apply_transaction_id != transaction_id
            ):
                raise WorkflowStoreError("Workflow recovery transaction is invalid.")
            connection.execute(
                """
                UPDATE workflows
                SET status = 'ready', apply_transaction_id = NULL,
                    updated_at = ?, last_wall_time = ?
                WHERE workflow_id = ?
                """,
                (current, current, workflow_id),
            )
            self._append_event(
                connection,
                workflow_id,
                event_type="apply_recovered",
                event_key=f"recovered:{transaction_id}",
                payload={"transactionId": transaction_id},
                now=current,
            )
            return self._selected_record(connection, workflow_id)

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
            record = self._selected_record(connection, workflow_id)
            self._check_clock(record, current)
            if record.status == "applied":
                raise WorkflowStoreError("Applied workflow cannot become conflicted.")
            if record.status != "conflicted":
                self._transition(
                    connection,
                    record,
                    status="conflicted",
                    event_type="source_drift_detected",
                    event_key=f"conflict:{reason_sha256}",
                    payload={"reasonSha256": reason_sha256},
                    now=current,
                )
            return self._selected_record(connection, workflow_id)

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
        if expected is not None and _database_identity(self.path) != expected:
            raise WorkflowStoreError("Workflow database identity changed.")
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            if expected is not None and _database_identity(self.path) != expected:
                raise WorkflowStoreError("Workflow database identity changed while opened.")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.execute("PRAGMA temp_store = MEMORY")
            connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
            journal = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
            connection.execute("PRAGMA synchronous = FULL")
            foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
            synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
            if journal.lower() != "wal" or foreign_keys != 1 or synchronous != 2:
                raise WorkflowStoreError(
                    "SQLite durability or referential-integrity mode is unavailable."
                )
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
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
                metadata_json TEXT NOT NULL,
                metadata_sha256 TEXT NOT NULL,
                issued_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                recorded_at REAL NOT NULL,
                UNIQUE(workflow_id, verifier_id),
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
            CREATE INDEX IF NOT EXISTS workflow_events_workflow_idx
                ON workflow_events(workflow_id, sequence);
            CREATE INDEX IF NOT EXISTS attestations_workflow_idx
                ON workflow_attestations(workflow_id, verifier_id);
            CREATE INDEX IF NOT EXISTS resume_confirmations_workflow_idx
                ON resume_confirmations(workflow_id, issued_at);
            INSERT INTO store_meta(key, value)
                VALUES ('schema_version', '1.0')
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

    def _selected_record(
        self, connection: sqlite3.Connection, workflow_id: str
    ) -> WorkflowRecord:
        row = connection.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise WorkflowStoreError("Workflow was not found.")
        return self._record(connection, row)

    def _record(
        self, connection: sqlite3.Connection, row: sqlite3.Row
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
            or binding.stage_idempotency_sha256
            != str(row["stage_idempotency_sha256"])
            or binding.created_at != float(row["created_at"])
            or binding.expires_at != float(row["expires_at"])
        ):
            raise WorkflowStoreError("Persisted workflow binding was tampered with.")
        attestation_rows = connection.execute(
            "SELECT * FROM workflow_attestations WHERE workflow_id = ? "
            "ORDER BY verifier_id",
            (binding.workflow_id,),
        ).fetchall()
        attestations = tuple(self._attestation(row) for row in attestation_rows)
        record = WorkflowRecord(
            workflow_id=str(row["workflow_id"]),
            status=str(row["status"]),
            binding=binding,
            workspace_root_sha256=str(row["workspace_root_sha256"]),
            attestations=attestations,
            apply_transaction_id=str(row["apply_transaction_id"] or ""),
            result_sha256=str(row["result_sha256"] or ""),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_wall_time=float(row["last_wall_time"]),
        )
        if record.status == "ready" and not record.quorum_satisfied:
            raise WorkflowStoreError("Ready workflow lost its attestation quorum.")
        if record.status in {"applying", "applied"} and not record.apply_transaction_id:
            raise WorkflowStoreError("Applying workflow lost its transaction id.")
        return record

    @staticmethod
    def _attestation(row: sqlite3.Row) -> RecordedAttestation:
        metadata_json = str(row["metadata_json"])
        if sha256_bytes(metadata_json.encode("utf-8")) != str(row["metadata_sha256"]):
            raise WorkflowStoreError("Persisted attestation metadata was tampered with.")
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise WorkflowStoreError("Persisted attestation metadata is invalid.") from exc
        if not isinstance(metadata, dict) or canonical_json_bytes(metadata).decode(
            "utf-8"
        ) != metadata_json:
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
            raise WorkflowStoreError("Persisted attestation metadata binding is invalid.")
        return RecordedAttestation(
            verifier_id=expected["verifierId"],
            adapter_id=expected["adapterId"],
            key_id=expected["keyId"],
            attestation_id=expected["attestationId"],
            evidence_sha256=expected["evidenceSha256"],
            statement_sha256=expected["statementSha256"],
            issued_at=expected["issuedAt"],
            expires_at=expected["expiresAt"],
            recorded_at=float(row["recorded_at"]),
        )

    def _resume_plan(
        self,
        record: WorkflowRecord,
        row: sqlite3.Row,
        *,
        idempotent_replay: bool,
    ) -> ResumePlan:
        plan_id = str(row["plan_id"])
        idempotency_sha256 = str(row["idempotency_sha256"])
        token = self._confirmation_token(record.workflow_id, plan_id, idempotency_sha256)
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

    def _check_clock(self, record: WorkflowRecord, now: float) -> None:
        if now < record.last_wall_time:
            raise WorkflowStoreError("Clock rollback detected for workflow.")

    def _check_live(self, record: WorkflowRecord, now: float) -> None:
        self._check_clock(record, now)
        if record.status == "expired" or now > record.binding.expires_at:
            raise WorkflowStoreError("Workflow expired.")

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


def _database_identity(path: Path) -> tuple[int, int]:
    state = _validate_database_file(path)
    return int(state.st_dev), int(state.st_ino)


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


def _load_or_create_secret(path: Path) -> bytes:
    flags = _READ_FLAGS
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
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
            raise WorkflowStoreError("Workflow state secret could not be created.") from exc
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
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            os.name == "posix" and stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise WorkflowStoreError("Workflow state secret is not private.")
        value = os.read(descriptor, _SECRET_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(value) != _SECRET_BYTES
        or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
        != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
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
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
