from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import time
from typing import Callable, Generic, Iterator, TypeVar

from filelock import FileLock, Timeout
from platformdirs import user_state_path

from .adaptive_advisor_service import AdaptiveAdvisorReceipt
from .adaptive_execution_gate import AdaptiveCellExecutionPreviewReceipt
from .adaptive_selector import CandidateAssessment
from .cell_contracts import AdaptiveCellCatalog, AdvisorProfile, CellPassport
from .cooperative_resource_lease_contracts import (
    CLAIM_BASIS,
    MAX_SQLITE_INTEGER,
    CooperativeResourceClaim,
    CooperativeResourceLeaseAdmissionReceipt,
    CooperativeResourceLeasePolicy,
    CooperativeResourceLeaseReleaseReceipt,
    CooperativeResourceLeaseTransitionReceipt,
)
from .resource_snapshot import ResourceSnapshot
from .verified_routing_contracts import (
    VerifiedRoutingError,
    now_utc,
    require_safe_id,
    require_sha256,
    sha256_json,
)


_STORE_SCHEMA_VERSION = "1"
_STORE_CONTRACT = "mymoe-cooperative-resource-lease-store"
_APPLICATION_ID = 0x4D594C53  # MYLS
_STORE_META_SQL = (
    "CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) "
    "STRICT, WITHOUT ROWID"
)
_ACTIVE_LEASES_SQL = """
CREATE TABLE active_leases (
    lease_id TEXT PRIMARY KEY,
    token_sha256 TEXT NOT NULL,
    admission_receipt_sha256 TEXT NOT NULL,
    claim_sha256 TEXT NOT NULL,
    resource_snapshot_sha256 TEXT NOT NULL,
    resource_class_sha256 TEXT NOT NULL,
    catalog_sha256 TEXT NOT NULL,
    profile_sha256 TEXT NOT NULL,
    pool TEXT NOT NULL CHECK (pool IN ('system', 'unified', 'discrete')),
    system_claim_bytes INTEGER NOT NULL CHECK (system_claim_bytes > 0),
    accelerator_claim_bytes INTEGER NOT NULL CHECK (accelerator_claim_bytes >= 0),
    accelerator_identity_sha256 TEXT,
    safety_reserve_bytes INTEGER NOT NULL CHECK (safety_reserve_bytes >= 0),
    acquired_at TEXT NOT NULL,
    acquired_monotonic_ns INTEGER NOT NULL CHECK (acquired_monotonic_ns >= 0),
    state TEXT NOT NULL CHECK (
        state IN ('reserved', 'delivery_armed', 'unknown_blocking')
    ),
    row_sha256 TEXT NOT NULL
) STRICT, WITHOUT ROWID
"""
_ROW_FIELDS = (
    "lease_id",
    "token_sha256",
    "admission_receipt_sha256",
    "claim_sha256",
    "resource_snapshot_sha256",
    "resource_class_sha256",
    "catalog_sha256",
    "profile_sha256",
    "pool",
    "system_claim_bytes",
    "accelerator_claim_bytes",
    "accelerator_identity_sha256",
    "safety_reserve_bytes",
    "acquired_at",
    "acquired_monotonic_ns",
    "state",
)
_TABLE_FIELDS = _ROW_FIELDS + ("row_sha256",)
_ACTIVE_STATES = frozenset({"reserved", "delivery_armed", "unknown_blocking"})
_T = TypeVar("_T")


class CooperativeResourceLeaseStoreError(RuntimeError):
    """Fail-closed local coordination-store failure with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CooperativeResourceLeasePaths:
    database: Path
    sentinels: Path


@dataclass(frozen=True)
class CooperativeResourceLeaseEvaluation(Generic[_T]):
    """Fresh evidence produced while the store owns its write transaction."""

    claim: CooperativeResourceClaim
    snapshot: ResourceSnapshot
    context: _T


@dataclass(slots=True)
class CooperativeResourceLeaseHandle:
    """In-memory ownership capability; the raw token is never persisted."""

    lease_id: str
    admission_receipt_sha256: str
    claim_sha256: str
    token: bytes = field(repr=False)
    _owner_lock: FileLock = field(repr=False, compare=False)
    _sentinel_path: Path = field(repr=False, compare=False)
    _released: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True)
class CooperativeResourceLeaseAcquisition(Generic[_T]):
    receipt: CooperativeResourceLeaseAdmissionReceipt
    handle: CooperativeResourceLeaseHandle | None
    context: _T


def default_cooperative_resource_lease_paths() -> CooperativeResourceLeasePaths:
    base = Path(user_state_path("myMoE", appauthor=False, ensure_exists=False))
    root = base / "resource-lease" / "v1"
    return CooperativeResourceLeasePaths(
        database=root / "leases.sqlite3",
        sentinels=root / "sentinels",
    )


def cooperative_resource_claim_from_preview(
    *,
    preview: AdaptiveCellExecutionPreviewReceipt,
    fresh_advisor: AdaptiveAdvisorReceipt,
    passport: CellPassport,
    catalog: AdaptiveCellCatalog,
    snapshot: ResourceSnapshot,
) -> CooperativeResourceClaim:
    """Derive a claim only from the exact, freshly previewed Advisor selection."""

    if not isinstance(preview, AdaptiveCellExecutionPreviewReceipt) or (
        preview.status != "admission_passed"
    ):
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "A passed execution preview is required."
        )
    if not isinstance(fresh_advisor, AdaptiveAdvisorReceipt):
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "The preview's fresh Advisor receipt is required."
        )
    if not isinstance(snapshot, ResourceSnapshot):
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "A verified resource snapshot is required."
        )
    if not isinstance(passport, CellPassport) or not isinstance(
        catalog, AdaptiveCellCatalog
    ):
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "Exact passport and catalog are required."
        )
    profile_name = fresh_advisor.request.profile
    profile = catalog.profiles.get(profile_name)
    advice = fresh_advisor.advice
    selected_id = advice.selected_cell_id
    selected_candidates = tuple(
        item for item in advice.candidates if item.cell_id == selected_id
    )
    if len(selected_candidates) != 1:
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "The fresh Advisor selection is not unique."
        )
    candidate = selected_candidates[0]
    matching_passports = tuple(
        item
        for item in catalog.cells
        if item.cell_id == passport.cell_id and item.digest == passport.digest
    )
    if (
        len(matching_passports) != 1
        or not isinstance(profile, AdvisorProfile)
        or not isinstance(candidate, CandidateAssessment)
        or not candidate.hard_eligible
        or not candidate.pareto_eligible
        or preview.fresh_advisor_receipt_sha256 != fresh_advisor.digest
        or preview.fresh_request_sha256 != fresh_advisor.request.digest
        or preview.fresh_resource_snapshot_sha256 != snapshot.digest
        or preview.evaluated_at != fresh_advisor.request.evaluated_at
        or preview.task_chars != fresh_advisor.task_chars
        or preview.fresh_selected_cell_id != selected_id
        or preview.fresh_passport_sha256 != passport.digest
        or advice.catalog_sha256 != catalog.digest
        or advice.resource_snapshot_sha256 != snapshot.digest
        or advice.profile != profile_name
        or advice.status != "recommended"
        or candidate.cell_id != passport.cell_id
        or candidate.passport_sha256 != passport.digest
        or candidate.memory_pool != passport.estimated.memory_pool
        or candidate.placement != passport.estimated.placement
    ):
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid",
            "Candidate, passport, catalog, and profile are not exactly linked.",
        )
    recomputed = _conservative_peak_resources(passport)
    selected_resources = (
        candidate.effective_peak_host_memory_bytes,
        candidate.effective_peak_unified_memory_bytes,
        candidate.effective_peak_accelerator_memory_bytes,
    )
    if recomputed != selected_resources:
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid",
            "Candidate conservative peaks do not match passport evidence.",
        )
    pool: str
    system_bytes: int | None
    accelerator_bytes = 0
    accelerator_identity: str | None = None
    if candidate.memory_pool == "host" and candidate.placement == "cpu":
        pool = "system"
        system_bytes = candidate.effective_peak_host_memory_bytes
    elif (
        candidate.memory_pool == "unified"
        and candidate.placement == "integrated_accelerator"
    ):
        if (
            snapshot.memory_topology != "unified"
            or snapshot.accelerator_kind != "integrated"
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_claim_invalid", "Unified candidate and snapshot do not match."
            )
        pool = "unified"
        system_bytes = candidate.effective_peak_unified_memory_bytes
    elif (
        candidate.memory_pool == "accelerator"
        and candidate.placement == "discrete_accelerator"
    ):
        # The contract supports a trusted singleton discrete pool.  The current
        # built-in Linux/Windows snapshot collector does not yet discover one,
        # so production collection remains fail-closed until that evidence exists.
        if (
            snapshot.memory_topology != "dedicated"
            or snapshot.accelerator_kind != "discrete"
            or snapshot.accelerator_identity_sha256 is None
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_claim_invalid", "Discrete candidate and snapshot do not match."
            )
        pool = "discrete"
        system_bytes = candidate.effective_peak_host_memory_bytes
        accelerator_bytes = candidate.effective_peak_accelerator_memory_bytes or 0
        accelerator_identity = snapshot.accelerator_identity_sha256
    else:
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "Candidate memory placement is unsupported."
        )
    if system_bytes is None or system_bytes <= 0 or accelerator_bytes < 0:
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "Candidate peak memory evidence is incomplete."
        )
    try:
        return CooperativeResourceClaim(
            preview_sha256=preview.digest,
            candidate_sha256=candidate.digest,
            passport_sha256=candidate.passport_sha256,
            resource_snapshot_sha256=snapshot.digest,
            resource_class_sha256=snapshot.resource_class_sha256,
            catalog_sha256=catalog.digest,
            profile_sha256=profile.digest,
            pool=pool,
            system_claim_bytes=system_bytes,
            accelerator_claim_bytes=accelerator_bytes,
            accelerator_identity_sha256=accelerator_identity,
            safety_reserve_bytes=profile.reserve_memory_bytes,
        )
    except Exception as exc:
        raise CooperativeResourceLeaseStoreError(
            "lease_claim_invalid", "Candidate peak claim is invalid."
        ) from exc


class SQLiteCooperativeResourceLeaseStore:
    """Same-user, same-host cooperative admission accounting.

    This store never reserves RAM or manages a model runtime.  Its only authority
    is serializing participants that use this same coordination domain.
    """

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        sentinel_root: str | Path | None = None,
        policy: CooperativeResourceLeasePolicy | None = None,
        clock: Callable[[], str] = now_utc,
        monotonic_clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.policy = policy or CooperativeResourceLeasePolicy()
        if not isinstance(self.policy, CooperativeResourceLeasePolicy):
            raise CooperativeResourceLeaseStoreError(
                "lease_policy_invalid", "Lease policy is invalid."
            )
        defaults = default_cooperative_resource_lease_paths()
        declared_database = (
            Path(database_path) if database_path is not None else defaults.database
        )
        declared_sentinels = (
            Path(sentinel_root)
            if sentinel_root is not None
            else (
                declared_database.parent / "sentinels"
                if database_path is not None
                else defaults.sentinels
            )
        )
        self.path = _prepare_database_path(declared_database)
        self.sentinel_root = _prepare_directory(declared_sentinels, "sentinel root")
        self.coordination_domain_sha256 = hashlib.sha256(
            (
                f"{_STORE_CONTRACT}\0{os.path.normcase(os.fspath(self.path))}"
                f"\0{os.path.normcase(os.fspath(self.sentinel_root))}"
            ).encode("utf-8")
        ).hexdigest()
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        initialization_lock = FileLock(
            self.sentinel_root / "store-init.lock",
            timeout=self.policy.busy_timeout_ms / 1_000,
            mode=0o600,
            thread_local=False,
            blocking=True,
            lifetime=None,
        )
        try:
            initialization_lock.acquire()
            self._initialize()
        except Timeout as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_busy", "Lease store initialization is busy."
            ) from exc
        finally:
            initialization_lock.release(force=True)

    def acquire(
        self,
        claim: CooperativeResourceClaim,
        snapshot: ResourceSnapshot,
    ) -> CooperativeResourceLeaseAcquisition[None]:
        return self.evaluate_and_acquire(
            lambda: CooperativeResourceLeaseEvaluation(claim, snapshot, None)
        )

    def evaluate_and_acquire(
        self,
        evaluator: Callable[[], CooperativeResourceLeaseEvaluation[_T]],
    ) -> CooperativeResourceLeaseAcquisition[_T]:
        """Evaluate fresh admission and insert its claim in one write transaction."""

        if not callable(evaluator):
            raise CooperativeResourceLeaseStoreError(
                "lease_evaluation_invalid", "Lease evaluator is invalid."
            )
        lease_id = secrets.token_hex(16)
        token = secrets.token_bytes(32)
        token_sha256 = hashlib.sha256(token).hexdigest()
        sentinel_path = self._sentinel_path(lease_id)
        owner_lock = _new_file_lock(sentinel_path)
        try:
            owner_lock.acquire(blocking=False)
        except Exception as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_owner_unavailable", "Lease owner sentinel is unavailable."
            ) from exc

        owner_transferred = False
        probe_locks: list[tuple[FileLock, Path]] = []
        context: _T
        try:
            with self._transaction() as connection:
                evaluation = evaluator()
                if not isinstance(evaluation, CooperativeResourceLeaseEvaluation):
                    raise CooperativeResourceLeaseStoreError(
                        "lease_evaluation_invalid",
                        "Lease evaluator returned an invalid result.",
                    )
                claim = evaluation.claim
                snapshot = evaluation.snapshot
                context = evaluation.context
                self._validate_evidence(claim, snapshot)
                rows, reaped, ownership_unknown, dead_locks = self._active_rows(
                    connection
                )
                probe_locks.extend(dead_locks)
                active_system = _checked_sum(
                    int(row["system_claim_bytes"]) for row in rows
                )
                accelerator_total, identity_conflict = _accelerator_total(rows, claim)
                applied_system_reserve = max(
                    (claim.safety_reserve_bytes,)
                    + tuple(int(row["safety_reserve_bytes"]) for row in rows)
                )
                applicable_accelerator_reserves = tuple(
                    int(row["safety_reserve_bytes"])
                    for row in rows
                    if claim.pool == "discrete"
                    and int(row["accelerator_claim_bytes"]) > 0
                    and row["accelerator_identity_sha256"]
                    == claim.accelerator_identity_sha256
                )
                applied_accelerator_reserve = (
                    max((claim.safety_reserve_bytes,) + applicable_accelerator_reserves)
                    if claim.pool == "discrete"
                    else 0
                )
                active_before = len(rows)
                system_available = _system_available(snapshot)
                accelerator_available = (
                    snapshot.accelerator_memory_available_bytes
                    if claim.pool == "discrete"
                    else None
                )
                reasons: set[str] = set()
                status = "denied"
                if ownership_unknown:
                    status = "unknown_blocking"
                    reasons.add("lease_owner_unknown")
                if identity_conflict:
                    status = "unknown_blocking"
                    reasons.add("accelerator_identity_conflict")
                if system_available is None or (
                    claim.pool == "discrete" and accelerator_available is None
                ):
                    status = "unknown_blocking"
                    reasons.add("resource_capacity_unknown")
                if active_before >= self.policy.max_active_leases:
                    reasons.add("active_lease_limit_reached")
                required_system = _checked_sum(
                    (
                        active_system,
                        claim.system_claim_bytes,
                        applied_system_reserve,
                    )
                )
                required_accelerator = _checked_sum(
                    (
                        accelerator_total,
                        claim.accelerator_claim_bytes,
                        applied_accelerator_reserve,
                    )
                )
                if system_available is not None and required_system > system_available:
                    reasons.add("system_capacity_insufficient")
                if (
                    claim.pool == "discrete"
                    and accelerator_available is not None
                    and required_accelerator > accelerator_available
                ):
                    reasons.add("accelerator_capacity_insufficient")
                if not reasons:
                    status = "acquired"
                evaluated_at = self._validated_clock()
                receipt = CooperativeResourceLeaseAdmissionReceipt(
                    policy_sha256=self.policy.digest,
                    claim_sha256=claim.digest,
                    resource_snapshot_sha256=snapshot.digest,
                    coordination_domain_sha256=self.coordination_domain_sha256,
                    status=status,
                    reason_codes=tuple(sorted(reasons)),
                    evaluated_at=evaluated_at,
                    lease_id=lease_id if status == "acquired" else None,
                    lease_token_sha256=(token_sha256 if status == "acquired" else None),
                    active_leases_before=active_before,
                    active_leases_after=(
                        active_before + 1 if status == "acquired" else active_before
                    ),
                    reaped_leases=reaped,
                    system_available_bytes=system_available,
                    accelerator_available_bytes=accelerator_available,
                    active_system_claim_bytes=active_system,
                    active_accelerator_claim_bytes=accelerator_total,
                    requested_system_claim_bytes=claim.system_claim_bytes,
                    requested_accelerator_claim_bytes=claim.accelerator_claim_bytes,
                    safety_reserve_bytes=claim.safety_reserve_bytes,
                    applied_system_reserve_bytes=applied_system_reserve,
                    applied_accelerator_reserve_bytes=applied_accelerator_reserve,
                )
                if status == "acquired":
                    acquired_at_tick = self._validated_monotonic_clock()
                    row = {
                        "lease_id": lease_id,
                        "token_sha256": token_sha256,
                        "admission_receipt_sha256": receipt.digest,
                        "claim_sha256": claim.digest,
                        "resource_snapshot_sha256": snapshot.digest,
                        "resource_class_sha256": snapshot.resource_class_sha256,
                        "catalog_sha256": claim.catalog_sha256,
                        "profile_sha256": claim.profile_sha256,
                        "pool": claim.pool,
                        "system_claim_bytes": claim.system_claim_bytes,
                        "accelerator_claim_bytes": claim.accelerator_claim_bytes,
                        "accelerator_identity_sha256": claim.accelerator_identity_sha256,
                        "safety_reserve_bytes": claim.safety_reserve_bytes,
                        "acquired_at": evaluated_at,
                        "acquired_monotonic_ns": acquired_at_tick,
                        "state": "reserved",
                    }
                    self._insert_row(connection, row)
            handle = (
                CooperativeResourceLeaseHandle(
                    lease_id=lease_id,
                    admission_receipt_sha256=receipt.digest,
                    claim_sha256=claim.digest,
                    token=token,
                    _owner_lock=owner_lock,
                    _sentinel_path=sentinel_path,
                )
                if status == "acquired"
                else None
            )
            result = CooperativeResourceLeaseAcquisition(receipt, handle, context)
            owner_transferred = handle is not None
            return result
        finally:
            for probe_lock, probe_path in probe_locks:
                _release_file_lock(probe_lock, probe_path)
            if not owner_transferred:
                _release_file_lock(owner_lock, sentinel_path)

    def arm_delivery(
        self, handle: CooperativeResourceLeaseHandle
    ) -> CooperativeResourceLeaseTransitionReceipt:
        """Authenticate and durably arm a reserved lease immediately before POST."""

        self._validate_handle(handle)
        token_sha256 = hashlib.sha256(handle.token).hexdigest()
        transitioned_at = self._validated_clock()
        applied = False
        reasons: set[str] = set()
        with self._transaction() as connection:
            raw = connection.execute(
                "SELECT * FROM active_leases WHERE lease_id = ?",
                (handle.lease_id,),
            ).fetchone()
            if raw is None:
                reasons.add("lease_not_found")
            else:
                row = self._validated_row(raw)
                if not _handle_matches_row(handle, token_sha256, row):
                    reasons.add("lease_token_mismatch")
                elif row["state"] == "delivery_armed":
                    reasons.add("lease_already_armed")
                elif row["state"] == "unknown_blocking":
                    reasons.add("lease_state_unknown")
                else:
                    row["state"] = "delivery_armed"
                    row["row_sha256"] = _row_digest(row)
                    updated = connection.execute(
                        "UPDATE active_leases SET state = ?, row_sha256 = ? "
                        "WHERE lease_id = ? AND state = 'reserved'",
                        (row["state"], row["row_sha256"], row["lease_id"]),
                    ).rowcount
                    if updated != 1:
                        raise CooperativeResourceLeaseStoreError(
                            "lease_store_invalid",
                            "Lease delivery fence was not atomic.",
                        )
                    applied = True
        return CooperativeResourceLeaseTransitionReceipt(
            policy_sha256=self.policy.digest,
            admission_receipt_sha256=handle.admission_receipt_sha256,
            claim_sha256=handle.claim_sha256,
            coordination_domain_sha256=self.coordination_domain_sha256,
            lease_id=handle.lease_id,
            state="delivery_armed",
            transition_applied=applied,
            reason_codes=tuple(sorted(reasons)),
            transitioned_at=transitioned_at,
        )

    def release(
        self,
        handle: CooperativeResourceLeaseHandle,
        *,
        delivery_status: str,
    ) -> CooperativeResourceLeaseReleaseReceipt:
        """Release only a known delivery outcome; ambiguous delivery stays fenced."""

        if delivery_status not in {
            "not_attempted",
            "attempted_unknown",
            "response_received",
        }:
            raise CooperativeResourceLeaseStoreError(
                "lease_delivery_status_invalid", "Delivery status is invalid."
            )
        self._validate_handle(handle)
        token_sha256 = hashlib.sha256(handle.token).hexdigest()
        released_at = self._validated_clock()
        status = "denied"
        reasons: set[str] = set()
        release_owner = False
        with self._transaction() as connection:
            raw = connection.execute(
                "SELECT * FROM active_leases WHERE lease_id = ?",
                (handle.lease_id,),
            ).fetchone()
            if raw is None:
                status = (
                    "unknown_blocking"
                    if delivery_status == "attempted_unknown"
                    else "already_absent"
                )
                reasons.add("lease_not_found")
                release_owner = delivery_status != "attempted_unknown"
                if delivery_status == "attempted_unknown":
                    self._quarantine_domain(connection, handle)
            else:
                row = self._validated_row(raw)
                if not _handle_matches_row(handle, token_sha256, row):
                    reasons.add("lease_token_mismatch")
                elif delivery_status == "attempted_unknown":
                    previous_state = str(row["state"])
                    if previous_state != "unknown_blocking":
                        row["state"] = "unknown_blocking"
                        row["row_sha256"] = _row_digest(row)
                        updated = connection.execute(
                            "UPDATE active_leases SET state = ?, row_sha256 = ? "
                            "WHERE lease_id = ? AND state = ?",
                            (
                                row["state"],
                                row["row_sha256"],
                                row["lease_id"],
                                previous_state,
                            ),
                        ).rowcount
                        if updated != 1:
                            raise CooperativeResourceLeaseStoreError(
                                "lease_store_invalid",
                                "Ambiguous lease fence was not atomic.",
                            )
                    status = "unknown_blocking"
                    reasons.add("delivery_outcome_unknown")
                elif row["state"] == "unknown_blocking":
                    status = "unknown_blocking"
                    reasons.add("lease_state_unknown")
                elif (
                    row["state"] == "reserved" and delivery_status == "not_attempted"
                ) or (
                    row["state"] == "delivery_armed"
                    and delivery_status == "response_received"
                ):
                    deleted = connection.execute(
                        "DELETE FROM active_leases WHERE lease_id = ?",
                        (handle.lease_id,),
                    ).rowcount
                    if deleted != 1:
                        raise CooperativeResourceLeaseStoreError(
                            "lease_store_invalid", "Lease release was not atomic."
                        )
                    status = "released"
                    release_owner = True
                else:
                    reasons.add("lease_state_outcome_mismatch")
            active_after = int(
                connection.execute("SELECT COUNT(*) FROM active_leases").fetchone()[0]
            )
        if release_owner:
            try:
                _release_file_lock(handle._owner_lock, handle._sentinel_path)
                handle._released = True
            except Exception:
                # The authenticated row deletion is already committed.  The
                # orphaned, untracked sentinel cannot block a later admission.
                status = "released_cleanup_deferred" if status == "released" else status
                reasons.add("owner_sentinel_cleanup_failed")
        return CooperativeResourceLeaseReleaseReceipt(
            policy_sha256=self.policy.digest,
            admission_receipt_sha256=handle.admission_receipt_sha256,
            claim_sha256=handle.claim_sha256,
            coordination_domain_sha256=self.coordination_domain_sha256,
            lease_id=handle.lease_id,
            status=status,
            reason_codes=tuple(sorted(reasons)),
            delivery_status=delivery_status,
            released_at=released_at,
            active_leases_after=active_after,
        )

    def _validate_handle(self, handle: CooperativeResourceLeaseHandle) -> None:
        if not isinstance(handle, CooperativeResourceLeaseHandle):
            raise CooperativeResourceLeaseStoreError(
                "lease_handle_invalid", "Lease handle is invalid."
            )
        if not isinstance(handle.token, bytes) or len(handle.token) != 32:
            raise CooperativeResourceLeaseStoreError(
                "lease_handle_invalid", "Lease handle token is invalid."
            )

    def _domain_is_quarantined(self, connection: sqlite3.Connection) -> bool:
        value = connection.execute(
            "SELECT value FROM store_meta WHERE key = 'quarantine_state'"
        ).fetchone()
        if value is None or value[0] not in {"clear", "unknown_blocking"}:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Lease quarantine state is invalid."
            )
        return value[0] == "unknown_blocking"

    def _quarantine_domain(
        self,
        connection: sqlite3.Connection,
        handle: CooperativeResourceLeaseHandle,
    ) -> None:
        evidence_sha256 = sha256_json(
            {
                "lease_id": handle.lease_id,
                "admission_receipt_sha256": handle.admission_receipt_sha256,
                "claim_sha256": handle.claim_sha256,
                "reason": "missing_ambiguous_lease",
            }
        )
        state_updates = connection.execute(
            "UPDATE store_meta SET value = 'unknown_blocking' "
            "WHERE key = 'quarantine_state'"
        ).rowcount
        evidence_updates = connection.execute(
            "UPDATE store_meta SET value = ? WHERE key = 'quarantine_evidence_sha256'",
            (evidence_sha256,),
        ).rowcount
        persisted = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT key, value FROM store_meta "
                "WHERE key IN ('quarantine_state', 'quarantine_evidence_sha256')"
            )
        }
        if (
            state_updates != 1
            or evidence_updates != 1
            or persisted
            != {
                "quarantine_state": "unknown_blocking",
                "quarantine_evidence_sha256": evidence_sha256,
            }
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Lease quarantine could not be verified."
            )

    def _active_rows(
        self, connection: sqlite3.Connection
    ) -> tuple[list[dict[str, object]], int, bool, list[tuple[FileLock, Path]]]:
        rows: list[dict[str, object]] = []
        reaped = 0
        ownership_unknown = self._domain_is_quarantined(connection)
        dead_locks: list[tuple[FileLock, Path]] = []
        for raw in connection.execute("SELECT * FROM active_leases ORDER BY lease_id"):
            row = self._validated_row(raw)
            if row["state"] == "unknown_blocking":
                ownership_unknown = True
                rows.append(row)
                continue
            owner_state, probe = self._probe_owner(row)
            if owner_state == "dead" and row["state"] == "reserved":
                deleted = connection.execute(
                    "DELETE FROM active_leases WHERE lease_id = ? AND state = 'reserved'",
                    (row["lease_id"],),
                ).rowcount
                if deleted != 1 or probe is None:
                    raise CooperativeResourceLeaseStoreError(
                        "lease_store_invalid", "Dead lease reap was not atomic."
                    )
                reaped += 1
                dead_locks.append((probe, self._sentinel_path(str(row["lease_id"]))))
                continue
            if owner_state in {"dead", "unknown"}:
                previous_state = str(row["state"])
                if probe is not None:
                    dead_locks.append(
                        (probe, self._sentinel_path(str(row["lease_id"])))
                    )
                row["state"] = "unknown_blocking"
                row["row_sha256"] = _row_digest(row)
                updated = connection.execute(
                    "UPDATE active_leases SET state = ?, row_sha256 = ? "
                    "WHERE lease_id = ? AND state = ?",
                    (
                        row["state"],
                        row["row_sha256"],
                        row["lease_id"],
                        previous_state,
                    ),
                ).rowcount
                if updated != 1:
                    raise CooperativeResourceLeaseStoreError(
                        "lease_store_invalid",
                        "Unknown lease transition was not atomic.",
                    )
                ownership_unknown = True
            rows.append(row)
        return rows, reaped, ownership_unknown, dead_locks

    def _probe_owner(self, row: dict[str, object]) -> tuple[str, FileLock | None]:
        sentinel_path = self._sentinel_path(str(row["lease_id"]))
        probe = _new_file_lock(sentinel_path)
        try:
            probe.acquire(blocking=False)
        except Timeout:
            return "active", None
        except Exception:
            return "unknown", None
        return "dead", probe

    def _validate_evidence(
        self, claim: CooperativeResourceClaim, snapshot: ResourceSnapshot
    ) -> None:
        if not isinstance(claim, CooperativeResourceClaim) or not isinstance(
            snapshot, ResourceSnapshot
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_evidence_invalid", "Lease evidence is invalid."
            )
        if (
            claim.resource_snapshot_sha256 != snapshot.digest
            or claim.resource_class_sha256 != snapshot.resource_class_sha256
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_evidence_invalid", "Claim and resource snapshot do not match."
            )
        if claim.pool == "unified" and (
            snapshot.memory_topology != "unified"
            or snapshot.accelerator_kind != "integrated"
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_evidence_invalid", "Unified resource topology is unavailable."
            )
        if claim.pool == "discrete" and (
            snapshot.memory_topology != "dedicated"
            or snapshot.accelerator_kind != "discrete"
            or snapshot.accelerator_identity_sha256 != claim.accelerator_identity_sha256
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_evidence_invalid", "Discrete resource topology is unavailable."
            )

    def _initialize(self) -> None:
        with self._connect(validate_schema=False) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                if not tables:
                    self._create_schema(connection)
                self._validate_schema(connection)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(_STORE_META_SQL)
        connection.execute(_ACTIVE_LEASES_SQL)
        connection.executemany(
            "INSERT INTO store_meta(key, value) VALUES (?, ?)",
            (
                ("schema_version", _STORE_SCHEMA_VERSION),
                ("contract", _STORE_CONTRACT),
                ("coordination_domain_sha256", self.coordination_domain_sha256),
                ("quarantine_state", "clear"),
                ("quarantine_evidence_sha256", ""),
            ),
        )
        connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {int(_STORE_SCHEMA_VERSION)}")

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        objects = {
            str(row[1]): (str(row[0]), _normalized_schema_sql(str(row[2])))
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'index', 'view', 'trigger') "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected_objects = {
            "store_meta": ("table", _normalized_schema_sql(_STORE_META_SQL)),
            "active_leases": ("table", _normalized_schema_sql(_ACTIVE_LEASES_SQL)),
        }
        if objects != expected_objects:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_schema_invalid",
                "Lease store schema objects are unsupported.",
            )
        columns = tuple(
            str(row[1])
            for row in connection.execute("PRAGMA table_info(active_leases)")
        )
        if columns != _TABLE_FIELDS:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_schema_invalid", "Lease store columns are unsupported."
            )
        meta = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key, value FROM store_meta")
        }
        expected_static = {
            "schema_version": _STORE_SCHEMA_VERSION,
            "contract": _STORE_CONTRACT,
            "coordination_domain_sha256": self.coordination_domain_sha256,
        }
        quarantine_state = meta.get("quarantine_state")
        quarantine_evidence = meta.get("quarantine_evidence_sha256")
        expected_keys = set(expected_static) | {
            "quarantine_state",
            "quarantine_evidence_sha256",
        }
        quarantine_valid = (
            quarantine_state == "clear" and quarantine_evidence == ""
        ) or (
            quarantine_state == "unknown_blocking"
            and isinstance(quarantine_evidence, str)
            and len(quarantine_evidence) == 64
            and all(
                character in "0123456789abcdef" for character in quarantine_evidence
            )
        )
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if (
            set(meta) != expected_keys
            or any(meta.get(key) != value for key, value in expected_static.items())
            or not quarantine_valid
            or application_id != _APPLICATION_ID
            or user_version != int(_STORE_SCHEMA_VERSION)
        ):
            raise CooperativeResourceLeaseStoreError(
                "lease_store_schema_invalid", "Lease store identity is unsupported."
            )

    @contextmanager
    def _connect(self, *, validate_schema: bool = True) -> Iterator[sqlite3.Connection]:
        expected_identity = _database_identity(self.path)
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.policy.busy_timeout_ms / 1_000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise _sqlite_error(exc) from exc
        connection.row_factory = sqlite3.Row
        try:
            if _database_identity(self.path) != expected_identity:
                raise CooperativeResourceLeaseStoreError(
                    "lease_store_identity_changed", "Lease database identity changed."
                )
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA trusted_schema = OFF")
                connection.execute("PRAGMA temp_store = MEMORY")
                connection.execute(
                    f"PRAGMA busy_timeout = {self.policy.busy_timeout_ms}"
                )
                journal = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
                if journal.lower() != "wal":
                    journal = str(
                        connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                    )
                connection.execute("PRAGMA synchronous = FULL")
                synchronous = int(
                    connection.execute("PRAGMA synchronous").fetchone()[0]
                )
                if journal.lower() != "wal" or synchronous != 2:
                    raise CooperativeResourceLeaseStoreError(
                        "lease_store_unavailable",
                        "Required SQLite durability mode is unavailable.",
                    )
                if validate_schema:
                    self._validate_schema(connection)
            except sqlite3.Error as exc:
                raise _sqlite_error(exc) from exc
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise _sqlite_error(exc) from exc
            except BaseException:
                connection.rollback()
                raise

    def _insert_row(
        self, connection: sqlite3.Connection, row: dict[str, object]
    ) -> None:
        row_sha256 = _row_digest(row)
        placeholders = ", ".join("?" for _ in _TABLE_FIELDS)
        inserted = connection.execute(
            f"INSERT INTO active_leases ({', '.join(_TABLE_FIELDS)}) "
            f"VALUES ({placeholders})",
            tuple(row[name] for name in _ROW_FIELDS) + (row_sha256,),
        ).rowcount
        raw = connection.execute(
            "SELECT * FROM active_leases WHERE lease_id = ?", (row["lease_id"],)
        ).fetchone()
        if inserted != 1 or raw is None:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Lease insert could not be verified."
            )
        persisted = self._validated_row(raw)
        expected = {**row, "row_sha256": row_sha256}
        if any(persisted[name] != expected[name] for name in _TABLE_FIELDS):
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Lease insert read-back does not match."
            )

    def _validated_row(self, raw: sqlite3.Row) -> dict[str, object]:
        row = {name: raw[name] for name in _TABLE_FIELDS}
        try:
            require_safe_id(row["lease_id"], "lease_id")
            for name in (
                "token_sha256",
                "admission_receipt_sha256",
                "claim_sha256",
                "resource_snapshot_sha256",
                "resource_class_sha256",
                "catalog_sha256",
                "profile_sha256",
                "row_sha256",
            ):
                require_sha256(row[name], name)
            if row["accelerator_identity_sha256"] is not None:
                require_sha256(
                    row["accelerator_identity_sha256"],
                    "accelerator_identity_sha256",
                )
            if row["pool"] not in {"system", "unified", "discrete"}:
                raise ValueError
            for name in (
                "system_claim_bytes",
                "accelerator_claim_bytes",
                "safety_reserve_bytes",
                "acquired_monotonic_ns",
            ):
                value = row[name]
                if type(value) is not int or not 0 <= value <= MAX_SQLITE_INTEGER:
                    raise ValueError
            if int(row["system_claim_bytes"]) == 0:
                raise ValueError
            if row["state"] not in _ACTIVE_STATES:
                raise ValueError
            if row["pool"] == "discrete":
                if (
                    int(row["accelerator_claim_bytes"]) == 0
                    or row["accelerator_identity_sha256"] is None
                ):
                    raise ValueError
            elif (
                int(row["accelerator_claim_bytes"]) != 0
                or row["accelerator_identity_sha256"] is not None
            ):
                raise ValueError
            if not hmac.compare_digest(str(row["row_sha256"]), _row_digest(row)):
                raise ValueError
        except (TypeError, ValueError, VerifiedRoutingError) as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Persisted lease row is invalid."
            ) from exc
        return row

    def _sentinel_path(self, lease_id: str) -> Path:
        try:
            safe = require_safe_id(lease_id, "lease_id")
        except VerifiedRoutingError as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Lease identifier is invalid."
            ) from exc
        path = self.sentinel_root / f"{safe}.lock"
        if path.parent != self.sentinel_root:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_invalid", "Lease sentinel is invalid."
            )
        return path

    def _validated_clock(self) -> str:
        try:
            value = self._clock()
            # Receipt validation normalizes and verifies this value.
            return str(value)
        except Exception as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_clock_invalid", "Lease wall clock is unavailable."
            ) from exc

    def _validated_monotonic_clock(self) -> int:
        try:
            value = self._monotonic_clock()
        except Exception as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_clock_invalid", "Lease monotonic clock is unavailable."
            ) from exc
        if type(value) is not int or not 0 <= value <= MAX_SQLITE_INTEGER:
            raise CooperativeResourceLeaseStoreError(
                "lease_clock_invalid", "Lease monotonic clock is invalid."
            )
        return value


def _row_digest(row: dict[str, object]) -> str:
    return sha256_json({name: row[name] for name in _ROW_FIELDS})


def _normalized_schema_sql(value: str) -> str:
    return " ".join(value.split())


def _conservative_peak_resources(
    passport: CellPassport,
) -> tuple[int | None, int | None, int | None]:
    """Take the larger applicable estimate/measurement without private helpers."""

    estimated, measured = passport.estimated, passport.measured
    if estimated.memory_pool is None or estimated.placement is None:
        return None, None, None
    measurement_applies = (
        measured.memory_pool == estimated.memory_pool
        and measured.placement == estimated.placement
    )
    values: list[int | None] = []
    for name in (
        "peak_host_memory_bytes",
        "peak_unified_memory_bytes",
        "peak_accelerator_memory_bytes",
    ):
        estimate = getattr(estimated, name)
        measurement = getattr(measured, name) if measurement_applies else None
        if estimate is None:
            values.append(measurement)
        elif measurement is None:
            values.append(estimate)
        else:
            values.append(max(estimate, measurement))
    return values[0], values[1], values[2]


def _handle_matches_row(
    handle: CooperativeResourceLeaseHandle,
    token_sha256: str,
    row: dict[str, object],
) -> bool:
    return (
        hmac.compare_digest(str(row["token_sha256"]), token_sha256)
        and hmac.compare_digest(str(row["claim_sha256"]), handle.claim_sha256)
        and hmac.compare_digest(
            str(row["admission_receipt_sha256"]),
            handle.admission_receipt_sha256,
        )
    )


def _checked_sum(values) -> int:
    total = 0
    for value in values:
        if type(value) is not int or value < 0:
            raise CooperativeResourceLeaseStoreError(
                "lease_accounting_invalid", "Lease byte accounting is invalid."
            )
        total += value
        if total > MAX_SQLITE_INTEGER:
            raise CooperativeResourceLeaseStoreError(
                "lease_accounting_invalid", "Lease byte accounting overflowed."
            )
    return total


def _system_available(snapshot: ResourceSnapshot) -> int | None:
    if (
        snapshot.available_memory_bytes is None
        or snapshot.effective_memory_limit_bytes is None
    ):
        return None
    return min(snapshot.available_memory_bytes, snapshot.effective_memory_limit_bytes)


def _accelerator_total(
    rows: list[dict[str, object]], claim: CooperativeResourceClaim
) -> tuple[int, bool]:
    if claim.pool != "discrete":
        return 0, False
    values: list[int] = []
    identity_conflict = False
    for row in rows:
        accelerator_bytes = int(row["accelerator_claim_bytes"])
        if accelerator_bytes == 0:
            continue
        if row["accelerator_identity_sha256"] != claim.accelerator_identity_sha256:
            identity_conflict = True
            continue
        values.append(accelerator_bytes)
    return _checked_sum(values), identity_conflict


def _new_file_lock(path: Path) -> FileLock:
    _validate_optional_regular_path(path, "lease sentinel")
    return FileLock(
        path,
        timeout=0,
        mode=0o600,
        thread_local=False,
        blocking=False,
        lifetime=None,
    )


def _release_file_lock(lock: FileLock, path: Path) -> None:
    lock.release(force=True)
    try:
        if path.is_symlink():
            return
        path.unlink(missing_ok=True)
    except OSError:
        # The native lock is already released. A leftover regular sentinel is
        # harmless because ownership is determined by the OS lock, not presence.
        return


def _prepare_database_path(value: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(value.expanduser())))
    if raw.name in {"", ".", ".."}:
        raise CooperativeResourceLeaseStoreError(
            "lease_store_path_invalid", "Lease database path is invalid."
        )
    parent = _prepare_directory(raw.parent, "database parent")
    path = parent / raw.name
    _validate_optional_regular_path(path, "lease database")
    if not path.exists():
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except FileExistsError:
            pass
        except OSError as exc:
            raise CooperativeResourceLeaseStoreError(
                "lease_store_path_invalid", "Lease database is unavailable."
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    _validate_database_file(path)
    return path


def _prepare_directory(value: Path, label: str) -> Path:
    raw = Path(os.path.abspath(os.fspath(value.expanduser())))
    try:
        raw.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = raw.resolve(strict=True)
        details = path.lstat()
        if _is_link_or_reparse(details) or not stat.S_ISDIR(details.st_mode):
            raise OSError
        if os.name == "posix":
            os.chmod(path, 0o700, follow_symlinks=False)
    except (OSError, RuntimeError) as exc:
        raise CooperativeResourceLeaseStoreError(
            "lease_store_path_invalid", f"Lease {label} is unavailable."
        ) from exc
    return path


def _validate_optional_regular_path(path: Path, label: str) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CooperativeResourceLeaseStoreError(
            "lease_store_path_invalid", f"{label.capitalize()} is unavailable."
        ) from exc
    if _is_link_or_reparse(details) or not stat.S_ISREG(details.st_mode):
        raise CooperativeResourceLeaseStoreError(
            "lease_store_path_invalid",
            f"{label.capitalize()} must be a regular non-link file.",
        )


def _validate_database_file(path: Path) -> os.stat_result:
    _validate_optional_regular_path(path, "lease database")
    try:
        details = path.lstat()
        if os.name == "posix":
            os.chmod(path, 0o600, follow_symlinks=False)
            details = path.lstat()
    except OSError as exc:
        raise CooperativeResourceLeaseStoreError(
            "lease_store_path_invalid", "Lease database is unavailable."
        ) from exc
    return details


def _database_identity(path: Path) -> tuple[int, int]:
    details = _validate_database_file(path)
    return int(details.st_dev), int(details.st_ino)


def _is_link_or_reparse(details: os.stat_result) -> bool:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    attributes = int(getattr(details, "st_file_attributes", 0))
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse_flag)


def _sqlite_error(exc: sqlite3.Error) -> CooperativeResourceLeaseStoreError:
    rendered = str(exc).lower()
    if "locked" in rendered or "busy" in rendered:
        return CooperativeResourceLeaseStoreError(
            "lease_store_busy", "Lease store is busy."
        )
    if "malformed" in rendered or "not a database" in rendered:
        return CooperativeResourceLeaseStoreError(
            "lease_store_corrupt", "Lease store is corrupt."
        )
    return CooperativeResourceLeaseStoreError(
        "lease_store_unavailable", "Lease store operation failed."
    )


__all__ = [
    "CLAIM_BASIS",
    "CooperativeResourceLeaseAcquisition",
    "CooperativeResourceLeaseEvaluation",
    "CooperativeResourceLeaseHandle",
    "CooperativeResourceLeasePaths",
    "CooperativeResourceLeaseStoreError",
    "SQLiteCooperativeResourceLeaseStore",
    "cooperative_resource_claim_from_preview",
    "default_cooperative_resource_lease_paths",
]
