"""Owner-bound durable ledger for process-bound runtime lifecycle metadata.

The store is deliberately independent from Cooperative Resource Lease.  It
serializes exact endpoint ownership claims and authenticates state changes; it
does not inspect, start, stop, or authorize a runtime process.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import stat
from typing import Callable, Iterator, Sequence

from filelock import FileLock, Timeout
from platformdirs import user_state_path

from .runtime_supervisor_contracts import (
    ACTIVE_RUNTIME_SUPERVISOR_STATES,
    MAX_SQLITE_INTEGER,
    RuntimeSupervisorContractError,
    RuntimeSupervisorLeaseBinding,
    RuntimeSupervisorLeasePolicy,
    RuntimeSupervisorLeaseReceipt,
    runtime_supervisor_transition_allowed,
)
from .verified_routing_contracts import (
    VerifiedRoutingError,
    canonical_json,
    now_utc,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


_STORE_SCHEMA_VERSION = "1"
_STORE_CONTRACT = "mymoe-runtime-supervisor-lease-store"
_APPLICATION_ID = 0x4D595253  # MYRS
_STORE_META_SQL = (
    "CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) "
    "STRICT, WITHOUT ROWID"
)
_ACTIVE_STATE_SQL = ", ".join(
    f"'{value}'" for value in sorted(ACTIVE_RUNTIME_SUPERVISOR_STATES)
)
_ACTIVE_LEASES_SQL = f"""
CREATE TABLE active_leases (
    lease_id TEXT PRIMARY KEY,
    token_sha256 TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    binding_sha256 TEXT NOT NULL,
    endpoint_authority_sha256 TEXT NOT NULL UNIQUE,
    coordination_domain_sha256 TEXT NOT NULL,
    owner_pid INTEGER NOT NULL CHECK (owner_pid > 0),
    state TEXT NOT NULL CHECK (state IN ({_ACTIVE_STATE_SQL})),
    reason_codes_json TEXT NOT NULL,
    transition_index INTEGER NOT NULL CHECK (transition_index >= 0),
    previous_receipt_sha256 TEXT,
    runtime_pid INTEGER CHECK (runtime_pid IS NULL OR runtime_pid > 0),
    runtime_create_time_ns INTEGER CHECK (
        runtime_create_time_ns IS NULL OR runtime_create_time_ns > 0
    ),
    runtime_executable_sha256 TEXT,
    process_tree_sha256 TEXT,
    endpoint_evidence_sha256 TEXT,
    updated_at TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    row_sha256 TEXT NOT NULL
) STRICT, WITHOUT ROWID
"""
_ROW_FIELDS = (
    "lease_id",
    "token_sha256",
    "policy_sha256",
    "binding_sha256",
    "endpoint_authority_sha256",
    "coordination_domain_sha256",
    "owner_pid",
    "state",
    "reason_codes_json",
    "transition_index",
    "previous_receipt_sha256",
    "runtime_pid",
    "runtime_create_time_ns",
    "runtime_executable_sha256",
    "process_tree_sha256",
    "endpoint_evidence_sha256",
    "updated_at",
    "receipt_sha256",
)
_TABLE_FIELDS = _ROW_FIELDS + ("row_sha256",)


class RuntimeSupervisorLeaseStoreError(RuntimeError):
    """Fail-closed store error with a stable non-sensitive reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeSupervisorLeasePaths:
    database: Path
    sentinels: Path


@dataclass(slots=True)
class RuntimeSupervisorLeaseHandle:
    """In-memory ownership capability; raw token bytes are never persisted."""

    lease_id: str
    binding_sha256: str
    endpoint_authority_sha256: str
    token: bytes = field(repr=False)
    _owner_lock: FileLock = field(repr=False, compare=False)
    _sentinel_path: Path = field(repr=False, compare=False)
    _released: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True)
class RuntimeSupervisorLeaseAcquisition:
    receipt: RuntimeSupervisorLeaseReceipt
    handle: RuntimeSupervisorLeaseHandle


def default_runtime_supervisor_lease_paths() -> RuntimeSupervisorLeasePaths:
    base = Path(user_state_path("myMoE", appauthor=False, ensure_exists=False))
    root = base / "runtime-supervisor" / "v1"
    return RuntimeSupervisorLeasePaths(
        database=root / "leases.sqlite3",
        sentinels=root / "sentinels",
    )


class SQLiteRuntimeSupervisorLeaseStore:
    """Same-user durable ownership ledger for exact local runtime endpoints."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        *,
        sentinel_root: str | Path | None = None,
        policy: RuntimeSupervisorLeasePolicy | None = None,
        clock: Callable[[], str] = now_utc,
    ) -> None:
        self.policy = policy or RuntimeSupervisorLeasePolicy()
        if not isinstance(self.policy, RuntimeSupervisorLeasePolicy):
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_policy_invalid", "Runtime lease policy is invalid."
            )
        defaults = default_runtime_supervisor_lease_paths()
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
        self.sentinel_root = _prepare_directory(
            declared_sentinels, "runtime lease sentinel root"
        )
        self.coordination_domain_sha256 = hashlib.sha256(
            (
                f"{_STORE_CONTRACT}\0{os.path.normcase(os.fspath(self.path))}"
                f"\0{os.path.normcase(os.fspath(self.sentinel_root))}"
            ).encode("utf-8")
        ).hexdigest()
        self._clock = clock
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
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_busy",
                "Runtime lease store initialization is busy.",
            ) from exc
        finally:
            initialization_lock.release(force=True)

    def acquire(
        self, binding: RuntimeSupervisorLeaseBinding
    ) -> RuntimeSupervisorLeaseAcquisition:
        """Atomically reserve one exact endpoint in the prepared state."""

        if not isinstance(binding, RuntimeSupervisorLeaseBinding):
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_binding_invalid", "Runtime lease binding is invalid."
            )
        lease_id = f"runtime-{secrets.token_hex(16)}"
        token = secrets.token_bytes(32)
        token_sha256 = hashlib.sha256(token).hexdigest()
        sentinel_path = self._sentinel_path(lease_id)
        owner_lock = _new_file_lock(sentinel_path)
        try:
            owner_lock.acquire(blocking=False)
        except Exception as exc:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_owner_unavailable",
                "Runtime lease owner sentinel is unavailable.",
            ) from exc

        owner_transferred = False
        probe_locks: list[tuple[FileLock, Path]] = []
        conflict_code: str | None = None
        receipt: RuntimeSupervisorLeaseReceipt | None = None
        try:
            with self._transaction() as connection:
                _, probes = self._mark_abandoned_rows(connection)
                probe_locks.extend(probes)
                active_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM active_leases "
                        "WHERE state != 'unknown_blocking'"
                    ).fetchone()[0]
                )
                if active_count >= self.policy.max_active_leases:
                    conflict_code = "runtime_lease_limit_reached"
                elif connection.execute(
                    "SELECT 1 FROM active_leases "
                    "WHERE endpoint_authority_sha256 = ?",
                    (binding.endpoint_authority_sha256,),
                ).fetchone() is not None:
                    conflict_code = "runtime_endpoint_already_leased"
                else:
                    receipt = RuntimeSupervisorLeaseReceipt(
                        policy_sha256=self.policy.digest,
                        binding_sha256=binding.digest,
                        endpoint_authority_sha256=(
                            binding.endpoint_authority_sha256
                        ),
                        coordination_domain_sha256=(
                            self.coordination_domain_sha256
                        ),
                        lease_id=lease_id,
                        lease_token_sha256=token_sha256,
                        owner_pid=os.getpid(),
                        state="prepared",
                        reason_codes=(),
                        transition_index=0,
                        previous_receipt_sha256=None,
                        runtime_pid=None,
                        runtime_create_time_ns=None,
                        runtime_executable_sha256=None,
                        process_tree_sha256=None,
                        endpoint_evidence_sha256=None,
                        updated_at=self._validated_clock(),
                    )
                    self._insert_receipt(connection, receipt)
            if conflict_code is not None:
                message = (
                    "Runtime endpoint already has an active or unknown lease."
                    if conflict_code == "runtime_endpoint_already_leased"
                    else "Runtime lease store reached its active lease limit."
                )
                raise RuntimeSupervisorLeaseStoreError(conflict_code, message)
            if receipt is None:
                raise RuntimeSupervisorLeaseStoreError(
                    "runtime_lease_store_invalid",
                    "Runtime lease acquisition produced no receipt.",
                )
            handle = RuntimeSupervisorLeaseHandle(
                lease_id=lease_id,
                binding_sha256=binding.digest,
                endpoint_authority_sha256=binding.endpoint_authority_sha256,
                token=token,
                _owner_lock=owner_lock,
                _sentinel_path=sentinel_path,
            )
            owner_transferred = True
            return RuntimeSupervisorLeaseAcquisition(receipt=receipt, handle=handle)
        finally:
            for probe_lock, probe_path in probe_locks:
                _release_file_lock(probe_lock, probe_path)
            if not owner_transferred:
                _release_file_lock(owner_lock, sentinel_path)

    def transition(
        self,
        handle: RuntimeSupervisorLeaseHandle,
        state: str,
        *,
        reason_codes: Sequence[str] = (),
        runtime_pid: int | None = None,
        runtime_create_time_ns: int | None = None,
        runtime_executable_sha256: str | None = None,
        process_tree_sha256: str | None = None,
        endpoint_evidence_sha256: str | None = None,
    ) -> RuntimeSupervisorLeaseReceipt:
        """Authenticate and durably apply one allowed lifecycle transition."""

        self._validate_handle(handle)
        token_sha256 = hashlib.sha256(handle.token).hexdigest()
        release_owner = False
        with self._transaction() as connection:
            raw = connection.execute(
                "SELECT * FROM active_leases WHERE lease_id = ?",
                (handle.lease_id,),
            ).fetchone()
            if raw is None:
                raise RuntimeSupervisorLeaseStoreError(
                    "runtime_lease_not_found", "Runtime lease is not active."
                )
            current = self._validated_row(raw)
            self._authenticate(handle, token_sha256, current)
            if not runtime_supervisor_transition_allowed(current.state, state):
                raise RuntimeSupervisorLeaseStoreError(
                    "runtime_lease_transition_invalid",
                    "Runtime lease transition is not allowed.",
                )
            receipt = RuntimeSupervisorLeaseReceipt(
                policy_sha256=self.policy.digest,
                binding_sha256=current.binding_sha256,
                endpoint_authority_sha256=current.endpoint_authority_sha256,
                coordination_domain_sha256=self.coordination_domain_sha256,
                lease_id=current.lease_id,
                lease_token_sha256=current.lease_token_sha256,
                owner_pid=current.owner_pid,
                state=state,
                reason_codes=tuple(reason_codes),
                transition_index=current.transition_index + 1,
                previous_receipt_sha256=current.digest,
                runtime_pid=(
                    current.runtime_pid if runtime_pid is None else runtime_pid
                ),
                runtime_create_time_ns=(
                    current.runtime_create_time_ns
                    if runtime_create_time_ns is None
                    else runtime_create_time_ns
                ),
                runtime_executable_sha256=(
                    current.runtime_executable_sha256
                    if runtime_executable_sha256 is None
                    else runtime_executable_sha256
                ),
                process_tree_sha256=(
                    current.process_tree_sha256
                    if process_tree_sha256 is None
                    else process_tree_sha256
                ),
                endpoint_evidence_sha256=(
                    current.endpoint_evidence_sha256
                    if endpoint_evidence_sha256 is None
                    else endpoint_evidence_sha256
                ),
                updated_at=self._validated_clock(),
            )
            if state == "stopped":
                deleted = connection.execute(
                    "DELETE FROM active_leases WHERE lease_id = ?",
                    (handle.lease_id,),
                ).rowcount
                if deleted != 1:
                    raise RuntimeSupervisorLeaseStoreError(
                        "runtime_lease_store_invalid",
                        "Stopped runtime lease was not removed atomically.",
                    )
                release_owner = True
            else:
                self._update_receipt(connection, current, receipt)
        if release_owner:
            _release_file_lock(handle._owner_lock, handle._sentinel_path)
            handle._released = True
        return receipt

    def get(self, lease_id: str) -> RuntimeSupervisorLeaseReceipt | None:
        """Return current metadata after fail-closed abandoned-owner detection."""

        try:
            safe_id = require_safe_id(lease_id, "lease_id")
        except VerifiedRoutingError as exc:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_id_invalid", "Runtime lease identifier is invalid."
            ) from exc
        probe_locks: list[tuple[FileLock, Path]] = []
        try:
            with self._transaction() as connection:
                _, probes = self._mark_abandoned_rows(connection)
                probe_locks.extend(probes)
                raw = connection.execute(
                    "SELECT * FROM active_leases WHERE lease_id = ?", (safe_id,)
                ).fetchone()
                return None if raw is None else self._validated_row(raw)
        finally:
            for probe_lock, probe_path in probe_locks:
                _release_file_lock(probe_lock, probe_path)

    def list_active(self) -> tuple[RuntimeSupervisorLeaseReceipt, ...]:
        """List active records after marking abandoned ownership unknown."""

        probe_locks: list[tuple[FileLock, Path]] = []
        try:
            with self._transaction() as connection:
                _, probes = self._mark_abandoned_rows(connection)
                probe_locks.extend(probes)
                return tuple(
                    self._validated_row(raw)
                    for raw in connection.execute(
                        "SELECT * FROM active_leases ORDER BY lease_id"
                    )
                )
        finally:
            for probe_lock, probe_path in probe_locks:
                _release_file_lock(probe_lock, probe_path)

    def mark_abandoned_owners(self) -> tuple[RuntimeSupervisorLeaseReceipt, ...]:
        """Persist sticky unknown-blocking states for every dead lease owner."""

        probe_locks: list[tuple[FileLock, Path]] = []
        try:
            with self._transaction() as connection:
                changed, probes = self._mark_abandoned_rows(connection)
                probe_locks.extend(probes)
                return tuple(changed)
        finally:
            for probe_lock, probe_path in probe_locks:
                _release_file_lock(probe_lock, probe_path)

    def _mark_abandoned_rows(
        self, connection: sqlite3.Connection
    ) -> tuple[
        list[RuntimeSupervisorLeaseReceipt], list[tuple[FileLock, Path]]
    ]:
        changed: list[RuntimeSupervisorLeaseReceipt] = []
        probes: list[tuple[FileLock, Path]] = []
        for raw in connection.execute("SELECT * FROM active_leases ORDER BY lease_id"):
            current = self._validated_row(raw)
            if current.state == "unknown_blocking":
                continue
            owner_state, probe = self._probe_owner(current.lease_id)
            if owner_state == "active":
                continue
            if probe is not None:
                probes.append((probe, self._sentinel_path(current.lease_id)))
            receipt = RuntimeSupervisorLeaseReceipt(
                policy_sha256=current.policy_sha256,
                binding_sha256=current.binding_sha256,
                endpoint_authority_sha256=current.endpoint_authority_sha256,
                coordination_domain_sha256=current.coordination_domain_sha256,
                lease_id=current.lease_id,
                lease_token_sha256=current.lease_token_sha256,
                owner_pid=current.owner_pid,
                state="unknown_blocking",
                reason_codes=("ownership_unknown",),
                transition_index=current.transition_index + 1,
                previous_receipt_sha256=current.digest,
                runtime_pid=current.runtime_pid,
                runtime_create_time_ns=current.runtime_create_time_ns,
                runtime_executable_sha256=current.runtime_executable_sha256,
                process_tree_sha256=current.process_tree_sha256,
                endpoint_evidence_sha256=current.endpoint_evidence_sha256,
                updated_at=self._validated_clock(),
            )
            self._update_receipt(connection, current, receipt)
            changed.append(receipt)
        return changed, probes

    def _probe_owner(
        self, lease_id: str
    ) -> tuple[str, FileLock | None]:
        path = self._sentinel_path(lease_id)
        probe = _new_file_lock(path)
        try:
            probe.acquire(blocking=False)
        except Timeout:
            return "active", None
        except Exception:
            return "unknown", None
        return "dead", probe

    def _authenticate(
        self,
        handle: RuntimeSupervisorLeaseHandle,
        token_sha256: str,
        receipt: RuntimeSupervisorLeaseReceipt,
    ) -> None:
        if not (
            hmac.compare_digest(receipt.lease_token_sha256, token_sha256)
            and hmac.compare_digest(receipt.binding_sha256, handle.binding_sha256)
            and hmac.compare_digest(
                receipt.endpoint_authority_sha256,
                handle.endpoint_authority_sha256,
            )
        ):
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_token_mismatch",
                "Runtime lease ownership capability does not match.",
            )

    def _validate_handle(self, handle: RuntimeSupervisorLeaseHandle) -> None:
        if (
            not isinstance(handle, RuntimeSupervisorLeaseHandle)
            or not isinstance(handle.token, bytes)
            or len(handle.token) != 32
            or handle._released
        ):
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_handle_invalid", "Runtime lease handle is invalid."
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
                ("policy_sha256", self.policy.digest),
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
            "active_leases": (
                "table",
                _normalized_schema_sql(_ACTIVE_LEASES_SQL),
            ),
        }
        if objects != expected_objects:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_schema_invalid",
                "Runtime lease store schema objects are unsupported.",
            )
        columns = tuple(
            str(row[1])
            for row in connection.execute("PRAGMA table_info(active_leases)")
        )
        meta = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key, value FROM store_meta")
        }
        expected_meta = {
            "schema_version": _STORE_SCHEMA_VERSION,
            "contract": _STORE_CONTRACT,
            "coordination_domain_sha256": self.coordination_domain_sha256,
            "policy_sha256": self.policy.digest,
        }
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if (
            columns != _TABLE_FIELDS
            or meta != expected_meta
            or application_id != _APPLICATION_ID
            or user_version != int(_STORE_SCHEMA_VERSION)
        ):
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_schema_invalid",
                "Runtime lease store identity is unsupported.",
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
                raise RuntimeSupervisorLeaseStoreError(
                    "runtime_lease_store_identity_changed",
                    "Runtime lease database identity changed.",
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
                    raise RuntimeSupervisorLeaseStoreError(
                        "runtime_lease_store_unavailable",
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

    def _insert_receipt(
        self, connection: sqlite3.Connection, receipt: RuntimeSupervisorLeaseReceipt
    ) -> None:
        row = _receipt_row(receipt)
        row_sha256 = _row_digest(row)
        placeholders = ", ".join("?" for _ in _TABLE_FIELDS)
        inserted = connection.execute(
            f"INSERT INTO active_leases ({', '.join(_TABLE_FIELDS)}) "
            f"VALUES ({placeholders})",
            tuple(row[name] for name in _ROW_FIELDS) + (row_sha256,),
        ).rowcount
        raw = connection.execute(
            "SELECT * FROM active_leases WHERE lease_id = ?", (receipt.lease_id,)
        ).fetchone()
        if inserted != 1 or raw is None or self._validated_row(raw) != receipt:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_invalid",
                "Runtime lease insert could not be verified.",
            )

    def _update_receipt(
        self,
        connection: sqlite3.Connection,
        current: RuntimeSupervisorLeaseReceipt,
        receipt: RuntimeSupervisorLeaseReceipt,
    ) -> None:
        row = _receipt_row(receipt)
        row_sha256 = _row_digest(row)
        assignments = ", ".join(f"{name} = ?" for name in _ROW_FIELDS[1:])
        updated = connection.execute(
            f"UPDATE active_leases SET {assignments}, row_sha256 = ? "
            "WHERE lease_id = ? AND receipt_sha256 = ?",
            tuple(row[name] for name in _ROW_FIELDS[1:])
            + (row_sha256, current.lease_id, current.digest),
        ).rowcount
        raw = connection.execute(
            "SELECT * FROM active_leases WHERE lease_id = ?", (current.lease_id,)
        ).fetchone()
        if updated != 1 or raw is None or self._validated_row(raw) != receipt:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_invalid",
                "Runtime lease transition could not be verified.",
            )

    def _validated_row(self, raw: sqlite3.Row) -> RuntimeSupervisorLeaseReceipt:
        row = {name: raw[name] for name in _TABLE_FIELDS}
        try:
            if row["state"] not in ACTIVE_RUNTIME_SUPERVISOR_STATES:
                raise ValueError
            for name in (
                "token_sha256",
                "policy_sha256",
                "binding_sha256",
                "endpoint_authority_sha256",
                "coordination_domain_sha256",
                "receipt_sha256",
                "row_sha256",
            ):
                require_sha256(row[name], name)
            require_safe_id(row["lease_id"], "lease_id")
            reason_codes = json.loads(str(row["reason_codes_json"]))
            if (
                not isinstance(reason_codes, list)
                or canonical_json(reason_codes) != row["reason_codes_json"]
            ):
                raise ValueError
            receipt = RuntimeSupervisorLeaseReceipt(
                policy_sha256=str(row["policy_sha256"]),
                binding_sha256=str(row["binding_sha256"]),
                endpoint_authority_sha256=str(row["endpoint_authority_sha256"]),
                coordination_domain_sha256=str(row["coordination_domain_sha256"]),
                lease_id=str(row["lease_id"]),
                lease_token_sha256=str(row["token_sha256"]),
                owner_pid=row["owner_pid"],
                state=str(row["state"]),
                reason_codes=tuple(reason_codes),
                transition_index=row["transition_index"],
                previous_receipt_sha256=row["previous_receipt_sha256"],
                runtime_pid=row["runtime_pid"],
                runtime_create_time_ns=row["runtime_create_time_ns"],
                runtime_executable_sha256=row["runtime_executable_sha256"],
                process_tree_sha256=row["process_tree_sha256"],
                endpoint_evidence_sha256=row["endpoint_evidence_sha256"],
                updated_at=str(row["updated_at"]),
                digest=str(row["receipt_sha256"]),
            )
            if (
                receipt.policy_sha256 != self.policy.digest
                or receipt.coordination_domain_sha256
                != self.coordination_domain_sha256
                or not hmac.compare_digest(str(row["row_sha256"]), _row_digest(row))
            ):
                raise ValueError
        except (
            json.JSONDecodeError,
            RuntimeSupervisorContractError,
            TypeError,
            ValueError,
            VerifiedRoutingError,
        ) as exc:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_invalid",
                "Persisted runtime lease row is invalid.",
            ) from exc
        return receipt

    def _sentinel_path(self, lease_id: str) -> Path:
        try:
            safe = require_safe_id(lease_id, "lease_id")
        except VerifiedRoutingError as exc:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_invalid",
                "Runtime lease sentinel identifier is invalid.",
            ) from exc
        path = self.sentinel_root / f"{safe}.lock"
        if path.parent != self.sentinel_root:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_invalid", "Runtime lease sentinel is invalid."
            )
        return path

    def _validated_clock(self) -> str:
        try:
            return require_utc_timestamp(self._clock(), "runtime lease clock")
        except (TypeError, ValueError, VerifiedRoutingError) as exc:
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_clock_invalid", "Runtime lease clock is unavailable."
            ) from exc


def _receipt_row(receipt: RuntimeSupervisorLeaseReceipt) -> dict[str, object]:
    if receipt.state == "stopped":
        raise RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_invalid",
            "Stopped receipts are terminal and cannot be persisted as active.",
        )
    return {
        "lease_id": receipt.lease_id,
        "token_sha256": receipt.lease_token_sha256,
        "policy_sha256": receipt.policy_sha256,
        "binding_sha256": receipt.binding_sha256,
        "endpoint_authority_sha256": receipt.endpoint_authority_sha256,
        "coordination_domain_sha256": receipt.coordination_domain_sha256,
        "owner_pid": receipt.owner_pid,
        "state": receipt.state,
        "reason_codes_json": canonical_json(list(receipt.reason_codes)),
        "transition_index": receipt.transition_index,
        "previous_receipt_sha256": receipt.previous_receipt_sha256,
        "runtime_pid": receipt.runtime_pid,
        "runtime_create_time_ns": receipt.runtime_create_time_ns,
        "runtime_executable_sha256": receipt.runtime_executable_sha256,
        "process_tree_sha256": receipt.process_tree_sha256,
        "endpoint_evidence_sha256": receipt.endpoint_evidence_sha256,
        "updated_at": receipt.updated_at,
        "receipt_sha256": receipt.digest,
    }


def _row_digest(row: dict[str, object]) -> str:
    return sha256_json({name: row[name] for name in _ROW_FIELDS})


def _normalized_schema_sql(value: str) -> str:
    return " ".join(value.split())


def _new_file_lock(path: Path) -> FileLock:
    _validate_optional_regular_path(path, "runtime lease sentinel")
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
        # Native ownership is already released.  An unlocked regular sentinel
        # cannot assert ownership and will be probed fail-closed later.
        return


def _prepare_database_path(value: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(value.expanduser())))
    if raw.name in {"", ".", ".."}:
        raise RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_path_invalid",
            "Runtime lease database path is invalid.",
        )
    parent = _prepare_directory(raw.parent, "runtime lease database parent")
    path = parent / raw.name
    _validate_optional_regular_path(path, "runtime lease database")
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
            raise RuntimeSupervisorLeaseStoreError(
                "runtime_lease_store_path_invalid",
                "Runtime lease database is unavailable.",
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
        raise RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_path_invalid", f"{label.capitalize()} is unavailable."
        ) from exc
    return path


def _validate_optional_regular_path(path: Path, label: str) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_path_invalid", f"{label.capitalize()} is unavailable."
        ) from exc
    if _is_link_or_reparse(details) or not stat.S_ISREG(details.st_mode):
        raise RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_path_invalid",
            f"{label.capitalize()} must be a regular non-link file.",
        )


def _validate_database_file(path: Path) -> os.stat_result:
    _validate_optional_regular_path(path, "runtime lease database")
    try:
        details = path.lstat()
        if os.name == "posix":
            os.chmod(path, 0o600, follow_symlinks=False)
            details = path.lstat()
    except OSError as exc:
        raise RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_path_invalid",
            "Runtime lease database is unavailable.",
        ) from exc
    return details


def _database_identity(path: Path) -> tuple[int, int]:
    details = _validate_database_file(path)
    return int(details.st_dev), int(details.st_ino)


def _is_link_or_reparse(details: os.stat_result) -> bool:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    attributes = int(getattr(details, "st_file_attributes", 0))
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse_flag)


def _sqlite_error(exc: sqlite3.Error) -> RuntimeSupervisorLeaseStoreError:
    rendered = str(exc).lower()
    if "locked" in rendered or "busy" in rendered:
        return RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_busy", "Runtime lease store is busy."
        )
    if "malformed" in rendered or "not a database" in rendered:
        return RuntimeSupervisorLeaseStoreError(
            "runtime_lease_store_corrupt", "Runtime lease store is corrupt."
        )
    return RuntimeSupervisorLeaseStoreError(
        "runtime_lease_store_unavailable", "Runtime lease store operation failed."
    )


__all__ = [
    "RuntimeSupervisorLeaseAcquisition",
    "RuntimeSupervisorLeaseHandle",
    "RuntimeSupervisorLeasePaths",
    "RuntimeSupervisorLeaseStoreError",
    "SQLiteRuntimeSupervisorLeaseStore",
    "default_runtime_supervisor_lease_paths",
]
