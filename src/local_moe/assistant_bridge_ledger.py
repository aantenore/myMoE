from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import threading
import time
from typing import Callable, Iterator
from uuid import uuid4

from .assistant_bridge_process import process_is_alive


LEDGER_SCHEMA_VERSION = "2.2"
_LEGACY_SCHEMA_VERSIONS = frozenset({"2.0", "2.1"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^confirm-v2-[A-Za-z0-9_-]{43}$")
_BUDGET_LEASE_TOKEN = re.compile(r"^budget-v2-[A-Za-z0-9_-]{43}$")
_IS_WINDOWS = os.name == "nt"


class BridgeLedgerError(ValueError):
    """Raised when authorization or budget state cannot be proven safely."""


class BridgeConfirmationNotReadyError(BridgeLedgerError):
    """Raised when a fresh one-shot confirmation must be planned."""

    code = "confirmation_not_ready"


@dataclass(frozen=True)
class ConfirmationTicket:
    token: str = field(repr=False)
    transaction_id: str
    expires_at: float

    def metadata_payload(self) -> dict[str, object]:
        return {
            "token_sha256": _sha256_text(self.token),
            "transaction_id": self.transaction_id,
            "expires_at": self.expires_at,
            "one_shot": True,
        }


@dataclass(frozen=True)
class PremiumBudgetLease:
    """Opaque pending premium authority reserved before process launch."""

    token: str = field(repr=False)
    key_sha256: str
    reserved_at: float
    expires_at: float

    def metadata_payload(self) -> dict[str, object]:
        return {
            "token_sha256": _sha256_text(self.token),
            "key_sha256": self.key_sha256,
            "reserved_at": self.reserved_at,
            "expires_at": self.expires_at,
            "state": "pending",
        }


class BridgeStateLedger:
    """Atomic namespaced budget and one-shot confirmation state."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        lock_timeout_seconds: float = 5.0,
        stale_lock_seconds: float = 120.0,
        budget_retention_seconds: float = 90 * 24 * 60 * 60,
        max_budget_entries: int = 4096,
        confirmation_retention_seconds: float = 24 * 60 * 60,
        max_confirmation_entries: int = 4096,
        budget_lease_ttl_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        declared = Path(path).expanduser().absolute()
        _reject_symlink_path(declared)
        self._declared_path = declared
        self.path = declared.resolve()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", namespace) is None:
            raise BridgeLedgerError("Bridge ledger namespace is invalid.")
        if not 0.1 <= lock_timeout_seconds <= 60:
            raise BridgeLedgerError("Bridge ledger lock timeout is invalid.")
        if not 1 <= stale_lock_seconds <= 86_400:
            raise BridgeLedgerError("Bridge ledger stale-lock TTL is invalid.")
        if (
            isinstance(budget_retention_seconds, bool)
            or not isinstance(budget_retention_seconds, (int, float))
            or not math.isfinite(float(budget_retention_seconds))
            or not 1 <= budget_retention_seconds <= 365 * 24 * 60 * 60
        ):
            raise BridgeLedgerError("Bridge budget retention is invalid.")
        if (
            isinstance(max_budget_entries, bool)
            or not isinstance(max_budget_entries, int)
            or not 1 <= max_budget_entries <= 65_536
        ):
            raise BridgeLedgerError("Bridge budget entry bound is invalid.")
        if (
            isinstance(confirmation_retention_seconds, bool)
            or not isinstance(confirmation_retention_seconds, (int, float))
            or not math.isfinite(float(confirmation_retention_seconds))
            or not 1 <= confirmation_retention_seconds <= 365 * 24 * 60 * 60
        ):
            raise BridgeLedgerError("Bridge confirmation retention is invalid.")
        if (
            isinstance(max_confirmation_entries, bool)
            or not isinstance(max_confirmation_entries, int)
            or not 1 <= max_confirmation_entries <= 65_536
        ):
            raise BridgeLedgerError("Bridge confirmation entry bound is invalid.")
        normalized_lease_ttl = _validate_lease_ttl(budget_lease_ttl_seconds)
        if not callable(clock):
            raise BridgeLedgerError("Bridge ledger clock must be callable.")
        self.namespace = namespace
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds
        self.budget_retention_seconds = float(budget_retention_seconds)
        self.max_budget_entries = max_budget_entries
        self.confirmation_retention_seconds = float(confirmation_retention_seconds)
        self.max_confirmation_entries = max_confirmation_entries
        self.budget_lease_ttl_seconds = normalized_lease_ttl
        self._clock = clock
        self._thread_lock = threading.Lock()

    def effective_descriptor(self) -> dict[str, object]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "namespace": self.namespace,
            "path_sha256": _sha256_text(str(self.path)),
            "lock_timeout_seconds": self.lock_timeout_seconds,
            "stale_lock_seconds": self.stale_lock_seconds,
            "budget_retention_seconds": self.budget_retention_seconds,
            "max_budget_entries": self.max_budget_entries,
            "confirmation_retention_seconds": self.confirmation_retention_seconds,
            "max_confirmation_entries": self.max_confirmation_entries,
            "budget_lease_ttl_seconds": self.budget_lease_ttl_seconds,
        }

    def issue_confirmation(
        self,
        binding_sha256: str,
        *,
        ttl_seconds: float,
        now: float | None = None,
    ) -> ConfirmationTicket:
        _require_sha256(binding_sha256, "confirmation binding")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(float(ttl_seconds))
            or not 1 <= ttl_seconds <= 3600
        ):
            raise BridgeLedgerError(
                "Confirmation TTL must be between 1 and 3600 seconds."
            )
        token = f"confirm-v2-{secrets.token_urlsafe(32)}"
        if _TOKEN.fullmatch(token) is None:
            raise BridgeLedgerError("Generated confirmation token is malformed.")
        transaction_id = secrets.token_hex(16)
        token_sha256 = _sha256_text(token)
        with self._locked_state(
            supplied_time=now,
            label="Confirmation issue time",
        ) as (state, issued_at):
            expires_at = issued_at + ttl_seconds
            confirmations = state["confirmations"]
            assert isinstance(confirmations, dict)
            _prune_confirmations(
                confirmations,
                now=issued_at,
                retention_seconds=self.confirmation_retention_seconds,
                max_entries=self.max_confirmation_entries,
                required_capacity=1,
            )
            confirmations[token_sha256] = {
                "binding_sha256": binding_sha256,
                "transaction_id": transaction_id,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "consumed_at": None,
            }
        return ConfirmationTicket(
            token=token,
            transaction_id=transaction_id,
            expires_at=expires_at,
        )

    def consume_confirmation(
        self,
        token: str,
        binding_sha256: str,
        *,
        now: float | None = None,
    ) -> str:
        if _TOKEN.fullmatch(token) is None:
            raise BridgeConfirmationNotReadyError("Confirmation token is malformed.")
        _require_sha256(binding_sha256, "confirmation binding")
        token_sha256 = _sha256_text(token)
        with self._locked_state(
            supplied_time=now,
            label="Confirmation consume time",
        ) as (state, consumed_at):
            confirmations = state["confirmations"]
            assert isinstance(confirmations, dict)
            raw = confirmations.get(token_sha256)
            if not isinstance(raw, dict):
                raise BridgeConfirmationNotReadyError("Confirmation token is unknown.")
            _validate_confirmation(raw)
            if raw["binding_sha256"] != binding_sha256:
                raise BridgeLedgerError("Confirmation binding does not match the plan.")
            if raw["consumed_at"] is not None:
                raise BridgeConfirmationNotReadyError(
                    "Confirmation token was already consumed."
                )
            if consumed_at < float(raw["issued_at"]):
                raise BridgeLedgerError("Confirmation clock moved before issue time.")
            if consumed_at > float(raw["expires_at"]):
                raise BridgeConfirmationNotReadyError(
                    "Confirmation token has expired."
                )
            raw["consumed_at"] = consumed_at
            transaction_id = str(raw["transaction_id"])
        return transaction_id

    def reserve_budget(
        self,
        key_sha256: str,
        limit: int,
        *,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> PremiumBudgetLease | None:
        """Atomically reserve pending premium authority before ``Popen``.

        Pending authority counts toward ``limit``.  A caller must commit the
        returned lease immediately after successful process creation, or call
        :meth:`release_budget_after_popen_failure` only when process creation
        itself failed.
        """

        _require_sha256(key_sha256, "budget key")
        _validate_budget_limit(limit)
        if limit == 0:
            return None
        requested_lease_ttl = (
            self.budget_lease_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        lease_ttl = _validate_lease_ttl(requested_lease_ttl)
        token = f"budget-v2-{secrets.token_urlsafe(32)}"
        if _BUDGET_LEASE_TOKEN.fullmatch(token) is None:
            raise BridgeLedgerError("Generated premium budget lease is malformed.")
        token_sha256 = _sha256_text(token)
        with self._locked_state(
            supplied_time=now,
            label="Premium budget reserve time",
        ) as (state, reserved_at):
            expires_at = reserved_at + lease_ttl
            budgets = state["budgets"]
            assert isinstance(budgets, dict)
            _maintain_budgets(
                budgets,
                now=reserved_at,
                retention_seconds=self.budget_retention_seconds,
            )
            raw = budgets.get(key_sha256)
            if raw is None:
                if len(budgets) >= self.max_budget_entries:
                    raise BridgeLedgerError(
                        "Bridge budget retention bound is exhausted; refusing unsafe eviction."
                    )
                raw = {"used": 0, "updated_at": reserved_at, "pending": {}}
                budgets[key_sha256] = raw
            _validate_budget(raw)
            assert isinstance(raw, dict)
            _require_non_regressing_clock(raw, reserved_at)
            pending = raw["pending"]
            assert isinstance(pending, dict)
            if int(raw["used"]) + len(pending) >= limit:
                return None
            pending[token_sha256] = {
                "reserved_at": reserved_at,
                "expires_at": expires_at,
            }
            raw["updated_at"] = reserved_at
        return PremiumBudgetLease(
            token=token,
            key_sha256=key_sha256,
            reserved_at=reserved_at,
            expires_at=expires_at,
        )

    def commit_budget(
        self,
        lease: PremiumBudgetLease,
        *,
        now: float | None = None,
    ) -> None:
        """Commit a pending lease immediately after successful ``Popen``."""

        key_sha256, token_sha256 = _validate_lease(lease)
        with self._locked_state(
            supplied_time=now,
            label="Premium budget commit time",
        ) as (state, committed_at):
            budgets = state["budgets"]
            assert isinstance(budgets, dict)
            raw = budgets.get(key_sha256)
            if raw is None:
                raise BridgeLedgerError("Premium budget lease is unknown.")
            _validate_budget(raw)
            assert isinstance(raw, dict)
            _require_non_regressing_clock(raw, committed_at)
            promoted = _promote_expired_pending(raw, now=committed_at)
            pending = raw["pending"]
            assert isinstance(pending, dict)
            reservation = pending.pop(token_sha256, None)
            if reservation is None:
                missing = True
                expired = token_sha256 in promoted
            else:
                missing = False
                expired = False
                raw["used"] = int(raw["used"]) + 1
                raw["updated_at"] = committed_at
        if missing:
            status = "expired" if expired else "unknown"
            raise BridgeLedgerError(f"Premium budget lease is {status}.")

    def release_budget_after_popen_failure(
        self,
        lease: PremiumBudgetLease,
        *,
        now: float | None = None,
    ) -> None:
        """Release pending authority only because ``Popen`` did not create a process."""

        key_sha256, token_sha256 = _validate_lease(lease)
        with self._locked_state(
            supplied_time=now,
            label="Premium budget release time",
        ) as (state, released_at):
            budgets = state["budgets"]
            assert isinstance(budgets, dict)
            raw = budgets.get(key_sha256)
            if raw is None:
                raise BridgeLedgerError("Premium budget lease is unknown.")
            _validate_budget(raw)
            assert isinstance(raw, dict)
            _require_non_regressing_clock(raw, released_at)
            promoted = _promote_expired_pending(raw, now=released_at)
            pending = raw["pending"]
            assert isinstance(pending, dict)
            reservation = pending.pop(token_sha256, None)
            if reservation is None:
                missing = True
                expired = token_sha256 in promoted
            else:
                missing = False
                expired = False
                raw["updated_at"] = released_at
                if int(raw["used"]) == 0 and not pending:
                    budgets.pop(key_sha256, None)
        if missing:
            status = "expired" if expired else "unknown"
            raise BridgeLedgerError(f"Premium budget lease is {status}.")

    def consume_budget(
        self,
        key_sha256: str,
        limit: int,
        *,
        now: float | None = None,
    ) -> bool:
        _require_sha256(key_sha256, "budget key")
        _validate_budget_limit(limit)
        if limit == 0:
            return False
        with self._locked_state(
            supplied_time=now,
            label="Premium budget consume time",
        ) as (state, consumed_at):
            budgets = state["budgets"]
            assert isinstance(budgets, dict)
            _maintain_budgets(
                budgets,
                now=consumed_at,
                retention_seconds=self.budget_retention_seconds,
            )
            raw = budgets.get(key_sha256)
            used = 0
            if raw is not None:
                _validate_budget(raw)
                assert isinstance(raw, dict)
                _require_non_regressing_clock(raw, consumed_at)
                _promote_expired_pending(raw, now=consumed_at)
                used = int(raw["used"])
                pending = raw["pending"]
                assert isinstance(pending, dict)
            else:
                pending = {}
            if used + len(pending) >= limit:
                return False
            if raw is None and len(budgets) >= self.max_budget_entries:
                raise BridgeLedgerError(
                    "Bridge budget retention bound is exhausted; refusing unsafe eviction."
                )
            budgets[key_sha256] = {
                "used": used + 1,
                "updated_at": consumed_at,
                "pending": pending,
            }
        return True

    def _timestamp(self, supplied: float | None, label: str) -> float:
        try:
            value = self._clock() if supplied is None else supplied
            timestamp = float(value)
        except Exception:
            raise BridgeLedgerError(f"{label} is unavailable.") from None
        if not math.isfinite(timestamp) or timestamp < 0:
            raise BridgeLedgerError(f"{label} is invalid.")
        return timestamp

    @contextmanager
    def _locked_state(
        self,
        *,
        supplied_time: float | None,
        label: str,
    ) -> Iterator[tuple[dict[str, object], float]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_time = self._timestamp(supplied_time, label)
        with self._thread_lock, self._file_lock(now=lock_time):
            observed_at = (
                lock_time if supplied_time is not None else self._timestamp(None, label)
            )
            state = self._read(migration_time=observed_at)
            try:
                yield state, observed_at
            except Exception:
                raise
            else:
                self._write(state)

    @contextmanager
    def _file_lock(self, *, now: float) -> Iterator[None]:
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                lock.mkdir(mode=0o700)
                break
            except FileExistsError:
                if _recover_stale_lock(
                    lock,
                    self.stale_lock_seconds,
                    now=now,
                ):
                    continue
                if time.monotonic() >= deadline:
                    raise BridgeLedgerError("Bridge ledger lock is busy.")
                time.sleep(0.02)
            except OSError as exc:
                # Windows may report a contended directory creation as
                # PermissionError instead of FileExistsError.  Keep the
                # acquisition fail-closed, but give that transient collision
                # the same bounded retry window as ordinary contention.
                retryable_contention = _IS_WINDOWS and isinstance(
                    exc,
                    PermissionError,
                )
                if not retryable_contention:
                    try:
                        retryable_contention = lock.is_dir()
                    except OSError:
                        retryable_contention = False
                if retryable_contention and time.monotonic() < deadline:
                    time.sleep(0.02)
                    continue
                raise BridgeLedgerError(
                    "Bridge ledger lock could not be acquired."
                ) from exc
        try:
            try:
                (lock / "owner.json").write_text(
                    json.dumps({"pid": os.getpid(), "created_at": now}),
                    encoding="utf-8",
                )
            except OSError:
                raise BridgeLedgerError(
                    "Bridge ledger lock owner cannot be persisted safely."
                ) from None
            yield
        finally:
            try:
                (lock / "owner.json").unlink(missing_ok=True)
                lock.rmdir()
            except OSError:
                pass

    def _read(self, *, migration_time: float) -> dict[str, object]:
        _reject_symlink_path(self._declared_path)
        if not self.path.exists():
            return {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "namespace": self.namespace,
                "budgets": {},
                "confirmations": {},
            }
        try:
            if self.path.stat().st_size > 4 * 1024 * 1024:
                raise BridgeLedgerError("Bridge ledger exceeds its size bound.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except BridgeLedgerError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise BridgeLedgerError("Bridge ledger cannot be read safely.") from exc
        if isinstance(raw, dict) and raw.get("schema_version") in (
            _LEGACY_SCHEMA_VERSIONS
        ):
            raw = _migrate_legacy_state(
                raw,
                self.namespace,
                migrated_at=migration_time,
            )
        _validate_state(raw, self.namespace)
        budgets = raw["budgets"]
        confirmations = raw["confirmations"]
        assert isinstance(budgets, dict)
        assert isinstance(confirmations, dict)
        if len(budgets) > 65_536:
            raise BridgeLedgerError("Bridge budget absolute bound is exceeded.")
        if len(confirmations) > 65_536:
            raise BridgeLedgerError("Bridge confirmation absolute bound is exceeded.")
        return raw

    def _write(self, state: dict[str, object]) -> None:
        _validate_state(state, self.namespace)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        temp = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            temp.chmod(0o600)
            os.replace(temp, self.path)
            try:
                directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                if os.name != "nt":
                    raise
        except OSError as exc:
            raise BridgeLedgerError(
                "Bridge ledger cannot be persisted safely."
            ) from exc
        finally:
            temp.unlink(missing_ok=True)


def budget_key(
    *,
    namespace: str,
    task_fingerprint: str,
    config_sha256: str,
    workspace_fingerprint: str,
) -> str:
    for label, value in (
        ("task fingerprint", task_fingerprint),
        ("config digest", config_sha256),
        ("workspace fingerprint", workspace_fingerprint),
    ):
        _require_sha256(value, label)
    return _sha256_text(
        json.dumps(
            {
                "namespace": namespace,
                "task_fingerprint": task_fingerprint,
                "config_sha256": config_sha256,
                "workspace_fingerprint": workspace_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _validate_state(raw: object, namespace: str) -> None:
    if not isinstance(raw, dict):
        raise BridgeLedgerError("Bridge ledger must be an object.")
    if set(raw) != {"schema_version", "namespace", "budgets", "confirmations"}:
        raise BridgeLedgerError("Bridge ledger keys are invalid.")
    if raw["schema_version"] != LEDGER_SCHEMA_VERSION:
        raise BridgeLedgerError("Bridge ledger schema is unsupported.")
    if raw["namespace"] != namespace:
        raise BridgeLedgerError("Bridge ledger namespace does not match configuration.")
    budgets = raw["budgets"]
    confirmations = raw["confirmations"]
    if not isinstance(budgets, dict) or not isinstance(confirmations, dict):
        raise BridgeLedgerError("Bridge ledger collections are malformed.")
    for key, value in budgets.items():
        _require_sha256(str(key), "stored budget key")
        _validate_budget(value)
    for key, value in confirmations.items():
        _require_sha256(str(key), "stored confirmation key")
        _validate_confirmation(value)


def _validate_confirmation(raw: object) -> None:
    if not isinstance(raw, dict) or set(raw) != {
        "binding_sha256",
        "transaction_id",
        "issued_at",
        "expires_at",
        "consumed_at",
    }:
        raise BridgeLedgerError("Stored confirmation is malformed.")
    _require_sha256(str(raw["binding_sha256"]), "stored confirmation binding")
    if re.fullmatch(r"[a-f0-9]{32}", str(raw["transaction_id"])) is None:
        raise BridgeLedgerError("Stored transaction id is malformed.")
    for label in ("issued_at", "expires_at"):
        if (
            isinstance(raw[label], bool)
            or not isinstance(raw[label], (int, float))
            or not math.isfinite(float(raw[label]))
        ):
            raise BridgeLedgerError(f"Stored confirmation {label} is malformed.")
    if float(raw["expires_at"]) <= float(raw["issued_at"]):
        raise BridgeLedgerError("Stored confirmation expiry is malformed.")
    consumed = raw["consumed_at"]
    if consumed is not None and (
        isinstance(consumed, bool) or not isinstance(consumed, (int, float))
    ):
        raise BridgeLedgerError("Stored confirmation consumed_at is malformed.")
    if consumed is not None and (
        not math.isfinite(float(consumed))
        or float(consumed) < float(raw["issued_at"])
        or float(consumed) > float(raw["expires_at"])
    ):
        raise BridgeLedgerError(
            "Stored confirmation consumed_at is outside its window."
        )


def _validate_budget(raw: object) -> None:
    if not isinstance(raw, dict) or set(raw) != {
        "used",
        "updated_at",
        "pending",
    }:
        raise BridgeLedgerError("Stored premium budget is malformed.")
    used = raw["used"]
    updated_at = raw["updated_at"]
    if isinstance(used, bool) or not isinstance(used, int) or used < 0:
        raise BridgeLedgerError("Stored premium budget usage is malformed.")
    if (
        isinstance(updated_at, bool)
        or not isinstance(updated_at, (int, float))
        or not math.isfinite(float(updated_at))
        or float(updated_at) < 0
    ):
        raise BridgeLedgerError("Stored premium budget timestamp is malformed.")
    pending = raw["pending"]
    if not isinstance(pending, dict):
        raise BridgeLedgerError("Stored premium budget pending leases are malformed.")
    for token_sha256, reservation in pending.items():
        _require_sha256(str(token_sha256), "stored premium budget lease")
        _validate_pending_reservation(reservation)
        assert isinstance(reservation, dict)
        if float(reservation["reserved_at"]) > float(updated_at):
            raise BridgeLedgerError(
                "Stored premium budget lease is newer than its budget clock."
            )


def _validate_pending_reservation(raw: object) -> None:
    if not isinstance(raw, dict) or set(raw) != {"reserved_at", "expires_at"}:
        raise BridgeLedgerError("Stored premium budget lease is malformed.")
    for label in ("reserved_at", "expires_at"):
        value = raw[label]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise BridgeLedgerError(
                f"Stored premium budget lease {label} is malformed."
            )
    if float(raw["expires_at"]) <= float(raw["reserved_at"]):
        raise BridgeLedgerError("Stored premium budget lease expiry is malformed.")


def _migrate_legacy_state(
    raw: dict[str, object],
    namespace: str,
    *,
    migrated_at: float,
) -> dict[str, object]:
    if set(raw) != {"schema_version", "namespace", "budgets", "confirmations"}:
        raise BridgeLedgerError("Legacy bridge ledger keys are invalid.")
    if raw.get("namespace") != namespace:
        raise BridgeLedgerError("Legacy bridge ledger namespace does not match.")
    budgets = raw.get("budgets")
    confirmations = raw.get("confirmations")
    if not isinstance(budgets, dict) or not isinstance(confirmations, dict):
        raise BridgeLedgerError("Legacy bridge ledger collections are malformed.")
    migrated_budgets: dict[str, object] = {}
    if raw.get("schema_version") == "2.0":
        for key, used in budgets.items():
            _require_sha256(str(key), "legacy stored budget key")
            if isinstance(used, bool) or not isinstance(used, int) or used < 0:
                raise BridgeLedgerError("Legacy premium budget is malformed.")
            migrated_budgets[str(key)] = {
                "used": used,
                # Version 2.0 had no clock. Bind it to the caller's injected
                # observation so deterministic migrations cannot move time.
                "updated_at": migrated_at,
                "pending": {},
            }
    elif raw.get("schema_version") == "2.1":
        for key, budget in budgets.items():
            _require_sha256(str(key), "legacy stored budget key")
            if not isinstance(budget, dict) or set(budget) != {
                "used",
                "updated_at",
            }:
                raise BridgeLedgerError("Legacy premium budget is malformed.")
            migrated = dict(budget)
            migrated["pending"] = {}
            _validate_budget(migrated)
            migrated_budgets[str(key)] = migrated
    else:  # pragma: no cover - guarded by the caller.
        raise BridgeLedgerError("Legacy bridge ledger schema is unsupported.")
    for key, value in confirmations.items():
        _require_sha256(str(key), "legacy stored confirmation key")
        _validate_confirmation(value)
    migrated: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "namespace": namespace,
        "budgets": migrated_budgets,
        "confirmations": dict(confirmations),
    }
    _validate_state(migrated, namespace)
    return migrated


def _maintain_budgets(
    budgets: dict[str, object],
    *,
    now: float,
    retention_seconds: float,
) -> None:
    cutoff = now - retention_seconds
    expired: list[tuple[float, str]] = []
    for key, raw in budgets.items():
        _validate_budget(raw)
        assert isinstance(raw, dict)
        if now < float(raw["updated_at"]):
            continue
        _promote_expired_pending(raw, now=now)
        updated_at = float(raw["updated_at"])
        pending = raw["pending"]
        assert isinstance(pending, dict)
        if not pending and updated_at < cutoff:
            expired.append((updated_at, key))
    for _, key in sorted(expired):
        budgets.pop(key, None)


def _prune_confirmations(
    confirmations: dict[str, object],
    *,
    now: float,
    retention_seconds: float,
    max_entries: int,
    required_capacity: int,
) -> None:
    terminal: list[tuple[float, str]] = []
    retained_terminal: list[tuple[float, str]] = []
    cutoff = now - retention_seconds
    for key, raw in confirmations.items():
        _require_sha256(str(key), "stored confirmation key")
        _validate_confirmation(raw)
        assert isinstance(raw, dict)
        consumed_at = raw["consumed_at"]
        terminal_at = (
            float(consumed_at) if consumed_at is not None else float(raw["expires_at"])
        )
        is_terminal = consumed_at is not None or float(raw["expires_at"]) < now
        if is_terminal and terminal_at < cutoff:
            terminal.append((terminal_at, key))
        elif is_terminal:
            retained_terminal.append((terminal_at, key))
    for _, key in sorted(terminal):
        confirmations.pop(key, None)
    overflow = len(confirmations) + required_capacity - max_entries
    if overflow > 0:
        for _, key in sorted(retained_terminal)[:overflow]:
            confirmations.pop(key, None)
    if len(confirmations) + required_capacity > max_entries:
        raise BridgeLedgerError("Bridge confirmation ledger reached its safe bound.")


def _promote_expired_pending(
    budget: dict[str, object],
    *,
    now: float,
) -> set[str]:
    _validate_budget(budget)
    _require_non_regressing_clock(budget, now)
    pending = budget["pending"]
    assert isinstance(pending, dict)
    expired = {
        token_sha256
        for token_sha256, reservation in pending.items()
        if isinstance(reservation, dict) and now >= float(reservation["expires_at"])
    }
    if expired:
        for token_sha256 in expired:
            pending.pop(token_sha256, None)
        budget["used"] = int(budget["used"]) + len(expired)
        budget["updated_at"] = now
    return expired


def _require_non_regressing_clock(budget: dict[str, object], now: float) -> None:
    if now < float(budget["updated_at"]):
        raise BridgeLedgerError(
            "Premium budget clock moved before its last observation."
        )


def _validate_budget_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 1:
        raise BridgeLedgerError("Premium call limit must be 0 or 1.")


def _validate_lease_ttl(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 1 <= float(value) <= 3600
    ):
        raise BridgeLedgerError(
            "Premium budget lease TTL must be between 1 and 3600 seconds."
        )
    return float(value)


def _validate_lease(lease: PremiumBudgetLease) -> tuple[str, str]:
    if not isinstance(lease, PremiumBudgetLease):
        raise BridgeLedgerError("Premium budget lease type is invalid.")
    if _BUDGET_LEASE_TOKEN.fullmatch(lease.token) is None:
        raise BridgeLedgerError("Premium budget lease token is malformed.")
    _require_sha256(lease.key_sha256, "premium budget lease key")
    for label, value in (
        ("reserved_at", lease.reserved_at),
        ("expires_at", lease.expires_at),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise BridgeLedgerError(f"Premium budget lease {label} is malformed.")
    if lease.expires_at <= lease.reserved_at:
        raise BridgeLedgerError("Premium budget lease expiry is malformed.")
    return lease.key_sha256, _sha256_text(lease.token)


def _recover_stale_lock(
    lock: Path,
    stale_seconds: float,
    *,
    now: float,
) -> bool:
    try:
        age = now - lock.stat().st_mtime
    except OSError:
        return False
    if age <= stale_seconds:
        return False
    pid = -1
    try:
        raw = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
        pid = int(raw.get("pid", -1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    if process_is_alive(pid):
        return False
    stale = lock.with_name(f"{lock.name}.stale-{uuid4().hex}")
    try:
        os.replace(lock, stale)
        shutil.rmtree(stale)
    except OSError:
        return False
    return True


def _require_sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise BridgeLedgerError(f"{label} must be a lowercase SHA-256 digest.")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_symlink_path(path: Path) -> None:
    # Reject the two locations the bridge mutates.  Walking all the way to the
    # filesystem root would make otherwise-safe paths unusable on systems such
    # as macOS, where ``/var`` is intentionally a platform-managed symlink.
    if path.is_symlink() or path.parent.is_symlink():
        raise BridgeLedgerError("Bridge ledger paths cannot contain symbolic links.")
    if path.exists() and not path.is_file():
        raise BridgeLedgerError("Bridge ledger path must be a regular file.")
