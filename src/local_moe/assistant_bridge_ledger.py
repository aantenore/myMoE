from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
from typing import Iterator
from uuid import uuid4


LEDGER_SCHEMA_VERSION = "2.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^confirm-v2-[A-Za-z0-9_-]{43}$")


class BridgeLedgerError(ValueError):
    """Raised when authorization or budget state cannot be proven safely."""


@dataclass(frozen=True)
class ConfirmationTicket:
    token: str
    transaction_id: str
    expires_at: float

    def metadata_payload(self) -> dict[str, object]:
        return {
            "token_sha256": _sha256_text(self.token),
            "transaction_id": self.transaction_id,
            "expires_at": self.expires_at,
            "one_shot": True,
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
        self.namespace = namespace
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds
        self._thread_lock = threading.Lock()

    def effective_descriptor(self) -> dict[str, object]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "namespace": self.namespace,
            "path_sha256": _sha256_text(str(self.path)),
            "lock_timeout_seconds": self.lock_timeout_seconds,
            "stale_lock_seconds": self.stale_lock_seconds,
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
            raise BridgeLedgerError("Confirmation TTL must be between 1 and 3600 seconds.")
        issued_at = time.time() if now is None else float(now)
        if not math.isfinite(issued_at):
            raise BridgeLedgerError("Confirmation issue time is invalid.")
        expires_at = issued_at + ttl_seconds
        token = f"confirm-v2-{secrets.token_urlsafe(32)}"
        if _TOKEN.fullmatch(token) is None:
            raise BridgeLedgerError("Generated confirmation token is malformed.")
        transaction_id = secrets.token_hex(16)
        token_sha256 = _sha256_text(token)
        with self._locked_state() as state:
            confirmations = state["confirmations"]
            assert isinstance(confirmations, dict)
            confirmations[token_sha256] = {
                "binding_sha256": binding_sha256,
                "transaction_id": transaction_id,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "consumed_at": None,
            }
            _prune_confirmations(confirmations, now=issued_at)
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
            raise BridgeLedgerError("Confirmation token is malformed.")
        _require_sha256(binding_sha256, "confirmation binding")
        consumed_at = time.time() if now is None else float(now)
        if not math.isfinite(consumed_at):
            raise BridgeLedgerError("Confirmation consume time is invalid.")
        token_sha256 = _sha256_text(token)
        with self._locked_state() as state:
            confirmations = state["confirmations"]
            assert isinstance(confirmations, dict)
            raw = confirmations.get(token_sha256)
            if not isinstance(raw, dict):
                raise BridgeLedgerError("Confirmation token is unknown.")
            _validate_confirmation(raw)
            if raw["binding_sha256"] != binding_sha256:
                raise BridgeLedgerError("Confirmation binding does not match the plan.")
            if raw["consumed_at"] is not None:
                raise BridgeLedgerError("Confirmation token was already consumed.")
            if consumed_at < float(raw["issued_at"]):
                raise BridgeLedgerError("Confirmation clock moved before issue time.")
            if consumed_at > float(raw["expires_at"]):
                raise BridgeLedgerError("Confirmation token has expired.")
            raw["consumed_at"] = consumed_at
            transaction_id = str(raw["transaction_id"])
        return transaction_id

    def consume_budget(self, key_sha256: str, limit: int) -> bool:
        _require_sha256(key_sha256, "budget key")
        if isinstance(limit, bool) or not 0 <= limit <= 1:
            raise BridgeLedgerError("Premium call limit must be 0 or 1.")
        if limit == 0:
            return False
        with self._locked_state() as state:
            budgets = state["budgets"]
            assert isinstance(budgets, dict)
            used = budgets.get(key_sha256, 0)
            if isinstance(used, bool) or not isinstance(used, int) or used < 0:
                raise BridgeLedgerError("Bridge budget state is malformed.")
            if used >= limit:
                return False
            budgets[key_sha256] = used + 1
        return True

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, object]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, self._file_lock():
            state = self._read()
            try:
                yield state
            except Exception:
                raise
            else:
                self._write(state)

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                lock.mkdir(mode=0o700)
                break
            except FileExistsError:
                if _recover_stale_lock(lock, self.stale_lock_seconds):
                    continue
                if time.monotonic() >= deadline:
                    raise BridgeLedgerError("Bridge ledger lock is busy.")
                time.sleep(0.02)
            except OSError as exc:
                raise BridgeLedgerError("Bridge ledger lock could not be acquired.") from exc
        try:
            (lock / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created_at": time.time()}),
                encoding="utf-8",
            )
            yield
        finally:
            try:
                (lock / "owner.json").unlink(missing_ok=True)
                lock.rmdir()
            except OSError:
                pass

    def _read(self) -> dict[str, object]:
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
        _validate_state(raw, self.namespace)
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
            raise BridgeLedgerError("Bridge ledger cannot be persisted safely.") from exc
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
    for key, used in budgets.items():
        _require_sha256(str(key), "stored budget key")
        if isinstance(used, bool) or not isinstance(used, int) or used < 0:
            raise BridgeLedgerError("Stored premium budget is malformed.")
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
        raise BridgeLedgerError("Stored confirmation consumed_at is outside its window.")


def _prune_confirmations(confirmations: dict[str, object], *, now: float) -> None:
    if len(confirmations) <= 4096:
        return
    removable = [
        key
        for key, raw in confirmations.items()
        if isinstance(raw, dict)
        and (
            float(raw.get("expires_at", 0)) < now - 86_400
            or raw.get("consumed_at") is not None
        )
    ]
    for key in removable[: len(confirmations) - 4096]:
        confirmations.pop(key, None)
    if len(confirmations) > 4096:
        raise BridgeLedgerError("Bridge confirmation ledger reached its safe bound.")


def _recover_stale_lock(lock: Path, stale_seconds: float) -> bool:
    try:
        age = time.time() - lock.stat().st_mtime
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
    if _pid_is_alive(pid):
        return False
    stale = lock.with_name(f"{lock.name}.stale-{uuid4().hex}")
    try:
        os.replace(lock, stale)
        shutil.rmtree(stale)
    except OSError:
        return False
    return True


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
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
