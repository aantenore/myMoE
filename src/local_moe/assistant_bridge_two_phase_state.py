from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from .assistant_bridge_integrity import sha256_bytes


TWO_PHASE_CONFIG_SCHEMA_VERSION = "1.0"
_MAX_CONFIG_BYTES = 1024 * 1024
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class TwoPhaseConfigError(ValueError):
    """Raised when lifecycle state or public trust configuration is invalid."""


@dataclass(frozen=True)
class TwoPhaseStateConfig:
    database_path: Path = field(repr=False)
    cas_path: Path = field(repr=False)
    transaction_state_dir: Path = field(repr=False)
    candidate_ttl_seconds: float = 24 * 60 * 60
    confirmation_ttl_seconds: float = 300
    transaction_lock_ttl_seconds: float = 120
    sqlite_timeout_seconds: float = 5

    def __post_init__(self) -> None:
        for value, label in (
            (self.database_path, "workflow database path"),
            (self.cas_path, "candidate store path"),
            (self.transaction_state_dir, "transaction state path"),
        ):
            if not isinstance(value, Path) or not str(value):
                raise TwoPhaseConfigError(f"{label} is required.")
        for value, minimum, maximum, label in (
            (self.candidate_ttl_seconds, 1, 7 * 24 * 60 * 60, "candidate TTL"),
            (self.confirmation_ttl_seconds, 1, 3600, "confirmation TTL"),
            (
                self.transaction_lock_ttl_seconds,
                1,
                86_400,
                "transaction lock TTL",
            ),
            (self.sqlite_timeout_seconds, 0.1, 60, "SQLite timeout"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise TwoPhaseConfigError(f"{label} is outside safe bounds.")

    def payload(self) -> dict[str, object]:
        return {
            "databasePath": str(self.database_path),
            "casPath": str(self.cas_path),
            "transactionStateDir": str(self.transaction_state_dir),
            "candidateTtlSeconds": float(self.candidate_ttl_seconds),
            "confirmationTtlSeconds": float(self.confirmation_ttl_seconds),
            "transactionLockTtlSeconds": float(self.transaction_lock_ttl_seconds),
            "sqliteTimeoutSeconds": float(self.sqlite_timeout_seconds),
        }


def load_two_phase_state_config(path: str | Path) -> TwoPhaseStateConfig:
    """Load durable state settings without importing or opening trust material."""

    source, raw, _ = _load_config_document(path)
    return _parse_state(raw, config_root=source.parent)


def read_bounded_regular_file(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read one stable regular file without following a final-path link."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise TwoPhaseConfigError("File size bound is invalid.")
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    try:
        before = target.lstat()
    except OSError as exc:
        raise TwoPhaseConfigError(f"{label} is unavailable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TwoPhaseConfigError(f"{label} must be a regular non-link file.")
    if not 0 < before.st_size <= max_bytes:
        raise TwoPhaseConfigError(f"{label} size is outside safe bounds.")
    descriptor = -1
    try:
        descriptor = os.open(target, _READ_FLAGS)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
        ):
            raise TwoPhaseConfigError(f"{label} identity changed before reading.")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise TwoPhaseConfigError(f"{label} is truncated.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TwoPhaseConfigError(f"{label} exceeds its size binding.")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise TwoPhaseConfigError(f"{label} could not be read.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise TwoPhaseConfigError(f"{label} changed while reading.")
    return b"".join(chunks)


def _load_config_document(
    path: str | Path,
) -> tuple[Path, dict[str, Any], str]:
    source = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    value = read_bounded_regular_file(
        source,
        max_bytes=_MAX_CONFIG_BYTES,
        label="two-phase configuration",
    )
    try:
        raw = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoPhaseConfigError("Two-phase configuration is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise TwoPhaseConfigError("Two-phase configuration must be an object.")
    _reject_unknown(
        "two-phase configuration",
        raw,
        {"schema_version", "state", "trust"},
    )
    if raw.get("schema_version") != TWO_PHASE_CONFIG_SCHEMA_VERSION:
        raise TwoPhaseConfigError("Two-phase configuration schema is unsupported.")
    return source, raw, sha256_bytes(value)


def _parse_state(raw: Mapping[str, Any], *, config_root: Path) -> TwoPhaseStateConfig:
    state = _object(raw.get("state"), "state")
    _reject_unknown(
        "state",
        state,
        {
            "candidate_ttl_seconds",
            "cas_path",
            "confirmation_ttl_seconds",
            "database_path",
            "sqlite_timeout_seconds",
            "transaction_lock_ttl_seconds",
            "transaction_state_dir",
        },
    )
    return TwoPhaseStateConfig(
        database_path=_configured_path(
            state.get("database_path"), "state.database_path", config_root
        ),
        cas_path=_configured_path(state.get("cas_path"), "state.cas_path", config_root),
        transaction_state_dir=_configured_path(
            state.get("transaction_state_dir"),
            "state.transaction_state_dir",
            config_root,
        ),
        candidate_ttl_seconds=_number(
            state.get("candidate_ttl_seconds", 24 * 60 * 60),
            "state.candidate_ttl_seconds",
        ),
        confirmation_ttl_seconds=_number(
            state.get("confirmation_ttl_seconds", 300),
            "state.confirmation_ttl_seconds",
        ),
        transaction_lock_ttl_seconds=_number(
            state.get("transaction_lock_ttl_seconds", 120),
            "state.transaction_lock_ttl_seconds",
        ),
        sqlite_timeout_seconds=_number(
            state.get("sqlite_timeout_seconds", 5),
            "state.sqlite_timeout_seconds",
        ),
    )


def _configured_path(value: object, label: str, root: Path) -> Path:
    text = _string(value, label)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TwoPhaseConfigError(f"{label} must be an object.")
    return dict(value)


def _reject_unknown(label: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise TwoPhaseConfigError(
            f"{label} contains unknown fields: {', '.join(unknown)}."
        )


def _string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise TwoPhaseConfigError(f"{label} must be a non-empty string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TwoPhaseConfigError(f"{label} must be an integer.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TwoPhaseConfigError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise TwoPhaseConfigError(f"{label} must be finite.")
    return result
