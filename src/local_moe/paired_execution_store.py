from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time
from typing import Any, Mapping

from filelock import FileLock, Timeout

from .paired_execution_contracts import (
    PairedOutcomeBinding,
    PairedRunCheckpoint,
    PairedRunClaim,
    PairedRunRoot,
    PairedRunSlot,
    validate_binding,
)
from .verified_routing_contracts import (
    VerifiedRoutingError,
    canonical_json,
    require_finite_number,
    sha256_json,
)


_LOCK_TIMEOUT_SECONDS = 10.0
_MAX_ROOT_BYTES = 128 * 1024
_MAX_EVENTS_BYTES = 1024 * 1024
_MAX_EVENT_BYTES = 256 * 1024
_EVENT_FIELDS = {"event", "payload"}
_EVENT_FILE = re.compile(
    r"^(?P<ordinal>[0-9]{6})-(?P<event>claim|checkpoint)-"
    r"(?P<digest>[0-9a-f]{64})\.json$"
)
_STATES = {
    "missing",
    "ready",
    "running",
    "indeterminate",
    "partial",
    "complete",
}


class PairedRunIndeterminateError(VerifiedRoutingError):
    """Raised when an invocation may have happened but was not checkpointed."""


@dataclass(frozen=True)
class PairedRunStatus:
    """Immutable read model for a paired execution journal."""

    state: str
    root: PairedRunRoot | None
    claims: tuple[PairedRunClaim, ...] = ()
    checkpoints: tuple[PairedRunCheckpoint, ...] = ()
    next_slot: PairedRunSlot | None = None
    current_claim: PairedRunClaim | None = None

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise VerifiedRoutingError("Paired run status is unsupported.")
        if self.state == "missing":
            if any(
                (
                    self.root is not None,
                    bool(self.claims),
                    bool(self.checkpoints),
                    self.next_slot is not None,
                    self.current_claim is not None,
                )
            ):
                raise VerifiedRoutingError("Missing paired status must be empty.")
            return
        if self.root is None:
            raise VerifiedRoutingError("Prepared paired status requires a root.")

    def payload(self) -> dict[str, object]:
        return {
            "state": self.state,
            "root": None if self.root is None else self.root.payload(),
            "claims": [claim.payload() for claim in self.claims],
            "checkpoints": [item.payload() for item in self.checkpoints],
            "next_slot": (
                None if self.next_slot is None else self.next_slot.payload()
            ),
            "current_claim": (
                None
                if self.current_claim is None
                else self.current_claim.payload()
            ),
        }


@dataclass(frozen=True)
class _JournalState:
    root: PairedRunRoot | None
    claims: tuple[PairedRunClaim, ...]
    checkpoints: tuple[PairedRunCheckpoint, ...]
    open_claim: PairedRunClaim | None


@dataclass(frozen=True)
class _ClaimAuthority:
    pid: int
    nonce: str


@dataclass(frozen=True)
class _EventArtifact:
    ordinal: int
    event: str
    digest: str
    path: Path
    metadata: os.stat_result


class PairedExecutionStore:
    """Fail-closed append-only filesystem journal for one paired run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        lock_timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
    ) -> None:
        declared = Path(run_dir).expanduser()
        self.run_dir = Path(os.path.abspath(declared))
        self.root_path = self.run_dir / "run.json"
        self.events_path = self.run_dir / "events"
        self._legacy_events_path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / "run.lock"
        timeout = require_finite_number(
            lock_timeout_seconds,
            "paired store lock_timeout_seconds",
            minimum=0.0,
        )
        if timeout == 0.0:
            raise VerifiedRoutingError(
                "paired store lock_timeout_seconds must be positive."
            )
        self.lock_timeout_seconds = timeout
        self._thread_lock = threading.Lock()
        self._process_lock = FileLock(
            str(self.lock_path),
            timeout=timeout,
            mode=0o600,
        )
        self._owner_pid = os.getpid()
        self._owner_nonce = secrets.token_hex(32)
        self._active_claims: dict[str, _ClaimAuthority] = {}

    def prepare(self, root: PairedRunRoot) -> bool:
        """Install a root once; an exact repeat is the only idempotent case."""

        if not isinstance(root, PairedRunRoot):
            raise TypeError("root must be a PairedRunRoot.")
        with self._locked(create=True):
            state = self._read_unlocked()
            if state.root is not None:
                if state.root != root:
                    raise VerifiedRoutingError(
                        "Paired run directory is bound to another root."
                    )
                return False
            if state.claims or state.checkpoints or state.open_claim is not None:
                raise VerifiedRoutingError(
                    "Paired run events exist without a prepared root."
                )
            encoded = (canonical_json(root.payload()) + "\n").encode("utf-8")
            if not _atomic_install(self.root_path, encoded):
                installed = self._read_unlocked().root
                if installed != root:
                    raise VerifiedRoutingError(
                        "Paired run root creation lost a conflicting race."
                    )
                return False
            return True

    def claim(self, slot: PairedRunSlot | str) -> PairedRunClaim:
        """Durably claim the next slot before any provider invocation."""

        self._refresh_process_identity_after_fork()
        with self._locked(create=False):
            state = self._require_prepared(self._read_unlocked())
            if state.open_claim is not None:
                if self._owns_claim(state.open_claim.claim_sha256):
                    raise VerifiedRoutingError(
                        "Paired run slot is already running in this owner."
                    )
                raise PairedRunIndeterminateError(
                    "Paired run has an uncheckpointed claim; retry is forbidden."
                )
            if len(state.checkpoints) >= len(state.root.slots):
                raise VerifiedRoutingError("Paired run is already complete.")
            expected = state.root.slots[len(state.checkpoints)]
            requested = self._resolve_slot(state.root, slot)
            if requested != expected:
                raise VerifiedRoutingError(
                    "Paired run slot claim is out of declared order."
                )
            claim = PairedRunClaim.build(state.root, expected)
            self._append_event_unlocked("claim", claim.payload())
            self._active_claims[claim.claim_sha256] = _ClaimAuthority(
                pid=self._owner_pid,
                nonce=self._owner_nonce,
            )
            return claim

    def binding_for(self, claim: PairedRunClaim) -> PairedOutcomeBinding:
        """Build the exact outcome lineage for the actively owned claim."""

        if not isinstance(claim, PairedRunClaim):
            raise TypeError("claim must be a PairedRunClaim.")
        self._assert_process_identity()
        with self._locked(create=False):
            state = self._require_prepared(self._read_unlocked())
            if state.open_claim != claim:
                raise VerifiedRoutingError("Paired claim is not the open slot.")
            if not self._owns_claim(claim.claim_sha256):
                raise PairedRunIndeterminateError(
                    "Only the process that wrote a claim may bind its outcome."
                )
            previous = (
                None
                if not state.checkpoints
                else state.checkpoints[-1].outcome_record_id
            )
            return PairedOutcomeBinding.build(
                state.root,
                claim,
                previous_record_id=previous,
            )

    def complete(
        self,
        binding: PairedOutcomeBinding,
        *,
        outcome_record_id: str,
        route_receipt_id: str,
        route_receipt_sha256: str,
        evidence_sha256: str,
    ) -> PairedRunCheckpoint:
        """Checkpoint one exact claimed outcome; conflicting repeats fail."""

        if not isinstance(binding, PairedOutcomeBinding):
            raise TypeError("binding must be a PairedOutcomeBinding.")
        self._assert_process_identity()
        checkpoint = PairedRunCheckpoint.build(
            binding,
            outcome_record_id=outcome_record_id,
            route_receipt_id=route_receipt_id,
            route_receipt_sha256=route_receipt_sha256,
            evidence_sha256=evidence_sha256,
        )
        with self._locked(create=False):
            state = self._require_prepared(self._read_unlocked())
            for existing in state.checkpoints:
                if existing.binding.claim_sha256 != binding.claim_sha256:
                    continue
                if existing == checkpoint:
                    return existing
                raise VerifiedRoutingError(
                    "Paired claim already has a different checkpoint."
                )
            if state.open_claim is None:
                raise VerifiedRoutingError("Paired run has no open claim.")
            if state.open_claim.claim_sha256 != binding.claim_sha256:
                raise VerifiedRoutingError(
                    "Paired checkpoint does not belong to the open claim."
                )
            validate_binding(state.root, state.open_claim, binding)
            previous = (
                None
                if not state.checkpoints
                else state.checkpoints[-1].outcome_record_id
            )
            if binding.previous_record_id != previous:
                raise VerifiedRoutingError(
                    "Paired checkpoint breaks the sequential outcome chain."
                )
            if not self._owns_claim(binding.claim_sha256):
                raise PairedRunIndeterminateError(
                    "An unowned or recovered claim cannot be checkpointed."
                )
            self._append_event_unlocked("checkpoint", checkpoint.payload())
            self._active_claims.pop(binding.claim_sha256, None)
            return checkpoint

    def status(self) -> PairedRunStatus:
        """Return missing/ready/running/indeterminate/partial/complete."""

        kind = _lstat_optional(self.run_dir)
        if kind is None:
            return PairedRunStatus(state="missing", root=None)
        _validate_run_directory(self.run_dir, kind)
        lock_metadata = _lstat_optional(self.lock_path)
        if lock_metadata is None:
            if _list_run_directory(self.run_dir):
                raise VerifiedRoutingError(
                    "Paired store is corrupt: journal artifacts exist without "
                    "run.lock."
                )
            return PairedRunStatus(state="missing", root=None)
        _validate_regular_file(
            self.lock_path,
            "paired store lock",
            lock_metadata,
        )
        with self._read_locked():
            state = self._read_unlocked()
            if state.root is None:
                return PairedRunStatus(state="missing", root=None)
            if state.open_claim is not None:
                status = (
                    "running"
                    if self._owns_claim(state.open_claim.claim_sha256)
                    else "indeterminate"
                )
                return PairedRunStatus(
                    state=status,
                    root=state.root,
                    claims=state.claims,
                    checkpoints=state.checkpoints,
                    current_claim=state.open_claim,
                )
            if len(state.checkpoints) == 2:
                status = "complete"
                next_slot = None
            elif len(state.checkpoints) == 1:
                status = "partial"
                next_slot = state.root.slots[1]
            else:
                status = "ready"
                next_slot = state.root.slots[0]
            return PairedRunStatus(
                state=status,
                root=state.root,
                claims=state.claims,
                checkpoints=state.checkpoints,
                next_slot=next_slot,
            )

    @contextmanager
    def _read_locked(self):
        """Acquire the existing journal lock without creating/truncating it."""

        with self._thread_lock:
            metadata = _lstat_optional(self.run_dir)
            if metadata is None:
                raise VerifiedRoutingError("Paired run is not prepared.")
            _validate_run_directory(self.run_dir, metadata)
            with _existing_read_lock(
                self.lock_path,
                timeout_seconds=self.lock_timeout_seconds,
            ):
                _validate_run_directory(self.run_dir)
                _validate_regular_file(self.lock_path, "paired store lock")
                yield

    def abandon(self, claim: PairedRunClaim) -> None:
        """Make a failed in-process invocation explicitly indeterminate."""

        if not isinstance(claim, PairedRunClaim):
            raise TypeError("claim must be a PairedRunClaim.")
        self._active_claims.pop(claim.claim_sha256, None)

    def _refresh_process_identity_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._owner_pid:
            return
        self._owner_pid = current_pid
        self._owner_nonce = secrets.token_hex(32)
        self._active_claims.clear()

    def _assert_process_identity(self) -> None:
        if os.getpid() != self._owner_pid:
            raise PairedRunIndeterminateError(
                "A forked process cannot inherit paired claim authority."
            )

    def _owns_claim(self, claim_sha256: str) -> bool:
        authority = self._active_claims.get(claim_sha256)
        return (
            authority is not None
            and os.getpid() == self._owner_pid == authority.pid
            and secrets.compare_digest(self._owner_nonce, authority.nonce)
        )

    @contextmanager
    def _locked(self, *, create: bool):
        with self._thread_lock:
            if create:
                _ensure_run_directory(self.run_dir)
            else:
                metadata = _lstat_optional(self.run_dir)
                if metadata is None:
                    raise VerifiedRoutingError("Paired run is not prepared.")
                _validate_run_directory(self.run_dir, metadata)
            _ensure_lock_file(self.lock_path)
            try:
                self._process_lock.acquire(timeout=self.lock_timeout_seconds)
            except Timeout as exc:
                raise VerifiedRoutingError(
                    "Paired store lock acquisition timed out."
                ) from exc
            except OSError as exc:
                raise VerifiedRoutingError(
                    "Paired store lock acquisition failed."
                ) from exc
            try:
                _validate_run_directory(self.run_dir)
                _validate_regular_file(self.lock_path, "paired store lock")
                yield
            finally:
                try:
                    self._process_lock.release()
                except OSError as exc:
                    raise VerifiedRoutingError(
                        "Paired store lock release failed."
                    ) from exc

    def _require_prepared(self, state: _JournalState) -> _JournalState:
        if state.root is None:
            raise VerifiedRoutingError("Paired run is not prepared.")
        return state

    def _resolve_slot(
        self, root: PairedRunRoot, value: PairedRunSlot | str
    ) -> PairedRunSlot:
        if isinstance(value, PairedRunSlot):
            matches = tuple(item for item in root.slots if item.slot == value.slot)
            if len(matches) != 1 or matches[0] != value:
                raise VerifiedRoutingError("Paired slot is not in the run root.")
            return value
        if isinstance(value, str):
            matches = tuple(item for item in root.slots if item.slot == value)
            if len(matches) == 1:
                return matches[0]
            raise VerifiedRoutingError("Paired slot must be A or B.")
        raise TypeError("slot must be a PairedRunSlot or slot id.")

    def _read_unlocked(self) -> _JournalState:
        root_metadata = _lstat_optional(self.root_path)
        events_metadata = _lstat_optional(self.events_path)
        if _lstat_optional(self._legacy_events_path) is not None:
            raise VerifiedRoutingError(
                "Paired store is corrupt: legacy events.jsonl is forbidden."
            )
        if root_metadata is None:
            if events_metadata is not None:
                raise VerifiedRoutingError(
                    "Paired store is corrupt: events exist without run.json."
                )
            return _JournalState(None, (), (), None)
        root_bytes = _read_secure_file(
            self.root_path,
            "paired run root",
            maximum_bytes=_MAX_ROOT_BYTES,
            metadata=root_metadata,
        )
        try:
            root_raw = _strict_json_loads(root_bytes)
            root = PairedRunRoot.from_payload(root_raw)
        except (UnicodeError, json.JSONDecodeError, VerifiedRoutingError) as exc:
            raise VerifiedRoutingError(
                f"Paired store run.json is corrupt: {exc}"
            ) from exc
        expected_root = (canonical_json(root.payload()) + "\n").encode("utf-8")
        if root_bytes != expected_root:
            raise VerifiedRoutingError("Paired store run.json is not canonical.")
        if events_metadata is None:
            return _JournalState(root, (), (), None)
        artifacts = _list_event_artifacts(self.events_path, events_metadata)
        claims: list[PairedRunClaim] = []
        checkpoints: list[PairedRunCheckpoint] = []
        open_claim: PairedRunClaim | None = None
        seen_claims: set[str] = set()
        seen_checkpoints: set[str] = set()
        for artifact in artifacts:
            line_number = artifact.ordinal + 1
            try:
                encoded = _read_secure_file(
                    artifact.path,
                    f"paired event {line_number}",
                    maximum_bytes=_MAX_EVENT_BYTES,
                    metadata=artifact.metadata,
                )
                raw = _mapping(
                    _strict_json_loads(encoded),
                    f"paired event {line_number}",
                )
                _require_exact_fields(
                    raw, _EVENT_FIELDS, f"paired event {line_number}"
                )
                event = raw["event"]
                if event != artifact.event:
                    raise VerifiedRoutingError(
                        "Paired event type disagrees with its immutable name."
                    )
                event_digest = sha256_json(
                    {"event": event, "payload": raw["payload"]}
                )
                if event_digest != artifact.digest:
                    raise VerifiedRoutingError(
                        "Paired event digest disagrees with its immutable name."
                    )
                if event == "claim":
                    claim = PairedRunClaim.from_payload(raw["payload"])
                    if claim.claim_sha256 in seen_claims:
                        raise VerifiedRoutingError("Duplicate paired claim event.")
                    if open_claim is not None:
                        raise VerifiedRoutingError(
                            "Paired claim follows an uncheckpointed claim."
                        )
                    if len(checkpoints) >= len(root.slots):
                        raise VerifiedRoutingError(
                            "Paired claim follows a complete run."
                        )
                    expected_slot = root.slots[len(checkpoints)]
                    expected_claim = PairedRunClaim.build(root, expected_slot)
                    if claim != expected_claim:
                        raise VerifiedRoutingError(
                            "Paired claim is out of order or has invalid lineage."
                        )
                    claims.append(claim)
                    seen_claims.add(claim.claim_sha256)
                    open_claim = claim
                elif event == "checkpoint":
                    checkpoint = PairedRunCheckpoint.from_payload(raw["payload"])
                    if checkpoint.checkpoint_sha256 in seen_checkpoints:
                        raise VerifiedRoutingError(
                            "Duplicate paired checkpoint event."
                        )
                    if open_claim is None:
                        raise VerifiedRoutingError(
                            "Paired checkpoint has no preceding claim."
                        )
                    if (
                        checkpoint.binding.claim_sha256
                        != open_claim.claim_sha256
                    ):
                        raise VerifiedRoutingError(
                            "Paired checkpoint belongs to another claim."
                        )
                    validate_binding(root, open_claim, checkpoint.binding)
                    expected_previous = (
                        None
                        if not checkpoints
                        else checkpoints[-1].outcome_record_id
                    )
                    if checkpoint.binding.previous_record_id != expected_previous:
                        raise VerifiedRoutingError(
                            "Paired checkpoint breaks the outcome chain."
                        )
                    checkpoints.append(checkpoint)
                    seen_checkpoints.add(checkpoint.checkpoint_sha256)
                    open_claim = None
                else:
                    raise VerifiedRoutingError("Unknown paired event type.")
                canonical = (canonical_json(
                    {"event": event, "payload": raw["payload"]}
                ) + "\n").encode("utf-8")
                if encoded != canonical:
                    raise VerifiedRoutingError("Paired event is not canonical.")
            except (UnicodeError, json.JSONDecodeError, VerifiedRoutingError) as exc:
                raise VerifiedRoutingError(
                    f"Paired store is corrupt at event {line_number}: {exc}"
                ) from exc
        if tuple(_event_artifact_identity(item) for item in artifacts) != tuple(
            _event_artifact_identity(item)
            for item in _list_event_artifacts(self.events_path)
        ):
            raise VerifiedRoutingError(
                "Paired store event set changed while it was read."
            )
        return _JournalState(
            root,
            tuple(claims),
            tuple(checkpoints),
            open_claim,
        )

    def _append_event_unlocked(
        self, event: str, payload: Mapping[str, object]
    ) -> None:
        encoded = (
            canonical_json({"event": event, "payload": dict(payload)}) + "\n"
        ).encode("utf-8")
        if len(encoded) > _MAX_EVENT_BYTES:
            raise VerifiedRoutingError("Paired event exceeds its size limit.")
        metadata = _lstat_optional(self.events_path)
        if metadata is None:
            _ensure_event_directory(self.events_path)
        else:
            _validate_event_directory(self.events_path, metadata)
        artifacts = _list_event_artifacts(self.events_path)
        ordinal = len(artifacts)
        digest = sha256_json({"event": event, "payload": dict(payload)})
        target = self.events_path / (
            f"{ordinal:06d}-{event}-{digest}.json"
        )
        if not _atomic_install(target, encoded):
            raise VerifiedRoutingError(
                "Paired immutable event creation lost a write race."
            )
        current = _list_event_artifacts(self.events_path)
        if len(current) != ordinal + 1 or current[-1].path != target:
            raise VerifiedRoutingError(
                "Paired immutable event was not installed contiguously."
            )


def _ensure_event_directory(path: Path) -> None:
    metadata = _lstat_optional(path)
    if metadata is None:
        try:
            path.mkdir(mode=0o700)
            if os.name != "nt":
                os.chmod(path, 0o700, follow_symlinks=False)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        except OSError as exc:
            raise VerifiedRoutingError(
                "Paired event directory cannot be created securely."
            ) from exc
    _validate_event_directory(path)


def _validate_event_directory(
    path: Path,
    metadata: os.stat_result | None = None,
) -> os.stat_result:
    metadata = (
        _lstat_required(path, "paired event directory")
        if metadata is None
        else metadata
    )
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise VerifiedRoutingError(
            "Paired events path must be a non-link directory."
        )
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise VerifiedRoutingError(
            "Paired event directory permissions must be 0700."
        )
    return metadata


def _list_event_artifacts(
    path: Path,
    metadata: os.stat_result | None = None,
) -> tuple[_EventArtifact, ...]:
    before = _validate_event_directory(path, metadata)
    try:
        with os.scandir(path) as entries:
            names = tuple(sorted(item.name for item in entries))
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired event directory cannot be listed safely."
        ) from exc
    after = _validate_event_directory(path)
    if _directory_identity(before) != _directory_identity(after):
        raise VerifiedRoutingError(
            "Paired event directory changed while it was listed."
        )
    if len(names) > 4:
        raise VerifiedRoutingError("Paired event directory contains extra files.")
    artifacts: list[_EventArtifact] = []
    total_bytes = 0
    for expected_ordinal, name in enumerate(names):
        match = _EVENT_FILE.fullmatch(name)
        if match is None:
            raise VerifiedRoutingError(
                f"Paired event directory contains extra file {name!r}."
            )
        ordinal = int(match.group("ordinal"))
        if ordinal != expected_ordinal:
            raise VerifiedRoutingError(
                "Paired event ordinals must be contiguous from zero."
            )
        event_path = path / name
        event_metadata = _validate_regular_file(
            event_path,
            f"paired event file {name!r}",
        )
        total_bytes += event_metadata.st_size
        if total_bytes > _MAX_EVENTS_BYTES:
            raise VerifiedRoutingError(
                "Paired event directory exceeds its size limit."
            )
        artifacts.append(
            _EventArtifact(
                ordinal=ordinal,
                event=match.group("event"),
                digest=match.group("digest"),
                path=event_path,
                metadata=event_metadata,
            )
        )
    return tuple(artifacts)


def _event_artifact_identity(
    artifact: _EventArtifact,
) -> tuple[int, str, str, str, tuple[int, int, int, int]]:
    return (
        artifact.ordinal,
        artifact.event,
        artifact.digest,
        artifact.path.name,
        _identity(artifact.metadata),
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns


def _ensure_run_directory(path: Path) -> None:
    parent = _lstat_optional(path.parent)
    if parent is None or stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(
        parent.st_mode
    ):
        raise VerifiedRoutingError(
            "Paired run parent must be an existing non-link directory."
        )
    metadata = _lstat_optional(path)
    if metadata is None:
        try:
            path.mkdir(mode=0o700)
            if os.name != "nt":
                os.chmod(path, 0o700, follow_symlinks=False)
        except FileExistsError:
            pass
        except OSError as exc:
            raise VerifiedRoutingError(
                "Paired run directory cannot be created securely."
            ) from exc
    _validate_run_directory(path)


def _validate_run_directory(
    path: Path, metadata: os.stat_result | None = None
) -> os.stat_result:
    metadata = (
        _lstat_required(path, "paired run directory")
        if metadata is None
        else metadata
    )
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise VerifiedRoutingError(
            "Paired run path must be a non-link directory."
        )
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o700:
        raise VerifiedRoutingError("Paired run directory permissions must be 0700.")
    return metadata


def _list_run_directory(path: Path) -> tuple[str, ...]:
    before = _validate_run_directory(path)
    try:
        with os.scandir(path) as entries:
            names = tuple(sorted(item.name for item in entries))
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired run directory cannot be listed safely."
        ) from exc
    after = _validate_run_directory(path)
    if _directory_identity(before) != _directory_identity(after):
        raise VerifiedRoutingError(
            "Paired run directory changed while it was inspected."
        )
    return names


@contextmanager
def _existing_read_lock(path: Path, *, timeout_seconds: float):
    """Lock an existing journal without create/truncate side effects."""

    before = _validate_regular_file(path, "paired store lock")
    flags = (
        (os.O_RDWR if os.name == "nt" else os.O_RDONLY)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    locked = False
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _validate_file_metadata(opened, "paired store lock")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise VerifiedRoutingError(
                "Paired store lock changed while it was opened."
            )
        deadline = time.monotonic() + timeout_seconds
        while not locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                if exc.errno not in {
                    errno.EACCES,
                    errno.EAGAIN,
                    getattr(errno, "EDEADLK", errno.EAGAIN),
                    getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
                }:
                    raise VerifiedRoutingError(
                        "Paired store read lock acquisition failed."
                    ) from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VerifiedRoutingError(
                        "Paired store lock acquisition timed out."
                    ) from exc
                time.sleep(min(0.05, remaining))
        current = _validate_regular_file(path, "paired store lock")
        if (current.st_dev, current.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise VerifiedRoutingError(
                "Paired store lock changed during acquisition."
            )
        yield
    except VerifiedRoutingError:
        raise
    except OSError as exc:
        raise VerifiedRoutingError(
            "Paired store read lock acquisition failed."
        ) from exc
    finally:
        if descriptor is not None:
            try:
                if locked:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError as exc:
                raise VerifiedRoutingError(
                    "Paired store read lock release failed."
                ) from exc
            finally:
                os.close(descriptor)


def _ensure_lock_file(path: Path) -> None:
    metadata = _lstat_optional(path)
    if metadata is None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, 0o600)
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        except FileExistsError:
            pass
        except OSError as exc:
            raise VerifiedRoutingError(
                "Paired store lock could not be created securely."
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        _fsync_directory(path.parent)
        metadata = _lstat_optional(path)
    if metadata is None:
        raise VerifiedRoutingError("Paired store lock could not be created.")
    _validate_regular_file(path, "paired store lock", metadata)


def _atomic_install(path: Path, content: bytes) -> bool:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    installed = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Atomic paired store write made no progress.")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except TypeError:
                os.link(temporary, path)
            installed = True
        except FileExistsError:
            installed = False
        _fsync_directory(path.parent)
    except VerifiedRoutingError:
        raise
    except OSError as exc:
        raise VerifiedRoutingError(
            f"Cannot install paired store file {path.name!r} securely."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise VerifiedRoutingError(
                "Cannot remove paired store installation artifact."
            ) from exc
    if installed:
        _validate_regular_file(path, f"paired store {path.name}")
        _fsync_directory(path.parent)
    return installed


def _read_secure_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    metadata: os.stat_result | None = None,
) -> bytes:
    before = _validate_regular_file(path, label, metadata)
    if not 0 < before.st_size <= maximum_bytes:
        raise VerifiedRoutingError(f"{label} size is invalid.")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            _validate_file_metadata(opened, label)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, maximum_bytes + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > maximum_bytes:
                    raise VerifiedRoutingError(f"{label} exceeds its size limit.")
            after = os.fstat(descriptor)
            _validate_file_metadata(after, label)
        finally:
            os.close(descriptor)
    except VerifiedRoutingError:
        raise
    except OSError as exc:
        raise VerifiedRoutingError(f"{label} cannot be read safely.") from exc
    current = _validate_regular_file(path, label)
    if not (
        _identity(before)
        == _identity(opened)
        == _identity(after)
        == _identity(current)
    ):
        raise VerifiedRoutingError(f"{label} changed while it was read.")
    return b"".join(chunks)


def _validate_regular_file(
    path: Path,
    label: str,
    metadata: os.stat_result | None = None,
) -> os.stat_result:
    metadata = _lstat_required(path, label) if metadata is None else metadata
    _validate_file_metadata(metadata, label)
    return metadata


def _validate_file_metadata(metadata: os.stat_result, label: str) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VerifiedRoutingError(f"{label} must be a regular non-link file.")
    if metadata.st_nlink != 1:
        raise VerifiedRoutingError(f"{label} must have exactly one hard link.")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise VerifiedRoutingError(f"{label} permissions must be 0600.")


def _lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise VerifiedRoutingError(f"Cannot inspect paired store path {path}.") from exc


def _lstat_required(path: Path, label: str) -> os.stat_result:
    metadata = _lstat_optional(path)
    if metadata is None:
        raise VerifiedRoutingError(f"{label} is missing.")
    return metadata


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _strict_json_loads(value: bytes) -> object:
    def reject_constant(token: str) -> object:
        raise VerifiedRoutingError(
            f"Non-finite JSON number {token!r} is forbidden."
        )

    def parse_float(token: str) -> float:
        rendered = float(token)
        if not math.isfinite(rendered):
            raise VerifiedRoutingError(
                f"Non-finite JSON number {token!r} is forbidden."
            )
        return rendered

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise VerifiedRoutingError(
                    f"Duplicate JSON key {key!r} is forbidden."
                )
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        parse_float=parse_float,
        object_pairs_hook=reject_duplicates,
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerifiedRoutingError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise VerifiedRoutingError(f"{label} keys must be strings.")
    return dict(value)


def _require_exact_fields(
    raw: dict[str, Any], fields: set[str], label: str
) -> None:
    unknown = sorted(str(key) for key in raw if key not in fields)
    missing = sorted(fields.difference(raw))
    if unknown:
        raise VerifiedRoutingError(
            f"Unknown {label} fields: {', '.join(unknown)}."
        )
    if missing:
        raise VerifiedRoutingError(
            f"Missing {label} fields: {', '.join(missing)}."
        )


def _fsync_directory(path: Path) -> None:
    """Fsync a directory, best-effort only where Windows cannot support it.

    Event and root files are always fsynced before installation. Windows may
    reject opening or flushing directory handles, so only its documented
    access/unsupported errors are tolerated here.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError as exc:
        windows_best_effort = {
            errno.EACCES,
            errno.EBADF,
            errno.EINVAL,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EPERM,
        }
        if os.name != "nt" or exc.errno not in windows_best_effort:
            raise VerifiedRoutingError(
                "Paired store directory synchronization failed."
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
