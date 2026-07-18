from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import Callable, Iterator, Sequence
import unicodedata
from uuid import uuid4

from .assistant_bridge_two_phase_contracts import (
    MAX_CANDIDATE_FILES,
    MAX_CANDIDATE_FILE_BYTES,
    MAX_CANDIDATE_TOTAL_BYTES,
)
from .assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ExecutableIdentity,
    ProcessCleanupError,
    ProcessExecutionPolicy,
    ProcessExecutionResult,
    execute_process,
    resolve_executable,
)
from .assistant_bridge_process import process_is_alive


class WorkspaceSecurityError(ValueError):
    """Raised when a workspace cannot be snapshotted or changed safely."""


class WorkspaceTransactionBusyError(WorkspaceSecurityError):
    """Raised when another live transaction owns the workspace lock."""


class WorkspaceRecoveryPolicyUnavailable(WorkspaceSecurityError):
    """Raised when an older recovery journal has no durable policy binding."""


class _SimulatedTransactionCrash(RuntimeError):
    pass


_SAFE_TRANSACTION_ID = re.compile(r"^[a-f0-9]{32,64}$")
_SAFE_BACKUP_NAME = re.compile(r"^[0-9]{8}\.bin$")
_SAFE_QUARANTINE_NAME = re.compile(r"^\.mymoe-before-[a-f0-9]{32,64}-[0-9]{8}$")
_SAFE_INSTALL_NAME = re.compile(r"^\.mymoe-install-[a-f0-9]{32,64}-[0-9]{8}$")
_SAFE_ROLLBACK_NAME = re.compile(r"^\.mymoe-rollback-[a-f0-9]{32,64}-[0-9]{8}$")
_SAFE_RESTORE_NAME = re.compile(r"^\.mymoe-restore-[a-f0-9]{32,64}-[0-9]{8}$")
_MAX_JOURNAL_BYTES = 16 * 1024 * 1024
_ROLLBACK_PHASES = (
    "pending",
    "detach_intent",
    "detached",
    "restore_intent",
    "restored",
    "restore_unlink_intent",
    "restore_unlinked",
    "rollback_unlink_intent",
    "rollback_unlinked",
    "quarantine_unlink_intent",
    "quarantine_unlinked",
    "complete",
)
_SECURE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_SECURE_DIRECTORY_FLAGS = _SECURE_OPEN_FLAGS | getattr(os, "O_DIRECTORY", 0)
_GIT_STDOUT_LIMIT_BYTES = 8 * 1024 * 1024
_GIT_STDERR_LIMIT_BYTES = 256 * 1024
_GIT_TIMEOUT_SECONDS = 30.0
_GIT_EXECUTION_POLICY = ProcessExecutionPolicy(
    stdin_limit_bytes=0,
    stdout_limit_bytes=_GIT_STDOUT_LIMIT_BYTES,
    stderr_limit_bytes=_GIT_STDERR_LIMIT_BYTES,
    require_tree_isolation=True,
)
_TRUSTED_DIFF_ATTRIBUTES = (
    b"** diff -text -filter -ident -working-tree-encoding "
    b"whitespace=trailing-space,space-before-tab,cr-at-eol\n"
)


@dataclass(frozen=True)
class GitIdentity:
    """Explicit identity used only for an internal synthetic Git commit."""

    name: str = "Workspace Materializer"
    email: str = "materializer@localhost"

    def __post_init__(self) -> None:
        for value, label, maximum in (
            (self.name, "Git identity name", 128),
            (self.email, "Git identity email", 254),
        ):
            if (
                not isinstance(value, str)
                or value != value.strip()
                or not 1 <= len(value) <= maximum
                or any(character in value for character in "\x00\r\n<>")
            ):
                raise WorkspaceSecurityError(f"{label} is invalid.")
        if self.email.count("@") != 1 or any(
            character.isspace() for character in self.email
        ):
            raise WorkspaceSecurityError("Git identity email is invalid.")

    def payload(self) -> dict[str, str]:
        return {"name": self.name, "email": self.email}


@dataclass(frozen=True)
class TrustedGitSession:
    """One bounded Git executable identity reused for read-only workspace queries."""

    root: Path = field(repr=False)
    executable: ExecutableIdentity = field(repr=False)

    def staged_diff(self, *, max_output_bytes: int) -> bytes:
        return self._query(
            (
                "diff",
                "--cached",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "--",
                ".",
            ),
            max_output_bytes=max_output_bytes,
        )

    def unstaged_diff(self, *, max_output_bytes: int) -> bytes:
        return self._query(
            (
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "--",
                ".",
            ),
            max_output_bytes=max_output_bytes,
        )

    def untracked_paths(self, *, max_output_bytes: int) -> bytes:
        return self._query(
            ("ls-files", "--others", "--exclude-standard", "-z"),
            max_output_bytes=max_output_bytes,
        )

    def diff_check(self, *, max_output_bytes: int) -> ProcessExecutionResult:
        """Expose untracked files, then run the fixed whitespace-error check."""

        if (
            isinstance(max_output_bytes, bool)
            or not 1 <= max_output_bytes <= 32 * 1024 * 1024
        ):
            raise WorkspaceSecurityError("Git output bound is outside safe limits.")
        intent_command = _trusted_git_command(
            ("add", "--intent-to-add", "-f", "--all", "--", "."),
            self.root,
        )
        intent = _execute_git(
            intent_command,
            self.root,
            commit_identity=None,
            git_identity=self.executable,
            stdout_limit_bytes=max_output_bytes,
        )
        if not intent.ok:
            return intent
        check_command = _trusted_git_command(
            (
                "diff",
                "--check",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
                ".",
            ),
            self.root,
        )
        return _execute_git(
            check_command,
            self.root,
            commit_identity=None,
            git_identity=self.executable,
            stdout_limit_bytes=max_output_bytes,
        )

    def _query(self, args: Sequence[str], *, max_output_bytes: int) -> bytes:
        if (
            isinstance(max_output_bytes, bool)
            or not 1 <= max_output_bytes <= 32 * 1024 * 1024
        ):
            raise WorkspaceSecurityError("Git output bound is outside safe limits.")
        return _run_git(
            args,
            self.root,
            capture_bytes=True,
            git_identity=self.executable,
            stdout_limit_bytes=max_output_bytes,
        )


@dataclass(frozen=True)
class WorkspaceWriteCapability:
    supported: bool
    backend: str
    reason: str = ""

    def payload(self) -> dict[str, object]:
        """Return the stable, content-free descriptor bound to execution authority."""

        return {
            "supported": self.supported,
            "backend": self.backend,
            "reason": self.reason or None,
        }


def workspace_write_capability() -> WorkspaceWriteCapability:
    """Report whether fail-closed workspace mutation is available before routing."""

    if os.name == "nt":
        try:
            import ctypes
            import msvcrt  # noqa: F401

            getattr(ctypes, "windll")
        except (AttributeError, ImportError):
            return WorkspaceWriteCapability(
                False,
                "windows",
                "required Win32 handle APIs are unavailable",
            )
        if not hasattr(os, "link"):
            return WorkspaceWriteCapability(
                False,
                "windows",
                "atomic no-replace hard links are unavailable",
            )
        return WorkspaceWriteCapability(True, "windows-handle-write-through")
    backend = _posix_no_replace_backend()
    required_dir_fd = (os.open, os.link, os.unlink, os.mkdir)
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or any(item not in os.supports_dir_fd for item in required_dir_fd)
        or not hasattr(os, "link")
        or backend is None
    ):
        return WorkspaceWriteCapability(
            False,
            "unsupported",
            "no-follow dir-fd and native atomic no-replace primitives are unavailable",
        )
    return WorkspaceWriteCapability(True, backend)


def trusted_git_session(workspace: str | Path) -> TrustedGitSession:
    """Attest the OS-owned Git executable and bind it to one live worktree."""

    root = _trusted_root(workspace, label="Git workspace")
    repository_marker = _has_repository_marker(root)
    if not repository_marker:
        raise WorkspaceSecurityError("Git workspace marker is unavailable.")
    identity = _resolve_trusted_git_identity(required=True)
    if identity is None:  # Defensive: required=True must resolve or raise.
        raise WorkspaceSecurityError("Trusted Git executable is unavailable.")
    if not _is_git_workspace(
        root,
        git_identity=identity,
        repository_marker=repository_marker,
    ):
        raise WorkspaceSecurityError("Git workspace could not be attested.")
    return TrustedGitSession(root=root, executable=identity)


def trusted_git_executable() -> ExecutableIdentity:
    """Attest the fixed Git executable without trusting a candidate repository."""

    identity = _resolve_trusted_git_identity(required=True)
    if identity is None:  # Defensive: required=True must resolve or raise.
        raise WorkspaceSecurityError("Trusted Git executable is unavailable.")
    return identity


def _posix_no_replace_backend() -> str | None:
    if os.name != "posix":
        return None
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
    except (ImportError, OSError):
        return None
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        return "darwin-renamex-excl"
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        return "linux-renameat2-noreplace"
    return None


@dataclass(frozen=True)
class IgnoredPathRule:
    path: str
    direction: str = "input_only"

    def __post_init__(self) -> None:
        clean = _safe_relative(self.path)
        if self.direction not in {"input_only", "round_trip"}:
            raise WorkspaceSecurityError(
                "Ignored path direction must be input_only or round_trip."
            )
        object.__setattr__(self, "path", clean)


@dataclass(frozen=True)
class WorkspaceScopePolicy:
    max_files: int = 5000
    max_total_bytes: int = 256 * 1024 * 1024
    max_file_bytes: int = 64 * 1024 * 1024
    ignored_paths: tuple[IgnoredPathRule, ...] = ()
    synthetic_git_identity: GitIdentity = field(default_factory=GitIdentity)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ignored_paths", tuple(self.ignored_paths))
        if not isinstance(self.synthetic_git_identity, GitIdentity):
            raise WorkspaceSecurityError("Workspace synthetic Git identity is invalid.")
        if not 1 <= self.max_files <= MAX_CANDIDATE_FILES:
            raise WorkspaceSecurityError("Workspace max_files is outside safe bounds.")
        if (
            not 1
            <= self.max_file_bytes
            <= min(
                self.max_total_bytes,
                MAX_CANDIDATE_FILE_BYTES,
            )
        ):
            raise WorkspaceSecurityError("Workspace byte bounds are invalid.")
        if not 1 <= self.max_total_bytes <= MAX_CANDIDATE_TOTAL_BYTES:
            raise WorkspaceSecurityError("Workspace total byte bound is invalid.")
        paths = [item.path.casefold() for item in self.ignored_paths]
        if len(paths) != len(set(paths)):
            raise WorkspaceSecurityError("Ignored path rules contain a collision.")


def _workspace_policy_recovery_payload(
    policy: WorkspaceScopePolicy,
) -> dict[str, object]:
    return {
        "max_files": policy.max_files,
        "max_total_bytes": policy.max_total_bytes,
        "max_file_bytes": policy.max_file_bytes,
        "ignored_paths": [
            {"path": item.path, "direction": item.direction}
            for item in policy.ignored_paths
        ],
        "synthetic_git_identity": policy.synthetic_git_identity.payload(),
    }


def _workspace_policy_from_recovery_journal(
    payload: dict[str, object],
) -> WorkspaceScopePolicy:
    raw = payload.get("workspace_policy")
    if raw is None:
        raise WorkspaceRecoveryPolicyUnavailable(
            "Workspace recovery policy is unavailable."
        )
    if not isinstance(raw, dict) or set(raw) != {
        "ignored_paths",
        "max_file_bytes",
        "max_files",
        "max_total_bytes",
        "synthetic_git_identity",
    }:
        raise WorkspaceSecurityError("Workspace recovery policy binding is invalid.")
    limits = (
        raw.get("max_files"),
        raw.get("max_total_bytes"),
        raw.get("max_file_bytes"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in limits):
        raise WorkspaceSecurityError("Workspace recovery policy bounds are invalid.")
    raw_rules = raw.get("ignored_paths")
    raw_identity = raw.get("synthetic_git_identity")
    if (
        not isinstance(raw_rules, list)
        or len(raw_rules) > 100_000
        or not isinstance(raw_identity, dict)
        or set(raw_identity) != {"email", "name"}
        or not isinstance(raw_identity.get("name"), str)
        or not isinstance(raw_identity.get("email"), str)
    ):
        raise WorkspaceSecurityError("Workspace recovery policy metadata is invalid.")
    rules: list[IgnoredPathRule] = []
    for item in raw_rules:
        if (
            not isinstance(item, dict)
            or set(item) != {"direction", "path"}
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("direction"), str)
        ):
            raise WorkspaceSecurityError(
                "Workspace recovery ignored-path policy is invalid."
            )
        rules.append(
            IgnoredPathRule(
                path=item["path"],
                direction=item["direction"],
            )
        )
    try:
        return WorkspaceScopePolicy(
            max_files=limits[0],
            max_total_bytes=limits[1],
            max_file_bytes=limits[2],
            ignored_paths=tuple(rules),
            synthetic_git_identity=GitIdentity(
                name=raw_identity["name"],
                email=raw_identity["email"],
            ),
        )
    except (TypeError, ValueError, WorkspaceSecurityError) as exc:
        raise WorkspaceSecurityError(
            "Workspace recovery policy binding is invalid."
        ) from exc


@dataclass(frozen=True, order=True)
class WorkspaceFile:
    path: str
    kind: str
    sha256: str
    size: int
    mode: int
    direction: str = "round_trip"

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size": self.size,
            "mode": self.mode,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class WorkspaceSnapshot:
    root: str = field(repr=False)
    git_repository: bool
    head_sha: str
    index_sha256: str
    status_sha256: str
    manifest_sha256: str
    fingerprint: str
    files: tuple[WorkspaceFile, ...] = field(repr=False)
    tracked_paths: tuple[str, ...] = field(repr=False)
    total_bytes: int

    def payload(self) -> dict[str, object]:
        return {
            "root_sha256": _sha256_text(self.root),
            "git_repository": self.git_repository,
            "head_sha": self.head_sha or None,
            "index_sha256": self.index_sha256,
            "status_sha256": self.status_sha256,
            "manifest_sha256": self.manifest_sha256,
            "fingerprint": self.fingerprint,
            "file_count": len(self.files),
            "tracked_file_count": len(self.tracked_paths),
            "total_bytes": self.total_bytes,
            "scope": "tracked_untracked_nonignored_plus_declared_ignored",
        }


@dataclass(frozen=True)
class WorkspaceChange:
    path: str
    before: WorkspaceFile | None
    after: WorkspaceFile | None


@dataclass(frozen=True)
class MaterializedWorkspace:
    root: Path
    source_snapshot: WorkspaceSnapshot
    baseline_files: tuple[WorkspaceFile, ...]
    policy: WorkspaceScopePolicy

    def snapshot(self) -> tuple[WorkspaceFile, ...]:
        return _canonical_materialized_files(
            self.source_snapshot,
            snapshot_materialized(self.root, self.policy),
        )


def snapshot_workspace(
    workspace: str | Path,
    policy: WorkspaceScopePolicy,
) -> WorkspaceSnapshot:
    root = _trusted_root(workspace, label="Workspace")
    repository_marker = _has_repository_marker(root)
    git_identity = (
        _resolve_trusted_git_identity(required=True) if repository_marker else None
    )
    first = _snapshot_once(
        root,
        policy,
        git_identity=git_identity,
        repository_marker=repository_marker,
    )
    if _has_repository_marker(root) != repository_marker:
        raise WorkspaceSecurityError(
            "Git repository marker changed while the workspace was being attested."
        )
    second = _snapshot_once(
        root,
        policy,
        git_identity=git_identity,
        repository_marker=repository_marker,
    )
    if _has_repository_marker(root) != repository_marker:
        raise WorkspaceSecurityError(
            "Git repository marker changed while the workspace was being attested."
        )
    if first.fingerprint != second.fingerprint:
        raise WorkspaceSecurityError("Workspace changed while it was being attested.")
    return second


def snapshot_materialized(
    root: str | Path,
    policy: WorkspaceScopePolicy,
) -> tuple[WorkspaceFile, ...]:
    base = _trusted_root(root, label="Materialized workspace")
    paths: list[str] = []
    for directory, directories, files in os.walk(base, topdown=True, followlinks=False):
        _assert_directory_entry(Path(directory))
        safe_directories: list[str] = []
        for name in sorted(directories):
            child = Path(directory) / name
            _assert_directory_entry(child)
            if name != ".git":
                safe_directories.append(name)
        directories[:] = safe_directories
        relative_dir = Path(directory).relative_to(base)
        for name in sorted(files):
            relative = (relative_dir / name).as_posix()
            paths.append(_safe_relative(relative))
    directions = {item.path: item.direction for item in policy.ignored_paths}
    for rule in policy.ignored_paths:
        if rule.path not in paths:
            paths.append(rule.path)
    return _manifest_files(base, paths, policy, directions=directions)


@contextmanager
def materialize_workspace(
    snapshot: WorkspaceSnapshot,
    policy: WorkspaceScopePolicy,
) -> Iterator[MaterializedWorkspace]:
    source = Path(snapshot.root)
    with tempfile.TemporaryDirectory(prefix="mymoe-workspace-") as tmp:
        root = Path(tmp) / "workspace"
        root.mkdir(mode=0o700)
        for item in snapshot.files:
            if item.kind == "missing":
                continue
            target = root / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            data = _read_attested_file(source, item, policy)
            _durable_write_new(target, data, item.mode)
        if snapshot_workspace(source, policy).fingerprint != snapshot.fingerprint:
            raise WorkspaceSecurityError(
                "Workspace changed while the materialized candidate was created."
            )
        baseline = _canonical_materialized_files(
            snapshot,
            snapshot_materialized(root, policy),
        )
        _initialize_synthetic_repository(
            root,
            identity=policy.synthetic_git_identity,
        )
        yield MaterializedWorkspace(
            root=root,
            source_snapshot=snapshot,
            baseline_files=baseline,
            policy=policy,
        )


def build_changeset(
    baseline: Sequence[WorkspaceFile],
    candidate: Sequence[WorkspaceFile],
) -> tuple[WorkspaceChange, ...]:
    before = {item.path: item for item in baseline}
    after = {item.path: item for item in candidate}
    changes: list[WorkspaceChange] = []
    for path in sorted(set(before) | set(after)):
        left = before.get(path)
        right = after.get(path)
        if left != right:
            if (left and left.direction == "input_only") or (
                right and right.direction == "input_only"
            ):
                raise WorkspaceSecurityError(
                    "An input_only ignored artifact was modified by the candidate."
                )
            changes.append(WorkspaceChange(path=path, before=left, after=right))
    return tuple(changes)


def _canonical_materialized_files(
    source_snapshot: WorkspaceSnapshot,
    candidate_files: Sequence[WorkspaceFile],
) -> tuple[WorkspaceFile, ...]:
    """Represent source-tracked deletions exactly as live Git snapshots do."""

    result = {item.path: item for item in candidate_files}
    source_files = {item.path: item for item in source_snapshot.files}
    empty_sha256 = _sha256_bytes(b"")
    for path in source_snapshot.tracked_paths:
        if path in result:
            continue
        previous = source_files.get(path)
        result[path] = WorkspaceFile(
            path=path,
            kind="missing",
            sha256=empty_sha256,
            size=0,
            mode=0,
            direction=(previous.direction if previous is not None else "round_trip"),
        )
    return tuple(sorted(result.values()))


def _with_expected_missing_files(
    observed_files: Sequence[WorkspaceFile],
    expected_files: Sequence[WorkspaceFile],
) -> tuple[WorkspaceFile, ...]:
    result = {item.path: item for item in observed_files}
    for item in expected_files:
        if item.kind == "missing" and item.path not in result:
            result[item.path] = item
    return tuple(sorted(result.values()))


def _canonical_changes(
    changes: Sequence[WorkspaceChange],
    candidate_files: Sequence[WorkspaceFile],
) -> tuple[WorkspaceChange, ...]:
    candidates = {item.path: item for item in candidate_files}
    result: list[WorkspaceChange] = []
    for change in changes:
        after = change.after
        canonical = candidates.get(change.path)
        if after is None and canonical is not None and canonical.kind == "missing":
            after = canonical
        result.append(
            WorkspaceChange(
                path=change.path,
                before=change.before,
                after=after,
            )
        )
    return tuple(result)


def apply_changeset(
    *,
    source_snapshot: WorkspaceSnapshot,
    candidate_root: str | Path,
    candidate_files: Sequence[WorkspaceFile],
    changes: Sequence[WorkspaceChange],
    policy: WorkspaceScopePolicy,
    state_dir: str | Path,
    transaction_id: str,
    lock_ttl_seconds: float = 120.0,
    retain_committed_journal: bool = False,
    on_committed: Callable[[WorkspaceSnapshot], None] | None = None,
    _fault_after_mutation: int | None = None,
    _fault_after_commit: bool = False,
    _fault_after_committed_callback: bool = False,
    _fault_after_install_step: str | None = None,
    _test_hook_after_detach: Callable[[Path], None] | None = None,
) -> WorkspaceSnapshot:
    _validate_transaction_id(transaction_id)
    if not isinstance(retain_committed_journal, bool):
        raise WorkspaceSecurityError("Workspace journal retention mode is invalid.")
    source = _trusted_root(source_snapshot.root, label="Source workspace")
    candidate = _trusted_root(candidate_root, label="Candidate workspace")
    candidate_files = _canonical_materialized_files(source_snapshot, candidate_files)
    changes = _canonical_changes(changes, candidate_files)
    if changes:
        _require_secure_apply_capabilities()
    state = _prepare_state_directory(state_dir)
    lock = state / f"workspace-{_sha256_text(str(source))[:24]}.lock"
    _acquire_transaction_lock(lock, lock_ttl_seconds)
    transaction = state / f"transaction-{transaction_id}"
    try:
        backup_dir, staged_dir = _prepare_transaction_directories(state, transaction)
    except Exception:
        _release_transaction_lock(lock)
        raise
    journal = transaction / "journal.json"
    records = [
        {
            "path": item.path,
            "before": item.before.payload() if item.before else None,
            "after": item.after.payload() if item.after else None,
            "backup": (
                f"{index:08d}.bin"
                if item.before is not None and item.before.kind != "missing"
                else None
            ),
            "backup_sha256": (
                item.before.sha256
                if item.before is not None and item.before.kind != "missing"
                else None
            ),
            "staged": (
                f"{index:08d}.bin"
                if item.after is not None and item.after.kind != "missing"
                else None
            ),
            "staged_sha256": (
                item.after.sha256
                if item.after is not None and item.after.kind != "missing"
                else None
            ),
            "quarantine": (
                f".mymoe-before-{transaction_id}-{index:08d}"
                if item.before is not None and item.before.kind != "missing"
                else None
            ),
            "install": f".mymoe-install-{transaction_id}-{index:08d}",
            "install_phase": "pending",
            "rollback": f".mymoe-rollback-{transaction_id}-{index:08d}",
            "restore": f".mymoe-restore-{transaction_id}-{index:08d}",
            "restore_identity": None,
            "restore_phase": "pending",
            "rollback_phase": "pending",
            "created_directories": [],
            "installed_identity": None,
            "status": "pending",
        }
        for index, item in enumerate(changes)
    ]
    journal_payload: dict[str, object] = {
        "schema_version": "1.0",
        "transaction_id": transaction_id,
        "source_root_sha256": _sha256_text(str(source)),
        "source_fingerprint": source_snapshot.fingerprint,
        "committed_fingerprint": None,
        "workspace_policy": _workspace_policy_recovery_payload(policy),
        "status": "prepared",
        "recovery_error": None,
        "changes": records,
    }
    try:
        for index, directories in enumerate(_plan_created_directories(source, changes)):
            records[index]["created_directories"] = list(directories)
        _assert_journal_capacity(journal_payload)
        _write_journal(journal, journal_payload)
    except Exception:
        _durable_rmtree(transaction, state)
        _release_transaction_lock(lock)
        raise
    try:
        current = snapshot_workspace(source, policy)
        if current.fingerprint != source_snapshot.fingerprint:
            raise WorkspaceSecurityError(
                "Workspace changed after confirmation; transaction was not applied."
            )
        expected_changes = build_changeset(source_snapshot.files, candidate_files)
        if tuple(changes) != expected_changes:
            raise WorkspaceSecurityError(
                "Workspace changes do not match the attested candidate manifest."
            )
        _stage_attested_candidate(
            candidate,
            candidate_files,
            changes,
            policy,
            staged_dir,
            records,
        )
        _preflight_transaction_artifacts(
            source,
            changes,
            records,
            staged_dir=staged_dir,
        )
        journal_payload["status"] = "prepared"
        _write_journal(journal, journal_payload)
        journal_payload["status"] = "applying"
        _write_journal(journal, journal_payload)
        _create_all_planned_directories(source, records)
        _preflight_transaction_artifacts(
            source,
            changes,
            records,
            staged_dir=staged_dir,
        )
        _prepare_install_artifacts(
            source,
            changes,
            records,
            staged_dir=staged_dir,
            journal_payload=journal_payload,
            journal=journal,
            fault_after_step=_fault_after_install_step,
        )
        for index, change in enumerate(changes):
            target = source / change.path
            _assert_current_entry(target, change.before)
            if change.before is not None and change.before.kind != "missing":
                backup = backup_dir / f"{index:08d}.bin"
                data = _read_attested_file(source, change.before, policy)
                _durable_write_new(backup, data, 0o600)
                _fsync_directory(backup_dir)
                records[index]["backup"] = backup.name
                records[index]["backup_sha256"] = _sha256_bytes(data)
            records[index]["status"] = "backed_up"
            _write_journal(journal, journal_payload)
            if change.before is not None and change.before.kind != "missing":
                records[index]["quarantine"] = (
                    f".mymoe-before-{transaction_id}-{index:08d}"
                )
            records[index]["status"] = "mutating"
            _write_journal(journal, journal_payload)
            staged_name = records[index].get("staged")
            staged = (
                staged_dir / str(staged_name) if isinstance(staged_name, str) else None
            )
            installed_identity = _apply_change_fail_closed(
                source=source,
                change=change,
                staged=staged,
                quarantine_name=records[index].get("quarantine"),
                install_name=records[index].get("install"),
                created_directories=records[index].get("created_directories"),
                record=records[index],
                journal_payload=journal_payload,
                journal=journal,
                fault_after_install_step=_fault_after_install_step,
                test_hook=_test_hook_after_detach,
            )
            _raise_install_fault(
                _fault_after_install_step,
                "before_caller_identity_persist",
            )
            records[index]["installed_identity"] = installed_identity
            _write_journal(journal, journal_payload)
            if _fault_after_mutation == index:
                raise _SimulatedTransactionCrash("simulated crash after mutation")
            records[index]["status"] = "applied"
            _write_journal(journal, journal_payload)
        result = snapshot_workspace(source, policy)
        if (
            result.git_repository != source_snapshot.git_repository
            or result.head_sha != source_snapshot.head_sha
            or result.index_sha256 != source_snapshot.index_sha256
        ):
            raise WorkspaceSecurityError(
                "Git HEAD or index changed during the workspace transaction."
            )
        expected = tuple(sorted(candidate_files))
        if tuple(sorted(result.files)) != expected:
            raise WorkspaceSecurityError(
                "Post-transaction workspace does not match the verified candidate."
            )
        journal_payload["committed_fingerprint"] = result.fingerprint
        journal_payload["status"] = "committed"
        _write_journal(journal, journal_payload)
        if _fault_after_commit:
            raise _SimulatedTransactionCrash("simulated crash after commit journal")
        if on_committed is not None:
            on_committed(result)
        if _fault_after_committed_callback:
            raise _SimulatedTransactionCrash("simulated crash after committed callback")
        if not retain_committed_journal:
            _durable_rmtree(transaction, state)
        return result
    except _SimulatedTransactionCrash:
        raise
    except Exception as original:
        if journal_payload.get("status") == "prepared":
            _durable_rmtree(transaction, state)
            raise
        if journal_payload.get("status") == "committed":
            raise
        try:
            _rollback_from_journal(
                source,
                journal_payload,
                backup_dir,
                journal=journal,
            )
            journal_payload["status"] = "rolled_back"
            _write_journal(journal, journal_payload)
        except (OSError, WorkspaceSecurityError) as recovery_error:
            journal_payload["status"] = "recovery_required"
            journal_payload["recovery_error"] = type(recovery_error).__name__
            _write_journal(journal, journal_payload)
            raise WorkspaceSecurityError(
                "Workspace transaction failed and requires journal recovery."
            ) from original
        raise
    finally:
        _release_transaction_lock(lock)


def recover_workspace_transaction(
    *,
    state_dir: str | Path,
    transaction_id: str,
    source_root: str | Path,
    lock_ttl_seconds: float = 120.0,
    expected_source_fingerprint: str | None = None,
    retain_recovered_journal: bool = False,
    fallback_workspace_policy: WorkspaceScopePolicy | None = None,
    _fault_after_rollback_step: str | None = None,
) -> None:
    _validate_transaction_id(transaction_id)
    if not isinstance(retain_recovered_journal, bool):
        raise WorkspaceSecurityError("Workspace recovery retention mode is invalid.")
    if retain_recovered_journal and expected_source_fingerprint is None:
        raise WorkspaceSecurityError(
            "Retained workspace recovery requires a source binding."
        )
    if fallback_workspace_policy is not None and not isinstance(
        fallback_workspace_policy,
        WorkspaceScopePolicy,
    ):
        raise WorkspaceSecurityError("Workspace recovery fallback policy is invalid.")
    state = _trusted_root(state_dir, label="Workspace state")
    source = _trusted_root(source_root, label="Source workspace")
    transaction = state / f"transaction-{transaction_id}"
    _assert_directory_entry(transaction)
    journal = transaction / "journal.json"
    if not journal.is_file():
        raise WorkspaceSecurityError("Workspace recovery journal does not exist.")
    lock = state / f"workspace-{_sha256_text(str(source))[:24]}.lock"
    _acquire_transaction_lock(lock, lock_ttl_seconds)
    try:
        raw_journal, _ = _read_regular_path_nofollow(journal, _MAX_JOURNAL_BYTES)
        payload = json.loads(raw_journal.decode("utf-8"))
        _validate_journal(payload, transaction_id)
        if payload.get("source_root_sha256") != _sha256_text(str(source)):
            raise WorkspaceSecurityError(
                "Workspace recovery journal targets another root."
            )
        recovery_policy: WorkspaceScopePolicy | None = None
        if expected_source_fingerprint is not None:
            if (
                re.fullmatch(r"[0-9a-f]{64}", expected_source_fingerprint) is None
                or payload.get("source_fingerprint") != expected_source_fingerprint
            ):
                raise WorkspaceSecurityError(
                    "Workspace recovery journal source binding is invalid."
                )
            try:
                recovery_policy = _workspace_policy_from_recovery_journal(payload)
            except WorkspaceRecoveryPolicyUnavailable:
                if fallback_workspace_policy is None:
                    raise
                recovery_policy = fallback_workspace_policy
        if payload.get("status") == "committed":
            raise WorkspaceSecurityError(
                "Committed workspace requires database finish, not rollback."
            )
        if payload.get("status") != "recovered":
            _rollback_from_journal(
                source,
                payload,
                transaction / "backups",
                journal=journal,
                _fault_after_step=_fault_after_rollback_step,
            )
        if recovery_policy is not None:
            restored = snapshot_workspace(source, recovery_policy)
            if restored.fingerprint != expected_source_fingerprint:
                raise WorkspaceSecurityError(
                    "Workspace recovery did not restore the bound source."
                )
        payload["status"] = "recovered"
        _write_journal(journal, payload)
        if not retain_recovered_journal:
            _durable_rmtree(transaction, state)
    finally:
        _release_transaction_lock(lock)


def finalize_recovered_workspace_transaction(
    *,
    state_dir: str | Path,
    transaction_id: str,
    source_root: str | Path,
    expected_source_fingerprint: str,
    lock_ttl_seconds: float = 120.0,
    _fault_after_cleanup_step: str | None = None,
) -> None:
    """Remove a recovered journal only after its database reset commits."""

    _validate_transaction_id(transaction_id)
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_fingerprint) is None:
        raise WorkspaceSecurityError("Workspace recovery source binding is invalid.")
    state = _trusted_root(state_dir, label="Workspace state")
    source = _trusted_root(source_root, label="Source workspace")
    transaction = state / f"transaction-{transaction_id}"
    lock = state / f"workspace-{_sha256_text(str(source))[:24]}.lock"
    _acquire_transaction_lock(lock, lock_ttl_seconds)
    try:
        _cleanup_transaction_state(
            state,
            transaction_id,
            transaction=transaction,
            source_root_sha256=_sha256_text(str(source)),
            source_fingerprint=expected_source_fingerprint,
            journal_status="recovered",
            fault_after_step=_fault_after_cleanup_step,
        )
    finally:
        _release_transaction_lock(lock)


def finish_committed_workspace_transaction(
    *,
    state_dir: str | Path,
    transaction_id: str,
    source_root: str | Path,
    expected_source_fingerprint: str,
    on_committed: Callable[[WorkspaceSnapshot], None],
    lock_ttl_seconds: float = 120.0,
    fallback_workspace_policy: WorkspaceScopePolicy | None = None,
    _fault_after_db_finish: bool = False,
) -> WorkspaceSnapshot | None:
    """Finish a committed filesystem apply while holding its workspace lock."""

    _validate_transaction_id(transaction_id)
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_fingerprint) is None:
        raise WorkspaceSecurityError("Committed workspace source binding is invalid.")
    state = _trusted_root(state_dir, label="Workspace state")
    source = _trusted_root(source_root, label="Source workspace")
    transaction = state / f"transaction-{transaction_id}"
    journal = transaction / "journal.json"
    lock = state / f"workspace-{_sha256_text(str(source))[:24]}.lock"
    _acquire_transaction_lock(lock, lock_ttl_seconds)
    try:
        _assert_directory_entry(transaction)
        raw_journal, _ = _read_regular_path_nofollow(journal, _MAX_JOURNAL_BYTES)
        payload = json.loads(raw_journal.decode("utf-8"))
        _validate_journal(payload, transaction_id)
        if payload.get("status") != "committed":
            return None
        if (
            payload.get("source_root_sha256") != _sha256_text(str(source))
            or payload.get("source_fingerprint") != expected_source_fingerprint
        ):
            raise WorkspaceSecurityError(
                "Committed workspace journal binding is invalid."
            )
        try:
            policy = _workspace_policy_from_recovery_journal(payload)
        except WorkspaceRecoveryPolicyUnavailable:
            if fallback_workspace_policy is None:
                raise
            policy = fallback_workspace_policy
        snapshot = snapshot_workspace(source, policy)
        if snapshot.fingerprint != payload.get("committed_fingerprint"):
            raise WorkspaceSecurityError(
                "Committed workspace changed before database finish."
            )
        on_committed(snapshot)
        if _fault_after_db_finish:
            raise _SimulatedTransactionCrash("simulated crash after database finish")
        try:
            _cleanup_committed_transaction(
                state,
                transaction_id,
                transaction=transaction,
                source_root_sha256=_sha256_text(str(source)),
                source_fingerprint=expected_source_fingerprint,
            )
        except (OSError, WorkspaceSecurityError):
            pass
        return snapshot
    finally:
        _release_transaction_lock(lock)


def finalize_committed_workspace_transaction(
    *,
    state_dir: str | Path,
    transaction_id: str,
    source_root: str | Path,
    expected_source_fingerprint: str,
    lock_ttl_seconds: float = 120.0,
    _fault_after_cleanup_step: str | None = None,
) -> None:
    """Idempotently remove a committed journal after database apply commits."""

    _validate_transaction_id(transaction_id)
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_fingerprint) is None:
        raise WorkspaceSecurityError("Committed workspace source binding is invalid.")
    state = _trusted_root(state_dir, label="Workspace state")
    source = _trusted_root(source_root, label="Source workspace")
    transaction = state / f"transaction-{transaction_id}"
    lock = state / f"workspace-{_sha256_text(str(source))[:24]}.lock"
    _acquire_transaction_lock(lock, lock_ttl_seconds)
    try:
        _cleanup_transaction_state(
            state,
            transaction_id,
            transaction=transaction,
            source_root_sha256=_sha256_text(str(source)),
            source_fingerprint=expected_source_fingerprint,
            journal_status="committed",
            fault_after_step=_fault_after_cleanup_step,
        )
    finally:
        _release_transaction_lock(lock)


def _cleanup_committed_transaction(
    state: Path,
    transaction_id: str,
    *,
    transaction: Path,
    source_root_sha256: str,
    source_fingerprint: str,
) -> None:
    _cleanup_transaction_state(
        state,
        transaction_id,
        transaction=transaction,
        source_root_sha256=source_root_sha256,
        source_fingerprint=source_fingerprint,
        journal_status="committed",
        fault_after_step=None,
    )


def _cleanup_transaction_state(
    state: Path,
    transaction_id: str,
    *,
    transaction: Path,
    source_root_sha256: str,
    source_fingerprint: str,
    journal_status: str,
    fault_after_step: str | None,
) -> None:
    cleanup = state / f"cleanup-{transaction_id}"
    marker = state / f"cleanup-{transaction_id}.json"
    marker_binding = {
        "schema_version": "1.0",
        "transaction_id": transaction_id,
        "source_root_sha256": source_root_sha256,
        "source_fingerprint": source_fingerprint,
        "journal_status": journal_status,
    }
    if _entry_exists(cleanup):
        if _entry_exists(transaction):
            raise WorkspaceSecurityError("Workspace cleanup state is ambiguous.")
        marker_payload = _validate_cleanup_marker(marker, marker_binding)
        _assert_directory_entry(cleanup)
        if marker_payload.get("transaction_identity") != _directory_identity(cleanup):
            raise WorkspaceSecurityError(
                "Workspace cleanup directory identity changed."
            )
        cleanup_journal = cleanup / "journal.json"
        if _entry_exists(cleanup_journal):
            cleanup_raw, _ = _read_regular_path_nofollow(
                cleanup_journal,
                _MAX_JOURNAL_BYTES,
            )
            if marker_payload.get("journal_sha256") != _sha256_bytes(cleanup_raw):
                raise WorkspaceSecurityError(
                    "Workspace cleanup journal digest changed."
                )
    elif _entry_exists(transaction):
        _assert_directory_entry(transaction)
        raw_journal, _ = _read_regular_path_nofollow(
            transaction / "journal.json",
            _MAX_JOURNAL_BYTES,
        )
        payload = json.loads(raw_journal.decode("utf-8"))
        _validate_journal(payload, transaction_id)
        if (
            payload.get("status") != journal_status
            or payload.get("source_root_sha256") != source_root_sha256
            or payload.get("source_fingerprint") != source_fingerprint
        ):
            raise WorkspaceSecurityError("Workspace cleanup binding is invalid.")
        marker_payload = {
            **marker_binding,
            "transaction_identity": _directory_identity(transaction),
            "journal_sha256": _sha256_bytes(raw_journal),
            "committed_fingerprint": payload.get("committed_fingerprint"),
        }
        _ensure_cleanup_marker(
            marker,
            marker_payload,
            fault_after_step=fault_after_step,
        )
        _raise_cleanup_fault(fault_after_step, "after_marker")
        _rename_no_replace(transaction, cleanup)
        _raise_cleanup_fault(fault_after_step, "after_rename")
    elif _entry_exists(marker):
        _validate_cleanup_marker(marker, marker_binding)
        _unlink_cleanup_marker(marker)
        return
    else:
        return
    _raise_cleanup_fault(fault_after_step, "before_rmtree")
    _durable_rmtree(cleanup, state)
    _raise_cleanup_fault(fault_after_step, "after_rmtree")
    _validate_cleanup_marker(marker, marker_binding)
    _unlink_cleanup_marker(marker)


def _ensure_cleanup_marker(
    path: Path,
    payload: dict[str, object],
    *,
    fault_after_step: str | None,
) -> None:
    if _entry_exists(path):
        _validate_cleanup_marker(path, payload)
        return
    data = _cleanup_marker_bytes(payload)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    temp_identity: dict[str, int] | None = None
    try:
        descriptor = _open_binary_new(temp, 0o600)
        metadata = os.fstat(descriptor)
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise WorkspaceSecurityError(
                "Workspace cleanup marker temporary file is invalid."
            )
        temp_identity = _identity_from_metadata(metadata)
        fchmod = getattr(os, "fchmod", None)
        if callable(fchmod):
            fchmod(descriptor, 0o600)
        first_length = max(1, len(data) // 2)
        _write_descriptor_bytes(descriptor, data[:first_length])
        os.fsync(descriptor)
        _raise_cleanup_fault(fault_after_step, "during_marker_write")
        _write_descriptor_bytes(descriptor, data[first_length:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if not callable(fchmod):
            if os.name != "nt":  # pragma: no cover - POSIX exposes fchmod.
                raise WorkspaceSecurityError(
                    "Secure cleanup marker creation is unavailable."
                )
            if _regular_file_identity(temp) != temp_identity:
                raise WorkspaceSecurityError(
                    "Workspace cleanup marker temporary identity changed."
                )
            temp.chmod(0o600)
            if _regular_file_identity(temp) != temp_identity:
                raise WorkspaceSecurityError(
                    "Workspace cleanup marker temporary identity changed."
                )
            _windows_flush_path(temp, directory=False)
        _fsync_directory(path.parent)
        _rename_no_replace(temp, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if _entry_exists(temp):
            if temp_identity is None:
                temp_identity = _regular_file_identity(temp)
            _unlink_recovery_artifact(temp, temp_identity)


def _validate_cleanup_marker(
    path: Path,
    expected: dict[str, object],
) -> dict[str, object]:
    try:
        raw, metadata = _read_regular_path_nofollow(path, 4096)
    except (OSError, WorkspaceSecurityError) as exc:
        raise WorkspaceSecurityError(
            "Workspace cleanup marker is unavailable."
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceSecurityError("Workspace cleanup marker is invalid.") from exc
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise WorkspaceSecurityError("Workspace cleanup marker binding is invalid.")
    if (
        not _private_artifact_mode_is_valid(metadata.st_mode)
        or _cleanup_marker_bytes(payload) != raw
    ):
        raise WorkspaceSecurityError("Workspace cleanup marker encoding is invalid.")
    identity = payload.get("transaction_identity")
    expected_keys = {
        *expected,
        "transaction_identity",
        "journal_sha256",
        "committed_fingerprint",
    }
    if set(payload) != expected_keys or not isinstance(identity, dict):
        raise WorkspaceSecurityError("Workspace cleanup marker identity is invalid.")
    _recorded_file_identity(identity)
    journal_sha256 = payload.get("journal_sha256")
    committed_fingerprint = payload.get("committed_fingerprint")
    if re.fullmatch(r"[0-9a-f]{64}", str(journal_sha256)) is None:
        raise WorkspaceSecurityError("Workspace cleanup marker digest is invalid.")
    if expected.get("journal_status") == "committed":
        if re.fullmatch(r"[0-9a-f]{64}", str(committed_fingerprint)) is None:
            raise WorkspaceSecurityError(
                "Workspace cleanup committed binding is invalid."
            )
    elif committed_fingerprint is not None:
        raise WorkspaceSecurityError("Workspace cleanup recovered binding is invalid.")
    return payload


def _cleanup_marker_bytes(payload: dict[str, object]) -> bytes:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(data) > 4096:
        raise WorkspaceSecurityError("Workspace cleanup marker exceeds its bound.")
    return data


def _unlink_cleanup_marker(path: Path) -> None:
    identity = _regular_file_identity(path)
    _unlink_recovery_artifact(path, identity)


def _raise_cleanup_fault(fault_after_step: str | None, step: str) -> None:
    if fault_after_step == step:
        raise _SimulatedTransactionCrash(f"simulated cleanup crash after {step}")


def _snapshot_once(
    root: Path,
    policy: WorkspaceScopePolicy,
    *,
    git_identity: ExecutableIdentity | None,
    repository_marker: bool,
) -> WorkspaceSnapshot:
    git_repository = _is_git_workspace(
        root,
        git_identity=git_identity,
        repository_marker=repository_marker,
    )
    if git_repository:
        raw_paths = _git(
            root,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            git_identity=git_identity,
        )
        paths = [os.fsdecode(item) for item in raw_paths.split(b"\0") if item]
        index_entries = _git(
            root,
            "ls-files",
            "--stage",
            "-z",
            git_identity=git_identity,
        )
        index = _git(
            root,
            "ls-files",
            "--stage",
            "--debug",
            "-z",
            git_identity=git_identity,
        )
        tracked_paths = _tracked_paths_from_index(index_entries)
        head_raw = _git_optional(
            root,
            "rev-parse",
            "--verify",
            "--quiet",
            "HEAD",
            git_identity=git_identity,
        )
        head = head_raw.decode("ascii").strip().lower() if head_raw else "unborn"
    else:
        paths = []
        for directory, directories, files in os.walk(
            root, topdown=True, followlinks=False
        ):
            _assert_directory_entry(Path(directory))
            safe_directories: list[str] = []
            for name in sorted(directories):
                child = Path(directory) / name
                _assert_directory_entry(child)
                if name != ".git":
                    safe_directories.append(name)
            directories[:] = safe_directories
            relative_dir = Path(directory).relative_to(root)
            paths.extend((relative_dir / name).as_posix() for name in sorted(files))
        index = b""
        head = ""
        tracked_paths = ()
    directions = {item.path: item.direction for item in policy.ignored_paths}
    for rule in policy.ignored_paths:
        if rule.path not in paths:
            paths.append(rule.path)
    files = _manifest_files(root, paths, policy, directions=directions)
    manifest = _canonical_json([item.payload() for item in files])
    index_sha256 = _sha256_bytes(index)
    manifest_sha256 = _sha256_text(manifest)
    status_sha256 = _sha256_text(
        _canonical_json(
            {
                "derivation": "head-index-manifest/v1",
                "head_sha": head,
                "index_sha256": index_sha256,
                "manifest_sha256": manifest_sha256,
            }
        )
    )
    fields = {
        "root_sha256": _sha256_text(str(root)),
        "git_repository": git_repository,
        "head_sha": head,
        "index_sha256": index_sha256,
        "status_sha256": status_sha256,
        "manifest_sha256": manifest_sha256,
    }
    return WorkspaceSnapshot(
        root=str(root),
        git_repository=git_repository,
        head_sha=head,
        index_sha256=str(fields["index_sha256"]),
        status_sha256=str(fields["status_sha256"]),
        manifest_sha256=str(fields["manifest_sha256"]),
        fingerprint=_sha256_text(_canonical_json(fields)),
        files=files,
        tracked_paths=tracked_paths,
        total_bytes=sum(item.size for item in files),
    )


def _tracked_paths_from_index(index: bytes) -> tuple[str, ...]:
    paths: set[str] = set()
    for record in (item for item in index.split(b"\0") if item):
        header, separator, raw_path = record.partition(b"\t")
        if (
            not separator
            or not raw_path
            or re.fullmatch(rb"[0-7]{6} [0-9a-fA-F]{40,64} [0-3]", header) is None
        ):
            raise WorkspaceSecurityError("Git index attestation is malformed.")
        paths.add(_safe_relative(os.fsdecode(raw_path)))
    return tuple(sorted(paths))


def _manifest_files(
    root: Path,
    paths: Sequence[str],
    policy: WorkspaceScopePolicy,
    *,
    directions: dict[str, str],
) -> tuple[WorkspaceFile, ...]:
    clean_paths = sorted({_safe_relative(item) for item in paths})
    folded = [item.casefold() for item in clean_paths]
    if len(folded) != len(set(folded)):
        raise WorkspaceSecurityError("Workspace contains case-colliding paths.")
    if len(clean_paths) > policy.max_files:
        raise WorkspaceSecurityError("Workspace file-count bound was exceeded.")
    result: list[WorkspaceFile] = []
    total = 0
    for relative in clean_paths:
        try:
            metadata, digest = _inspect_regular_file(root, relative, policy)
        except FileNotFoundError:
            result.append(
                WorkspaceFile(
                    path=relative,
                    kind="missing",
                    sha256=_sha256_bytes(b""),
                    size=0,
                    mode=0,
                    direction=directions.get(relative, "round_trip"),
                )
            )
            continue
        size = int(metadata.st_size)
        total += size
        if total > policy.max_total_bytes:
            raise WorkspaceSecurityError("Workspace total byte bound was exceeded.")
        result.append(
            WorkspaceFile(
                path=relative,
                kind="file",
                sha256=digest,
                size=size,
                mode=stat.S_IMODE(metadata.st_mode),
                direction=directions.get(relative, "round_trip"),
            )
        )
    return tuple(result)


def _initialize_synthetic_repository(
    root: Path,
    *,
    identity: GitIdentity,
) -> None:
    git_identity = _resolve_trusted_git_identity(required=True)
    if git_identity is None:  # Defensive: required=True must either resolve or raise.
        raise WorkspaceSecurityError("Trusted Git executable is unavailable.")
    _run_git(
        ("init", "-q", "--template="),
        root,
        commit_identity=None,
        git_identity=git_identity,
    )
    _install_trusted_diff_attributes(root)
    _run_git(
        ("add", "-f", "--all"),
        root,
        commit_identity=None,
        git_identity=git_identity,
    )
    _run_git(
        ("commit", "-q", "--allow-empty", "-m", "materialized baseline"),
        root,
        commit_identity=identity,
        git_identity=git_identity,
    )
    remotes = _run_git(
        ("remote",),
        root,
        commit_identity=None,
        capture=True,
        git_identity=git_identity,
    ).splitlines()
    if remotes:
        raise WorkspaceSecurityError(
            "Synthetic repository unexpectedly contains remotes."
        )


def _install_trusted_diff_attributes(root: Path) -> None:
    """Pin diff classification above candidate-controlled attributes."""

    git_directory = root / ".git"
    _assert_directory_entry(git_directory)
    info_directory = git_directory / "info"
    try:
        info_directory.mkdir(mode=0o700)
    except FileExistsError:
        _assert_directory_entry(info_directory)
    else:
        _fsync_directory(git_directory)
    attributes = info_directory / "attributes"
    _durable_write_new(attributes, _TRUSTED_DIFF_ATTRIBUTES, 0o600)


def _assert_current_entry(path: Path, expected: WorkspaceFile | None) -> None:
    if expected is None or expected.kind == "missing":
        try:
            path.lstat()
        except FileNotFoundError:
            return
        else:
            raise WorkspaceSecurityError("Workspace path failed compare-and-swap.")
    try:
        data, metadata = _read_regular_path_nofollow(path, expected.size)
    except (OSError, WorkspaceSecurityError) as exc:
        raise WorkspaceSecurityError("Workspace path failed compare-and-swap.") from exc
    if (
        int(metadata.st_size) != expected.size
        or stat.S_IMODE(metadata.st_mode) != expected.mode
        or _sha256_bytes(data) != expected.sha256
    ):
        raise WorkspaceSecurityError("Workspace path failed compare-and-swap.")


def _stage_attested_candidate(
    candidate_root: Path,
    candidate_files: Sequence[WorkspaceFile],
    changes: Sequence[WorkspaceChange],
    policy: WorkspaceScopePolicy,
    staged_dir: Path,
    records: list[dict[str, object]],
) -> None:
    expected = tuple(sorted(candidate_files))
    first = _with_expected_missing_files(
        snapshot_materialized(candidate_root, policy),
        expected,
    )
    second = _with_expected_missing_files(
        snapshot_materialized(candidate_root, policy),
        expected,
    )
    if first != second or second != expected:
        raise WorkspaceSecurityError(
            "Candidate workspace changed or does not match its attested manifest."
        )
    for index, change in enumerate(changes):
        if change.after is None or change.after.kind == "missing":
            continue
        data = _read_attested_file(candidate_root, change.after, policy)
        staged = staged_dir / f"{index:08d}.bin"
        _durable_write_new(staged, data, 0o600)
        _fsync_directory(staged_dir)
        records[index]["staged"] = staged.name
        records[index]["staged_sha256"] = _sha256_bytes(data)


def _preflight_transaction_artifacts(
    root: Path,
    changes: Sequence[WorkspaceChange],
    records: Sequence[dict[str, object]],
    *,
    staged_dir: Path,
) -> None:
    for change, record in zip(changes, records, strict=True):
        target = root / change.path
        raw_names = (
            (record.get("quarantine"), _SAFE_QUARANTINE_NAME),
            (record.get("install"), _SAFE_INSTALL_NAME),
            (record.get("rollback"), _SAFE_ROLLBACK_NAME),
            (record.get("restore"), _SAFE_RESTORE_NAME),
        )
        paths: list[Path] = [target]
        for raw_name, validator in raw_names:
            if raw_name is None:
                continue
            if not isinstance(raw_name, str) or validator.fullmatch(raw_name) is None:
                raise WorkspaceSecurityError(
                    "Workspace transaction artifact path is malformed."
                )
            paths.append(target.with_name(raw_name))
        _assert_planned_path_ancestry(root, change.path)
        staged_name = record.get("staged")
        if staged_name is not None:
            if change.after is None or change.after.kind == "missing":
                raise WorkspaceSecurityError("Workspace staged artifact is unexpected.")
            if (
                not isinstance(staged_name, str)
                or _SAFE_BACKUP_NAME.fullmatch(staged_name) is None
            ):
                raise WorkspaceSecurityError(
                    "Workspace staged artifact path is malformed."
                )
            _read_staged_file(staged_dir / staged_name, change.after)
        for artifact in paths[1:]:
            if _entry_exists(artifact):
                raise WorkspaceSecurityError(
                    "Workspace transaction artifact path already exists."
                )


def _assert_planned_path_ancestry(root: Path, relative: str) -> Path:
    current = root
    _assert_directory_entry(current)
    for part in Path(_safe_relative(relative)).parts[:-1]:
        candidate = current / part
        try:
            candidate.lstat()
        except FileNotFoundError:
            return current
        _assert_directory_entry(candidate)
        current = candidate
    return current


def _prepare_install_artifacts(
    root: Path,
    changes: Sequence[WorkspaceChange],
    records: Sequence[dict[str, object]],
    *,
    staged_dir: Path,
    journal_payload: dict[str, object],
    journal: Path,
    fault_after_step: str | None,
) -> None:
    for index, (change, record) in enumerate(zip(changes, records, strict=True)):
        if change.after is None or change.after.kind == "missing":
            continue
        target = root / change.path
        install = _journal_install(
            target,
            record,
            transaction_id=str(journal_payload.get("transaction_id", "")),
            index=index,
        )
        staged_name = record.get("staged")
        if not isinstance(staged_name, str):
            raise WorkspaceSecurityError("Workspace staged artifact path is malformed.")
        staged = staged_dir / staged_name
        data = _read_staged_file(staged, change.after)
        _assert_recovery_path_ancestry(root, change.path, target, install)
        record["install_phase"] = "create_intent"
        _write_journal(journal, journal_payload)
        identity = _create_journaled_artifact(
            install,
            data,
            record=record,
            identity_key="installed_identity",
            phase_key="install_phase",
            journal_payload=journal_payload,
            journal=journal,
            before_identity_fault=lambda: _raise_install_fault(
                fault_after_step,
                "before_identity_persist",
            ),
            during_write_fault=lambda: _raise_install_fault(
                fault_after_step,
                "during_temp_write",
            ),
        )
        _assert_install_artifact(install, change.after, identity)
        record["install_phase"] = "created"
        _write_journal(journal, journal_payload)
        _raise_install_fault(fault_after_step, "after_temp_creation")


def _create_journaled_artifact(
    path: Path,
    data: bytes,
    *,
    record: dict[str, object],
    identity_key: str,
    phase_key: str,
    journal_payload: dict[str, object],
    journal: Path,
    before_identity_fault: Callable[[], None],
    during_write_fault: Callable[[], None],
) -> dict[str, int]:
    descriptor = _open_binary_new(path, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise WorkspaceSecurityError(
                "Workspace transaction artifact is not a regular file."
            )
        identity = _identity_from_metadata(metadata)
        fchmod = getattr(os, "fchmod", None)
        if callable(fchmod):
            fchmod(descriptor, 0o600)
        elif os.name == "nt":
            if _regular_file_identity(path) != identity:
                raise WorkspaceSecurityError(
                    "Workspace transaction artifact identity changed at creation."
                )
            path.chmod(0o600)
            if _regular_file_identity(path) != identity:
                raise WorkspaceSecurityError(
                    "Workspace transaction artifact identity changed during setup."
                )
            _windows_flush_path(path, directory=False)
        else:  # pragma: no cover - supported POSIX runtimes expose fchmod.
            raise WorkspaceSecurityError(
                "Secure transaction artifact creation is unavailable."
            )
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        before_identity_fault()
        record[identity_key] = identity
        record[phase_key] = "write_intent"
        _write_journal(journal, journal_payload)

        first_length = max(1, len(data) // 2) if data else 0
        _write_descriptor_bytes(descriptor, data[:first_length])
        os.fsync(descriptor)
        during_write_fault()
        _write_descriptor_bytes(descriptor, data[first_length:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _regular_file_identity(path) != identity:
        raise WorkspaceSecurityError(
            "Workspace transaction artifact identity changed during write."
        )
    return identity


def _write_descriptor_bytes(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("Workspace transaction artifact write made no progress.")
        offset += written


def _apply_change_fail_closed(
    *,
    source: Path,
    change: WorkspaceChange,
    staged: Path | None,
    quarantine_name: object,
    install_name: object,
    created_directories: object,
    record: dict[str, object],
    journal_payload: dict[str, object],
    journal: Path,
    fault_after_install_step: str | None,
    test_hook: Callable[[Path], None] | None,
) -> dict[str, int] | None:
    _validate_created_directories(created_directories)
    target = source / change.path
    _assert_safe_relative_ancestry(source, change.path)
    quarantine: Path | None = None
    if change.before is not None and change.before.kind != "missing":
        if not isinstance(quarantine_name, str) or not _SAFE_QUARANTINE_NAME.fullmatch(
            quarantine_name
        ):
            raise WorkspaceSecurityError(
                "Workspace transaction quarantine name is malformed."
            )
        quarantine = target.with_name(quarantine_name)
        try:
            quarantine.lstat()
        except FileNotFoundError:
            pass
        else:
            raise WorkspaceSecurityError(
                "Workspace transaction quarantine path already exists."
            )
        _rename_no_replace(target, quarantine)
        try:
            _assert_current_entry(quarantine, change.before)
        except WorkspaceSecurityError:
            _restore_quarantine_no_replace(quarantine, target)
            raise WorkspaceSecurityError(
                "Workspace path changed during compare-and-swap acquisition."
            ) from None
    else:
        _assert_current_entry(target, change.before)

    if test_hook is not None:
        test_hook(target)

    installed_identity: dict[str, int] | None = None
    if change.after is not None and change.after.kind != "missing":
        if staged is None:
            raise WorkspaceSecurityError("Candidate staging artifact is missing.")
        data = _read_staged_file(staged, change.after)
        installed_identity = _install_bytes_no_replace(
            data,
            target,
            change.after.mode,
            root=source,
            relative=change.path,
            install_name=install_name,
            expected=change.after,
            record=record,
            journal_payload=journal_payload,
            journal=journal,
            fault_after_step=fault_after_install_step,
        )
    else:
        _assert_current_entry(target, None)

    _assert_current_entry(target, change.after)
    if installed_identity is not None:
        if _regular_file_identity(target) != installed_identity:
            raise WorkspaceSecurityError(
                "Workspace installation ownership changed before commit."
            )
    if quarantine is not None:
        _assert_current_entry(quarantine, change.before)
        quarantine_identity = _regular_file_identity(quarantine)
        _unlink_recovery_artifact(quarantine, quarantine_identity)
    return installed_identity


def _restore_quarantine_no_replace(quarantine: Path, target: Path) -> None:
    quarantine_identity = _regular_file_identity(quarantine)
    try:
        os.link(quarantine, target)
    except FileExistsError as exc:
        raise WorkspaceSecurityError(
            "Concurrent workspace value was preserved; journal recovery is required."
        ) from exc
    if _regular_file_identity(target) != quarantine_identity:
        raise WorkspaceSecurityError(
            "Workspace quarantine ownership changed during restoration."
        )
    _unlink_recovery_artifact(quarantine, quarantine_identity)


def _plan_created_directories(
    source: Path,
    changes: Sequence[WorkspaceChange],
) -> tuple[tuple[str, ...], ...]:
    claimed: set[str] = set()
    result: list[tuple[str, ...]] = []
    for change in changes:
        planned: list[str] = []
        if change.after is not None and change.after.kind != "missing":
            parts = Path(change.path).parts[:-1]
            for depth in range(1, len(parts) + 1):
                relative = Path(*parts[:depth]).as_posix()
                if relative in claimed:
                    continue
                target = source / relative
                if _entry_exists(target):
                    _assert_directory_entry(target)
                    continue
                claimed.add(relative)
                planned.append(relative)
        result.append(tuple(planned))
    return tuple(result)


def _create_planned_directories(source: Path, raw_directories: object) -> None:
    if not isinstance(raw_directories, list):
        raise WorkspaceSecurityError(
            "Workspace transaction directory plan is malformed."
        )
    for raw in raw_directories:
        if not isinstance(raw, str):
            raise WorkspaceSecurityError(
                "Workspace transaction directory plan is malformed."
            )
        relative = _safe_relative(raw)
        directory = source / relative
        parent_relative = Path(relative).parent.as_posix()
        if parent_relative != ".":
            _assert_safe_relative_ancestry(source, f"{parent_relative}/entry")
        else:
            _assert_directory_entry(source)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise WorkspaceSecurityError(
                "Workspace directory changed during no-replace creation."
            ) from exc
        _fsync_directory(directory.parent)
        _assert_directory_entry(directory)


def _create_all_planned_directories(
    source: Path,
    records: Sequence[dict[str, object]],
) -> None:
    for record in records:
        _create_planned_directories(source, record.get("created_directories"))


def _validate_created_directories(raw_directories: object) -> None:
    if not isinstance(raw_directories, list) or not all(
        isinstance(item, str) for item in raw_directories
    ):
        raise WorkspaceSecurityError(
            "Workspace transaction directory plan is malformed."
        )
    for raw in raw_directories:
        _safe_relative(raw)


def _install_bytes_no_replace(
    data: bytes,
    target: Path,
    mode: int,
    *,
    root: Path,
    relative: str,
    install_name: object,
    expected: WorkspaceFile,
    record: dict[str, object],
    journal_payload: dict[str, object],
    journal: Path,
    fault_after_step: str | None,
) -> dict[str, int]:
    if (
        not isinstance(install_name, str)
        or _SAFE_INSTALL_NAME.fullmatch(install_name) is None
    ):
        raise WorkspaceSecurityError("Workspace install path is malformed.")
    temp = target.with_name(install_name)
    _assert_recovery_path_ancestry(root, relative, target, temp)
    installed_identity = _recorded_file_identity(record.get("installed_identity"))
    if installed_identity is None:
        raise WorkspaceSecurityError("Workspace install identity is unavailable.")
    _assert_install_artifact(temp, expected, installed_identity)
    record["install_phase"] = "link_intent"
    _write_journal(journal, journal_payload)
    _assert_recovery_path_ancestry(root, relative, target, temp)
    try:
        os.link(temp, target)
    except FileExistsError as exc:
        raise WorkspaceSecurityError(
            "Workspace path changed during no-replace installation."
        ) from exc
    _fsync_directory(target.parent)
    if _regular_file_identity(target) != installed_identity:
        raise WorkspaceSecurityError(
            "Workspace no-replace installation lost object ownership."
        )
    record["install_phase"] = "linked"
    _write_journal(journal, journal_payload)
    _raise_install_fault(fault_after_step, "after_link")
    _assert_recovery_path_ancestry(root, relative, target, temp)
    _unlink_recovery_artifact(temp, installed_identity)
    record["install_phase"] = "temp_unlinked"
    _write_journal(journal, journal_payload)
    _raise_install_fault(fault_after_step, "after_temp_unlink")
    _set_file_mode_durable(target, mode, installed_identity)
    record["install_phase"] = "mode_applied"
    _write_journal(journal, journal_payload)
    _raise_install_fault(fault_after_step, "after_mode_apply")
    _assert_current_entry(target, expected)
    return installed_identity


def _assert_recovery_path_ancestry(
    root: Path,
    relative: str,
    *paths: Path,
) -> None:
    safe_relative = _safe_relative(relative)
    _assert_safe_relative_ancestry(root, safe_relative)
    expected_parent = (root / safe_relative).parent
    _assert_directory_entry(expected_parent)
    for path in paths:
        if path.parent != expected_parent:
            raise WorkspaceSecurityError(
                "Workspace recovery artifact escaped its target directory."
            )


def _assert_install_artifact(
    path: Path,
    expected: WorkspaceFile,
    installed_identity: dict[str, int],
) -> None:
    data, metadata = _read_regular_path_nofollow(path, expected.size)
    if (
        int(metadata.st_size) != expected.size
        or not (
            _private_artifact_mode_is_valid(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) == expected.mode
        )
        or _sha256_bytes(data) != expected.sha256
        or _regular_file_identity(path) != installed_identity
    ):
        raise WorkspaceSecurityError("Workspace install artifact changed unexpectedly.")


def _assert_install_content(path: Path, expected: WorkspaceFile) -> None:
    data, metadata = _read_regular_path_nofollow(path, expected.size)
    if (
        int(metadata.st_size) != expected.size
        or not _private_artifact_mode_is_valid(metadata.st_mode)
        or _sha256_bytes(data) != expected.sha256
    ):
        raise WorkspaceSecurityError("Workspace install artifact changed unexpectedly.")


def _assert_owned_partial_artifact(
    path: Path,
    expected: WorkspaceFile,
    expected_identity: dict[str, int],
) -> None:
    data, metadata = _read_regular_path_nofollow(path, expected.size)
    if (
        len(data) > expected.size
        or not _private_artifact_mode_is_valid(metadata.st_mode)
        or _regular_file_identity(path) != expected_identity
    ):
        raise WorkspaceSecurityError(
            "Workspace partial transaction artifact changed unexpectedly."
        )


def _entry_matches_installed(
    path: Path,
    expected: WorkspaceFile | None,
    installed_identity: dict[str, int],
) -> bool:
    if expected is None or expected.kind == "missing":
        return False
    try:
        _assert_install_artifact(path, expected, installed_identity)
    except (OSError, WorkspaceSecurityError):
        return False
    return True


def _set_file_mode_durable(
    path: Path,
    mode: int,
    expected_identity: dict[str, int],
) -> None:
    descriptor = os.open(path, _SECURE_OPEN_FLAGS)
    try:
        metadata = os.fstat(descriptor)
        if (
            _is_link_or_reparse(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or _identity_from_metadata(metadata) != expected_identity
        ):
            raise WorkspaceSecurityError(
                "Workspace file identity changed before mode application."
            )
        fchmod = getattr(os, "fchmod", None)
        if os.name == "nt":
            writable_descriptor = _windows_open_fd(
                path,
                directory=False,
                writable=True,
            )
            try:
                writable_metadata = os.fstat(writable_descriptor)
                if (
                    _is_link_or_reparse(writable_metadata)
                    or not stat.S_ISREG(writable_metadata.st_mode)
                    or _identity_from_metadata(writable_metadata)
                    != expected_identity
                    or _regular_file_identity(path) != expected_identity
                ):
                    raise WorkspaceSecurityError(
                        "Workspace file identity changed before mode application."
                    )
                if callable(fchmod):
                    fchmod(writable_descriptor, mode)
                else:
                    path.chmod(mode)
                if (
                    _identity_from_metadata(os.fstat(writable_descriptor))
                    != expected_identity
                    or _regular_file_identity(path) != expected_identity
                ):
                    raise WorkspaceSecurityError(
                        "Workspace file identity changed during mode application."
                    )
                os.fsync(writable_descriptor)
            finally:
                os.close(writable_descriptor)
        elif callable(fchmod):
            fchmod(descriptor, mode)
            os.fsync(descriptor)
        else:  # pragma: no cover - supported POSIX runtimes expose fchmod.
            raise WorkspaceSecurityError("Secure file mode application is unavailable.")
    finally:
        os.close(descriptor)
    if _regular_file_identity(path) != expected_identity:
        raise WorkspaceSecurityError(
            "Workspace file identity changed after mode application."
        )
    _fsync_directory(path.parent)


def _unlink_recovery_artifact(
    path: Path,
    expected_identity: dict[str, int],
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _SECURE_OPEN_FLAGS)
        metadata = os.fstat(descriptor)
        if (
            _is_link_or_reparse(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or _identity_from_metadata(metadata) != expected_identity
        ):
            raise WorkspaceSecurityError(
                "Workspace recovery artifact identity changed before cleanup."
            )
        fchmod = getattr(os, "fchmod", None)
        writable_mode = stat.S_IMODE(metadata.st_mode) | stat.S_IWUSR
        if callable(fchmod):
            fchmod(descriptor, writable_mode)
        elif os.name == "nt":
            if _regular_file_identity(path) != expected_identity:
                raise WorkspaceSecurityError(
                    "Workspace recovery artifact identity changed before cleanup."
                )
            path.chmod(writable_mode)
            if _regular_file_identity(path) != expected_identity:
                raise WorkspaceSecurityError(
                    "Workspace recovery artifact identity changed during cleanup."
                )
            _windows_flush_path(path, directory=False)
        else:  # pragma: no cover - supported POSIX runtimes expose fchmod.
            raise WorkspaceSecurityError(
                "Secure recovery artifact cleanup is unavailable."
            )
        os.close(descriptor)
        descriptor = None
        if _regular_file_identity(path) != expected_identity:
            raise WorkspaceSecurityError(
                "Workspace recovery artifact identity changed before unlink."
            )
        path.unlink()
    except FileNotFoundError:
        return
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _raise_install_fault(fault_after_step: str | None, step: str) -> None:
    if fault_after_step == step:
        raise _SimulatedTransactionCrash(f"simulated install crash after {step}")


def _read_staged_file(path: Path, expected: WorkspaceFile) -> bytes:
    data, metadata = _read_regular_path_nofollow(path, expected.size)
    if (
        int(metadata.st_size) != expected.size
        or not _private_artifact_mode_is_valid(metadata.st_mode)
        or _sha256_bytes(data) != expected.sha256
    ):
        raise WorkspaceSecurityError("Candidate staging artifact changed unexpectedly.")
    return data


def _rollback_from_journal(
    root: Path,
    payload: dict[str, object],
    backup_dir: Path,
    *,
    journal: Path,
    _fault_after_step: str | None = None,
) -> None:
    raw_changes = payload.get("changes")
    if not isinstance(raw_changes, list):
        raise WorkspaceSecurityError("Workspace recovery journal is malformed.")
    transaction_id = payload.get("transaction_id")
    if not isinstance(transaction_id, str):
        raise WorkspaceSecurityError("Workspace recovery transaction is malformed.")
    _validate_transaction_id(transaction_id)
    for index in reversed(range(len(raw_changes))):
        raw = raw_changes[index]
        if not isinstance(raw, dict):
            raise WorkspaceSecurityError("Workspace recovery record is malformed.")
        status_value = raw.get("status")
        if status_value == "rolled_back":
            continue
        if status_value not in {"pending", "backed_up", "mutating", "applied"}:
            raise WorkspaceSecurityError("Workspace recovery status is invalid.")
        change = _change_from_payload(raw)
        target = root / change.path
        install = _journal_install(
            target,
            raw,
            transaction_id=transaction_id,
            index=index,
        )
        rollback = _journal_rollback(
            target,
            raw,
            transaction_id=transaction_id,
            index=index,
        )
        restore = _journal_restore(
            target,
            raw,
            transaction_id=transaction_id,
            index=index,
        )
        _assert_recovery_path_ancestry(
            root,
            change.path,
            target,
            install,
            rollback,
            restore,
        )
        if raw.get("rollback_phase") is None:
            raw["rollback_phase"] = "pending"
        _rollback_phase(raw)
        _write_journal(journal, payload)
        quarantine = _journal_quarantine(target, raw.get("quarantine"))
        if quarantine is not None and not _entry_matches(quarantine, change.before):
            raise WorkspaceSecurityError(
                "Workspace recovery quarantine does not match the recorded value."
            )
        before_exists = change.before is not None and change.before.kind != "missing"
        after_exists = change.after is not None and change.after.kind != "missing"
        restore_identity = (
            _recorded_file_identity(raw.get("restore_identity"))
            if before_exists
            else None
        )
        installed_identity = (
            _recorded_file_identity(raw.get("installed_identity"))
            if after_exists
            else None
        )
        if after_exists and _entry_exists(install):
            assert change.after is not None
            install_prewrite_state = (
                status_value == "pending"
                and raw.get("install_phase")
                in {"create_intent", "write_intent"}
                and _entry_matches(target, change.before)
                and not _entry_exists(rollback)
                and not _entry_exists(restore)
                and quarantine is None
            )
            if installed_identity is None:
                if not install_prewrite_state:
                    raise WorkspaceSecurityError(
                        "Workspace recovery install identity is unavailable."
                    )
                installed_identity = _regular_file_identity(install)
                raw["installed_identity"] = installed_identity
                try:
                    _assert_install_content(install, change.after)
                except WorkspaceSecurityError:
                    _assert_owned_partial_artifact(
                        install,
                        change.after,
                        installed_identity,
                    )
                    raw["install_phase"] = "write_intent"
                else:
                    raw["install_phase"] = "created"
                _write_journal(journal, payload)
            if raw.get("install_phase") == "write_intent":
                if not install_prewrite_state:
                    raise WorkspaceSecurityError(
                        "Workspace recovery partial install state is invalid."
                    )
                _assert_owned_partial_artifact(
                    install,
                    change.after,
                    installed_identity,
                )
                _unlink_recovery_artifact(install, installed_identity)
                raw["installed_identity"] = None
                raw["install_phase"] = "recovered_partial_unlinked"
                installed_identity = None
                _write_journal(journal, payload)
            else:
                _assert_install_artifact(install, change.after, installed_identity)
        elif not after_exists and _entry_exists(install):
            raise WorkspaceSecurityError(
                "Workspace recovery install artifact is unexpected."
            )
        if before_exists and _entry_exists(restore):
            assert change.before is not None
            restore_creation_state = (
                _rollback_phase(raw) == "restore_intent"
                and not _entry_exists(target)
                and not _entry_exists(install)
            )
            if restore_creation_state and _entry_exists(rollback):
                if not after_exists or installed_identity is None:
                    restore_creation_state = False
                else:
                    _assert_owned_rollback(
                        rollback,
                        change.after,
                        installed_identity,
                    )
            elif restore_creation_state and after_exists:
                restore_creation_state = (
                    status_value == "mutating" and quarantine is not None
                )
            elif restore_creation_state:
                restore_creation_state = (
                    not _entry_exists(rollback)
                    and (
                        quarantine is None
                        or status_value == "mutating"
                    )
                )
            if restore_identity is None:
                if not restore_creation_state:
                    raise WorkspaceSecurityError(
                        "Workspace recovery restore identity is unavailable."
                    )
                restore_identity = _regular_file_identity(restore)
                raw["restore_identity"] = restore_identity
                try:
                    _assert_install_content(restore, change.before)
                except WorkspaceSecurityError:
                    _assert_owned_partial_artifact(
                        restore,
                        change.before,
                        restore_identity,
                    )
                    raw["restore_phase"] = "write_intent"
                else:
                    raw["restore_phase"] = "created"
                _write_journal(journal, payload)
            if raw.get("restore_phase") == "write_intent":
                if not restore_creation_state:
                    raise WorkspaceSecurityError(
                        "Workspace recovery partial restore state is invalid."
                    )
                _assert_owned_partial_artifact(
                    restore,
                    change.before,
                    restore_identity,
                )
                _unlink_recovery_artifact(restore, restore_identity)
                raw["restore_identity"] = None
                raw["restore_phase"] = "recovered_partial_unlinked"
                restore_identity = None
                _write_journal(journal, payload)
            else:
                _assert_install_artifact(restore, change.before, restore_identity)
        target_is_before = _entry_matches(target, change.before)
        target_is_after = _entry_matches(target, change.after)
        target_is_installed = (
            after_exists
            and installed_identity is not None
            and _entry_matches_installed(
                target,
                change.after,
                installed_identity,
            )
        )
        target_is_restoring = (
            before_exists
            and restore_identity is not None
            and _entry_matches_installed(
                target,
                change.before,
                restore_identity,
            )
        )
        if (
            after_exists
            and target_is_after
            and installed_identity is not None
            and not target_is_installed
        ):
            raise WorkspaceSecurityError(
                "Workspace recovery cannot prove ownership of the installed value."
            )
        if status_value in {"pending", "backed_up"}:
            if (
                not target_is_before
                or target_is_installed
                or _entry_exists(rollback)
                or _entry_exists(restore)
                or quarantine is not None
            ):
                raise WorkspaceSecurityError(
                    "Workspace recovery found mutation before its durable intent."
                )
            if _entry_exists(install):
                if (
                    not after_exists
                    or installed_identity is None
                    or change.after is None
                ):
                    raise WorkspaceSecurityError(
                        "Workspace recovery install artifact is unexpected."
                    )
                _assert_recovery_path_ancestry(root, change.path, target, install)
                _assert_install_artifact(install, change.after, installed_identity)
                _unlink_recovery_artifact(install, installed_identity)
            raw["install_phase"] = "recovered_unlinked"
            raw["rollback_phase"] = "complete"
            raw["status"] = "rolled_back"
            _write_journal(journal, payload)
            continue
        if (
            after_exists
            and installed_identity is None
            and (
                target_is_after
                or target_is_installed
                or _entry_exists(rollback)
                or status_value == "applied"
            )
        ):
            raise WorkspaceSecurityError(
                "Workspace recovery installed identity is unavailable."
            )
        if _entry_exists(rollback):
            _assert_owned_rollback(rollback, change.after, installed_identity)
        if after_exists and target_is_installed:
            if _entry_exists(rollback):
                raise WorkspaceSecurityError(
                    "Workspace recovery rollback path is already occupied."
                )
            if _rollback_phase_index(raw) >= _rollback_phase_number("detached"):
                raise WorkspaceSecurityError(
                    "Workspace recovery target regressed after detachment."
                )
            _advance_rollback_phase(
                raw,
                "detach_intent",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
            _assert_recovery_path_ancestry(
                root,
                change.path,
                target,
                rollback,
            )
            if _regular_file_identity(target) != installed_identity:
                raise WorkspaceSecurityError(
                    "Workspace recovery cannot prove ownership of the installed value."
                )
            _rename_no_replace(target, rollback)
            _raise_rollback_fault(_fault_after_step, "detach_action")
            _assert_owned_rollback(rollback, change.after, installed_identity)
            _advance_rollback_phase(
                raw,
                "detached",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
        elif after_exists and _entry_exists(rollback):
            if (
                _entry_exists(target)
                and not target_is_before
                and not target_is_restoring
            ):
                raise WorkspaceSecurityError(
                    "Workspace recovery found a concurrent target value."
                )
            _advance_rollback_phase(
                raw,
                "detached",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
        elif (
            after_exists
            and not before_exists
            and target_is_before
            and installed_identity is not None
            and _rollback_phase_index(raw)
            < _rollback_phase_number("rollback_unlink_intent")
        ):
            raise WorkspaceSecurityError(
                "Workspace recovery lost ownership of the installed value."
            )
        elif not target_is_before and not target_is_restoring and _entry_exists(target):
            raise WorkspaceSecurityError(
                "Workspace recovery found a value matching neither before nor after."
            )
        if _entry_exists(install):
            if not after_exists or installed_identity is None:
                raise WorkspaceSecurityError(
                    "Workspace recovery install artifact is unexpected."
                )
            assert change.after is not None
            _assert_recovery_path_ancestry(root, change.path, target, install)
            _assert_install_artifact(install, change.after, installed_identity)
            raw["install_phase"] = "unlink_intent"
            _write_journal(journal, payload)
            _raise_rollback_fault(_fault_after_step, "install_unlink_intent")
            _unlink_recovery_artifact(install, installed_identity)
            _raise_rollback_fault(_fault_after_step, "install_unlink_action")
        raw["install_phase"] = "recovered_unlinked"
        _write_journal(journal, payload)
        _raise_rollback_fault(_fault_after_step, "install_unlinked")
        if (
            before_exists
            and not target_is_before
            and _rollback_phase_index(raw) >= _rollback_phase_number("restored")
        ):
            raise WorkspaceSecurityError(
                "Workspace recovery target regressed after restoration."
            )

        restore_data: bytes | None = None
        if before_exists and not target_is_before:
            assert change.before is not None
            restore_data = _read_recovery_backup(
                raw,
                backup_dir=backup_dir,
                before=change.before,
            )
        elif not before_exists and _entry_exists(restore):
            raise WorkspaceSecurityError(
                "Workspace recovery restore path is unexpected."
            )
        if not before_exists:
            if _entry_exists(restore):
                raise WorkspaceSecurityError(
                    "Workspace recovery restore path is unexpected."
                )
            _advance_rollback_phase(
                raw,
                "restored",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
        elif not target_is_before:
            assert change.before is not None
            _advance_rollback_phase(
                raw,
                "restore_intent",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
            if not target_is_restoring:
                if _entry_exists(target):
                    raise WorkspaceSecurityError(
                        "Workspace recovery target is occupied before restoration."
                    )
                assert restore_data is not None
                if not _entry_exists(restore):
                    _assert_recovery_path_ancestry(
                        root,
                        change.path,
                        target,
                        restore,
                    )
                    restore_identity = _create_journaled_artifact(
                        restore,
                        restore_data,
                        record=raw,
                        identity_key="restore_identity",
                        phase_key="restore_phase",
                        journal_payload=payload,
                        journal=journal,
                        before_identity_fault=lambda: _raise_rollback_fault(
                            _fault_after_step,
                            "restore_before_identity_persist",
                        ),
                        during_write_fault=lambda: _raise_rollback_fault(
                            _fault_after_step,
                            "restore_during_write",
                        ),
                    )
                    _assert_install_artifact(
                        restore,
                        change.before,
                        restore_identity,
                    )
                    raw["restore_phase"] = "created"
                    _write_journal(journal, payload)
                    _raise_rollback_fault(
                        _fault_after_step,
                        "restore_prepare_action",
                    )
                if restore_identity is None:
                    raise WorkspaceSecurityError(
                        "Workspace recovery restore identity is unavailable."
                    )
                _assert_install_artifact(restore, change.before, restore_identity)
                _assert_recovery_path_ancestry(
                    root,
                    change.path,
                    target,
                    restore,
                )
                try:
                    os.link(restore, target)
                except FileExistsError as exc:
                    raise WorkspaceSecurityError(
                        "Workspace recovery target changed during restoration."
                    ) from exc
                _fsync_directory(target.parent)
                _raise_rollback_fault(_fault_after_step, "restore_action")
            if restore_identity is None or not _entry_matches_installed(
                target,
                change.before,
                restore_identity,
            ):
                raise WorkspaceSecurityError(
                    "Workspace recovery restore ownership changed."
                )
            if _entry_exists(restore):
                _assert_recovery_path_ancestry(
                    root,
                    change.path,
                    target,
                    restore,
                )
                _assert_install_artifact(restore, change.before, restore_identity)
                raw["restore_phase"] = "unlink_intent"
                _write_journal(journal, payload)
                _raise_rollback_fault(_fault_after_step, "restore_unlink_intent")
                _unlink_recovery_artifact(restore, restore_identity)
                _raise_rollback_fault(_fault_after_step, "restore_unlink_action")
            raw["restore_phase"] = "temp_unlinked"
            _write_journal(journal, payload)
            _raise_rollback_fault(_fault_after_step, "restore_unlinked")
            _assert_recovery_path_ancestry(root, change.path, target)
            _set_file_mode_durable(target, change.before.mode, restore_identity)
            _raise_rollback_fault(_fault_after_step, "restore_mode_action")
            _advance_rollback_phase(
                raw,
                "restored",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
        else:
            if _entry_exists(restore):
                if restore_identity is None or change.before is None:
                    raise WorkspaceSecurityError(
                        "Workspace recovery restore identity is unavailable."
                    )
                _assert_recovery_path_ancestry(
                    root,
                    change.path,
                    target,
                    restore,
                )
                _assert_install_artifact(restore, change.before, restore_identity)
                _unlink_recovery_artifact(restore, restore_identity)
            _advance_rollback_phase(
                raw,
                "restored",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
        _assert_recovery_path_ancestry(root, change.path, target)
        _assert_current_entry(target, change.before)

        _advance_rollback_phase(
            raw,
            "restore_unlinked",
            payload=payload,
            journal=journal,
            fault_after_step=_fault_after_step,
        )

        if _entry_exists(rollback):
            if _rollback_phase_index(raw) >= _rollback_phase_number(
                "rollback_unlinked"
            ):
                raise WorkspaceSecurityError(
                    "Workspace recovery rollback path reappeared after cleanup."
                )
            _assert_owned_rollback(rollback, change.after, installed_identity)
            _advance_rollback_phase(
                raw,
                "rollback_unlink_intent",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
            _assert_recovery_path_ancestry(root, change.path, target, rollback)
            if installed_identity is None:
                raise WorkspaceSecurityError(
                    "Workspace recovery rollback identity is unavailable."
                )
            _unlink_recovery_artifact(rollback, installed_identity)
            _raise_rollback_fault(_fault_after_step, "rollback_unlink_action")
        _advance_rollback_phase(
            raw,
            "rollback_unlinked",
            payload=payload,
            journal=journal,
            fault_after_step=_fault_after_step,
        )

        quarantine = _journal_quarantine(target, raw.get("quarantine"))
        if quarantine is not None:
            if _rollback_phase_index(raw) >= _rollback_phase_number(
                "quarantine_unlinked"
            ):
                raise WorkspaceSecurityError(
                    "Workspace recovery quarantine reappeared after cleanup."
                )
            if not _entry_matches(quarantine, change.before):
                raise WorkspaceSecurityError(
                    "Workspace recovery quarantine does not match the recorded value."
                )
            _advance_rollback_phase(
                raw,
                "quarantine_unlink_intent",
                payload=payload,
                journal=journal,
                fault_after_step=_fault_after_step,
            )
            _assert_recovery_path_ancestry(root, change.path, target, quarantine)
            quarantine_identity = _regular_file_identity(quarantine)
            _unlink_recovery_artifact(quarantine, quarantine_identity)
            _raise_rollback_fault(_fault_after_step, "quarantine_unlink_action")
        _advance_rollback_phase(
            raw,
            "quarantine_unlinked",
            payload=payload,
            journal=journal,
            fault_after_step=_fault_after_step,
        )
        _rollback_created_directories(root, raw.get("created_directories"))
        raw["status"] = "rolled_back"
        raw["rollback_phase"] = "complete"
        _write_journal(journal, payload)
        _raise_rollback_fault(_fault_after_step, "record_rolled_back")


def _journal_install(
    target: Path,
    raw: dict[str, object],
    *,
    transaction_id: str,
    index: int,
) -> Path:
    expected = f".mymoe-install-{transaction_id}-{index:08d}"
    raw_name = raw.get("install")
    if raw_name is None:
        raw["install"] = expected
        raw_name = expected
    if (
        not isinstance(raw_name, str)
        or raw_name != expected
        or _SAFE_INSTALL_NAME.fullmatch(raw_name) is None
    ):
        raise WorkspaceSecurityError("Workspace recovery install path is malformed.")
    return target.with_name(raw_name)


def _journal_rollback(
    target: Path,
    raw: dict[str, object],
    *,
    transaction_id: str,
    index: int,
) -> Path:
    expected = f".mymoe-rollback-{transaction_id}-{index:08d}"
    raw_name = raw.get("rollback")
    if raw_name is None:
        raw["rollback"] = expected
        raw_name = expected
    if (
        not isinstance(raw_name, str)
        or raw_name != expected
        or _SAFE_ROLLBACK_NAME.fullmatch(raw_name) is None
    ):
        raise WorkspaceSecurityError("Workspace recovery rollback path is malformed.")
    return target.with_name(raw_name)


def _journal_restore(
    target: Path,
    raw: dict[str, object],
    *,
    transaction_id: str,
    index: int,
) -> Path:
    expected = f".mymoe-restore-{transaction_id}-{index:08d}"
    raw_name = raw.get("restore")
    if raw_name is None:
        raw["restore"] = expected
        raw_name = expected
    if (
        not isinstance(raw_name, str)
        or raw_name != expected
        or _SAFE_RESTORE_NAME.fullmatch(raw_name) is None
    ):
        raise WorkspaceSecurityError("Workspace recovery restore path is malformed.")
    return target.with_name(raw_name)


def _read_recovery_backup(
    raw: dict[str, object],
    *,
    backup_dir: Path,
    before: WorkspaceFile,
) -> bytes:
    _assert_directory_entry(backup_dir)
    backup_name = raw.get("backup")
    if (
        not isinstance(backup_name, str)
        or _SAFE_BACKUP_NAME.fullmatch(backup_name) is None
    ):
        raise WorkspaceSecurityError(
            "Workspace recovery backup is missing or malformed."
        )
    backup = backup_dir / backup_name
    data, backup_metadata = _read_regular_path_nofollow(backup, before.size)
    if _sha256_bytes(data) != raw.get("backup_sha256"):
        raise WorkspaceSecurityError("Workspace recovery backup digest mismatch.")
    if not (
        _private_artifact_mode_is_valid(backup_metadata.st_mode)
        or stat.S_IMODE(backup_metadata.st_mode) == before.mode
    ):
        raise WorkspaceSecurityError("Workspace recovery backup mode mismatch.")
    return data


def _rollback_phase(raw: dict[str, object]) -> str:
    phase = raw.get("rollback_phase", "pending")
    if not isinstance(phase, str) or phase not in _ROLLBACK_PHASES:
        raise WorkspaceSecurityError("Workspace recovery rollback phase is invalid.")
    return phase


def _rollback_phase_number(phase: str) -> int:
    try:
        return _ROLLBACK_PHASES.index(phase)
    except ValueError as exc:
        raise WorkspaceSecurityError(
            "Workspace recovery rollback phase is invalid."
        ) from exc


def _rollback_phase_index(raw: dict[str, object]) -> int:
    return _rollback_phase_number(_rollback_phase(raw))


def _advance_rollback_phase(
    raw: dict[str, object],
    phase: str,
    *,
    payload: dict[str, object],
    journal: Path,
    fault_after_step: str | None,
) -> None:
    desired = _rollback_phase_number(phase)
    current = _rollback_phase_index(raw)
    if desired <= current:
        return
    raw["rollback_phase"] = phase
    _write_journal(journal, payload)
    _raise_rollback_fault(fault_after_step, phase)


def _raise_rollback_fault(fault_after_step: str | None, step: str) -> None:
    if fault_after_step == step:
        raise _SimulatedTransactionCrash(f"simulated rollback crash after {step}")


def _assert_owned_rollback(
    rollback: Path,
    expected: WorkspaceFile | None,
    installed_identity: dict[str, int] | None,
) -> None:
    if (
        expected is None
        or expected.kind == "missing"
        or installed_identity is None
        or not _entry_matches_installed(rollback, expected, installed_identity)
    ):
        raise WorkspaceSecurityError(
            "Workspace recovery rollback path lost transaction ownership."
        )


def _journal_quarantine(target: Path, raw_name: object) -> Path | None:
    if raw_name is None:
        return None
    if not isinstance(raw_name, str) or not _SAFE_QUARANTINE_NAME.fullmatch(raw_name):
        raise WorkspaceSecurityError("Workspace recovery quarantine is malformed.")
    quarantine = target.with_name(raw_name)
    return quarantine if _entry_exists(quarantine) else None


def _remove_matching_quarantine(
    target: Path,
    before: WorkspaceFile | None,
    raw_name: object,
) -> None:
    quarantine = _journal_quarantine(target, raw_name)
    if quarantine is None:
        return
    if not _entry_matches(quarantine, before):
        raise WorkspaceSecurityError(
            "Workspace recovery quarantine does not match the recorded value."
        )
    quarantine_identity = _regular_file_identity(quarantine)
    _unlink_recovery_artifact(quarantine, quarantine_identity)


def _rollback_created_directories(root: Path, raw_directories: object) -> None:
    if raw_directories is None:
        return
    if not isinstance(raw_directories, list) or not all(
        isinstance(item, str) for item in raw_directories
    ):
        raise WorkspaceSecurityError("Workspace recovery directory plan is malformed.")
    # A path-only journal cannot prove that an empty directory is still the
    # directory created by this transaction. Preserve it rather than deleting
    # a same-name concurrent directory. Empty directories are outside the file
    # manifest and can be cleaned explicitly after operator inspection.
    for raw in raw_directories:
        _safe_relative(raw)


def _regular_file_identity(path: Path) -> dict[str, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise WorkspaceSecurityError(
            "Workspace file identity could not be inspected."
        ) from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise WorkspaceSecurityError(
            "Workspace file identity requires a regular non-link file."
        )
    return _identity_from_metadata(metadata)


def _directory_identity(path: Path) -> dict[str, int]:
    metadata = path.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise WorkspaceSecurityError(
            "Workspace cleanup identity requires a real directory."
        )
    return _identity_from_metadata(metadata)


def _identity_from_metadata(metadata: os.stat_result) -> dict[str, int]:
    device_id = int(metadata.st_dev)
    inode = int(metadata.st_ino)
    if not 0 <= device_id <= 2**64 - 1 or not 0 <= inode <= 2**64 - 1:
        raise WorkspaceSecurityError("Workspace file identity is outside safe bounds.")
    return {"device_id": device_id, "inode": inode}


def _private_artifact_mode_is_valid(raw_mode: int) -> bool:
    mode = stat.S_IMODE(raw_mode)
    if os.name == "nt":
        return bool(mode & stat.S_IWUSR) and not bool(mode & 0o111)
    return mode == 0o600


def _recorded_file_identity(raw: object) -> dict[str, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"device_id", "inode"}:
        raise WorkspaceSecurityError(
            "Workspace recovery installed identity is malformed."
        )
    device_id = raw.get("device_id")
    inode = raw.get("inode")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 2**64 - 1
        for value in (device_id, inode)
    ):
        raise WorkspaceSecurityError(
            "Workspace recovery installed identity is malformed."
        )
    return {"device_id": device_id, "inode": inode}


def _has_repository_marker(root: Path) -> bool:
    for current in (root, *root.parents):
        marker = current / ".git"
        try:
            metadata = marker.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise WorkspaceSecurityError(
                "Git repository marker could not be inspected safely."
            ) from exc
        if _is_link_or_reparse(metadata):
            raise WorkspaceSecurityError(
                "Git repository marker cannot be a symbolic or reparse path."
            )
        return True
    return False


def _trusted_git_directories() -> tuple[Path, ...]:
    raw_directories: list[Path] = []
    if os.name == "posix":
        raw_directories.extend(
            (
                Path("/usr/bin"),
                Path("/bin"),
            )
        )
    elif os.name == "nt":
        raw_directories.extend(
            (
                Path("C:/Program Files/Git/cmd"),
                Path("C:/Program Files/Git/bin"),
            )
        )
    result: list[Path] = []
    seen: set[str] = set()
    for raw in raw_directories:
        if not raw.is_absolute():
            continue
        try:
            resolved = raw.resolve(strict=True)
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        result.append(resolved)
    return tuple(result)


def _resolve_trusted_git_identity(
    *,
    required: bool,
    configured_executable: str | Path | None = None,
) -> ExecutableIdentity | None:
    candidates: tuple[Path, ...]
    if configured_executable is not None:
        configured = Path(configured_executable).expanduser()
        if not configured.is_absolute():
            raise WorkspaceSecurityError(
                "Configured Git executable must be an absolute path."
            )
        candidates = (configured,)
    else:
        executable_name = "git.exe" if os.name == "nt" else "git"
        candidates = tuple(
            directory / executable_name for directory in _trusted_git_directories()
        )
    for candidate in candidates:
        try:
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            environment = _sanitized_git_environment(
                commit_identity=None,
                executable_path=candidate,
            )
            return resolve_executable(str(candidate), env=environment)
        except (OSError, ValueError, AssistantBridgeRuntimeError):
            continue
    if required:
        raise WorkspaceSecurityError(
            "A trusted, content-attested Git executable is unavailable."
        )
    return None


def _is_git_workspace(
    root: Path,
    *,
    git_identity: ExecutableIdentity | None,
    repository_marker: bool,
) -> bool:
    if git_identity is None:
        if repository_marker:
            raise WorkspaceSecurityError(
                "Git repository attestation is unavailable for a detected repository."
            )
        return False
    result = _execute_git(
        ("rev-parse", "--is-inside-work-tree"),
        root,
        commit_identity=None,
        git_identity=git_identity,
    )
    if result.returncode != 0:
        if repository_marker:
            raise WorkspaceSecurityError(
                "Detected Git repository could not be attested."
            )
        return False
    output = result.stdout.strip()
    if output == b"true":
        return True
    if output == b"false" and not repository_marker:
        return False
    raise WorkspaceSecurityError("Git repository probe returned an invalid result.")


def _git(
    root: Path,
    *args: str,
    git_identity: ExecutableIdentity | None = None,
) -> bytes:
    return _run_git(
        args,
        root,
        capture_bytes=True,
        git_identity=git_identity,
    )


def _git_optional(
    root: Path,
    *args: str,
    git_identity: ExecutableIdentity | None = None,
) -> bytes | None:
    selected = git_identity or _resolve_trusted_git_identity(required=True)
    if selected is None:
        raise WorkspaceSecurityError("Trusted Git executable is unavailable.")
    result = _execute_git(
        args,
        root,
        commit_identity=None,
        git_identity=selected,
    )
    if result.returncode == 0:
        return result.stdout
    if result.returncode == 1:
        return None
    raise WorkspaceSecurityError("Git workspace attestation failed.")


def _run_git(
    argv: Sequence[str],
    cwd: Path,
    *,
    capture: bool = False,
    capture_bytes: bool = False,
    commit_identity: GitIdentity | None = None,
    git_identity: ExecutableIdentity | None = None,
    stdout_limit_bytes: int = _GIT_STDOUT_LIMIT_BYTES,
) -> str | bytes:
    selected = git_identity or _resolve_trusted_git_identity(required=True)
    if selected is None:
        raise WorkspaceSecurityError("Trusted Git executable is unavailable.")
    command = _trusted_git_command(argv, cwd)
    result = _execute_git(
        command,
        cwd,
        commit_identity=commit_identity,
        git_identity=selected,
        stdout_limit_bytes=stdout_limit_bytes,
    )
    if result.returncode != 0:
        raise WorkspaceSecurityError("Git workspace attestation failed.")
    if capture_bytes:
        return result.stdout
    return result.stdout.decode("utf-8", errors="replace") if capture else ""


def _trusted_git_command(argv: Sequence[str], cwd: Path) -> tuple[str, ...]:
    null_config = os.devnull
    command: tuple[str, ...] = (
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        f"core.excludesFile={null_config}",
        "-c",
        f"core.hooksPath={null_config}",
        "-c",
        "credential.helper=",
        "-c",
        "core.sshCommand=false",
        "-c",
        "diff.external=",
        "-c",
        "protocol.ext.allow=never",
        "-c",
        "protocol.file.allow=never",
    )
    if _entry_exists(cwd / ".git"):
        command = (*command, "-c", f"core.worktree={cwd}")
    return (*command, "-C", str(cwd), *tuple(argv))


def _execute_git(
    argv: Sequence[str],
    cwd: Path,
    *,
    commit_identity: GitIdentity | None,
    git_identity: ExecutableIdentity,
    stdout_limit_bytes: int = _GIT_STDOUT_LIMIT_BYTES,
) -> ProcessExecutionResult:
    environment = _sanitized_git_environment(
        commit_identity=commit_identity,
        executable_path=Path(git_identity.resolved_path),
    )
    try:
        execution_policy = (
            _GIT_EXECUTION_POLICY
            if stdout_limit_bytes == _GIT_STDOUT_LIMIT_BYTES
            else ProcessExecutionPolicy(
                stdin_limit_bytes=0,
                stdout_limit_bytes=stdout_limit_bytes,
                stderr_limit_bytes=_GIT_STDERR_LIMIT_BYTES,
                require_tree_isolation=True,
            )
        )
        result = execute_process(
            git_identity,
            argv,
            cwd=cwd,
            env=environment,
            timeout_seconds=_GIT_TIMEOUT_SECONDS,
            policy=execution_policy,
        )
    except ProcessCleanupError:
        raise
    except (AssistantBridgeRuntimeError, OSError, ValueError) as exc:
        raise WorkspaceSecurityError(
            "Git workspace attestation failed safely."
        ) from exc
    if result.code in {"stdout_limit_exceeded", "stderr_limit_exceeded"}:
        raise WorkspaceSecurityError(
            "Git workspace attestation exceeded its output bound."
        )
    if result.code == "timed_out":
        raise WorkspaceSecurityError("Git workspace attestation timed out.")
    if result.code not in {"completed", "nonzero_exit"}:
        raise WorkspaceSecurityError("Git workspace attestation failed safely.")
    return result


def _sanitized_git_environment(
    *,
    commit_identity: GitIdentity | None,
    executable_path: Path,
) -> dict[str, str]:
    allowed = {
        "SystemRoot",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    trusted_path = (executable_path.parent, *_trusted_git_directories())
    env["PATH"] = os.pathsep.join(dict.fromkeys(str(item) for item in trusted_path))
    env.update(
        {
            # HOME and XDG_CONFIG_HOME are deliberately absent from this
            # allowlisted environment, so Git cannot discover user-controlled
            # global configuration. System configuration remains in the OS
            # trust boundary; command-local -c overrides below disable the
            # execution-capable features used by these fixed Git operations.
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_PAGER": "cat",
            "GIT_PROTOCOL_FROM_USER": "0",
        }
    )
    if commit_identity is not None:
        env.update(
            {
                "GIT_AUTHOR_NAME": commit_identity.name,
                "GIT_AUTHOR_EMAIL": commit_identity.email,
                "GIT_COMMITTER_NAME": commit_identity.name,
                "GIT_COMMITTER_EMAIL": commit_identity.email,
            }
        )
    return env


def _safe_relative(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/")).strip("/")
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in normalized
    ):
        raise WorkspaceSecurityError("Workspace scope contains an unsafe path.")
    if path.parts[0] == ".git":
        raise WorkspaceSecurityError(
            "Real Git metadata cannot enter the workspace scope."
        )
    return path.as_posix()


def _change_from_payload(raw: dict[str, object]) -> WorkspaceChange:
    path = _safe_relative(str(raw.get("path", "")))
    return WorkspaceChange(
        path=path,
        before=_file_from_payload(raw.get("before")),
        after=_file_from_payload(raw.get("after")),
    )


def _file_from_payload(raw: object) -> WorkspaceFile | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise WorkspaceSecurityError("Workspace journal file entry is malformed.")
    item = WorkspaceFile(
        path=_safe_relative(str(raw.get("path", ""))),
        kind=str(raw.get("kind", "")),
        sha256=str(raw.get("sha256", "")),
        size=int(raw.get("size", -1)),
        mode=int(raw.get("mode", -1)),
        direction=str(raw.get("direction", "round_trip")),
    )
    if (
        item.kind not in {"file", "missing"}
        or re.fullmatch(r"[a-f0-9]{64}", item.sha256) is None
        or item.size < 0
        or not 0 <= item.mode <= 0o7777
        or item.direction not in {"input_only", "round_trip"}
    ):
        raise WorkspaceSecurityError("Workspace journal file entry is invalid.")
    return item


def _journal_bytes(payload: dict[str, object]) -> bytes:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(data) > _MAX_JOURNAL_BYTES:
        raise WorkspaceSecurityError("Workspace recovery journal exceeds its bound.")
    return data


def _assert_journal_capacity(payload: dict[str, object]) -> None:
    if _journal_worst_case_size(payload) > _MAX_JOURNAL_BYTES:
        raise WorkspaceSecurityError("Workspace recovery journal exceeds its bound.")


def _journal_worst_case_size(payload: dict[str, object]) -> int:
    raw_changes = payload.get("changes")
    if not isinstance(raw_changes, list):
        raise WorkspaceSecurityError("Workspace recovery journal is malformed.")
    worst = dict(payload)
    worst["status"] = "recovery_required"
    worst["committed_fingerprint"] = "f" * 64
    worst["recovery_error"] = "X" * 128
    worst_changes: list[object] = []
    maximum_identity = {
        "device_id": 18_446_744_073_709_551_615,
        "inode": 18_446_744_073_709_551_615,
    }
    for raw in raw_changes:
        if not isinstance(raw, dict):
            raise WorkspaceSecurityError("Workspace recovery record is malformed.")
        item = dict(raw)
        item["status"] = "rolled_back"
        item["rollback_phase"] = "quarantine_unlink_intent"
        item["install_phase"] = "recovered_partial_unlinked"
        item["restore_phase"] = "recovered_partial_unlinked"
        after = _file_from_payload(item.get("after"))
        before = _file_from_payload(item.get("before"))
        if after is not None and after.kind != "missing":
            item["installed_identity"] = maximum_identity
        if before is not None and before.kind != "missing":
            item["restore_identity"] = maximum_identity
        worst_changes.append(item)
    worst["changes"] = worst_changes
    return len(json.dumps(worst, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _write_journal(path: Path, payload: dict[str, object]) -> None:
    temp = path.with_suffix(f".{uuid4().hex}.tmp")
    data = _journal_bytes(payload)
    try:
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_path_durable(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _acquire_transaction_lock(path: Path, ttl_seconds: float) -> None:
    if ttl_seconds < 1:
        raise WorkspaceSecurityError("Workspace lock TTL is invalid.")
    try:
        path.mkdir(mode=0o700)
        _fsync_directory(path.parent)
    except FileExistsError:
        owner = path / "owner.json"
        age = time.time() - path.stat().st_mtime
        pid = -1
        try:
            raw = json.loads(owner.read_text(encoding="utf-8"))
            pid = int(raw.get("pid", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        if age <= ttl_seconds or process_is_alive(pid):
            raise WorkspaceTransactionBusyError(
                "Workspace transaction lock is busy."
            )
        stale = path.with_name(f"{path.name}.stale-{uuid4().hex}")
        try:
            _rename_no_replace(path, stale)
            _durable_rmtree(stale, path.parent)
            path.mkdir(mode=0o700)
            _fsync_directory(path.parent)
        except OSError as exc:
            raise WorkspaceSecurityError(
                "Stale workspace transaction lock could not be recovered."
            ) from exc
    owner = path / "owner.json"
    _durable_write_new(
        owner,
        json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode("utf-8"),
        0o600,
    )


def _release_transaction_lock(path: Path) -> None:
    try:
        (path / "owner.json").unlink(missing_ok=True)
        _fsync_directory(path)
        path.rmdir()
        _fsync_directory(path.parent)
    except OSError:
        pass


def _entry_matches(path: Path, expected: WorkspaceFile | None) -> bool:
    try:
        _assert_current_entry(path, expected)
    except (OSError, WorkspaceSecurityError):
        return False
    return True


def _validate_transaction_id(value: str) -> None:
    if _SAFE_TRANSACTION_ID.fullmatch(value) is None:
        raise WorkspaceSecurityError("Workspace transaction_id is invalid.")


def _validate_journal(payload: object, transaction_id: str) -> None:
    if not isinstance(payload, dict):
        raise WorkspaceSecurityError("Workspace recovery journal must be an object.")
    if payload.get("schema_version") != "1.0":
        raise WorkspaceSecurityError("Workspace recovery schema is unsupported.")
    if payload.get("transaction_id") != transaction_id:
        raise WorkspaceSecurityError("Workspace recovery transaction id mismatch.")
    if payload.get("status") not in {
        "prepared",
        "applying",
        "recovery_required",
        "rolled_back",
        "recovered",
        "committed",
    }:
        raise WorkspaceSecurityError("Workspace recovery journal status is invalid.")


def _trusted_root(value: str | Path, *, label: str) -> Path:
    root = Path(os.path.abspath(Path(value).expanduser()))
    try:
        metadata = root.lstat()
    except FileNotFoundError as exc:
        raise WorkspaceSecurityError(f"{label} must be an existing directory.") from exc
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise WorkspaceSecurityError(
            f"{label} must be a real directory, not a symbolic or reparse path."
        )
    if os.name == "nt":
        _windows_assert_safe_path(root, directory=True)
    else:
        descriptor = os.open(root, _SECURE_DIRECTORY_FLAGS)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or _is_link_or_reparse(opened):
                raise WorkspaceSecurityError(f"{label} directory changed during open.")
        finally:
            os.close(descriptor)
    canonical = root.resolve(strict=True)
    _assert_directory_entry(canonical)
    return canonical


def _prepare_state_directory(value: str | Path) -> Path:
    state = Path(os.path.abspath(Path(value).expanduser()))
    missing: list[Path] = []
    current = state
    while not _entry_exists(current):
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    _assert_directory_entry(current)
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        _fsync_directory(directory.parent)
        _assert_directory_entry(directory)
    return _trusted_root(state, label="Workspace state")


def _prepare_transaction_directories(
    state: Path,
    transaction: Path,
) -> tuple[Path, Path]:
    try:
        transaction.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise WorkspaceSecurityError(
            "Workspace transaction id has already been used."
        ) from exc
    _fsync_directory(state)
    backup_dir = transaction / "backups"
    backup_dir.mkdir(mode=0o700)
    staged_dir = transaction / "staged"
    staged_dir.mkdir(mode=0o700)
    _fsync_directory(transaction)
    return backup_dir, staged_dir


def _assert_directory_entry(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise WorkspaceSecurityError(
            "Workspace directory changed during traversal."
        ) from exc
    if _is_link_or_reparse(metadata):
        raise WorkspaceSecurityError(
            "Workspace symbolic-link or reparse directories are not supported."
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise WorkspaceSecurityError("Workspace path ancestor is not a directory.")
    if os.name == "nt":
        _windows_assert_safe_path(path, directory=True)


def _assert_safe_relative_ancestry(root: Path, relative: str) -> None:
    current = root
    _assert_directory_entry(current)
    for part in Path(_safe_relative(relative)).parts[:-1]:
        current /= part
        _assert_directory_entry(current)


def _inspect_regular_file(
    root: Path,
    relative: str,
    policy: WorkspaceScopePolicy,
) -> tuple[os.stat_result, str]:
    data, metadata = _read_workspace_path_nofollow(
        root, relative, policy.max_file_bytes
    )
    return metadata, _sha256_bytes(data)


def _read_attested_file(
    root: Path,
    expected: WorkspaceFile,
    policy: WorkspaceScopePolicy,
) -> bytes:
    if expected.kind != "file" or not 0 <= expected.size <= policy.max_file_bytes:
        raise WorkspaceSecurityError("Attested workspace file is outside safe bounds.")
    data, metadata = _read_workspace_path_nofollow(
        root, expected.path, policy.max_file_bytes
    )
    if (
        int(metadata.st_size) != expected.size
        or stat.S_IMODE(metadata.st_mode) != expected.mode
        or _sha256_bytes(data) != expected.sha256
    ):
        raise WorkspaceSecurityError(
            "Workspace file changed while its attested bytes were reopened."
        )
    return data


def _read_workspace_path_nofollow(
    root: Path,
    relative: str,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    clean = _safe_relative(relative)
    if os.name == "nt":
        _assert_safe_relative_ancestry(root, clean)
        return _read_regular_path_nofollow(root / clean, max_bytes)
    if _supports_secure_dirfd_reads():
        root_fd = os.open(root, _SECURE_DIRECTORY_FLAGS)
        current_fd = root_fd
        try:
            parts = Path(clean).parts
            for part in parts[:-1]:
                next_fd = os.open(part, _SECURE_DIRECTORY_FLAGS, dir_fd=current_fd)
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = next_fd
                opened = os.fstat(current_fd)
                if not stat.S_ISDIR(opened.st_mode) or _is_link_or_reparse(opened):
                    raise WorkspaceSecurityError(
                        "Workspace path ancestor is symbolic or not a directory."
                    )
            descriptor = os.open(parts[-1], _SECURE_OPEN_FLAGS, dir_fd=current_fd)
            try:
                return _read_stable_descriptor(descriptor, max_bytes)
            finally:
                os.close(descriptor)
        except OSError as exc:
            if isinstance(exc, FileNotFoundError):
                raise
            raise WorkspaceSecurityError(
                "Workspace symbolic or inaccessible path failed no-follow open."
            ) from exc
        finally:
            if current_fd != root_fd:
                os.close(current_fd)
            os.close(root_fd)
    _assert_safe_relative_ancestry(root, clean)
    return _read_regular_path_nofollow(root / clean, max_bytes)


def _read_regular_path_nofollow(
    path: Path,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    if os.name == "nt":
        descriptor = _windows_open_fd(path, directory=False)
    else:
        try:
            descriptor = os.open(path, _SECURE_OPEN_FLAGS)
        except OSError as exc:
            if isinstance(exc, FileNotFoundError):
                raise
            raise WorkspaceSecurityError(
                "Workspace symbolic or inaccessible path failed no-follow open."
            ) from exc
    try:
        return _read_stable_descriptor(descriptor, max_bytes)
    finally:
        os.close(descriptor)


def _read_stable_descriptor(
    descriptor: int,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if _is_link_or_reparse(before):
        raise WorkspaceSecurityError("Workspace symbolic links are not supported.")
    if not stat.S_ISREG(before.st_mode):
        raise WorkspaceSecurityError("Workspace contains a special file.")
    if int(before.st_size) > max_bytes:
        raise WorkspaceSecurityError("Workspace per-file byte bound was exceeded.")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise WorkspaceSecurityError("Workspace per-file byte bound was exceeded.")
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mode",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, item, None) != getattr(after, item, None)
        for item in stable_fields
    ):
        raise WorkspaceSecurityError("Workspace file changed while it was read.")
    data = b"".join(chunks)
    if len(data) != int(after.st_size):
        raise WorkspaceSecurityError("Workspace file size changed while it was read.")
    return data, after


def _supports_secure_dirfd_reads() -> bool:
    return (
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _durable_write_new(path: Path, data: bytes, mode: int) -> None:
    descriptor = _open_binary_new(path, mode)
    fchmod = getattr(os, "fchmod", None)
    try:
        identity = _identity_from_metadata(os.fstat(descriptor))
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        if callable(fchmod):
            fchmod(descriptor, mode)
        else:
            if os.name == "nt" and _regular_file_identity(path) != identity:
                raise WorkspaceSecurityError(
                    "Workspace file identity changed before mode application."
                )
            path.chmod(mode)
            if os.name == "nt" and _regular_file_identity(path) != identity:
                raise WorkspaceSecurityError(
                    "Workspace file identity changed during mode application."
                )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _open_binary_new(path: Path, mode: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, mode)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        _windows_flush_path(path, directory=True)
        return
    descriptor = os.open(path, _SECURE_DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_path_durable(source: Path, target: Path) -> None:
    if os.name == "nt":
        _windows_move(source, target, replace=True)
    else:
        os.replace(source, target)
        _fsync_directory(target.parent)


def _rename_no_replace(source: Path, target: Path) -> None:
    if os.name == "nt":
        _windows_move(source, target, replace=False)
        return
    backend = _posix_no_replace_backend()
    if backend is None:
        raise WorkspaceSecurityError(
            "Native atomic no-replace rename is unavailable on this platform."
        )
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_target = os.fsencode(target)
    if backend == "darwin-renamex-excl":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(encoded_source, encoded_target, 0x00000004)
    elif backend == "linux-renameat2-noreplace":
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, encoded_source, -100, encoded_target, 0x00000001)
    else:  # pragma: no cover - the backend detector is deliberately closed.
        raise WorkspaceSecurityError(
            "Native atomic no-replace rename backend is unsupported."
        )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(target))
    _fsync_directory(target.parent)
    if source.parent != target.parent:
        _fsync_directory(source.parent)


def _durable_rmtree(path: Path, parent: Path) -> None:
    def make_writable_and_retry(
        operation: Callable[..., object],
        raw_path: str,
        _: object,
    ) -> None:
        target = Path(raw_path)
        try:
            target.chmod(stat.S_IMODE(target.lstat().st_mode) | stat.S_IWUSR)
        except FileNotFoundError:
            return
        operation(raw_path)

    shutil.rmtree(path, onerror=make_writable_and_retry)
    _fsync_directory(parent)


def _require_secure_apply_capabilities() -> None:
    capability = workspace_write_capability()
    if not capability.supported:
        raise WorkspaceSecurityError(
            f"Workspace write capability is unavailable: {capability.reason}."
        )


def _windows_open_fd(
    path: Path,
    *,
    directory: bool,
    writable: bool = False,
) -> int:
    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    flags = 0x00200000
    if directory:
        flags |= 0x02000000
    desired_access = 0x40000000 if writable else 0x80000000
    handle = create_file(
        str(path),
        desired_access,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()
        if error in {2, 3}:
            raise FileNotFoundError(error, "Workspace path does not exist", str(path))
        raise WorkspaceSecurityError(f"Win32 no-follow open failed with error {error}.")
    descriptor: int | None = None
    try:
        access_flag = os.O_WRONLY if writable else os.O_RDONLY
        flags_value = access_flag | getattr(os, "O_BINARY", 0)
        descriptor = msvcrt.open_osfhandle(int(handle), flags_value)
        information = os.fstat(descriptor)
        if _is_link_or_reparse(information):
            raise WorkspaceSecurityError(
                "Workspace symbolic-link or reparse paths are not supported."
            )
        return descriptor
    except Exception:
        if descriptor is None:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
        else:
            os.close(descriptor)
        raise


def _windows_assert_safe_path(path: Path, *, directory: bool) -> None:
    descriptor = _windows_open_fd(path, directory=directory)
    os.close(descriptor)


def _windows_flush_path(path: Path, *, directory: bool) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    flags = 0x02000000 if directory else 0
    desired_access = 0x80000000 if directory else 0x40000000
    handle = create_file(
        str(path),
        desired_access,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise WorkspaceSecurityError("Win32 durability handle could not be opened.")
    try:
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [ctypes.c_void_p]
        flush.restype = ctypes.c_int
        if not flush(ctypes.c_void_p(handle)):
            error = ctypes.get_last_error()
            if directory and error in {1, 5, 6, 87}:
                return
            raise WorkspaceSecurityError(
                f"Win32 durability flush failed with error {error}."
            )
    finally:
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        close(ctypes.c_void_p(handle))


def _windows_move(source: Path, target: Path, *, replace: bool) -> None:
    import ctypes

    flags = 0x00000008
    if replace:
        flags |= 0x00000001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move = kernel32.MoveFileExW
    move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    move.restype = ctypes.c_int
    if not move(str(source), str(target), flags):
        error = ctypes.get_last_error()
        if error in {80, 183}:
            raise FileExistsError(error, "Workspace target already exists", str(target))
        raise WorkspaceSecurityError(f"Win32 durable move failed with error {error}.")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
