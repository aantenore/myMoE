from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Iterator, Sequence


class WorkspaceSecurityError(ValueError):
    """Raised when a workspace cannot be snapshotted or changed safely."""


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "ignored_paths", tuple(self.ignored_paths))
        if not 1 <= self.max_files <= 100_000:
            raise WorkspaceSecurityError("Workspace max_files is outside safe bounds.")
        if not 1 <= self.max_file_bytes <= self.max_total_bytes:
            raise WorkspaceSecurityError("Workspace byte bounds are invalid.")
        paths = [item.path.casefold() for item in self.ignored_paths]
        if len(paths) != len(set(paths)):
            raise WorkspaceSecurityError("Ignored path rules contain a collision.")


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
        return snapshot_materialized(self.root, self.policy)


def snapshot_workspace(
    workspace: str | Path,
    policy: WorkspaceScopePolicy,
) -> WorkspaceSnapshot:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise WorkspaceSecurityError("Workspace must be an existing directory.")
    first = _snapshot_once(root, policy)
    second = _snapshot_once(root, policy)
    if first.fingerprint != second.fingerprint:
        raise WorkspaceSecurityError("Workspace changed while it was being attested.")
    return second


def snapshot_materialized(
    root: str | Path,
    policy: WorkspaceScopePolicy,
) -> tuple[WorkspaceFile, ...]:
    base = Path(root).resolve()
    paths: list[str] = []
    for directory, directories, files in os.walk(base, topdown=True, followlinks=False):
        directories[:] = sorted(item for item in directories if item != ".git")
        relative_dir = Path(directory).relative_to(base)
        for name in sorted(files):
            relative = (relative_dir / name).as_posix()
            paths.append(_safe_relative(relative))
    directions = {item.path: item.direction for item in policy.ignored_paths}
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
            source_path = source / item.path
            target = root / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_attested_file(source_path, target, item)
        if snapshot_workspace(source, policy).fingerprint != snapshot.fingerprint:
            raise WorkspaceSecurityError(
                "Workspace changed while the materialized candidate was created."
            )
        baseline = snapshot_materialized(root, policy)
        _initialize_synthetic_repository(root)
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


def apply_changeset(
    *,
    source_snapshot: WorkspaceSnapshot,
    candidate_root: str | Path,
    candidate_files: Sequence[WorkspaceFile],
    changes: Sequence[WorkspaceChange],
    policy: WorkspaceScopePolicy,
    state_dir: str | Path,
    transaction_id: str,
) -> WorkspaceSnapshot:
    source = Path(source_snapshot.root)
    state = Path(state_dir).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    lock = state / f"workspace-{_sha256_text(str(source))[:24]}.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise WorkspaceSecurityError("Workspace transaction lock is busy.") from exc
    journal = state / f"transaction-{transaction_id}.json"
    backups: dict[str, tuple[bytes, int] | None] = {}
    applied: list[WorkspaceChange] = []
    try:
        current = snapshot_workspace(source, policy)
        if current.fingerprint != source_snapshot.fingerprint:
            raise WorkspaceSecurityError(
                "Workspace changed after confirmation; transaction was not applied."
            )
        journal.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "transaction_id": transaction_id,
                    "source_fingerprint": source_snapshot.fingerprint,
                    "change_count": len(changes),
                    "status": "applying",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        for change in changes:
            target = source / change.path
            _assert_current_entry(target, change.before)
            if target.exists():
                backups[change.path] = (
                    target.read_bytes(),
                    stat.S_IMODE(target.stat().st_mode),
                )
            else:
                backups[change.path] = None
            if change.after is None:
                target.unlink()
            else:
                candidate = Path(candidate_root) / change.path
                _atomic_copy(candidate, target, change.after.mode)
            applied.append(change)
        result = snapshot_workspace(source, policy)
        expected = tuple(sorted(candidate_files))
        if tuple(sorted(result.files)) != expected:
            raise WorkspaceSecurityError(
                "Post-transaction workspace does not match the verified candidate."
            )
        journal.unlink(missing_ok=True)
        return result
    except Exception:
        _rollback(source, applied, backups)
        raise
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def _snapshot_once(root: Path, policy: WorkspaceScopePolicy) -> WorkspaceSnapshot:
    git_repository = _is_git_workspace(root)
    if git_repository:
        raw_paths = _git(
            root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        )
        paths = [os.fsdecode(item) for item in raw_paths.split(b"\0") if item]
        status = _git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        index = _git(root, "ls-files", "--stage", "-z")
        head_raw = _git_optional(root, "rev-parse", "--verify", "HEAD")
        head = head_raw.decode("ascii").strip().lower() if head_raw else "unborn"
    else:
        paths = []
        for directory, directories, files in os.walk(
            root, topdown=True, followlinks=False
        ):
            directories[:] = sorted(item for item in directories if item != ".git")
            relative_dir = Path(directory).relative_to(root)
            paths.extend((relative_dir / name).as_posix() for name in sorted(files))
        status = b""
        index = b""
        head = ""
    directions = {item.path: item.direction for item in policy.ignored_paths}
    for rule in policy.ignored_paths:
        if rule.path not in paths:
            paths.append(rule.path)
    files = _manifest_files(root, paths, policy, directions=directions)
    manifest = _canonical_json([item.payload() for item in files])
    fields = {
        "root_sha256": _sha256_text(str(root)),
        "git_repository": git_repository,
        "head_sha": head,
        "index_sha256": _sha256_bytes(index),
        "status_sha256": _sha256_bytes(status),
        "manifest_sha256": _sha256_text(manifest),
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
        total_bytes=sum(item.size for item in files),
    )


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
        target = root / relative
        try:
            metadata = target.lstat()
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
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkspaceSecurityError("Workspace symbolic links are not supported.")
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkspaceSecurityError("Workspace contains a special file.")
        size = int(metadata.st_size)
        if size > policy.max_file_bytes:
            raise WorkspaceSecurityError("Workspace per-file byte bound was exceeded.")
        total += size
        if total > policy.max_total_bytes:
            raise WorkspaceSecurityError("Workspace total byte bound was exceeded.")
        result.append(
            WorkspaceFile(
                path=relative,
                kind="file",
                sha256=_hash_file(target),
                size=size,
                mode=stat.S_IMODE(metadata.st_mode),
                direction=directions.get(relative, "round_trip"),
            )
        )
    return tuple(result)


def _initialize_synthetic_repository(root: Path) -> None:
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(root.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Antonio Antenore",
        "GIT_AUTHOR_EMAIL": "ant_ant95@hotmail.it",
        "GIT_COMMITTER_NAME": "Antonio Antenore",
        "GIT_COMMITTER_EMAIL": "ant_ant95@hotmail.it",
    }
    _run(("git", "init", "-q", "--template="), root, env)
    _run(("git", "add", "-f", "--all"), root, env)
    _run(
        ("git", "commit", "-q", "--allow-empty", "-m", "materialized baseline"),
        root,
        env,
    )
    remotes = _run(("git", "remote"), root, env, capture=True).splitlines()
    if remotes:
        raise WorkspaceSecurityError(
            "Synthetic repository unexpectedly contains remotes."
        )


def _copy_attested_file(source: Path, target: Path, expected: WorkspaceFile) -> None:
    if _hash_file(source) != expected.sha256:
        raise WorkspaceSecurityError("Workspace file changed during materialization.")
    shutil.copyfile(source, target)
    target.chmod(expected.mode)
    if _hash_file(target) != expected.sha256:
        raise WorkspaceSecurityError("Materialized file digest mismatch.")


def _assert_current_entry(path: Path, expected: WorkspaceFile | None) -> None:
    if expected is None or expected.kind == "missing":
        if path.exists():
            raise WorkspaceSecurityError("Workspace path failed compare-and-swap.")
        return
    if not path.is_file() or path.is_symlink():
        raise WorkspaceSecurityError("Workspace path failed compare-and-swap.")
    if _hash_file(path) != expected.sha256:
        raise WorkspaceSecurityError("Workspace path failed compare-and-swap.")


def _atomic_copy(source: Path, target: Path, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=".mymoe-", dir=target.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(mode)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _rollback(
    root: Path,
    applied: Sequence[WorkspaceChange],
    backups: dict[str, tuple[bytes, int] | None],
) -> None:
    for change in reversed(applied):
        target = root / change.path
        try:
            _assert_current_entry(target, change.after)
            backup = backups[change.path]
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                data, mode = backup
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(mode)
        except OSError as exc:
            raise WorkspaceSecurityError(
                "Workspace rollback could not prove safe restoration; inspect the journal."
            ) from exc


def _is_git_workspace(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == b"true"


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkspaceSecurityError("Git workspace attestation failed.")
    return completed.stdout


def _git_optional(root: Path, *args: str) -> bytes | None:
    try:
        return _git(root, *args)
    except WorkspaceSecurityError:
        return None


def _run(
    argv: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    *,
    capture: bool = False,
) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkspaceSecurityError("Synthetic repository initialization failed.")
    return completed.stdout if capture else ""


def _safe_relative(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
