"""Fail-closed OS containment plans for assistant-bridge verifiers.

The bridge treats verifier commands as untrusted candidate code.  This module
therefore exposes a capability, not a label: a command is runnable only when a
fixed, OS-owned containment executable can be attested and an exact wrapper
plan can be bound to the confirmation receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Iterable, Sequence

from .assistant_bridge_runtime import (
    AssistantBridgeRuntimeError,
    ExecutableIdentity,
    resolve_executable,
)


_MACOS_BACKEND = Path("/usr/bin/sandbox-exec")
_LINUX_BACKEND = Path("/usr/bin/bwrap")
_SCHEMA = "assistant-bridge-verifier-isolation/v1"


class VerifierIsolationError(ValueError):
    """Raised when a verifier containment contract is unsafe or incomplete."""


@dataclass(frozen=True)
class VerifierIsolationPolicy:
    """Provider-neutral selection policy for supported OS containment backends."""

    required: bool = True
    macos_backend: str = str(_MACOS_BACKEND)
    linux_backend: str = str(_LINUX_BACKEND)

    def __post_init__(self) -> None:
        if self.required is not True:
            raise VerifierIsolationError(
                "Verifier hard isolation cannot be disabled."
            )
        if Path(self.macos_backend) != _MACOS_BACKEND:
            raise VerifierIsolationError(
                "The macOS verifier backend must be /usr/bin/sandbox-exec."
            )
        if Path(self.linux_backend) != _LINUX_BACKEND:
            raise VerifierIsolationError(
                "The Linux verifier backend must be /usr/bin/bwrap."
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "required": self.required,
            "macos_backend_sha256": _sha256_text(self.macos_backend),
            "linux_backend_sha256": _sha256_text(self.linux_backend),
            "network": "denied",
            "filesystem": "deny_default_declared_roots_only",
        }


@dataclass(frozen=True)
class VerifierIsolationCapability:
    """Attested host support for one hard-containment backend."""

    supported: bool
    backend: str
    reason: str = ""
    executable: ExecutableIdentity | None = field(default=None, repr=False)

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "supported": self.supported,
            "backend": self.backend,
            "reason": self.reason or None,
            "executable": (
                None
                if self.executable is None
                else self.executable.binding_payload()
            ),
            "hard_containment": self.supported,
            "network": "denied" if self.supported else "unavailable",
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "supported": self.supported,
            "backend": self.backend,
            "reason": self.reason or None,
            "executable": (
                None if self.executable is None else self.executable.payload()
            ),
            "hard_containment": self.supported,
            "network": "denied" if self.supported else "unavailable",
        }


@dataclass(frozen=True)
class RuntimeRootIdentity:
    """Metadata binding for one declared read-only runtime root."""

    requested_sha256: str
    resolved_path: str = field(repr=False)
    resolved_path_sha256: str
    kind: str
    content_sha256: str
    device_id: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)
    owner_uid: int | None = field(repr=False)
    owner_gid: int | None = field(repr=False)

    def binding_payload(self) -> dict[str, object]:
        return {
            "requested_sha256": self.requested_sha256,
            "resolved_path": self.resolved_path,
            "resolved_path_sha256": self.resolved_path_sha256,
            "kind": self.kind,
            "content_sha256": self.content_sha256,
            "device_id": self.device_id,
            "inode": self.inode,
            "mode": self.mode,
            "owner_uid": self.owner_uid,
            "owner_gid": self.owner_gid,
        }

    def payload(self) -> dict[str, object]:
        return {
            "requested_sha256": self.requested_sha256,
            "resolved_path_sha256": self.resolved_path_sha256,
            "kind": self.kind,
            "content_sha256": self.content_sha256,
            "identity_sha256": _sha256_json(self.binding_payload()),
        }


@dataclass(frozen=True)
class VerifierIsolationPlan:
    """Exact outer sandbox invocation bound without exposing local paths."""

    policy: VerifierIsolationPolicy
    capability: VerifierIsolationCapability
    runtime_roots: tuple[RuntimeRootIdentity, ...] = field(repr=False)
    argv: tuple[str, ...] = field(repr=False)
    internal_temp: str = field(repr=False)
    profile_sha256: str
    argv_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_roots", tuple(self.runtime_roots))
        object.__setattr__(self, "argv", tuple(self.argv))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "policy": self.policy.payload(),
            "capability": self.capability.payload(),
            "runtime_roots": [item.payload() for item in self.runtime_roots],
            "profile_sha256": self.profile_sha256 or None,
            "sandbox_argv_sha256": self.argv_sha256 or None,
            "binding_sha256": self.binding_sha256,
            "workspace": "read_write_disposable",
            "runtime_roots_access": "read_only",
            "system_roots_access": "read_only_minimal",
            "temporary_storage": (
                "workspace_internal" if self.capability.backend == "sandbox-exec" else "tmpfs"
            ),
            "network": "denied" if self.capability.supported else "unavailable",
        }


def verifier_isolation_capability(
    policy: VerifierIsolationPolicy,
) -> VerifierIsolationCapability:
    """Attest the only accepted backend for the current platform."""

    if sys.platform == "darwin":
        return _attest_backend(Path(policy.macos_backend), "sandbox-exec")
    if sys.platform.startswith("linux"):
        return _attest_backend(Path(policy.linux_backend), "bwrap")
    return VerifierIsolationCapability(
        supported=False,
        backend="unsupported",
        reason="no supported verifier hard-sandbox backend for this platform",
    )


def expand_runtime_read_roots(values: Sequence[str]) -> tuple[Path, ...]:
    """Expand configured runtime tokens into canonical, duplicate-free roots."""

    expanded: list[Path] = []
    for value in values:
        if value == "{python_runtime}":
            expanded.extend(
                Path(item)
                for item in (sys.prefix, sys.base_prefix, sys.executable)
                if item
            )
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise VerifierIsolationError(
                "Verifier runtime read roots must be absolute paths or supported tokens."
            )
        expanded.append(candidate)
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in expanded:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise VerifierIsolationError(
                "A declared verifier runtime read root is unavailable."
            ) from exc
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return tuple(result)


def build_verifier_isolation_plan(
    policy: VerifierIsolationPolicy,
    capability: VerifierIsolationCapability,
    *,
    workspace: str | Path,
    command_argv: Sequence[str],
    runtime_read_roots: Sequence[str],
    temp_namespace: str,
    attested_read_artifacts: Sequence[str | Path] = (),
) -> VerifierIsolationPlan:
    """Build a macOS Seatbelt or Linux bubblewrap invocation.

    Unsupported capabilities still produce a bindable plan with no executable
    argv.  Callers must route-block such a plan and this module never falls back
    to direct host execution.
    """

    root = Path(workspace).expanduser().resolve(strict=True)
    if not temp_namespace or any(
        item not in "0123456789abcdef" for item in temp_namespace
    ):
        raise VerifierIsolationError(
            "Verifier temporary namespace must be a lowercase hexadecimal token."
        )
    semantic_root = "<ephemeral-workspace>"
    runtime_paths = expand_runtime_read_roots(runtime_read_roots)
    identities = tuple(_attest_runtime_root(path) for path in runtime_paths)
    temp_root = root / f".mymoe-verifier-tmp-{temp_namespace}"
    if not capability.supported or capability.executable is None:
        binding = {
            "policy": policy.payload(),
            "capability": capability.binding_payload(),
            "runtime_roots": [item.binding_payload() for item in identities],
            "temp_namespace": temp_namespace,
            "profile_sha256": None,
            "sandbox_argv_sha256": None,
        }
        return VerifierIsolationPlan(
            policy=policy,
            capability=capability,
            runtime_roots=identities,
            argv=(),
            internal_temp=str(temp_root),
            profile_sha256="",
            argv_sha256="",
            binding_sha256=_sha256_json(binding),
        )

    artifacts = tuple(
        _absolute_existing_path(item) for item in attested_read_artifacts
    )
    if capability.backend == "sandbox-exec":
        profile, semantic_profile = _macos_profiles(
            workspace=root,
            temp_root=temp_root,
            runtime_roots=runtime_paths,
            attested_read_artifacts=artifacts,
        )
        argv = ("-p", profile, *tuple(command_argv))
        profile_sha256 = _sha256_text(semantic_profile)
        semantic_argv = (
            "-p",
            semantic_profile,
            "<attested-verifier-executable>",
            *tuple(
                item.replace(str(root), semantic_root)
                for item in command_argv[1:]
            ),
        )
    elif capability.backend == "bwrap":
        argv = _bubblewrap_argv(
            workspace=root,
            runtime_roots=runtime_paths,
            attested_read_artifacts=artifacts,
            command_argv=tuple(command_argv),
        )
        semantic_argv = _semantic_bubblewrap_argv(
            workspace=root,
            runtime_roots=runtime_paths,
            attested_read_artifacts=artifacts,
            command_argv=tuple(command_argv),
            semantic_workspace=semantic_root,
        )
        separator = semantic_argv.index("--")
        profile_sha256 = _sha256_json(list(semantic_argv[: separator + 1]))
    else:  # Defensive against fabricated capabilities.
        raise VerifierIsolationError("Unsupported verifier isolation backend.")

    argv_sha256 = _sha256_json(list(semantic_argv))
    binding = {
        "policy": policy.payload(),
        "capability": capability.binding_payload(),
        "runtime_roots": [item.binding_payload() for item in identities],
        "temp_namespace": temp_namespace,
        "profile_sha256": profile_sha256,
        "sandbox_argv": list(semantic_argv),
    }
    return VerifierIsolationPlan(
        policy=policy,
        capability=capability,
        runtime_roots=identities,
        argv=argv,
        internal_temp=str(temp_root),
        profile_sha256=profile_sha256,
        argv_sha256=argv_sha256,
        binding_sha256=_sha256_json(binding),
    )


def _attest_backend(path: Path, backend: str) -> VerifierIsolationCapability:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
            raise OSError("backend is not an executable regular file")
        if metadata.st_uid != 0 or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise OSError("backend is not OS-owned and immutable to unprivileged users")
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
        identity = resolve_executable(str(path), env=environment)
        if Path(identity.resolved_path) != path.resolve(strict=True):
            raise OSError("backend resolved outside its fixed OS path")
    except (AssistantBridgeRuntimeError, OSError, ValueError) as exc:
        return VerifierIsolationCapability(
            supported=False,
            backend=backend,
            reason=f"required OS-owned {backend} backend is unavailable",
        )
    return VerifierIsolationCapability(
        supported=True,
        backend=backend,
        executable=identity,
    )


def _absolute_existing_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.lstat()
    except OSError as exc:
        raise VerifierIsolationError(
            "An attested verifier read artifact is unavailable."
        ) from exc
    return path


def _attest_runtime_root(path: Path) -> RuntimeRootIdentity:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise VerifierIsolationError(
            "A declared verifier runtime read root cannot be attested."
        ) from exc
    if stat.S_ISREG(metadata.st_mode):
        content_sha256 = _hash_file(path)
        kind = "file"
    elif stat.S_ISDIR(metadata.st_mode):
        content_sha256 = _sha256_json(
            {
                "device_id": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode": stat.S_IMODE(metadata.st_mode),
                "mtime_ns": metadata.st_mtime_ns,
            }
        )
        kind = "directory"
    else:
        raise VerifierIsolationError(
            "Verifier runtime roots must be regular files or directories."
        )
    resolved = str(path.resolve(strict=True))
    return RuntimeRootIdentity(
        requested_sha256=_sha256_text(str(path)),
        resolved_path=resolved,
        resolved_path_sha256=_sha256_text(resolved),
        kind=kind,
        content_sha256=content_sha256,
        device_id=metadata.st_dev,
        inode=metadata.st_ino,
        mode=stat.S_IMODE(metadata.st_mode),
        owner_uid=getattr(metadata, "st_uid", None),
        owner_gid=getattr(metadata, "st_gid", None),
    )


def _macos_profile(
    *,
    workspace: Path,
    temp_root: Path,
    runtime_roots: Sequence[Path],
    attested_read_artifacts: Sequence[Path],
) -> str:
    return _macos_profiles(
        workspace=workspace,
        temp_root=temp_root,
        runtime_roots=runtime_roots,
        attested_read_artifacts=attested_read_artifacts,
    )[0]


def _macos_profiles(
    *,
    workspace: Path,
    temp_root: Path,
    runtime_roots: Sequence[Path],
    attested_read_artifacts: Sequence[Path],
) -> tuple[str, str]:
    system_roots = tuple(
        path
        for path in (
            Path("/System"),
            Path("/usr/bin"),
            Path("/usr/lib"),
            Path("/usr/share/locale"),
            Path("/bin"),
            Path("/sbin"),
            Path("/private/etc/hosts"),
            Path("/private/etc/resolv.conf"),
            Path("/private/etc/services"),
            Path("/private/var/db/timezone"),
            Path("/dev/null"),
            Path("/dev/random"),
            Path("/dev/urandom"),
        )
        if path.exists()
    )
    fixed = (
        "(version 1)",
        "(deny default)",
        "(deny network*)",
        "(allow process*)",
    )
    system_filters = _sandbox_filters(_darwin_paths(system_roots))
    runtime_filters = _sandbox_filters(_darwin_paths(runtime_roots))
    workspace_filters = _sandbox_filters(
        _darwin_paths((workspace, temp_root))
    )
    write_filters = _sandbox_filters(
        _darwin_paths((workspace, temp_root)),
        include_ancestors=False,
    )
    artifact_filters: list[tuple[str, str]] = []
    for artifact in attested_read_artifacts:
        actual = _sandbox_filters(_darwin_paths((artifact,)))
        if _path_is_covered(artifact, runtime_roots):
            semantic = actual
        elif _path_is_covered(artifact, (workspace,)):
            relative = artifact.relative_to(workspace).as_posix()
            semantic = f"<workspace-artifact:{relative}>"
        else:
            semantic = actual
        artifact_filters.append((actual, semantic))

    def read_clauses(filters: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            clause
            for value in filters
            if value
            for clause in (
                f"(allow file-read* {value})",
                f"(allow file-map-executable {value})",
            )
        )

    actual_filters = (
        system_filters,
        workspace_filters,
        runtime_filters,
        *(item[0] for item in artifact_filters),
    )
    semantic_filters = (
        system_filters,
        "<workspace-read-grants>",
        runtime_filters,
        *(item[1] for item in artifact_filters),
    )
    actual = "".join(
        (*fixed, *read_clauses(actual_filters), f"(allow file-write* {write_filters})")
    )
    semantic = "".join(
        (
            *fixed,
            *read_clauses(semantic_filters),
            "(allow file-write* <workspace-write-grants>)",
        )
    )
    return actual, semantic


def _bubblewrap_argv(
    *,
    workspace: Path,
    runtime_roots: Sequence[Path],
    attested_read_artifacts: Sequence[Path],
    command_argv: tuple[str, ...],
) -> tuple[str, ...]:
    read_roots = _bubblewrap_read_roots(
        workspace=workspace,
        runtime_roots=runtime_roots,
        attested_read_artifacts=attested_read_artifacts,
    )
    argv = list(_bubblewrap_prefix())
    for path in read_roots:
        argv.extend(("--ro-bind", str(path), str(path)))
    argv.extend(
        (
            "--bind",
            str(workspace),
            str(workspace),
            "--chdir",
            str(workspace),
            "--",
            *command_argv,
        )
    )
    return tuple(argv)


def _bubblewrap_read_roots(
    *,
    workspace: Path,
    runtime_roots: Sequence[Path],
    attested_read_artifacts: Sequence[Path],
) -> tuple[Path, ...]:
    system_roots = tuple(
        path
        for path in (
            Path("/usr/bin"),
            Path("/usr/lib"),
            Path("/usr/lib64"),
            Path("/usr/share/locale"),
            Path("/usr/share/zoneinfo"),
            Path("/bin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc/ld.so.cache"),
            Path("/etc/ssl/certs"),
        )
        if path.exists()
    )
    declared_roots = _remove_contained_paths(
        _deduplicate_paths((*system_roots, *runtime_roots))
    )
    return _remove_contained_paths(
        _deduplicate_paths(
            (
                *declared_roots,
                *(
                    item
                    for item in attested_read_artifacts
                    if not _path_is_covered(
                        item, (*declared_roots, workspace)
                    )
                ),
            )
        )
    )


def _bubblewrap_prefix() -> tuple[str, ...]:
    return (
        "--die-with-parent",
        "--unshare-all",
        "--unshare-net",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    )


def _semantic_bubblewrap_argv(
    *,
    workspace: Path,
    runtime_roots: Sequence[Path],
    attested_read_artifacts: Sequence[Path],
    command_argv: tuple[str, ...],
    semantic_workspace: str,
) -> tuple[str, ...]:
    read_roots = _bubblewrap_read_roots(
        workspace=workspace,
        runtime_roots=runtime_roots,
        attested_read_artifacts=attested_read_artifacts,
    )
    argv = list(_bubblewrap_prefix())
    for path in read_roots:
        argv.extend(("--ro-bind", str(path), str(path)))
    argv.extend(
        (
            "--bind",
            semantic_workspace,
            semantic_workspace,
            "--chdir",
            semantic_workspace,
            "--",
            "<attested-verifier-executable>",
            *tuple(
                item.replace(str(workspace), semantic_workspace)
                for item in command_argv[1:]
            ),
        )
    )
    return tuple(argv)


def _darwin_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    expanded: list[Path] = []
    for path in paths:
        for candidate in _symlink_resolution_paths(path):
            expanded.append(candidate)
            resolved = candidate.resolve(strict=False)
            expanded.append(resolved)
            for visible in (candidate, resolved):
                raw = str(visible)
                if raw.startswith("/") and not raw.startswith(
                    "/System/Volumes/Data/"
                ):
                    data_alias = Path("/System/Volumes/Data" + raw)
                    if data_alias.exists() or raw.startswith(
                        ("/private/", "/Library/", "/Users/")
                    ):
                        expanded.append(data_alias)
    return _deduplicate_paths(expanded)


def _symlink_resolution_paths(path: Path) -> tuple[Path, ...]:
    """Return every caller-visible link in a bounded resolution chain."""

    current = Path(os.path.abspath(os.fspath(path.expanduser())))
    result: list[Path] = [current]
    seen: set[str] = set()
    for _ in range(64):
        prefix = Path(current.anchor)
        rewritten: Path | None = None
        parts = current.parts[1:] if current.is_absolute() else current.parts
        for index, part in enumerate(parts):
            candidate = prefix / part
            try:
                metadata = candidate.lstat()
            except OSError:
                return tuple(result)
            if stat.S_ISLNK(metadata.st_mode):
                key = str(candidate)
                if key in seen:
                    raise VerifierIsolationError(
                        "A verifier read path contains a symlink cycle."
                    )
                seen.add(key)
                result.append(candidate)
                target = Path(os.readlink(candidate))
                base = target if target.is_absolute() else candidate.parent / target
                rewritten = Path(
                    os.path.abspath(
                        os.fspath(base.joinpath(*parts[index + 1 :]))
                    )
                )
                result.append(rewritten)
                break
            prefix = candidate
        if rewritten is None:
            return tuple(result)
        current = rewritten
    raise VerifierIsolationError(
        "A verifier read path exceeded the symlink resolution bound."
    )


def _sandbox_filters(
    paths: Sequence[Path],
    *,
    include_ancestors: bool = True,
) -> str:
    literals: set[str] = {"/"} if include_ancestors else set()
    subpaths: set[str] = set()
    for path in paths:
        raw = str(path)
        literals.add(raw)
        if include_ancestors:
            current = path.parent
            while True:
                literals.add(str(current))
                if current.parent == current:
                    break
                current = current.parent
        if path.is_dir() or not path.exists():
            subpaths.add(raw)
    literal_filters = "".join(
        f"(literal {json.dumps(item)})" for item in sorted(literals)
    )
    subpath_filters = "".join(
        f"(subpath {json.dumps(item)})" for item in sorted(subpaths)
    )
    return literal_filters + subpath_filters


def _deduplicate_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return tuple(result)


def _remove_contained_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    for candidate in sorted(paths, key=lambda item: (len(item.parts), str(item))):
        if not _path_is_covered(candidate, result):
            result.append(candidate)
    return tuple(result)


def _path_is_covered(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        if path == root:
            return True
        try:
            is_directory = root.is_dir()
        except OSError:
            is_directory = False
        if is_directory:
            try:
                path.relative_to(root)
            except ValueError:
                pass
            else:
                return True
    return False


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
