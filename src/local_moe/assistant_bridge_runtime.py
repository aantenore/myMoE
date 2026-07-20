"""Hardened process runtime primitives for the hybrid assistant bridge.

The module is intentionally independent from :mod:`local_moe.assistant_bridge` so
the bridge can adopt it behind its existing public API in a later, reviewable
change.  Its observed-cleanup contract is explicit:

* POSIX launches receive a fresh session/process group.  Every process that
  remains in that group is terminated and its absence is verified.
* When ``psutil`` is installed, descendants observed recursively are tracked and
  cleaned even if they later leave the original process group.  Discovery is
  best effort because a process can detach and exit between observations.
* Windows observed-tree execution requires ``psutil``.  Without it, execution
  is rejected before launch unless a caller explicitly opts out of tree cleanup.

Cleanup verification includes the root process, the POSIX process group,
psutil-observed descendants, and all stdin/stdout/stderr worker threads.  A
verification failure raises :class:`ProcessCleanupError`; partial cleanup is
never reported as a successful execution.  These controls are not an OS
containment primitive: an unobserved process that deliberately detaches can
escape.  Callers needing hard containment must add a job object, cgroup, or
equivalent supervisor outside this module.

Executable and working-directory identities are checked immediately before and
after process creation.  This is fail-closed change detection, not a race-free
launch primitive: a same-user concurrent replacement could begin executing
briefly before the post-launch mismatch is detected and the observed tree is
cleaned.  Native descriptor-based execution or an external OS supervisor is
required to remove that residual window without changing launcher semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import stat
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

try:  # Optional by design; capability reporting below makes absence observable.
    import psutil as _psutil
except ImportError:  # pragma: no cover - availability depends on selected extras.
    _psutil = None


_HASH_CHUNK_BYTES = 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024
_MAX_IO_BYTES = 256 * 1024 * 1024
_CAPABILITIES_SCHEMA_VERSION = "assistant-bridge-runtime-capabilities/v1"
_LAUNCHER_CHAIN_SCHEMA_VERSION = "assistant-bridge-launcher-chain/v1"
_SCRIPT_SUFFIXES = frozenset(
    {
        ".bat",
        ".cmd",
        ".js",
        ".mjs",
        ".pl",
        ".ps1",
        ".py",
        ".rb",
        ".sh",
    }
)
_NATIVE_EXECUTABLE_SUFFIXES = frozenset({".com", ".exe"})
_DANGEROUS_ENVIRONMENT_KEYS = frozenset(
    {
        "BASH_ENV",
        "CLASSPATH",
        "DOTNET_STARTUP_HOOKS",
        "ENV",
        "GCONV_PATH",
        "GIT_ASKPASS",
        "GIT_EXEC_PATH",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_TEMPLATE_DIR",
        "IFS",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "NODE_OPTIONS",
        "NODE_PATH",
        "NODE_REPL_EXTERNAL_MODULE",
        "NPM_CONFIG_NODE_OPTIONS",
        "PERL5LIB",
        "PERL5OPT",
        "PERLLIB",
        "PROMPT_COMMAND",
        "PYTHONBREAKPOINT",
        "PYTHONCASEOK",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONPLATLIBDIR",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONWARNINGS",
        "RUBYLIB",
        "RUBYOPT",
        "SHELLOPTS",
        "SSLKEYLOGFILE",
        "SSH_ASKPASS",
        "SUDO_ASKPASS",
        "ZDOTDIR",
        "_JAVA_OPTIONS",
    }
)
_DANGEROUS_ENVIRONMENT_PREFIXES = (
    "BASH_FUNC_",
    "COMPLUS_",
    "CORECLR_",
    "DYLD_",
    "GIT_CONFIG_",
    "LD_",
)


class AssistantBridgeRuntimeError(RuntimeError):
    """Base error for runtime contract failures."""


class ExecutableResolutionError(AssistantBridgeRuntimeError):
    """The configured executable could not be resolved and attested."""


class ExecutableChangedError(AssistantBridgeRuntimeError):
    """The executable no longer matches its previously resolved identity."""


class WorkingDirectoryChangedError(AssistantBridgeRuntimeError):
    """The execution directory no longer matches its attested identity."""


class LauncherChainError(AssistantBridgeRuntimeError):
    """A declared launcher chain cannot be resolved or safely represented."""


class LauncherChainChangedError(AssistantBridgeRuntimeError):
    """A declared launcher artifact no longer matches its identity."""


class ProcessTreeUnavailableError(AssistantBridgeRuntimeError):
    """The requested process-tree containment contract is unavailable."""


class ProcessObservationError(AssistantBridgeRuntimeError):
    """Required process-tree observation was lost after launch."""


class ProcessLaunchError(AssistantBridgeRuntimeError):
    """The attested executable could not be launched."""


class ProcessLaunchNotAuthorizedError(AssistantBridgeRuntimeError):
    """A launch reservation was denied before process creation."""


class ProcessLaunchLifecycleError(AssistantBridgeRuntimeError):
    """Launch reservation state could not be transitioned safely."""


class ProcessCleanupError(AssistantBridgeRuntimeError):
    """Process or pipe cleanup could not be verified before returning."""

    def __init__(self, message: str, *, details: Mapping[str, object]) -> None:
        super().__init__(message)
        self.details = MappingProxyType(dict(details))


@dataclass(frozen=True)
class _FileAttestation:
    sha256: str
    size_bytes: int
    mtime_ns: int
    device_id: int
    inode: int
    mode: int
    owner_uid: int | None
    owner_gid: int | None

    def binding_tuple(self) -> tuple[object, ...]:
        return (
            self.sha256,
            self.size_bytes,
            self.mtime_ns,
            self.device_id,
            self.inode,
            self.mode,
            self.owner_uid,
            self.owner_gid,
        )


@dataclass(frozen=True)
class ProcessLaunchPermit:
    """Provider-neutral callbacks for one reserved process launch."""

    commit_after_popen: Callable[[], None] = field(repr=False)
    release_after_popen_failure: Callable[[], None] = field(repr=False)

    def __post_init__(self) -> None:
        if not callable(self.commit_after_popen) or not callable(
            self.release_after_popen_failure
        ):
            raise TypeError("Process launch permit callbacks must be callable")

@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Opaque digest of a validated, platform-normalized environment."""

    sha256: str
    variable_count: int
    key_semantics: str
    schema_version: str = "assistant-bridge-environment/v1"

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "variable_count": self.variable_count,
            "key_semantics": self.key_semantics,
        }


@dataclass(frozen=True, init=False)
class RuntimeCapabilities:
    """Current host support and the exact descendant-observation boundary."""

    platform: str
    posix_process_groups: bool
    psutil_available: bool
    strict_tree_supported: bool
    detached_descendant_contract: str
    schema_version: str = _CAPABILITIES_SCHEMA_VERSION

    def __init__(
        self,
        platform: str,
        posix_process_groups: bool,
        psutil_available: bool,
        strict_tree_supported: bool | None = None,
        detached_descendant_contract: str = "",
        *,
        observed_tree_cleanup_supported: bool | None = None,
        schema_version: str = _CAPABILITIES_SCHEMA_VERSION,
    ) -> None:
        """Accept both the legacy strict field and the observed-tree alias."""

        if strict_tree_supported is None and observed_tree_cleanup_supported is None:
            raise TypeError(
                "strict_tree_supported or observed_tree_cleanup_supported is required"
            )
        if (
            strict_tree_supported is not None
            and observed_tree_cleanup_supported is not None
            and strict_tree_supported != observed_tree_cleanup_supported
        ):
            raise ValueError("Runtime capability aliases cannot disagree")
        selected = (
            observed_tree_cleanup_supported
            if strict_tree_supported is None
            else strict_tree_supported
        )
        if not all(
            isinstance(value, bool)
            for value in (posix_process_groups, psutil_available, selected)
        ):
            raise TypeError("Runtime capability flags must be boolean")
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "posix_process_groups", posix_process_groups)
        object.__setattr__(self, "psutil_available", psutil_available)
        object.__setattr__(self, "strict_tree_supported", selected)
        object.__setattr__(
            self, "detached_descendant_contract", detached_descendant_contract
        )
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def observed_tree_cleanup_supported(self) -> bool:
        """Observed-tree name for the legacy strict capability field."""

        return self.strict_tree_supported

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "platform": self.platform,
            "posix_process_groups": self.posix_process_groups,
            "psutil_available": self.psutil_available,
            "observed_tree_cleanup_supported": self.observed_tree_cleanup_supported,
            "strict_tree_supported": self.strict_tree_supported,
            "hard_containment_supported": False,
            "race_free_launch_binding": False,
            "launch_change_detection": "pre_and_post_process_creation",
            "detached_descendant_contract": self.detached_descendant_contract,
        }


@dataclass(frozen=True)
class ExecutableVersionMetadata:
    """Bounded result of an explicit executable version probe."""

    args: tuple[str, ...] = field(repr=False)
    status: str
    returncode: int | None
    text: str = field(repr=False)
    output_sha256: str
    output_bytes: int
    truncated: bool

    def binding_payload(self) -> dict[str, object]:
        """Return the complete private value used for execution bindings."""

        return {
            "args": list(self.args),
            "status": self.status,
            "returncode": self.returncode,
            "text": self.text,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "truncated": self.truncated,
        }

    def payload(self) -> dict[str, object]:
        """Return metadata safe to expose without argv or version output text."""

        return {
            "args_sha256": hashlib.sha256(
                json.dumps(
                    list(self.args),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "status": self.status,
            "returncode": self.returncode,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ExecutableIdentity:
    """Absolute, content-addressed executable selected from a specific PATH."""

    requested: str = field(repr=False)
    launch_path: str = field(repr=False)
    resolved_path: str = field(repr=False)
    sha256: str
    size_bytes: int
    mtime_ns: int = field(repr=False)
    device_id: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)
    owner_uid: int | None = field(repr=False)
    owner_gid: int | None = field(repr=False)
    launch_path_binding_sha256: str = field(repr=False)
    resolution_environment: EnvironmentFingerprint
    version: ExecutableVersionMetadata | None = field(default=None, repr=False)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.binding_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def binding_payload(self) -> dict[str, object]:
        """Return the complete private identity used for execution bindings."""

        return {
            "requested": self.requested,
            "launch_path": self.launch_path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device_id": self.device_id,
            "inode": self.inode,
            "mode": self.mode,
            "owner_uid": self.owner_uid,
            "owner_gid": self.owner_gid,
            "launch_path_binding_sha256": self.launch_path_binding_sha256,
            "resolution_environment": self.resolution_environment.payload(),
            "version": (
                None if self.version is None else self.version.binding_payload()
            ),
        }

    def payload(self) -> dict[str, object]:
        """Return a public identity containing digests instead of local details."""

        metadata_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "device_id": self.device_id,
                    "inode": self.inode,
                    "mode": self.mode,
                    "mtime_ns": self.mtime_ns,
                    "owner_gid": self.owner_gid,
                    "owner_uid": self.owner_uid,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "requested_sha256": hashlib.sha256(
                self.requested.encode("utf-8")
            ).hexdigest(),
            "launch_path_sha256": hashlib.sha256(
                self.launch_path.encode("utf-8")
            ).hexdigest(),
            "resolved_path_sha256": hashlib.sha256(
                self.resolved_path.encode("utf-8")
            ).hexdigest(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata_sha256": metadata_sha256,
            "resolution_environment": self.resolution_environment.payload(),
            "version": None if self.version is None else self.version.payload(),
        }


@dataclass(frozen=True)
class LauncherArtifactIdentity:
    """One explicitly declared, in-place launcher file or companion."""

    role: str
    requested_path: str = field(repr=False)
    launch_path: str = field(repr=False)
    resolved_path: str = field(repr=False)
    sha256: str
    size_bytes: int
    mtime_ns: int = field(repr=False)
    device_id: int = field(repr=False)
    inode: int = field(repr=False)
    mode: int = field(repr=False)
    owner_uid: int | None = field(repr=False)
    owner_gid: int | None = field(repr=False)
    launch_path_binding_sha256: str = field(repr=False)

    def binding_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "requested_path": self.requested_path,
            "launch_path": self.launch_path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device_id": self.device_id,
            "inode": self.inode,
            "mode": self.mode,
            "owner_uid": self.owner_uid,
            "owner_gid": self.owner_gid,
            "launch_path_binding_sha256": self.launch_path_binding_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path_sha256": hashlib.sha256(self.launch_path.encode("utf-8")).hexdigest(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "identity_sha256": hashlib.sha256(
                json.dumps(
                    self.binding_payload(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }


@dataclass(frozen=True)
class LauncherChainIdentity:
    """Provider-neutral binding for an interpreter, entrypoint, and companions."""

    executable_fingerprint: str
    argv: tuple[str, ...] = field(repr=False)
    cwd: str = field(repr=False)
    environment: EnvironmentFingerprint
    entrypoint: LauncherArtifactIdentity | None = field(default=None, repr=False)
    interpreter: ExecutableIdentity | None = field(default=None, repr=False)
    env_launcher: ExecutableIdentity | None = field(default=None, repr=False)
    companions: tuple[LauncherArtifactIdentity, ...] = field(default=(), repr=False)
    shebang: tuple[str, ...] = field(default=(), repr=False)
    strict: bool = True
    schema_version: str = _LAUNCHER_CHAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "companions", tuple(self.companions))
        object.__setattr__(self, "shebang", tuple(self.shebang))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.binding_payload(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "executable_fingerprint": self.executable_fingerprint,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment": self.environment.payload(),
            "entrypoint": (
                None if self.entrypoint is None else self.entrypoint.binding_payload()
            ),
            "interpreter": (
                None if self.interpreter is None else self.interpreter.binding_payload()
            ),
            "env_launcher": (
                None
                if self.env_launcher is None
                else self.env_launcher.binding_payload()
            ),
            "companions": [item.binding_payload() for item in self.companions],
            "shebang": list(self.shebang),
            "strict": self.strict,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "environment": self.environment.payload(),
            "entrypoint": None
            if self.entrypoint is None
            else self.entrypoint.payload(),
            "interpreter": (
                None if self.interpreter is None else self.interpreter.payload()
            ),
            "env_launcher": (
                None if self.env_launcher is None else self.env_launcher.payload()
            ),
            "companions": [item.payload() for item in self.companions],
            "strict": self.strict,
        }


@dataclass(frozen=True)
class ProcessExecutionPolicy:
    """Resource and cleanup bounds for one process-tree execution."""

    stdin_limit_bytes: int = 8 * 1024 * 1024
    stdout_limit_bytes: int = 1024 * 1024
    stderr_limit_bytes: int = 1024 * 1024
    pipe_settle_seconds: float = 0.05
    cleanup_grace_seconds: float = 0.25
    cleanup_kill_seconds: float = 0.75
    poll_interval_seconds: float = 0.01
    require_tree_isolation: bool = True
    require_psutil: bool = False
    require_launcher_chain: bool = False

    def __post_init__(self) -> None:
        for name in (
            "require_tree_isolation",
            "require_psutil",
            "require_launcher_chain",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in (
            "stdin_limit_bytes",
            "stdout_limit_bytes",
            "stderr_limit_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0 <= value <= _MAX_IO_BYTES:
                raise ValueError(f"{name} must be between 0 and {_MAX_IO_BYTES} bytes")
        for name, minimum, maximum in (
            ("pipe_settle_seconds", 0.0, 5.0),
            ("cleanup_grace_seconds", 0.0, 10.0),
            ("cleanup_kill_seconds", 0.05, 10.0),
            ("poll_interval_seconds", 0.001, 0.25),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True)
class CleanupReport:
    """Evidence that the observed tree and all owned pipe workers are gone."""

    attempted: bool
    verified: bool
    methods: tuple[str, ...]
    observed_descendants: int
    process_group_verified: bool | None
    psutil_verified: bool | None
    root_reaped: bool
    pipe_threads_joined: bool
    observation_verified: bool | None = None
    fallback_used: bool = False

    def payload(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "verified": self.verified,
            "verification_scope": "observed_process_tree",
            "hard_containment": False,
            "methods": list(self.methods),
            "observed_descendants": self.observed_descendants,
            "process_group_verified": self.process_group_verified,
            "psutil_verified": self.psutil_verified,
            "observation_verified": self.observation_verified,
            "root_reaped": self.root_reaped,
            "pipe_threads_joined": self.pipe_threads_joined,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class ProcessExecutionResult:
    """Bounded process outcome returned only after cleanup verification."""

    code: str
    returncode: int | None
    timed_out: bool
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)
    stdout_bytes: int
    stderr_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    stdout_truncated: bool
    stderr_truncated: bool
    stdin_bytes_written: int
    execution_duration_ms: int
    duration_ms: int
    executable: ExecutableIdentity = field(repr=False)
    environment: EnvironmentFingerprint
    cleanup: CleanupReport

    @property
    def ok(self) -> bool:
        return self.code == "completed"

    def payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "stdin_bytes_written": self.stdin_bytes_written,
            "execution_duration_ms": self.execution_duration_ms,
            "duration_ms": self.duration_ms,
            "executable": self.executable.payload(),
            "environment": self.environment.payload(),
            "cleanup": self.cleanup.payload(),
        }


def runtime_capabilities() -> RuntimeCapabilities:
    """Describe host tree-control support without performing a launch."""

    posix_groups = os.name == "posix"
    psutil_available = _psutil is not None
    if psutil_available:
        detached_contract = (
            "psutil recursively observes descendants while their ancestry is visible; "
            "already observed processes remain cleanup targets after detaching"
        )
    else:
        detached_contract = (
            "psutil is absent; detached descendants are not observable beyond the "
            "POSIX process-group boundary"
        )
    return RuntimeCapabilities(
        platform=platform.system() or os.name,
        posix_process_groups=posix_groups,
        psutil_available=psutil_available,
        observed_tree_cleanup_supported=posix_groups or psutil_available,
        detached_descendant_contract=detached_contract,
    )


def fingerprint_environment(
    env: Mapping[str, str] | None = None,
) -> EnvironmentFingerprint:
    """Hash environment semantics without retaining keys or values in the result."""

    normalized = _normalize_environment(os.environ if env is None else env)
    semantics = "case-insensitive-uppercase" if os.name == "nt" else "case-sensitive"
    serialized = json.dumps(
        {
            "schema_version": "assistant-bridge-environment/v1",
            "key_semantics": semantics,
            "variables": sorted(normalized.items()),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return EnvironmentFingerprint(
        sha256=hashlib.sha256(serialized).hexdigest(),
        variable_count=len(normalized),
        key_semantics=semantics,
    )


def resolve_executable(
    executable: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
) -> ExecutableIdentity:
    """Resolve one bare name or absolute path to a content-addressed identity."""

    requested = os.fspath(executable)
    if not requested or "\x00" in requested:
        raise ExecutableResolutionError("Executable name is empty or contains NUL")
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) != 1:
        raise ExecutableResolutionError(
            "Executable must be an absolute path or a bare PATH entry name"
        )
    launch_env = _normalize_environment(os.environ if env is None else env)
    search_path = launch_env.get("PATH", os.defpath)
    selected = shutil.which(requested, path=search_path)
    if selected is None:
        raise ExecutableResolutionError(f"Executable is unavailable: {requested}")
    try:
        launch_path = Path(os.path.abspath(os.fspath(Path(selected).expanduser())))
        resolved = launch_path.resolve(strict=True)
    except OSError as exc:
        raise ExecutableResolutionError(
            f"Executable could not be resolved: {requested}"
        ) from exc
    if not resolved.is_absolute():
        raise ExecutableResolutionError("Resolved executable path is not absolute")
    attestation = _attest_executable_file(resolved)
    return ExecutableIdentity(
        requested=requested,
        launch_path=str(launch_path),
        resolved_path=str(resolved),
        sha256=attestation.sha256,
        size_bytes=attestation.size_bytes,
        mtime_ns=attestation.mtime_ns,
        device_id=attestation.device_id,
        inode=attestation.inode,
        mode=attestation.mode,
        owner_uid=attestation.owner_uid,
        owner_gid=attestation.owner_gid,
        launch_path_binding_sha256=_launch_path_binding(
            launch_path,
            expected_resolved=resolved,
        ),
        resolution_environment=fingerprint_environment(launch_env),
    )


def inspect_executable(
    executable: str | os.PathLike[str],
    *,
    env: Mapping[str, str] | None = None,
    version_args: Sequence[str] = ("--version",),
    version_timeout_seconds: float = 3.0,
    version_output_limit_bytes: int = 32 * 1024,
    policy: ProcessExecutionPolicy | None = None,
) -> ExecutableIdentity:
    """Resolve, hash, and run a bounded version probe for an executable."""

    identity = resolve_executable(executable, env=env)
    args = _validate_args(version_args)
    if not args:
        raise ValueError("version_args cannot be empty")
    if not 1 <= version_output_limit_bytes <= _MAX_IO_BYTES:
        raise ValueError("version_output_limit_bytes is outside the supported range")
    probe_policy = policy or ProcessExecutionPolicy(
        stdin_limit_bytes=0,
        stdout_limit_bytes=version_output_limit_bytes,
        stderr_limit_bytes=version_output_limit_bytes,
    )
    result = execute_process(
        identity,
        args,
        env=env,
        timeout_seconds=version_timeout_seconds,
        policy=probe_policy,
    )
    combined = (
        result.stdout
        + (b"\n" if result.stdout and result.stderr else b"")
        + result.stderr
    )
    version = ExecutableVersionMetadata(
        args=args,
        status=result.code,
        returncode=result.returncode,
        text=_normalize_version_text(combined),
        output_sha256=hashlib.sha256(combined).hexdigest(),
        output_bytes=result.stdout_bytes + result.stderr_bytes,
        truncated=result.stdout_truncated or result.stderr_truncated,
    )
    return replace(identity, version=version)


def resolve_launcher_chain(
    executable: ExecutableIdentity,
    args: Sequence[str] = (),
    *,
    entrypoint: str | os.PathLike[str] | None = None,
    companions: Sequence[str | os.PathLike[str]] = (),
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    strict: bool = True,
) -> LauncherChainIdentity:
    """Resolve an explicit, provider-neutral launcher chain in place.

    The returned identity never stages or rewrites the executable, entrypoint,
    or companions.  Direct shebang scripts therefore retain their original
    ``$0``/``dirname`` behavior, while interpreter-driven scripts retain the
    exact argv token supplied by the caller.  In strict mode, every argv value
    that resolves to a script or executable path must match the declared
    entrypoint or one of its companions, and direct scripts must expose a
    resolvable interpreter before launch.
    """

    if not isinstance(strict, bool):
        raise TypeError("strict must be boolean")
    argv = _validate_args(args)
    launch_env = _normalize_environment(os.environ if env is None else env)
    root = Path.cwd().resolve() if cwd is None else _validate_cwd(cwd)
    environment = fingerprint_environment(launch_env)
    executable_path = Path(executable.resolved_path)
    direct_shebang = _read_shebang(executable_path)
    declared_entrypoint: LauncherArtifactIdentity | None = None
    interpreter: ExecutableIdentity | None = None
    env_launcher: ExecutableIdentity | None = None
    shebang: tuple[str, ...] = ()

    if direct_shebang:
        declared_entrypoint = _attest_launcher_artifact(
            executable.launch_path,
            cwd=root,
            role="entrypoint",
        )
        if entrypoint is not None:
            explicit = _attest_launcher_artifact(
                entrypoint,
                cwd=root,
                role="entrypoint",
            )
            if explicit.resolved_path != declared_entrypoint.resolved_path:
                raise LauncherChainError(
                    "Declared entrypoint does not match the direct script executable"
                )
        shebang, env_launcher, interpreter = _resolve_shebang_chain(
            direct_shebang,
            env=launch_env,
            strict=strict,
        )
    elif entrypoint is not None:
        declared_entrypoint = _attest_launcher_artifact(
            entrypoint,
            cwd=root,
            role="entrypoint",
        )
        if not _argv_declares_artifact(argv, declared_entrypoint, cwd=root):
            raise LauncherChainError(
                "Declared launcher entrypoint is not present in process arguments"
            )
        interpreter = executable
    elif strict:
        if executable_path.suffix.lower() in {".bat", ".cmd"}:
            raise LauncherChainError(
                "Direct command scripts require an explicit native interpreter"
            )

    companion_identities = tuple(
        _attest_launcher_artifact(value, cwd=root, role="companion")
        for value in companions
    )
    resolved_paths = [item.resolved_path for item in companion_identities]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise LauncherChainError("Launcher companions contain duplicate files")
    if declared_entrypoint is not None and declared_entrypoint.resolved_path in set(
        resolved_paths
    ):
        raise LauncherChainError("Launcher entrypoint cannot also be a companion")
    if strict:
        declared = (
            (() if declared_entrypoint is None else (declared_entrypoint,))
            + companion_identities
        )
        undeclared = _undeclared_executable_argument(
            argv,
            cwd=root,
            declared=declared,
        )
        if undeclared is not None:
            raise LauncherChainError(
                "Script or executable launcher arguments must declare an attested "
                "entrypoint or companion"
            )

    return LauncherChainIdentity(
        executable_fingerprint=executable.fingerprint,
        argv=argv,
        cwd=str(root),
        environment=environment,
        entrypoint=declared_entrypoint,
        interpreter=interpreter,
        env_launcher=env_launcher,
        companions=companion_identities,
        shebang=shebang,
        strict=strict,
    )


def verify_launcher_chain(
    chain: LauncherChainIdentity,
    executable: ExecutableIdentity,
    args: Sequence[str] = (),
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Verify a declared chain without changing any launch path or argv token."""

    argv = _validate_args(args)
    launch_env = _normalize_environment(os.environ if env is None else env)
    root = Path.cwd().resolve() if cwd is None else _validate_cwd(cwd)
    if executable.fingerprint != chain.executable_fingerprint:
        raise LauncherChainChangedError(
            "Launcher chain is bound to a different executable identity"
        )
    if argv != chain.argv or str(root) != chain.cwd:
        raise LauncherChainChangedError("Launcher argv or working directory changed")
    if fingerprint_environment(launch_env).sha256 != chain.environment.sha256:
        raise LauncherChainChangedError("Launcher environment changed")
    if chain.entrypoint is not None:
        _verify_launcher_artifact(chain.entrypoint)
    for companion in chain.companions:
        _verify_launcher_artifact(companion)
    if chain.env_launcher is not None:
        _verify_executable_identity(chain.env_launcher)
    if chain.interpreter is not None:
        _verify_executable_identity(chain.interpreter)


def execute_process(
    executable: ExecutableIdentity,
    args: Sequence[str] = (),
    *,
    stdin: bytes = b"",
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float,
    policy: ProcessExecutionPolicy | None = None,
    launcher_chain: LauncherChainIdentity | None = None,
    reserve_launch: Callable[[], ProcessLaunchPermit | None] | None = None,
) -> ProcessExecutionResult:
    """Execute an attested binary under one end-to-end monotonic deadline.

    The execution deadline begins before executable re-attestation and includes
    process creation plus concurrent stdin, stdout, and stderr handling.  Safety
    cleanup has its own small, bounded grace/kill budget so a timed-out process
    can still be removed and verified before this function returns.
    """

    if isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= 86_400:
        raise ValueError(
            "timeout_seconds must be greater than zero and at most one day"
        )
    selected_policy = policy or ProcessExecutionPolicy()
    capabilities = runtime_capabilities()
    if selected_policy.require_psutil and not capabilities.psutil_available:
        raise ProcessTreeUnavailableError(
            "Execution policy requires psutil, but the optional dependency is absent"
        )
    if (
        selected_policy.require_tree_isolation
        and not capabilities.observed_tree_cleanup_supported
    ):
        raise ProcessTreeUnavailableError(
            "Observed process-tree cleanup is unavailable on this host without psutil"
        )
    argv_tail = _validate_args(args)
    if not isinstance(stdin, bytes):
        raise TypeError("stdin must be bytes")
    if len(stdin) > selected_policy.stdin_limit_bytes:
        raise ValueError("stdin exceeds the configured input bound")
    launch_env = _normalize_environment(os.environ if env is None else env)
    environment = fingerprint_environment(launch_env)
    launch_cwd = None if cwd is None else _validate_cwd(cwd)
    cwd_identity = None if launch_cwd is None else _attest_working_directory(launch_cwd)
    started = time.monotonic()
    deadline = started + timeout_seconds
    _verify_executable_identity(executable)
    if launcher_chain is not None:
        if selected_policy.require_launcher_chain and not launcher_chain.strict:
            raise LauncherChainError(
                "Execution policy requires a strict launcher-chain attestation"
            )
        verify_launcher_chain(
            launcher_chain,
            executable,
            argv_tail,
            cwd=launch_cwd,
            env=launch_env,
        )
    elif selected_policy.require_launcher_chain:
        raise LauncherChainError(
            "Execution policy requires an explicitly attested launcher chain"
        )
    if time.monotonic() >= deadline:
        raise ProcessLaunchError(
            "Execution deadline elapsed during executable attestation"
        )

    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": None if launch_cwd is None else str(launch_cwd),
        "env": launch_env,
        "shell": False,
        "bufsize": 0,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    process: subprocess.Popen[bytes] | None = None
    tracker: _ProcessTracker | None = None
    workers: tuple[threading.Thread, ...] = ()
    stdout_state: _BoundedStream | None = None
    stderr_state: _BoundedStream | None = None
    stdin_state: _StdinState | None = None
    root_exited_at: float | None = None
    terminal = "completed"
    cleanup: CleanupReport | None = None
    pending_error: BaseException | None = None
    execution_ended = started
    try:
        permit: ProcessLaunchPermit | None = None
        try:
            if launch_cwd is not None and cwd_identity is not None:
                _verify_working_directory(launch_cwd, cwd_identity)
            if reserve_launch is not None:
                try:
                    permit = reserve_launch()
                except Exception:
                    raise ProcessLaunchLifecycleError(
                        "Process launch reservation failed"
                    ) from None
                if permit is None:
                    raise ProcessLaunchNotAuthorizedError(
                        "Process launch was not authorized"
                    )
                if not isinstance(permit, ProcessLaunchPermit):
                    raise ProcessLaunchLifecycleError(
                        "Process launch reservation returned an invalid permit"
                    )
            try:
                process = subprocess.Popen(
                    [executable.launch_path, *argv_tail],
                    **popen_kwargs,
                )
            except OSError:
                if permit is not None:
                    try:
                        permit.release_after_popen_failure()
                    except Exception:
                        raise ProcessLaunchLifecycleError(
                            "Process launch reservation release failed"
                        ) from None
                raise ProcessLaunchError(
                    "Could not launch the attested executable"
                ) from None
        except WorkingDirectoryChangedError:
            raise

        # From the assignment above onward the process is owned by this outer
        # try/finally.  Tracker, state, and worker construction may all fail;
        # none may bypass the tracker-independent emergency reaper.
        tracker = _ProcessTracker(process.pid)
        if selected_policy.require_psutil and not tracker.observation_verified:
            raise ProcessObservationError(
                "Required psutil observation failed during process ownership setup"
            )
        if permit is not None:
            try:
                permit.commit_after_popen()
            except Exception:
                raise ProcessLaunchLifecycleError(
                    "Process launch reservation commit failed"
                ) from None
        try:
            _verify_executable_identity(executable)
            if launcher_chain is not None:
                verify_launcher_chain(
                    launcher_chain,
                    executable,
                    argv_tail,
                    cwd=launch_cwd,
                    env=launch_env,
                )
            if launch_cwd is not None and cwd_identity is not None:
                _verify_working_directory(launch_cwd, cwd_identity)
        except (
            ExecutableChangedError,
            LauncherChainChangedError,
            WorkingDirectoryChangedError,
        ) as exc:
            subject = (
                "working directory"
                if isinstance(exc, WorkingDirectoryChangedError)
                else "launcher chain"
                if isinstance(exc, LauncherChainChangedError)
                else "executable identity"
            )
            raise ProcessLaunchError(
                f"Execution {subject} changed during process creation"
            ) from exc

        wake = threading.Event()
        stdout_state = _BoundedStream(selected_policy.stdout_limit_bytes)
        stderr_state = _BoundedStream(selected_policy.stderr_limit_bytes)
        stdin_state = _StdinState(len(stdin))
        workers = (
            threading.Thread(
                target=_read_pipe,
                name=f"bridge-stdout-{process.pid}",
                args=(process.stdout, stdout_state, wake),
                daemon=True,
            ),
            threading.Thread(
                target=_read_pipe,
                name=f"bridge-stderr-{process.pid}",
                args=(process.stderr, stderr_state, wake),
                daemon=True,
            ),
            threading.Thread(
                target=_write_stdin,
                name=f"bridge-stdin-{process.pid}",
                args=(process.stdin, stdin, stdin_state, wake),
                daemon=True,
            ),
        )
        for worker in workers:
            worker.start()
        while True:
            tracker.observe()
            if selected_policy.require_psutil and not tracker.observation_verified:
                raise ProcessObservationError(
                    "Required psutil process-tree observation was lost"
                )
            now = time.monotonic()
            if stdout_state.overflowed:
                terminal = "stdout_limit_exceeded"
                break
            if stderr_state.overflowed:
                terminal = "stderr_limit_exceeded"
                break
            if stdout_state.failed or stderr_state.failed or stdin_state.failed:
                terminal = "io_failed"
                break
            if now >= deadline:
                terminal = "timed_out"
                break
            returncode = process.poll()
            if returncode is not None:
                if all(not worker.is_alive() for worker in workers):
                    terminal = "completed" if returncode == 0 else "nonzero_exit"
                    break
                if root_exited_at is None:
                    root_exited_at = now
                elif now - root_exited_at >= selected_policy.pipe_settle_seconds:
                    terminal = "completed" if returncode == 0 else "nonzero_exit"
                    break
            wait_seconds = min(
                selected_policy.poll_interval_seconds,
                max(deadline - now, 0.001),
            )
            wake.wait(wait_seconds)
            wake.clear()
    except BaseException as exc:
        pending_error = exc
    finally:
        execution_ended = time.monotonic()
        if process is not None:
            try:
                if pending_error is not None:
                    cleanup = _emergency_cleanup_process(
                        process,
                        tracker=tracker,
                        workers=workers,
                        policy=selected_policy,
                    )
                    if not cleanup.verified:
                        raise ProcessCleanupError(
                            "Emergency process cleanup could not be verified",
                            details=cleanup.payload(),
                        ) from pending_error
                else:
                    cleanup = _cleanup_owned_process(
                        process,
                        tracker=tracker,
                        workers=workers,
                        policy=selected_policy,
                    )
            except BaseException as exc:
                if pending_error is not None:
                    try:
                        exc.add_note(
                            f"Execution error before cleanup: {type(pending_error).__name__}"
                        )
                    except AttributeError:  # Python 3.10 has no BaseException.add_note.
                        pass
                pending_error = exc

    if pending_error is not None:
        if isinstance(pending_error, (KeyboardInterrupt, SystemExit)):
            raise pending_error
        if isinstance(pending_error, AssistantBridgeRuntimeError):
            raise pending_error
        raise ProcessCleanupError(
            "Process execution failed after ownership was acquired",
            details=(
                cleanup.payload()
                if cleanup is not None
                else _unverified_cleanup_payload(process, tracker, workers)
            ),
        ) from pending_error

    assert process is not None
    assert cleanup is not None
    assert stdout_state is not None
    assert stderr_state is not None
    assert stdin_state is not None
    if terminal in {"completed", "nonzero_exit"}:
        # A short-lived process can exit before a contended reader observes all
        # buffered bytes. Cleanup joins the readers, so classify their final
        # state here as well as in the live polling loop.
        if stdout_state.overflowed:
            terminal = "stdout_limit_exceeded"
        elif stderr_state.overflowed:
            terminal = "stderr_limit_exceeded"
        elif (
            stdout_state.failed
            or stderr_state.failed
            or not stdout_state.reached_eof
            or not stderr_state.reached_eof
            or not stdin_state.complete
        ):
            terminal = "io_failed"

    stdout = stdout_state.snapshot()
    stderr = stderr_state.snapshot()
    finished = time.monotonic()
    return ProcessExecutionResult(
        code=terminal,
        returncode=process.returncode,
        timed_out=terminal == "timed_out",
        stdout=stdout,
        stderr=stderr,
        stdout_bytes=stdout_state.count,
        stderr_bytes=stderr_state.count,
        stdout_sha256=stdout_state.hexdigest,
        stderr_sha256=stderr_state.hexdigest,
        stdout_truncated=stdout_state.overflowed,
        stderr_truncated=stderr_state.overflowed,
        stdin_bytes_written=stdin_state.count,
        execution_duration_ms=_duration_ms(started, execution_ended),
        duration_ms=_duration_ms(started, finished),
        executable=executable,
        environment=environment,
        cleanup=cleanup,
    )


class _BoundedStream:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.overflowed = False
        self.failed = False
        self.reached_eof = False
        self._retained = bytearray()
        self._digest = hashlib.sha256()

    def update(self, chunk: bytes) -> None:
        self.count += len(chunk)
        self._digest.update(chunk)
        remaining = self.limit - len(self._retained)
        if remaining > 0:
            self._retained.extend(chunk[:remaining])
        if self.count > self.limit:
            self.overflowed = True

    def snapshot(self) -> bytes:
        return bytes(self._retained)

    def fail(self) -> None:
        self.failed = True

    def finish(self) -> None:
        self.reached_eof = True

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _StdinState:
    def __init__(self, expected: int) -> None:
        self.count = 0
        self.expected = expected
        self.failed = False
        self.flushed = expected == 0

    @property
    def complete(self) -> bool:
        return not self.failed and self.count == self.expected and self.flushed

    def fail(self) -> None:
        self.failed = True


class _ProcessTracker:
    """Retain psutil process handles after their ancestry becomes unavailable."""

    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self._observed: dict[tuple[int, float], Any] = {}
        self._observation_failed = False
        self.observe()

    @property
    def observed_descendants(self) -> int:
        return max(len(self._observed) - 1, 0)

    @property
    def observation_verified(self) -> bool:
        return not self._observation_failed

    @property
    def observed_handles(self) -> tuple[Any, ...]:
        return tuple(self._observed.values())

    def observe(self) -> None:
        if _psutil is None:
            return
        seeds: list[Any] = list(self._observed.values())
        try:
            seeds.append(_psutil.Process(self.root_pid))
        except _psutil.NoSuchProcess:
            pass
        except _psutil.Error:
            self._observation_failed = True
        for process in seeds:
            self._remember(process)
            try:
                children = process.children(recursive=True)
            except _psutil.NoSuchProcess:
                continue
            except _psutil.Error:
                self._observation_failed = True
                continue
            for child in children:
                self._remember(child)

    def live(self) -> tuple[Any, ...]:
        if _psutil is None:
            return ()
        live: list[Any] = []
        for process in self._observed.values():
            try:
                if not process.is_running():
                    continue
                if process.status() == _psutil.STATUS_ZOMBIE:
                    continue
            except (_psutil.NoSuchProcess, _psutil.ZombieProcess):
                continue
            except _psutil.Error:
                self._observation_failed = True
                live.append(process)
                continue
            live.append(process)
        return tuple(live)

    def _remember(self, process: Any) -> None:
        try:
            key = (int(process.pid), float(process.create_time()))
        except _psutil.NoSuchProcess:
            return
        except _psutil.Error:
            self._observation_failed = True
            return
        self._observed[key] = process


def _cleanup_owned_process(
    process: subprocess.Popen[bytes],
    *,
    tracker: _ProcessTracker | None,
    workers: Sequence[threading.Thread],
    policy: ProcessExecutionPolicy,
) -> CleanupReport:
    """Run primary cleanup, then a tracker-independent emergency fallback."""

    primary_error: BaseException | None = None
    if tracker is not None:
        try:
            return _cleanup_process_tree(
                process,
                tracker=tracker,
                workers=workers,
                policy=policy,
            )
        except BaseException as exc:
            primary_error = exc
    else:
        primary_error = RuntimeError("Process tracker was not initialized")

    report = _emergency_cleanup_process(
        process,
        tracker=tracker,
        workers=workers,
        policy=policy,
    )
    details = report.payload()
    details["primary_cleanup_failed"] = True
    raise ProcessCleanupError(
        "Primary process cleanup failed; emergency reaping was applied",
        details=details,
    ) from primary_error


def _emergency_cleanup_process(
    process: subprocess.Popen[bytes],
    *,
    tracker: _ProcessTracker | None,
    workers: Sequence[threading.Thread],
    policy: ProcessExecutionPolicy,
) -> CleanupReport:
    methods: list[str] = []
    if os.name == "posix":
        try:
            _signal_process_group(process.pid, signal.SIGKILL)
            methods.append("emergency-posix-process-group-kill")
        except BaseException:
            pass
    if tracker is not None and _psutil is not None:
        for tracked in tracker.observed_handles:
            try:
                tracked.kill()
                methods.append("emergency-psutil-kill")
            except BaseException:
                pass
    try:
        if process.poll() is None:
            process.kill()
            methods.append("emergency-root-kill")
    except BaseException:
        pass
    for stream in (process.stdin, process.stdout, process.stderr):
        try:
            _close_pipe(stream)
        except BaseException:
            pass
    try:
        process.wait(timeout=policy.cleanup_kill_seconds)
    except BaseException:
        pass
    for worker in workers:
        try:
            if worker.ident is not None:
                worker.join(timeout=policy.poll_interval_seconds * 2)
        except BaseException:
            pass

    try:
        root_reaped = process.poll() is not None
    except BaseException:
        root_reaped = False
    if os.name == "posix":
        try:
            group_verified: bool | None = not _process_group_alive(process.pid)
        except BaseException:
            group_verified = False
    else:
        group_verified = None
    observation_verified = (
        None if tracker is None or _psutil is None else tracker.observation_verified
    )
    psutil_verified: bool | None
    if tracker is None or _psutil is None:
        psutil_verified = None
    else:
        try:
            tracked_live = tracker.live()
            observation_verified = tracker.observation_verified
            psutil_verified = bool(observation_verified and not tracked_live)
        except BaseException:
            observation_verified = False
            psutil_verified = False
    pipe_threads_joined = True
    for worker in workers:
        try:
            pipe_threads_joined = pipe_threads_joined and not worker.is_alive()
        except BaseException:
            pipe_threads_joined = False
    verified = bool(
        root_reaped
        and group_verified is not False
        and pipe_threads_joined
        and psutil_verified is not False
        and (
            os.name == "posix"
            or not policy.require_tree_isolation
            or psutil_verified is True
        )
        and (not policy.require_psutil or psutil_verified is True)
    )
    return CleanupReport(
        attempted=True,
        verified=verified,
        methods=tuple(dict.fromkeys(methods)),
        observed_descendants=(0 if tracker is None else tracker.observed_descendants),
        process_group_verified=group_verified,
        psutil_verified=psutil_verified,
        root_reaped=root_reaped,
        pipe_threads_joined=pipe_threads_joined,
        observation_verified=observation_verified,
        fallback_used=True,
    )


def _unverified_cleanup_payload(
    process: subprocess.Popen[bytes] | None,
    tracker: _ProcessTracker | None,
    workers: Sequence[threading.Thread],
) -> dict[str, object]:
    return {
        "attempted": process is not None,
        "verified": False,
        "verification_scope": "observed_process_tree",
        "hard_containment": False,
        "methods": [],
        "observed_descendants": (
            0 if tracker is None else tracker.observed_descendants
        ),
        "process_group_verified": None,
        "psutil_verified": None,
        "observation_verified": (
            None if tracker is None else tracker.observation_verified
        ),
        "root_reaped": False,
        "pipe_threads_joined": False,
        "fallback_used": True,
    }


def _cleanup_process_tree(
    process: subprocess.Popen[bytes],
    *,
    tracker: _ProcessTracker,
    workers: Sequence[threading.Thread],
    policy: ProcessExecutionPolicy,
) -> CleanupReport:
    methods: list[str] = []
    tracker.observe()
    group_alive = _process_group_alive(process.pid) if os.name == "posix" else False
    tracked_alive = tracker.live()
    root_alive = process.poll() is None
    pipe_alive = any(worker.is_alive() for worker in workers)
    attempted = root_alive or group_alive or bool(tracked_alive) or pipe_alive

    group_signaled = False
    if os.name == "posix" and group_alive:
        group_signaled = _attempt_process_group_signal(
            process.pid,
            signal.SIGTERM,
        )
        if group_signaled:
            methods.append("posix-process-group-term")
    if root_alive and not group_signaled:
        try:
            process.terminate()
            methods.append("root-terminate")
        except OSError:
            pass
    if _psutil is not None:
        for tracked in tracked_alive:
            try:
                tracked.terminate()
            except _psutil.Error:
                continue
        if tracked_alive:
            methods.append("psutil-descendant-term")

    _wait_for_cleanup(
        process,
        tracker=tracker,
        workers=workers,
        timeout_seconds=policy.cleanup_grace_seconds,
        poll_interval_seconds=policy.poll_interval_seconds,
    )
    tracker.observe()
    group_alive = _process_group_alive(process.pid) if os.name == "posix" else False
    tracked_alive = tracker.live()
    root_alive = process.poll() is None
    group_signaled = False
    if os.name == "posix" and group_alive:
        group_signaled = _attempt_process_group_signal(
            process.pid,
            signal.SIGKILL,
        )
        if group_signaled:
            methods.append("posix-process-group-kill")
    if root_alive and not group_signaled:
        try:
            process.kill()
            methods.append("root-kill")
        except OSError:
            pass
    if _psutil is not None:
        for tracked in tracked_alive:
            try:
                tracked.kill()
            except _psutil.Error:
                continue
        if tracked_alive:
            methods.append("psutil-descendant-kill")

    _wait_for_cleanup(
        process,
        tracker=tracker,
        workers=workers,
        timeout_seconds=policy.cleanup_kill_seconds,
        poll_interval_seconds=policy.poll_interval_seconds,
    )
    _close_pipe(process.stdin)
    _close_pipe(process.stdout)
    _close_pipe(process.stderr)
    for worker in workers:
        if worker.ident is not None:
            worker.join(timeout=policy.poll_interval_seconds * 2)
    try:
        if process.poll() is None:
            process.wait(timeout=policy.poll_interval_seconds)
    except subprocess.TimeoutExpired:
        pass

    tracker.observe()
    root_reaped = process.poll() is not None
    group_verified = (
        not _process_group_alive(process.pid) if os.name == "posix" else None
    )
    final_tracked_alive = tracker.live() if _psutil is not None else ()
    observation_verified = tracker.observation_verified if _psutil is not None else None
    psutil_verified = (
        bool(observation_verified and not final_tracked_alive)
        if _psutil is not None
        else None
    )
    pipe_threads_joined = all(not worker.is_alive() for worker in workers)
    verified = bool(
        root_reaped
        and (group_verified is not False)
        and (psutil_verified is not False)
        and pipe_threads_joined
    )
    report = CleanupReport(
        attempted=attempted,
        verified=verified,
        methods=tuple(dict.fromkeys(methods)),
        observed_descendants=tracker.observed_descendants,
        process_group_verified=group_verified,
        psutil_verified=psutil_verified,
        root_reaped=root_reaped,
        pipe_threads_joined=pipe_threads_joined,
        observation_verified=observation_verified,
        fallback_used=False,
    )
    if not verified:
        raise ProcessCleanupError(
            "Process-tree cleanup could not be verified",
            details=report.payload(),
        )
    return report


def _wait_for_cleanup(
    process: subprocess.Popen[bytes],
    *,
    tracker: _ProcessTracker,
    workers: Sequence[threading.Thread],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        tracker.observe()
        root_alive = process.poll() is None
        group_alive = _process_group_alive(process.pid) if os.name == "posix" else False
        tracked_alive = bool(tracker.live())
        pipe_alive = any(worker.is_alive() for worker in workers)
        if not root_alive and not group_alive and not tracked_alive and not pipe_alive:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(poll_interval_seconds, remaining))


def _read_pipe(
    stream: Any,
    state: _BoundedStream,
    wake: threading.Event,
) -> None:
    try:
        if stream is None:
            return
        while True:
            try:
                chunk = stream.read(_PIPE_CHUNK_BYTES)
            except (OSError, ValueError):
                state.fail()
                return
            if not chunk:
                state.finish()
                return
            state.update(chunk)
            wake.set()
    finally:
        wake.set()


def _write_stdin(
    stream: Any,
    content: bytes,
    state: _StdinState,
    wake: threading.Event,
) -> None:
    try:
        if stream is None:
            if state.expected:
                state.fail()
            return
        if not content:
            return
        view = memoryview(content)
        while state.count < len(view):
            try:
                written = stream.write(
                    view[state.count : state.count + _PIPE_CHUNK_BYTES]
                )
            except (BrokenPipeError, OSError, ValueError):
                state.fail()
                return
            if written is None:
                written = 0
            if written <= 0:
                state.fail()
                return
            state.count += written
        try:
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            state.fail()
            return
        state.flushed = True
    finally:
        _close_pipe(stream)
        wake.set()


def _normalize_environment(env: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Environment keys and values must be strings")
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise ValueError("Environment contains an invalid key or value")
        validate_environment_name(key)
        normalized_key = key.upper() if os.name == "nt" else key
        if normalized_key in normalized and normalized[normalized_key] != value:
            raise ValueError(
                "Environment contains conflicting platform-equivalent keys"
            )
        normalized[normalized_key] = value
    return normalized


def validate_environment_name(name: str) -> None:
    """Reject environment variables that can inject code or runtime loaders."""

    if not isinstance(name, str):
        raise TypeError("Environment variable name must be a string")
    normalized = name.upper()
    if normalized in _DANGEROUS_ENVIRONMENT_KEYS or any(
        normalized.startswith(prefix) for prefix in _DANGEROUS_ENVIRONMENT_PREFIXES
    ):
        raise ValueError(
            "Environment variable is denied by the process injection policy"
        )


def _validate_args(args: Sequence[str]) -> tuple[str, ...]:
    if isinstance(args, (str, bytes)):
        raise TypeError("Process arguments must be a sequence of strings")
    validated: list[str] = []
    for argument in args:
        if not isinstance(argument, str):
            raise TypeError("Process arguments must be strings")
        if "\x00" in argument:
            raise ValueError("Process argument contains NUL")
        validated.append(argument)
    return tuple(validated)


def _validate_cwd(cwd: str | os.PathLike[str]) -> Path:
    try:
        resolved = Path(cwd).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("Execution working directory is unavailable") from exc
    if not resolved.is_dir():
        raise ValueError("Execution working directory is not a directory")
    return resolved


def _attest_working_directory(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise WorkingDirectoryChangedError(
            "Execution working directory is unavailable"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise WorkingDirectoryChangedError(
            "Execution working directory is not a stable directory"
        )
    return int(metadata.st_dev), int(metadata.st_ino)


def _verify_working_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = _attest_working_directory(path)
    except WorkingDirectoryChangedError:
        raise
    if current != identity:
        raise WorkingDirectoryChangedError(
            "Execution working directory changed after attestation"
        )


def _regular_file_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )


def _stat_identity(metadata: os.stat_result) -> tuple[object, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        getattr(metadata, "st_uid", None),
        getattr(metadata, "st_gid", None),
    )


def _open_regular_file_descriptor(path: Path) -> tuple[int, os.stat_result]:
    descriptor = -1
    try:
        descriptor = os.open(path, _regular_file_open_flags())
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(errno.EINVAL, "Path is not a regular file", path)
        return descriptor, metadata
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _read_stable_first_line(path: Path) -> bytes:
    descriptor, before = _open_regular_file_descriptor(path)
    first_line = bytearray()
    try:
        while len(first_line) < 4097:
            chunk = os.read(descriptor, 4097 - len(first_line))
            if not chunk:
                break
            newline = chunk.find(b"\n")
            if newline >= 0:
                first_line.extend(chunk[: newline + 1])
                break
            first_line.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise OSError(
            getattr(errno, "ESTALE", errno.EIO),
            "File changed while its first line was read",
            path,
        )
    return bytes(first_line)


def _parse_shebang(first_line: bytes) -> tuple[str, ...]:
    if not first_line.startswith(b"#!"):
        return ()
    if len(first_line) > 4096 or not first_line.endswith((b"\n", b"\r")):
        raise LauncherChainError("Launcher shebang is not safely bounded")
    try:
        declaration = first_line[2:].decode("utf-8").strip()
        words = tuple(shlex.split(declaration, posix=True))
    except (UnicodeDecodeError, ValueError):
        raise LauncherChainError("Launcher shebang cannot be parsed safely") from None
    if not words or any("\x00" in word for word in words):
        raise LauncherChainError("Launcher shebang is empty or malformed")
    return words


def _read_shebang(path: Path) -> tuple[str, ...]:
    try:
        first_line = _read_stable_first_line(path)
    except OSError as exc:
        raise LauncherChainError("Launcher shebang cannot be read") from exc
    return _parse_shebang(first_line)


def _resolve_shebang_chain(
    shebang: Sequence[str],
    *,
    env: Mapping[str, str],
    strict: bool,
) -> tuple[tuple[str, ...], ExecutableIdentity | None, ExecutableIdentity | None]:
    words = tuple(shebang)
    if not words:
        return (), None, None
    requested = Path(words[0]).expanduser()
    if not requested.is_absolute():
        if strict:
            raise LauncherChainError(
                "Direct shebang interpreters must use an absolute path"
            )
        return words, None, None
    try:
        launcher = resolve_executable(str(requested), env=env)
    except ExecutableResolutionError as exc:
        raise LauncherChainError("Shebang interpreter cannot be resolved") from exc
    if Path(launcher.resolved_path).name.lower() != "env":
        if strict and _read_shebang(Path(launcher.resolved_path)):
            raise LauncherChainError(
                "Nested script interpreters are not declarable in strict mode"
            )
        return words, None, launcher

    tail = list(words[1:])
    if tail[:1] == ["-S"]:
        tail = tail[1:]
    elif tail and tail[0].startswith("-"):
        if strict:
            raise LauncherChainError(
                "env shebang options require the declarable -S form"
            )
        return words, launcher, None
    interpreter_env = dict(env)
    while tail and "=" in tail[0] and not tail[0].startswith("="):
        name, _, value = tail.pop(0).partition("=")
        if not name or "\x00" in value:
            raise LauncherChainError("env shebang assignment is malformed")
        try:
            validate_environment_name(name)
        except (TypeError, ValueError) as exc:
            raise LauncherChainError(
                "env shebang assignment violates environment policy"
            ) from exc
        interpreter_env[name.upper() if os.name == "nt" else name] = value
    if not tail:
        if strict:
            raise LauncherChainError("env shebang does not declare an interpreter")
        return words, launcher, None
    try:
        interpreter = resolve_executable(tail[0], env=interpreter_env)
    except ExecutableResolutionError as exc:
        raise LauncherChainError("env shebang interpreter cannot be resolved") from exc
    if strict and _read_shebang(Path(interpreter.resolved_path)):
        raise LauncherChainError(
            "Nested script interpreters are not declarable in strict mode"
        )
    return words, launcher, interpreter


def _normalized_declared_path(
    value: str | os.PathLike[str],
    *,
    cwd: Path,
) -> Path:
    raw = os.fspath(value)
    if not raw or "\x00" in raw:
        raise LauncherChainError("Launcher artifact path is empty or malformed")
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        return Path(os.path.abspath(os.fspath(candidate)))
    except (OSError, ValueError) as exc:
        raise LauncherChainError(
            "Launcher artifact path cannot be normalized"
        ) from exc


def _attest_launcher_artifact(
    value: str | os.PathLike[str],
    *,
    cwd: Path,
    role: str,
) -> LauncherArtifactIdentity:
    if role not in {"entrypoint", "companion"}:
        raise LauncherChainError("Launcher artifact role is unsupported")
    launch_path = _normalized_declared_path(value, cwd=cwd)
    try:
        resolved = launch_path.resolve(strict=True)
        attestation = _attest_executable_file(resolved)
        launch_binding = _launch_path_binding(
            launch_path,
            expected_resolved=resolved,
        )
    except (ExecutableResolutionError, OSError) as exc:
        raise LauncherChainError("Launcher artifact cannot be attested") from exc
    return LauncherArtifactIdentity(
        role=role,
        requested_path=os.fspath(value),
        launch_path=str(launch_path),
        resolved_path=str(resolved),
        sha256=attestation.sha256,
        size_bytes=attestation.size_bytes,
        mtime_ns=attestation.mtime_ns,
        device_id=attestation.device_id,
        inode=attestation.inode,
        mode=attestation.mode,
        owner_uid=attestation.owner_uid,
        owner_gid=attestation.owner_gid,
        launch_path_binding_sha256=launch_binding,
    )


def _verify_launcher_artifact(identity: LauncherArtifactIdentity) -> None:
    try:
        attestation = _attest_executable_file(Path(identity.resolved_path))
        launch_binding = _launch_path_binding(
            Path(identity.launch_path),
            expected_resolved=Path(identity.resolved_path),
        )
    except ExecutableResolutionError as exc:
        raise LauncherChainChangedError(
            "Launcher artifact can no longer be verified"
        ) from exc
    if (
        attestation.binding_tuple()
        != (
            identity.sha256,
            identity.size_bytes,
            identity.mtime_ns,
            identity.device_id,
            identity.inode,
            identity.mode,
            identity.owner_uid,
            identity.owner_gid,
        )
        or launch_binding != identity.launch_path_binding_sha256
    ):
        raise LauncherChainChangedError("Launcher artifact changed after attestation")


def _argv_declares_artifact(
    argv: Sequence[str],
    identity: LauncherArtifactIdentity,
    *,
    cwd: Path,
) -> bool:
    for _, value in _argv_artifact_values(argv):
        try:
            candidate = _normalized_declared_path(value, cwd=cwd)
        except LauncherChainError:
            continue
        if str(candidate) == identity.launch_path:
            return True
    return False


def _argv_artifact_values(argv: Sequence[str]) -> Iterator[tuple[str, str]]:
    for argument in argv:
        if argument.startswith("-"):
            if "=" not in argument:
                continue
            _, _, value = argument.partition("=")
            if value:
                yield argument, value
            continue
        yield argument, argument


def _is_unusable_windows_path_syntax(error: OSError) -> bool:
    """Return whether Windows rejected a value before it could name a path."""

    winerror = getattr(error, "winerror", None)
    return winerror in {123, 161} or (os.name == "nt" and error.errno == errno.EINVAL)


def _undeclared_executable_argument(
    argv: Sequence[str],
    *,
    cwd: Path,
    declared: Sequence[LauncherArtifactIdentity],
) -> str | None:
    declared_paths = {item.launch_path for item in declared}
    for argument, value in _argv_artifact_values(argv):
        suffix = Path(value).suffix.lower()
        known_executable_suffix = suffix in (
            _SCRIPT_SUFFIXES | _NATIVE_EXECUTABLE_SUFFIXES
        )
        try:
            normalized = _normalized_declared_path(value, cwd=cwd)
        except LauncherChainError:
            if known_executable_suffix:
                return argument
            continue
        if str(normalized) in declared_paths:
            continue
        try:
            descriptor, before = _open_regular_file_descriptor(normalized)
        except FileNotFoundError:
            if known_executable_suffix:
                return argument
            continue
        except OSError as exc:
            try:
                metadata = normalized.lstat()
            except FileNotFoundError:
                if known_executable_suffix:
                    return argument
                continue
            except OSError as inspection_error:
                if _is_unusable_windows_path_syntax(inspection_error):
                    if known_executable_suffix:
                        return argument
                    continue
                raise LauncherChainError(
                    "Path-shaped launcher argument cannot be inspected"
                ) from inspection_error
            if stat.S_ISDIR(metadata.st_mode):
                continue
            raise LauncherChainError(
                "Path-shaped launcher arguments must be regular files"
            ) from exc
        first_line = bytearray()
        try:
            while len(first_line) < 4097:
                chunk = os.read(descriptor, 4097 - len(first_line))
                if not chunk:
                    break
                newline = chunk.find(b"\n")
                if newline >= 0:
                    first_line.extend(chunk[: newline + 1])
                    break
                first_line.extend(chunk)
            after = os.fstat(descriptor)
        except OSError as exc:
            raise LauncherChainError(
                "Path-shaped launcher argument cannot be inspected"
            ) from exc
        finally:
            os.close(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise LauncherChainError(
                "Path-shaped launcher argument changed during inspection"
            )
        executable_mode = bool(
            before.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        )
        if (
            known_executable_suffix
            or executable_mode
            or bool(_parse_shebang(bytes(first_line)))
        ):
            return argument
    return None


def _launch_path_binding(
    path: Path,
    *,
    expected_resolved: Path | None = None,
) -> str:
    """Bind the caller-visible launch path without resolving away symlinks."""

    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        link_target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None
    except OSError as exc:
        raise ExecutableResolutionError(
            "Executable launch path cannot be attested"
        ) from exc
    if expected_resolved is not None and resolved != expected_resolved:
        raise ExecutableResolutionError(
            "Executable launch path resolves to an unexpected target"
        )
    payload = {
        "device_id": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "link_target": link_target,
        "mode": int(metadata.st_mode),
        "resolved_path": str(resolved),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _attest_executable_file(path: Path) -> _FileAttestation:
    try:
        descriptor, before = _open_regular_file_descriptor(path)
    except OSError as exc:
        raise ExecutableResolutionError(f"Could not open executable: {path}") from exc
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ExecutableResolutionError(f"Could not attest executable: {path}") from exc
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise ExecutableResolutionError(
            "Executable changed while it was being attested"
        )
    return _FileAttestation(
        sha256=digest.hexdigest(),
        size_bytes=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device_id=int(after.st_dev),
        inode=int(after.st_ino),
        mode=int(after.st_mode),
        owner_uid=(
            None if not hasattr(after, "st_uid") else int(getattr(after, "st_uid"))
        ),
        owner_gid=(
            None if not hasattr(after, "st_gid") else int(getattr(after, "st_gid"))
        ),
    )


def _verify_executable_identity(identity: ExecutableIdentity) -> None:
    path = Path(identity.resolved_path)
    if not path.is_absolute():
        raise ExecutableChangedError("Executable identity path is not absolute")
    try:
        attestation = _attest_executable_file(path)
        launch_path_binding = _launch_path_binding(
            Path(identity.launch_path),
            expected_resolved=path,
        )
    except ExecutableResolutionError as exc:
        raise ExecutableChangedError(
            "Executable identity can no longer be verified"
        ) from exc
    if (
        attestation.binding_tuple()
        != (
            identity.sha256,
            identity.size_bytes,
            identity.mtime_ns,
            identity.device_id,
            identity.inode,
            identity.mode,
            identity.owner_uid,
            identity.owner_gid,
        )
        or launch_path_binding != identity.launch_path_binding_sha256
    ):
        raise ExecutableChangedError("Executable changed after identity resolution")


def _process_group_alive(process_group: int) -> bool:
    if os.name != "posix" or process_group <= 0:
        return False
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _signal_process_group(process_group: int, requested_signal: int) -> None:
    try:
        os.killpg(process_group, requested_signal)
    except ProcessLookupError:
        return
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise


def _attempt_process_group_signal(
    process_group: int,
    requested_signal: int,
) -> bool:
    """Attempt a group signal while deferring transient Darwin permission races.

    Darwin can report ``EPERM`` while an exited process group contains only
    zombies that are waiting to be reaped.  The normal cleanup waits and final
    liveness probe remain authoritative: a persistent group still fails closed.
    """

    try:
        _signal_process_group(process_group, requested_signal)
    except PermissionError:
        return False
    return True


def _close_pipe(stream: Any) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _normalize_version_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return "\n".join(line.rstrip() for line in decoded.strip().splitlines())


def _duration_ms(started: float, finished: float) -> int:
    return max(0, int(round((finished - started) * 1000)))


__all__ = [
    "AssistantBridgeRuntimeError",
    "CleanupReport",
    "EnvironmentFingerprint",
    "ExecutableChangedError",
    "ExecutableIdentity",
    "ExecutableResolutionError",
    "ExecutableVersionMetadata",
    "LauncherArtifactIdentity",
    "LauncherChainChangedError",
    "LauncherChainError",
    "LauncherChainIdentity",
    "ProcessCleanupError",
    "ProcessExecutionPolicy",
    "ProcessExecutionResult",
    "ProcessLaunchError",
    "ProcessLaunchLifecycleError",
    "ProcessLaunchNotAuthorizedError",
    "ProcessLaunchPermit",
    "ProcessObservationError",
    "ProcessTreeUnavailableError",
    "RuntimeCapabilities",
    "WorkingDirectoryChangedError",
    "execute_process",
    "fingerprint_environment",
    "inspect_executable",
    "resolve_executable",
    "resolve_launcher_chain",
    "runtime_capabilities",
    "validate_environment_name",
    "verify_launcher_chain",
]
