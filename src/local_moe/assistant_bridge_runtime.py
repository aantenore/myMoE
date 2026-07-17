"""Hardened process runtime primitives for the hybrid assistant bridge.

The module is intentionally independent from :mod:`local_moe.assistant_bridge` so
the bridge can adopt it behind its existing public API in a later, reviewable
change.  Its containment contract is explicit:

* POSIX launches receive a fresh session/process group.  Every process that
  remains in that group is terminated and its absence is verified.
* When ``psutil`` is installed, descendants observed recursively are tracked and
  cleaned even if they later leave the original process group.  Discovery is
  best effort because a process can detach and exit between observations.
* Windows strict execution requires ``psutil``.  Without it, execution is
  rejected before launch unless a caller explicitly opts out of tree isolation.

Cleanup verification includes the root process, the POSIX process group,
psutil-observed descendants, and all stdin/stdout/stderr worker threads.  A
verification failure raises :class:`ProcessCleanupError`; partial cleanup is
never reported as a successful execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

try:  # Optional by design; capability reporting below makes absence observable.
    import psutil as _psutil
except ImportError:  # pragma: no cover - availability depends on selected extras.
    _psutil = None


_HASH_CHUNK_BYTES = 1024 * 1024
_PIPE_CHUNK_BYTES = 64 * 1024
_MAX_IO_BYTES = 256 * 1024 * 1024


class AssistantBridgeRuntimeError(RuntimeError):
    """Base error for runtime contract failures."""


class ExecutableResolutionError(AssistantBridgeRuntimeError):
    """The configured executable could not be resolved and attested."""


class ExecutableChangedError(AssistantBridgeRuntimeError):
    """The executable no longer matches its previously resolved identity."""


class ProcessTreeUnavailableError(AssistantBridgeRuntimeError):
    """The requested process-tree containment contract is unavailable."""


class ProcessLaunchError(AssistantBridgeRuntimeError):
    """The attested executable could not be launched."""


class ProcessCleanupError(AssistantBridgeRuntimeError):
    """Process or pipe cleanup could not be verified before returning."""

    def __init__(self, message: str, *, details: Mapping[str, object]) -> None:
        super().__init__(message)
        self.details = MappingProxyType(dict(details))


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


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Current host support and the exact descendant-observation boundary."""

    platform: str
    posix_process_groups: bool
    psutil_available: bool
    strict_tree_supported: bool
    detached_descendant_contract: str

    def payload(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "posix_process_groups": self.posix_process_groups,
            "psutil_available": self.psutil_available,
            "strict_tree_supported": self.strict_tree_supported,
            "detached_descendant_contract": self.detached_descendant_contract,
        }


@dataclass(frozen=True)
class ExecutableVersionMetadata:
    """Bounded result of an explicit executable version probe."""

    args: tuple[str, ...]
    status: str
    returncode: int | None
    text: str
    output_sha256: str
    output_bytes: int
    truncated: bool

    def payload(self) -> dict[str, object]:
        return {
            "args": list(self.args),
            "status": self.status,
            "returncode": self.returncode,
            "text": self.text,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ExecutableIdentity:
    """Absolute, content-addressed executable selected from a specific PATH."""

    requested: str
    resolved_path: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    resolution_environment: EnvironmentFingerprint
    version: ExecutableVersionMetadata | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "resolution_environment": self.resolution_environment.payload(),
            "version": None if self.version is None else self.version.payload(),
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

    def __post_init__(self) -> None:
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
    """Evidence that the launched tree and all owned pipe workers are gone."""

    attempted: bool
    verified: bool
    methods: tuple[str, ...]
    observed_descendants: int
    process_group_verified: bool | None
    psutil_verified: bool | None
    root_reaped: bool
    pipe_threads_joined: bool

    def payload(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "verified": self.verified,
            "methods": list(self.methods),
            "observed_descendants": self.observed_descendants,
            "process_group_verified": self.process_group_verified,
            "psutil_verified": self.psutil_verified,
            "root_reaped": self.root_reaped,
            "pipe_threads_joined": self.pipe_threads_joined,
        }


@dataclass(frozen=True)
class ProcessExecutionResult:
    """Bounded process outcome returned only after cleanup verification."""

    code: str
    returncode: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    stdout_bytes: int
    stderr_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    stdout_truncated: bool
    stderr_truncated: bool
    stdin_bytes_written: int
    execution_duration_ms: int
    duration_ms: int
    executable: ExecutableIdentity
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
        strict_tree_supported=posix_groups or psutil_available,
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
        resolved = Path(selected).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExecutableResolutionError(
            f"Executable could not be resolved: {requested}"
        ) from exc
    if not resolved.is_absolute():
        raise ExecutableResolutionError("Resolved executable path is not absolute")
    sha256, size_bytes, mtime_ns = _attest_executable_file(resolved)
    return ExecutableIdentity(
        requested=requested,
        resolved_path=str(resolved),
        sha256=sha256,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
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


def execute_process(
    executable: ExecutableIdentity,
    args: Sequence[str] = (),
    *,
    stdin: bytes = b"",
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float,
    policy: ProcessExecutionPolicy | None = None,
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
        and not capabilities.strict_tree_supported
    ):
        raise ProcessTreeUnavailableError(
            "Strict process-tree cleanup is unavailable on this host without psutil"
        )
    argv_tail = _validate_args(args)
    if not isinstance(stdin, bytes):
        raise TypeError("stdin must be bytes")
    if len(stdin) > selected_policy.stdin_limit_bytes:
        raise ValueError("stdin exceeds the configured input bound")
    launch_env = _normalize_environment(os.environ if env is None else env)
    environment = fingerprint_environment(launch_env)
    launch_cwd = None if cwd is None else str(_validate_cwd(cwd))
    started = time.monotonic()
    deadline = started + timeout_seconds
    _verify_executable_identity(executable)
    if time.monotonic() >= deadline:
        raise ProcessLaunchError(
            "Execution deadline elapsed during executable attestation"
        )

    popen_kwargs: dict[str, object] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "cwd": launch_cwd,
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
    try:
        process = subprocess.Popen(
            [executable.resolved_path, *argv_tail],
            **popen_kwargs,
        )
    except OSError as exc:
        raise ProcessLaunchError(
            f"Could not launch attested executable: {executable.resolved_path}"
        ) from exc

    tracker = _ProcessTracker(process.pid)
    wake = threading.Event()
    stdout_state = _BoundedStream(selected_policy.stdout_limit_bytes)
    stderr_state = _BoundedStream(selected_policy.stderr_limit_bytes)
    stdin_state = _StdinState()
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
    root_exited_at: float | None = None
    terminal = "completed"
    try:
        for worker in workers:
            worker.start()
        while True:
            tracker.observe()
            now = time.monotonic()
            if stdout_state.overflowed:
                terminal = "stdout_limit_exceeded"
                break
            if stderr_state.overflowed:
                terminal = "stderr_limit_exceeded"
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
    finally:
        execution_ended = time.monotonic()
        cleanup = _cleanup_process_tree(
            process,
            tracker=tracker,
            workers=workers,
            policy=selected_policy,
        )

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

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _StdinState:
    def __init__(self) -> None:
        self.count = 0


class _ProcessTracker:
    """Retain psutil process handles after their ancestry becomes unavailable."""

    def __init__(self, root_pid: int) -> None:
        self.root_pid = root_pid
        self._observed: dict[tuple[int, float], Any] = {}
        self.observe()

    @property
    def observed_descendants(self) -> int:
        return max(len(self._observed) - 1, 0)

    def observe(self) -> None:
        if _psutil is None:
            return
        seeds: list[Any] = list(self._observed.values())
        try:
            seeds.append(_psutil.Process(self.root_pid))
        except _psutil.Error:
            pass
        for process in seeds:
            self._remember(process)
            try:
                children = process.children(recursive=True)
            except _psutil.Error:
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
            except _psutil.Error:
                continue
            live.append(process)
        return tuple(live)

    def _remember(self, process: Any) -> None:
        try:
            key = (int(process.pid), float(process.create_time()))
        except _psutil.Error:
            return
        self._observed[key] = process


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

    if os.name == "posix" and group_alive:
        _signal_process_group(process.pid, signal.SIGTERM)
        methods.append("posix-process-group-term")
    elif root_alive:
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
    if os.name == "posix" and group_alive:
        _signal_process_group(process.pid, signal.SIGKILL)
        methods.append("posix-process-group-kill")
    elif root_alive:
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
    psutil_verified = not tracker.live() if _psutil is not None else None
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
                return
            if not chunk:
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
            return
        view = memoryview(content)
        while state.count < len(view):
            try:
                written = stream.write(
                    view[state.count : state.count + _PIPE_CHUNK_BYTES]
                )
            except (BrokenPipeError, OSError, ValueError):
                return
            if written is None:
                written = 0
            if written <= 0:
                return
            state.count += written
        try:
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            return
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
        normalized_key = key.upper() if os.name == "nt" else key
        if normalized_key in normalized and normalized[normalized_key] != value:
            raise ValueError(
                "Environment contains conflicting platform-equivalent keys"
            )
        normalized[normalized_key] = value
    return normalized


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


def _attest_executable_file(path: Path) -> tuple[str, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExecutableResolutionError(f"Could not open executable: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExecutableResolutionError("Executable is not a regular file")
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
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ExecutableResolutionError(
            "Executable changed while it was being attested"
        )
    return digest.hexdigest(), int(after.st_size), int(after.st_mtime_ns)


def _verify_executable_identity(identity: ExecutableIdentity) -> None:
    path = Path(identity.resolved_path)
    if not path.is_absolute():
        raise ExecutableChangedError("Executable identity path is not absolute")
    try:
        sha256, size_bytes, mtime_ns = _attest_executable_file(path)
    except ExecutableResolutionError as exc:
        raise ExecutableChangedError(
            "Executable identity can no longer be verified"
        ) from exc
    if (
        sha256 != identity.sha256
        or size_bytes != identity.size_bytes
        or mtime_ns != identity.mtime_ns
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
    "ProcessCleanupError",
    "ProcessExecutionPolicy",
    "ProcessExecutionResult",
    "ProcessLaunchError",
    "ProcessTreeUnavailableError",
    "RuntimeCapabilities",
    "execute_process",
    "fingerprint_environment",
    "inspect_executable",
    "resolve_executable",
    "runtime_capabilities",
]
