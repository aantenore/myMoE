"""Minimal OS evidence for a directly owned local inference process.

The v1 contract admits a root-only tree.  It enumerates descendants twice and
fails closed if any appear; it does not attest a hostile process or replace OS
containment.  The separately owned launcher is responsible for terminating
the process group on teardown.

``psutil`` is imported only when the default observer is first used.  Tests and
platform-specific implementations can inject a compatible backend instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from ipaddress import ip_address
import json
import math
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Callable, Protocol


_HASH_CHUNK_BYTES = 1024 * 1024
_LSOF_MAX_BYTES = 256 * 1024


class RuntimeProcessObservationError(RuntimeError):
    """Stable failure raised when required OS evidence cannot be collected."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pids_digest(pids: tuple[int, ...]) -> str:
    return _sha256_json({"pids": list(pids)})


def _require_sha256(value: object, label: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return rendered


def _require_pid(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return value


def normalize_numeric_loopback_host(value: object) -> str:
    """Return a canonical numeric loopback address and reject hostnames."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("host must be a numeric loopback address")
    try:
        parsed = ip_address(value)
    except ValueError as exc:
        raise ValueError("host must be a numeric loopback address") from exc
    if not parsed.is_loopback:
        raise ValueError("host must be a numeric loopback address")
    return parsed.compressed


@dataclass(frozen=True)
class ProcessTreeEvidence:
    """Identity of the one root process observed by the v1 adapter."""

    root_pid: int
    create_time_ns: int
    process_count: int
    pids_digest: str
    root_executable_sha256: str
    root_only: bool

    def __post_init__(self) -> None:
        _require_pid(self.root_pid, "root_pid")
        if (
            isinstance(self.create_time_ns, bool)
            or not isinstance(self.create_time_ns, int)
            or self.create_time_ns < 1
        ):
            raise ValueError("create_time_ns must be a positive integer")
        if self.process_count != 1:
            raise ValueError("v1 process evidence must contain exactly the root process")
        if self.root_only is not True:
            raise ValueError("v1 process evidence must be marked root_only")
        if _require_sha256(self.pids_digest, "pids_digest") != _pids_digest(
            (self.root_pid,)
        ):
            raise ValueError("pids_digest does not match root_pid")
        _require_sha256(self.root_executable_sha256, "root_executable_sha256")

    def content_payload(self) -> dict[str, object]:
        """Return metadata-only process evidence suitable for content addressing."""

        return {
            "contract": "mymoe-runtime-process-tree-evidence/v1",
            "root_pid": self.root_pid,
            "create_time_ns": self.create_time_ns,
            "process_count": self.process_count,
            "pids_digest": self.pids_digest,
            "root_executable_sha256": self.root_executable_sha256,
            "root_only": self.root_only,
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.content_payload())

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class EndpointOwnershipEvidence:
    """PID ownership evidence for one numeric-loopback TCP listener."""

    host: str
    port: int
    listener_pids: tuple[int, ...]
    listener_pids_digest: str
    owned_by_root: bool
    ambiguous: bool

    def __post_init__(self) -> None:
        normalized_host = normalize_numeric_loopback_host(self.host)
        object.__setattr__(self, "host", normalized_host)
        _require_port(self.port)
        pids = tuple(self.listener_pids)
        if pids != tuple(sorted(set(pids))) or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in pids
        ):
            raise ValueError("listener_pids must be sorted unique positive integers")
        object.__setattr__(self, "listener_pids", pids)
        if _require_sha256(
            self.listener_pids_digest, "listener_pids_digest"
        ) != _pids_digest(pids):
            raise ValueError("listener_pids_digest does not match listener_pids")
        if not isinstance(self.owned_by_root, bool) or not isinstance(
            self.ambiguous, bool
        ):
            raise TypeError("ownership flags must be boolean")
        if self.owned_by_root and (self.ambiguous or len(pids) != 1):
            raise ValueError("owned_by_root requires one unambiguous listener PID")

    def content_payload(self) -> dict[str, object]:
        """Return listener metadata without commands, paths, or response bodies."""

        return {
            "contract": "mymoe-runtime-endpoint-ownership-evidence/v1",
            "host": self.host,
            "port": self.port,
            "listener_pids": list(self.listener_pids),
            "listener_pids_digest": self.listener_pids_digest,
            "owned_by_root": self.owned_by_root,
            "ambiguous": self.ambiguous,
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.content_payload())

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


class RuntimeProcessObserver(Protocol):
    """Injected OS boundary used by the llama.cpp runtime supervisor."""

    def observe_process_tree(self, root_pid: int) -> ProcessTreeEvidence: ...

    def observe_endpoint_ownership(
        self,
        *,
        host: str,
        port: int,
        root_pid: int | None,
    ) -> EndpointOwnershipEvidence: ...


def _load_psutil() -> object:
    try:
        return importlib.import_module("psutil")
    except ImportError as exc:
        raise RuntimeProcessObservationError(
            "psutil_unavailable",
            "psutil is required for listener PID ownership evidence.",
        ) from exc


def _sha256_regular_file(path: str) -> str:
    candidate = Path(path)
    try:
        before = candidate.stat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeProcessObservationError(
                "executable_invalid", "Observed executable is not a regular file."
            )
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        after = candidate.stat()
    except RuntimeProcessObservationError:
        raise
    except OSError as exc:
        raise RuntimeProcessObservationError(
            "executable_unreadable", "Unable to hash the observed executable."
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeProcessObservationError(
            "executable_changed", "Observed executable changed while it was hashed."
        )
    return digest.hexdigest()


def _listener_address(connection: object) -> tuple[str, int] | None:
    address = getattr(connection, "laddr", None)
    if address in (None, (), ""):
        return None
    raw_host = getattr(address, "ip", None)
    raw_port = getattr(address, "port", None)
    if raw_host is None or raw_port is None:
        try:
            raw_host, raw_port = address[0], address[1]  # type: ignore[index]
        except (IndexError, TypeError):
            return None
    if not isinstance(raw_host, str) or isinstance(raw_port, bool):
        return None
    try:
        return ip_address(raw_host).compressed, int(raw_port)
    except (TypeError, ValueError):
        return None


def _macos_lsof_listener_pids(host: str, port: int) -> tuple[int, ...]:
    executable = Path("/usr/sbin/lsof")
    try:
        metadata = executable.lstat()
    except OSError as exc:
        raise RuntimeProcessObservationError(
            "listeners_unobservable", "The macOS listener observer is unavailable."
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
    ):
        raise RuntimeProcessObservationError(
            "listeners_unobservable",
            "The macOS listener observer does not have a trusted file identity.",
        )
    selector = f"-iTCP@{host}:{port}"
    try:
        completed = subprocess.run(
            (
                str(executable),
                "-nP",
                "-a",
                selector,
                "-sTCP:LISTEN",
                "-Fp",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            shell=False,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeProcessObservationError(
            "listeners_unobservable", "Unable to inspect macOS TCP listeners."
        ) from exc
    if len(completed.stdout) > _LSOF_MAX_BYTES or completed.returncode not in {0, 1}:
        raise RuntimeProcessObservationError(
            "listeners_unobservable", "The macOS listener observation is invalid."
        )
    pids: set[int] = set()
    try:
        for raw_line in completed.stdout.decode("ascii", errors="strict").splitlines():
            if not raw_line.startswith("p"):
                continue
            pid = int(raw_line[1:])
            if pid < 1:
                raise ValueError("invalid PID")
            pids.add(pid)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeProcessObservationError(
            "listeners_unobservable", "The macOS listener observation is invalid."
        ) from exc
    return tuple(sorted(pids))


class PsutilRuntimeProcessObserver:
    """Root-only process and TCP-listener observer backed lazily by psutil."""

    def __init__(
        self,
        *,
        psutil_loader: Callable[[], object] | None = None,
        executable_sha256: Callable[[str], str] | None = None,
        socket_factory: Callable[..., object] | None = None,
        global_listener_reader: Callable[[str, int], tuple[int, ...]] | None = None,
    ) -> None:
        self._psutil_loader = psutil_loader or _load_psutil
        self._executable_sha256 = executable_sha256 or _sha256_regular_file
        self._socket_factory = socket_factory or socket.socket
        self._global_listener_reader = global_listener_reader or (
            _macos_lsof_listener_pids if sys.platform == "darwin" else None
        )
        self._psutil_backend: object | None = None

    def _backend(self) -> object:
        if self._psutil_backend is None:
            try:
                self._psutil_backend = self._psutil_loader()
            except RuntimeProcessObservationError:
                raise
            except ImportError as exc:
                raise RuntimeProcessObservationError(
                    "psutil_unavailable",
                    "psutil is required for listener PID ownership evidence.",
                ) from exc
            except Exception as exc:
                raise RuntimeProcessObservationError(
                    "observer_unavailable", "Unable to initialize the process observer."
                ) from exc
        return self._psutil_backend

    def observe_process_tree(self, root_pid: int) -> ProcessTreeEvidence:
        pid = _require_pid(root_pid, "root_pid")
        backend = self._backend()
        try:
            process_factory = getattr(backend, "Process")
            process = process_factory(pid)
            observed_pid = int(getattr(process, "pid"))
            if observed_pid != pid:
                raise RuntimeProcessObservationError(
                    "process_identity_mismatch",
                    "The process observer returned a different root PID.",
                )
            first_create_time = float(process.create_time())
            first_children = tuple(
                sorted(
                    int(getattr(child, "pid"))
                    for child in process.children(recursive=True)
                )
            )
            executable = process.exe()
            if not isinstance(executable, str) or not executable:
                raise RuntimeProcessObservationError(
                    "executable_unknown", "The root executable path is unavailable."
                )
            executable_digest = self._executable_sha256(executable)
            second_create_time = float(process.create_time())
            second_children = tuple(
                sorted(
                    int(getattr(child, "pid"))
                    for child in process.children(recursive=True)
                )
            )
        except RuntimeProcessObservationError:
            raise
        except Exception as exc:
            raise RuntimeProcessObservationError(
                "process_unobservable", "Unable to observe the root process identity."
            ) from exc
        if (
            not math.isfinite(first_create_time)
            or first_create_time <= 0
            or first_create_time != second_create_time
        ):
            raise RuntimeProcessObservationError(
                "process_identity_changed",
                "The root process identity changed during observation.",
            )
        if first_children != second_children or first_children:
            raise RuntimeProcessObservationError(
                "unexpected_descendant",
                "The direct runtime process tree is not stably root-only.",
            )
        create_time_ns = int(round(first_create_time * 1_000_000_000))
        return ProcessTreeEvidence(
            root_pid=pid,
            create_time_ns=create_time_ns,
            process_count=1,
            pids_digest=_pids_digest((pid,)),
            root_executable_sha256=_require_sha256(
                executable_digest, "root executable digest"
            ),
            root_only=True,
        )

    def observe_endpoint_ownership(
        self,
        *,
        host: str,
        port: int,
        root_pid: int | None,
    ) -> EndpointOwnershipEvidence:
        normalized_host = normalize_numeric_loopback_host(host)
        normalized_port = _require_port(port)
        expected_pid = (
            None if root_pid is None else _require_pid(root_pid, "root_pid")
        )
        backend = self._backend()
        try:
            connections = backend.net_connections(kind="tcp")  # type: ignore[attr-defined]
            listen_status = str(getattr(backend, "CONN_LISTEN", "LISTEN")).upper()
        except Exception as exc:
            access_denied = getattr(backend, "AccessDenied", None)
            if not isinstance(access_denied, type) or not isinstance(
                exc, access_denied
            ):
                raise RuntimeProcessObservationError(
                    "listeners_unobservable",
                    "Unable to enumerate TCP listener ownership.",
                ) from exc
            return self._observe_endpoint_without_global_listing(
                host=normalized_host,
                port=normalized_port,
                root_pid=expected_pid,
            )

        listener_pids: set[int] = set()
        unknown_owner = False
        try:
            for connection in connections:
                if str(getattr(connection, "status", "")).upper() != listen_status:
                    continue
                address = _listener_address(connection)
                if address != (normalized_host, normalized_port):
                    continue
                owner = getattr(connection, "pid", None)
                if isinstance(owner, bool) or not isinstance(owner, int) or owner < 1:
                    unknown_owner = True
                else:
                    listener_pids.add(owner)
        except Exception as exc:
            raise RuntimeProcessObservationError(
                "listeners_unobservable", "Unable to inspect TCP listener ownership."
            ) from exc

        ordered = tuple(sorted(listener_pids))
        ambiguous = unknown_owner or len(ordered) > 1
        owned_by_root = (
            expected_pid is not None
            and not ambiguous
            and ordered == (expected_pid,)
        )
        return EndpointOwnershipEvidence(
            host=normalized_host,
            port=normalized_port,
            listener_pids=ordered,
            listener_pids_digest=_pids_digest(ordered),
            owned_by_root=owned_by_root,
            ambiguous=ambiguous,
        )

    def _observe_endpoint_without_global_listing(
        self,
        *,
        host: str,
        port: int,
        root_pid: int | None,
    ) -> EndpointOwnershipEvidence:
        if self._global_listener_reader is None:
            raise RuntimeProcessObservationError(
                "listeners_unobservable",
                "No global listener ownership fallback is available.",
            )
        try:
            pids = tuple(self._global_listener_reader(host, port))
        except Exception as exc:
            if isinstance(exc, RuntimeProcessObservationError):
                raise
            raise RuntimeProcessObservationError(
                "listeners_unobservable",
                "Unable to inspect global TCP listener ownership.",
            ) from exc
        if pids != tuple(sorted(set(pids))) or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid < 1
            for pid in pids
        ):
            raise RuntimeProcessObservationError(
                "listeners_unobservable",
                "Global listener ownership evidence is invalid.",
            )
        if root_pid is None and not pids:
            return self._probe_endpoint_vacancy(host=host, port=port)
        ambiguous = len(pids) > 1
        return EndpointOwnershipEvidence(
            host=host,
            port=port,
            listener_pids=pids,
            listener_pids_digest=_pids_digest(pids),
            owned_by_root=root_pid is not None and pids == (root_pid,),
            ambiguous=ambiguous,
        )

    def _probe_endpoint_vacancy(
        self, *, host: str, port: int
    ) -> EndpointOwnershipEvidence:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        probe = None
        occupied = False
        try:
            probe = self._socket_factory(family, socket.SOCK_STREAM)
            probe.bind((host, port))  # type: ignore[attr-defined]
        except OSError:
            occupied = True
        except Exception as exc:
            raise RuntimeProcessObservationError(
                "listeners_unobservable",
                "Unable to sample loopback endpoint vacancy.",
            ) from exc
        finally:
            if probe is not None:
                try:
                    probe.close()  # type: ignore[attr-defined]
                except Exception as exc:
                    raise RuntimeProcessObservationError(
                        "listeners_unobservable",
                        "Unable to close the loopback vacancy probe.",
                    ) from exc
        return EndpointOwnershipEvidence(
            host=host,
            port=port,
            listener_pids=(),
            listener_pids_digest=_pids_digest(()),
            owned_by_root=False,
            ambiguous=occupied,
        )
