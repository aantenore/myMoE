"""Owned, direct, single-model llama.cpp runtime supervision.

This adapter intentionally does less than the upstream llama.cpp router.  It
launches one local GGUF model as one directly owned ``llama-server`` process,
requires a numeric-loopback listener owned by that root PID, and probes only
three bounded GET endpoints: ``/health``, ``/props``, and ``/v1/models``.

There is no attach mode, router mode, model download/delete API, or POST control
probe in v1. The owned server still has its native inference API. The observer
rejects descendants before evidence is accepted;
this is still local OS evidence, not cryptographic process attestation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence
from urllib import error, request

from .runtime_process_observer import (
    EndpointOwnershipEvidence,
    ProcessTreeEvidence,
    RuntimeProcessObservationError,
    RuntimeProcessObserver,
    normalize_numeric_loopback_host,
)


_ALLOWED_GET_PATHS = frozenset({"/health", "/props", "/v1/models"})
_DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024
_FIXED_ENVIRONMENT = {
    "DO_NOT_TRACK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "LANG": "C",
    "LC_ALL": "C",
    "TRANSFORMERS_OFFLINE": "1",
    "TZ": "UTC",
}
_TERMINAL_APPLICATION_REASONS = frozenset(
    {
        "model_count_invalid",
        "model_id_mismatch",
        "models_invalid",
        "props_invalid",
        "props_model_mismatch",
    }
)


class LlamaCppRuntimeSupervisorError(RuntimeError):
    """Stable failure at the direct llama.cpp lifecycle boundary."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        reason_codes: Sequence[str] = (),
    ) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.reason_codes = tuple(sorted(set(str(item) for item in reason_codes)))
        super().__init__(f"{self.code}: {self.detail}")


class LlamaCppTransportError(RuntimeError):
    """Bounded HTTP GET failure without response-body disclosure."""

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


def _require_sha256(value: object, label: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return rendered


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    rendered = float(value)
    if rendered <= 0 or rendered == float("inf") or rendered != rendered:
        raise ValueError(f"{label} must be finite and positive")
    return rendered


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _model_id(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("model_id must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise ValueError("model_id cannot contain control characters")
    return value


def _absolute_path(value: object, label: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be an absolute path")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if suffix is not None and candidate.suffix.lower() != suffix:
        raise ValueError(f"{label} must end with {suffix}")
    return str(candidate)


@dataclass(frozen=True)
class LlamaCppApplicationEvidence:
    """Bounded application-layer evidence returned by the three GET probes."""

    ready: bool
    model_ids: tuple[str, ...]
    models_digest: str
    props_digest: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be boolean")
        model_ids = tuple(self.model_ids)
        if any(_model_id(item) != item for item in model_ids):
            raise ValueError("model_ids are invalid")
        if model_ids != tuple(sorted(set(model_ids))):
            raise ValueError("model_ids must be sorted and unique")
        object.__setattr__(self, "model_ids", model_ids)
        if _require_sha256(self.models_digest, "models_digest") != _sha256_json(
            {"model_ids": list(model_ids)}
        ):
            raise ValueError("models_digest does not match model_ids")
        _require_sha256(self.props_digest, "props_digest")
        reasons = tuple(sorted(set(self.reason_codes)))
        if any(not isinstance(item, str) or not item for item in reasons):
            raise ValueError("reason_codes are invalid")
        object.__setattr__(self, "reason_codes", reasons)
        if self.ready != (not reasons):
            raise ValueError("ready must be true exactly when reason_codes is empty")

    def content_payload(self) -> dict[str, object]:
        """Return bounded digests and advertised IDs, never raw HTTP bodies."""

        return {
            "contract": "mymoe-llama-cpp-application-evidence/v1",
            "ready": self.ready,
            "model_ids": list(self.model_ids),
            "models_digest": self.models_digest,
            "props_digest": self.props_digest,
            "reason_codes": list(self.reason_codes),
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.content_payload())

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class LlamaCppRuntimeEvidence:
    """One verified snapshot of process, listener, and application state."""

    process: ProcessTreeEvidence
    endpoint: EndpointOwnershipEvidence
    application: LlamaCppApplicationEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.process, ProcessTreeEvidence):
            raise TypeError("process evidence is invalid")
        if not isinstance(self.endpoint, EndpointOwnershipEvidence):
            raise TypeError("endpoint evidence is invalid")
        if not isinstance(self.application, LlamaCppApplicationEvidence):
            raise TypeError("application evidence is invalid")
        if not self.application.ready:
            raise ValueError("runtime evidence requires a ready application snapshot")
        if not self.endpoint.owned_by_root or self.endpoint.ambiguous:
            raise ValueError("runtime evidence requires unambiguous root listener ownership")
        if self.endpoint.listener_pids != (self.process.root_pid,):
            raise ValueError("listener PID does not match the observed root process")

    def content_payload(self) -> dict[str, object]:
        """Bind the three independently collected evidence snapshots."""

        return {
            "contract": "mymoe-llama-cpp-runtime-evidence/v1",
            "process_tree_sha256": self.process.digest,
            "endpoint_evidence_sha256": self.endpoint.digest,
            "application_evidence_sha256": self.application.digest,
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.content_payload())

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class JsonHttpResponse:
    status_code: int
    payload: object

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code is invalid")


class LoopbackJsonTransport(Protocol):
    def get_json(
        self,
        *,
        host: str,
        port: int,
        path: str,
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> JsonHttpResponse: ...


class SpawnedProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ProcessLauncher(Protocol):
    def launch(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        working_directory: str,
    ) -> SpawnedProcess: ...


@dataclass(frozen=True)
class LlamaCppRuntimeSpec:
    """Configuration for one direct local llama.cpp worker."""

    executable_path: str
    executable_sha256: str
    model_path: str
    model_id: str
    working_directory: str
    host: str
    port: int
    context_size: int = 4096
    parallel: int = 1
    fit_mode: str = "off"
    sleep_idle_seconds: int | None = None
    extra_args: tuple[str, ...] = ()
    startup_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 2.0
    shutdown_timeout_seconds: float = 5.0
    poll_interval_seconds: float = 0.05
    maximum_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executable_path",
            _absolute_path(self.executable_path, "executable_path"),
        )
        object.__setattr__(
            self,
            "executable_sha256",
            _require_sha256(self.executable_sha256, "executable_sha256"),
        )
        object.__setattr__(
            self,
            "model_path",
            _absolute_path(self.model_path, "model_path", suffix=".gguf"),
        )
        object.__setattr__(self, "model_id", _model_id(self.model_id))
        object.__setattr__(
            self,
            "working_directory",
            _absolute_path(self.working_directory, "working_directory"),
        )
        object.__setattr__(self, "host", normalize_numeric_loopback_host(self.host))
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        _positive_int(self.context_size, "context_size")
        _positive_int(self.parallel, "parallel")
        if self.fit_mode != "off":
            raise ValueError("fit_mode must be 'off' for the process-bound v1 profile")
        if self.sleep_idle_seconds is not None:
            _positive_int(self.sleep_idle_seconds, "sleep_idle_seconds")
        arguments = tuple(self.extra_args)
        if any(
            not isinstance(item, str)
            or not item
            or "\x00" in item
            for item in arguments
        ):
            raise ValueError("extra_args are invalid")
        if arguments:
            if len(arguments) != 2 or arguments[0] != "--n-gpu-layers":
                raise ValueError(
                    "process-bound v1 permits only an explicit --n-gpu-layers pair"
                )
            try:
                gpu_layers = int(arguments[1])
            except ValueError as exc:
                raise ValueError("--n-gpu-layers must be a positive integer") from exc
            if str(gpu_layers) != arguments[1] or gpu_layers < 1:
                raise ValueError("--n-gpu-layers must be a positive integer")
        object.__setattr__(self, "extra_args", arguments)
        for name in (
            "startup_timeout_seconds",
            "request_timeout_seconds",
            "shutdown_timeout_seconds",
            "poll_interval_seconds",
        ):
            object.__setattr__(
                self,
                name,
                _positive_number(getattr(self, name), name),
            )
        _positive_int(self.maximum_response_bytes, "maximum_response_bytes")

    def argv(self) -> tuple[str, ...]:
        arguments = [
            self.executable_path,
            "-m",
            self.model_path,
            "--alias",
            self.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--offline",
            "--no-ui",
            "--no-ui-mcp-proxy",
            "--no-agent",
            "--no-slots",
            "--fit",
            self.fit_mode,
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            str(self.parallel),
        ]
        if self.sleep_idle_seconds is not None:
            arguments.extend(
                ("--sleep-idle-seconds", str(self.sleep_idle_seconds))
            )
        arguments.extend(self.extra_args)
        return tuple(arguments)


class _NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class BoundedLoopbackJsonTransport:
    """No-proxy, no-redirect, bounded GET transport for fixed loopback paths."""

    def __init__(self, *, opener: object | None = None) -> None:
        self._opener = opener or request.build_opener(
            request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def get_json(
        self,
        *,
        host: str,
        port: int,
        path: str,
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> JsonHttpResponse:
        normalized_host = normalize_numeric_loopback_host(host)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise LlamaCppTransportError("endpoint_invalid", "Invalid loopback port.")
        if path not in _ALLOWED_GET_PATHS:
            raise LlamaCppTransportError("path_forbidden", "GET path is outside the v1 contract.")
        timeout = _positive_number(timeout_seconds, "timeout_seconds")
        limit = _positive_int(maximum_bytes, "maximum_bytes")
        rendered_host = (
            f"[{normalized_host}]" if ":" in normalized_host else normalized_host
        )
        target = f"http://{rendered_host}:{port}{path}"
        outbound = request.Request(
            target,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
            method="GET",
        )
        try:
            response = self._opener.open(outbound, timeout=timeout)  # type: ignore[attr-defined]
        except error.HTTPError as exc:
            if 300 <= exc.code < 400:
                exc.close()
                raise LlamaCppTransportError(
                    "redirect_forbidden", "Runtime probes never follow redirects."
                ) from exc
            response = exc
        except (OSError, ValueError) as exc:
            raise LlamaCppTransportError(
                "request_failed", "Runtime GET probe failed."
            ) from exc
        try:
            status_code = int(
                getattr(response, "status", None) or response.getcode()
            )
            headers = getattr(response, "headers", {})
            content_type = str(headers.get("Content-Type", ""))
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                raise LlamaCppTransportError(
                    "content_type_invalid",
                    "Runtime probe response must be application/json.",
                )
            content_encoding = str(headers.get("Content-Encoding", "")).lower()
            if content_encoding not in {"", "identity"}:
                raise LlamaCppTransportError(
                    "encoding_forbidden", "Compressed runtime probe responses are disabled."
                )
            content_length = headers.get("Content-Length")
            if content_length not in (None, ""):
                try:
                    declared = int(content_length)
                except (TypeError, ValueError) as exc:
                    raise LlamaCppTransportError(
                        "response_invalid", "Invalid Content-Length header."
                    ) from exc
                if declared < 0 or declared > limit:
                    raise LlamaCppTransportError(
                        "response_too_large", "Runtime probe response exceeds its bound."
                    )
            raw = response.read(limit + 1)
        except LlamaCppTransportError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise LlamaCppTransportError(
                "response_invalid", "Unable to read the runtime probe response."
            ) from exc
        finally:
            response.close()
        if len(raw) > limit:
            raise LlamaCppTransportError(
                "response_too_large", "Runtime probe response exceeds its bound."
            )
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (
            RecursionError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise LlamaCppTransportError(
                "response_invalid", "Runtime probe response is not valid JSON."
            ) from exc
        return JsonHttpResponse(status_code=status_code, payload=payload)


class SubprocessLauncher:
    """Direct POSIX process-group launcher; it never invokes a shell."""

    def launch(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        working_directory: str,
    ) -> SpawnedProcess:
        if os.name != "posix":
            raise LlamaCppRuntimeSupervisorError(
                "platform_unsupported",
                "The process-bound v1 production launcher requires POSIX.",
            )
        try:
            return subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(environment),
                cwd=working_directory,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise LlamaCppRuntimeSupervisorError(
                "launch_failed", "Unable to launch the configured llama-server binary."
            ) from exc


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _sanitized_environment(source: Mapping[str, str]) -> Mapping[str, str]:
    del source
    return MappingProxyType(dict(_FIXED_ENVIRONMENT))


class LlamaCppRuntimeSupervisor:
    """Own one direct llama-server process and refuse all attach semantics."""

    def __init__(
        self,
        spec: LlamaCppRuntimeSpec,
        *,
        observer: RuntimeProcessObserver,
        launcher: ProcessLauncher | None = None,
        transport: LoopbackJsonTransport | None = None,
        environment: Mapping[str, str] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not isinstance(spec, LlamaCppRuntimeSpec):
            raise TypeError("spec must be LlamaCppRuntimeSpec")
        self._spec = spec
        self._observer = observer
        self._launcher = launcher or SubprocessLauncher()
        self._transport = transport or BoundedLoopbackJsonTransport()
        self._environment = _sanitized_environment(
            os.environ if environment is None else environment
        )
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._process: SpawnedProcess | None = None
        self._root_identity: tuple[int, int, str] | None = None
        self._cleanup_unknown = False
        self._lock = threading.RLock()

    @property
    def owns_process(self) -> bool:
        with self._lock:
            return self._process is not None

    @property
    def cleanup_unknown(self) -> bool:
        """Whether shutdown ownership is ambiguous and restart is blocked."""

        with self._lock:
            return self._cleanup_unknown

    def start(
        self,
        *,
        on_process_launched: Callable[[], None] | None = None,
        on_process_observed: Callable[[ProcessTreeEvidence], None] | None = None,
    ) -> LlamaCppRuntimeEvidence:
        with self._lock:
            if self._cleanup_unknown:
                raise LlamaCppRuntimeSupervisorError(
                    "cleanup_unverified",
                    "A previous cleanup is unverified; restart is blocked.",
                )
            if self._process is not None:
                raise LlamaCppRuntimeSupervisorError(
                    "already_started", "This supervisor already owns a process."
                )
            vacant = self._observe_endpoint(root_pid=None)
            if vacant.ambiguous or vacant.listener_pids:
                raise LlamaCppRuntimeSupervisorError(
                    "endpoint_in_use",
                    "The configured loopback endpoint already has a listener; "
                    "attach is forbidden.",
                )
            process = self._launcher.launch(
                self._spec.argv(),
                environment=self._environment,
                working_directory=self._spec.working_directory,
            )
            if on_process_launched is not None:
                on_process_launched()
            if (
                isinstance(process.pid, bool)
                or not isinstance(process.pid, int)
                or process.pid < 1
            ):
                self._terminate_process(process)
                raise LlamaCppRuntimeSupervisorError(
                    "launch_invalid", "The launcher returned an invalid root PID."
                )
            self._process = process
            deadline = self._monotonic() + self._spec.startup_timeout_seconds
            last_reasons: tuple[str, ...] = ("startup_pending",)
            starting_notified = False
            try:
                while True:
                    if process.poll() is not None:
                        raise LlamaCppRuntimeSupervisorError(
                            "process_exited", "llama-server exited before readiness."
                        )
                    process_evidence = self._observe_process(process.pid)
                    if not starting_notified and on_process_observed is not None:
                        on_process_observed(process_evidence)
                        starting_notified = True
                    endpoint_evidence = self._observe_endpoint(
                        root_pid=process.pid
                    )

                    if endpoint_evidence.ambiguous:
                        raise LlamaCppRuntimeSupervisorError(
                            "endpoint_ambiguous",
                            "Loopback listener ownership became ambiguous.",
                        )
                    if endpoint_evidence.listener_pids and not endpoint_evidence.owned_by_root:
                        raise LlamaCppRuntimeSupervisorError(
                            "endpoint_owner_mismatch",
                            "The loopback listener is not owned by the launched root PID.",
                        )
                    if endpoint_evidence.owned_by_root:
                        application = self._probe_application()
                        last_reasons = application.reason_codes
                        if application.ready:
                            # Bind successful application probes to the same
                            # root process and listener on both sides of I/O.
                            process_evidence = self._observe_process(process.pid)
                            endpoint_evidence = self._observe_endpoint(
                                root_pid=process.pid
                            )
                            if endpoint_evidence.ambiguous:
                                raise LlamaCppRuntimeSupervisorError(
                                    "endpoint_ambiguous",
                                    "Loopback listener ownership changed during readiness probes.",
                                )
                            if not endpoint_evidence.owned_by_root:
                                raise LlamaCppRuntimeSupervisorError(
                                    "endpoint_owner_mismatch",
                                    "The loopback listener changed owner during readiness probes.",
                                )
                            return LlamaCppRuntimeEvidence(
                                process=process_evidence,
                                endpoint=endpoint_evidence,
                                application=application,
                            )
                        if _TERMINAL_APPLICATION_REASONS.intersection(
                            application.reason_codes
                        ):
                            raise LlamaCppRuntimeSupervisorError(
                                "application_identity_mismatch",
                                "llama-server application identity is invalid.",
                                reason_codes=application.reason_codes,
                            )
                    now = self._monotonic()
                    if now >= deadline:
                        raise LlamaCppRuntimeSupervisorError(
                            "startup_timeout",
                            "llama-server did not become ready before the deadline.",
                            reason_codes=last_reasons,
                        )
                    self._sleeper(
                        min(self._spec.poll_interval_seconds, deadline - now)
                    )
            except Exception as original:
                try:
                    self._terminate_process(process)
                    endpoint = self._observe_endpoint(root_pid=process.pid)
                    if endpoint.ambiguous or endpoint.listener_pids:
                        raise LlamaCppRuntimeSupervisorError(
                            "cleanup_unverified",
                            "Startup cleanup did not prove endpoint vacancy.",
                        )
                except Exception as cleanup_error:
                    self._cleanup_unknown = True
                    raise LlamaCppRuntimeSupervisorError(
                        "cleanup_unverified",
                        "Startup failed and owned-process cleanup is unverified.",
                    ) from cleanup_error
                self._process = None
                self._root_identity = None
                raise original

    def inspect(self) -> LlamaCppRuntimeEvidence:
        """Inspect only the process launched by this supervisor; never attach."""

        with self._lock:
            process = self._process
            if process is None:
                raise LlamaCppRuntimeSupervisorError(
                    "process_not_owned", "This supervisor has no owned process."
                )
            if process.poll() is not None:
                raise LlamaCppRuntimeSupervisorError(
                    "process_exited", "The owned llama-server process has exited."
                )
            process_evidence = self._observe_process(process.pid)
            endpoint_evidence = self._observe_endpoint(root_pid=process.pid)
            if endpoint_evidence.ambiguous:
                raise LlamaCppRuntimeSupervisorError(
                    "endpoint_ambiguous", "Loopback listener ownership is ambiguous."
                )
            if not endpoint_evidence.owned_by_root:
                raise LlamaCppRuntimeSupervisorError(
                    "endpoint_ownership_lost",
                    "The configured listener is no longer owned by the root process.",
                )
            application = self._probe_application()
            if not application.ready:
                code = (
                    "application_identity_mismatch"
                    if _TERMINAL_APPLICATION_REASONS.intersection(
                        application.reason_codes
                    )
                    else "application_not_ready"
                )
                raise LlamaCppRuntimeSupervisorError(
                    code,
                    "llama-server application readiness is no longer valid.",
                    reason_codes=application.reason_codes,
                )
            process_evidence = self._observe_process(process.pid)
            endpoint_evidence = self._observe_endpoint(root_pid=process.pid)
            if endpoint_evidence.ambiguous or not endpoint_evidence.owned_by_root:
                raise LlamaCppRuntimeSupervisorError(
                    "endpoint_ownership_lost",
                    "Listener ownership changed during application probes.",
                )
            return LlamaCppRuntimeEvidence(
                process=process_evidence,
                endpoint=endpoint_evidence,
                application=application,
            )

    def stop(self) -> None:
        """Terminate the owned POSIX process group and prove endpoint vacancy."""

        with self._lock:
            process = self._process
            if process is None:
                if self._cleanup_unknown:
                    raise LlamaCppRuntimeSupervisorError(
                        "cleanup_unverified",
                        "Owned-process cleanup remains unverified.",
                    )
                return
            root_pid = process.pid
            try:
                self._terminate_process(process)
                endpoint = self._observe_endpoint(root_pid=root_pid)
                if endpoint.ambiguous or endpoint.listener_pids:
                    self._cleanup_unknown = True
                    raise LlamaCppRuntimeSupervisorError(
                        "cleanup_unverified",
                        "The endpoint is not vacant after owned-root shutdown.",
                    )
            except Exception:
                self._cleanup_unknown = True
                raise
            else:
                self._process = None
                self._root_identity = None
                self._cleanup_unknown = False

    def _observe_process(self, root_pid: int) -> ProcessTreeEvidence:
        try:
            evidence = self._observer.observe_process_tree(root_pid)
        except RuntimeProcessObservationError as exc:
            raise LlamaCppRuntimeSupervisorError(
                "process_observation_failed", "Unable to observe the owned root process."
            ) from exc
        except Exception as exc:
            raise LlamaCppRuntimeSupervisorError(
                "process_observation_failed", "Unable to observe the owned root process."
            ) from exc
        if evidence.root_pid != root_pid or not evidence.root_only or evidence.process_count != 1:
            raise LlamaCppRuntimeSupervisorError(
                "process_contract_mismatch", "Process evidence exceeds the root-only v1 contract."
            )
        if evidence.root_executable_sha256 != self._spec.executable_sha256:
            raise LlamaCppRuntimeSupervisorError(
                "executable_identity_mismatch",
                "Observed root executable does not match the configured digest.",
            )
        identity = (
            evidence.root_pid,
            evidence.create_time_ns,
            evidence.root_executable_sha256,
        )
        if self._root_identity is None:
            self._root_identity = identity
        elif self._root_identity != identity:
            raise LlamaCppRuntimeSupervisorError(
                "process_identity_changed", "Owned root process identity changed."
            )
        return evidence

    def _observe_endpoint(self, *, root_pid: int | None) -> EndpointOwnershipEvidence:
        try:
            return self._observer.observe_endpoint_ownership(
                host=self._spec.host,
                port=self._spec.port,
                root_pid=root_pid,
            )
        except RuntimeProcessObservationError as exc:
            raise LlamaCppRuntimeSupervisorError(
                "endpoint_observation_failed", "Unable to observe listener ownership."
            ) from exc
        except Exception as exc:
            raise LlamaCppRuntimeSupervisorError(
                "endpoint_observation_failed", "Unable to observe listener ownership."
            ) from exc

    def _probe_application(self) -> LlamaCppApplicationEvidence:
        reasons: list[str] = []
        responses: dict[str, JsonHttpResponse | None] = {}
        for label, path in (
            ("health", "/health"),
            ("props", "/props"),
            ("models", "/v1/models"),
        ):
            try:
                responses[label] = self._transport.get_json(
                    host=self._spec.host,
                    port=self._spec.port,
                    path=path,
                    timeout_seconds=self._spec.request_timeout_seconds,
                    maximum_bytes=self._spec.maximum_response_bytes,
                )
            except (LlamaCppTransportError, OSError, ValueError):
                responses[label] = None
                reasons.append(f"{label}_unreachable")

        health = responses["health"]
        health_ready = bool(
            health is not None
            and health.status_code == 200
            and isinstance(health.payload, dict)
            and health.payload.get("status") == "ok"
        )
        if health is not None and not health_ready:
            reasons.append("health_not_ready")

        props = responses["props"]
        props_payload: object = {}
        if props is not None and health_ready:
            props_payload = props.payload
            if props.status_code != 200 or not isinstance(props.payload, dict):
                reasons.append("props_invalid")
            else:
                reported_path = props.payload.get("model_path")
                build_info = props.payload.get("build_info")
                sleeping = props.payload.get("is_sleeping")
                if (
                    not isinstance(reported_path, str)
                    or not isinstance(build_info, str)
                    or not build_info
                    or not isinstance(sleeping, bool)
                ):
                    reasons.append("props_invalid")
                else:
                    if os.path.realpath(reported_path) != os.path.realpath(
                        self._spec.model_path
                    ):
                        reasons.append("props_model_mismatch")
                    if sleeping:
                        reasons.append("runtime_sleeping")
        try:
            props_digest = _sha256_json(props_payload)
        except (OverflowError, RecursionError, TypeError, ValueError):
            props_digest = _sha256_json({})
            reasons.append("props_invalid")

        model_ids: tuple[str, ...] = ()
        models = responses["models"]
        if models is not None and health_ready:
            if models.status_code != 200 or not isinstance(models.payload, dict):
                reasons.append("models_invalid")
            else:
                entries = models.payload.get("data")
                if not isinstance(entries, list):
                    reasons.append("models_invalid")
                else:
                    parsed_ids: list[str] = []
                    for entry in entries:
                        if (
                            not isinstance(entry, dict)
                            or not isinstance(entry.get("id"), str)
                            or not entry["id"]
                        ):
                            reasons.append("models_invalid")
                            parsed_ids = []
                            break
                        parsed_ids.append(entry["id"])
                    if parsed_ids:
                        model_ids = tuple(sorted(set(parsed_ids)))
                        if len(parsed_ids) != 1 or len(model_ids) != 1:
                            reasons.append("model_count_invalid")
                        elif model_ids[0] != self._spec.model_id:
                            reasons.append("model_id_mismatch")
                    elif not reasons or "models_invalid" not in reasons:
                        reasons.append("model_count_invalid")

        normalized_reasons = tuple(sorted(set(reasons)))
        return LlamaCppApplicationEvidence(
            ready=not normalized_reasons,
            model_ids=model_ids,
            models_digest=_sha256_json({"model_ids": list(model_ids)}),
            props_digest=props_digest,
            reason_codes=normalized_reasons,
        )

    def _terminate_process(self, process: SpawnedProcess) -> None:
        if isinstance(process, subprocess.Popen):
            self._terminate_posix_group(process)
            return
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=self._spec.shutdown_timeout_seconds)
            return
        except (subprocess.TimeoutExpired, TimeoutError):
            pass
        except OSError as exc:
            raise LlamaCppRuntimeSupervisorError(
                "cleanup_failed", "Unable to terminate the owned root process."
            ) from exc
        try:
            process.kill()
            process.wait(timeout=self._spec.shutdown_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired, TimeoutError) as exc:
            raise LlamaCppRuntimeSupervisorError(
                "cleanup_failed", "Unable to kill the owned root process."
            ) from exc

    def _terminate_posix_group(self, process: subprocess.Popen[bytes]) -> None:
        if os.name != "posix":
            raise LlamaCppRuntimeSupervisorError(
                "cleanup_failed", "POSIX process-group cleanup is unavailable."
            )
        try:
            process_group = process.pid
            root_running = process.poll() is None
            if root_running:
                observed_group = os.getpgid(process.pid)
                if observed_group != process_group:
                    raise LlamaCppRuntimeSupervisorError(
                        "cleanup_failed",
                        "Owned process is no longer its expected process-group leader.",
                    )
            if root_running or self._process_group_exists(process_group):
                os.killpg(process_group, signal.SIGTERM)
            if root_running:
                try:
                    process.wait(timeout=self._spec.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process_group, signal.SIGKILL)
                    process.wait(timeout=self._spec.shutdown_timeout_seconds)
            if self._process_group_exists(process_group):
                os.killpg(process_group, signal.SIGKILL)
                deadline = self._monotonic() + self._spec.shutdown_timeout_seconds
                while self._process_group_exists(process_group):
                    now = self._monotonic()
                    if now >= deadline:
                        raise LlamaCppRuntimeSupervisorError(
                            "cleanup_failed",
                            "The owned process group did not become empty.",
                        )
                    self._sleeper(min(0.05, deadline - now))
        except LlamaCppRuntimeSupervisorError:
            raise
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LlamaCppRuntimeSupervisorError(
                "cleanup_failed", "Unable to terminate the owned process group."
            ) from exc

    @staticmethod
    def _process_group_exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise LlamaCppRuntimeSupervisorError(
                "cleanup_failed", "Unable to verify the owned process group."
            ) from exc
        return True
