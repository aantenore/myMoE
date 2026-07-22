"""Process-bound lifecycle orchestration for one verified local llama.cpp cell.

The module composes the static Bound Cell inspection, the owner-bound metadata
lease, and the direct llama.cpp adapter. It never adopts a process, invokes a
model, or turns lifecycle evidence into inference authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Callable, Protocol, Sequence

from .llama_cpp_runtime_supervisor import (
    LlamaCppRuntimeEvidence,
    LlamaCppRuntimeSpec,
    LlamaCppRuntimeSupervisor,
    LlamaCppRuntimeSupervisorError,
)
from .runtime_binding_inspector import (
    ResolvedCellRuntimeLaunch,
    RuntimeBindingInspectionError,
    resolve_verified_cell_runtime_launch,
)
from .runtime_process_observer import (
    ProcessTreeEvidence,
    PsutilRuntimeProcessObserver,
    RuntimeProcessObserver,
)
from .runtime_supervisor_contracts import (
    RuntimeSupervisorLeaseBinding,
    RuntimeSupervisorLeaseReceipt,
)
from .runtime_supervisor_store import (
    RuntimeSupervisorLeaseAcquisition,
    RuntimeSupervisorLeaseHandle,
    RuntimeSupervisorLeaseStoreError,
    SQLiteRuntimeSupervisorLeaseStore,
)


_ADAPTER_ID = "mymoe_llama_cpp_process_bound_v1"
_DEFAULT_STATIC_RECHECK_SECONDS = 300.0
_ALLOWED_FINAL_REASONS = frozenset(
    {
        "binding_changed",
        "cleanup_unverified",
        "endpoint_already_occupied",
        "health_probe_failed",
        "listener_missing",
        "model_advertisement_changed",
        "ownership_unknown",
        "pid_reused",
        "port_substituted",
        "process_tree_changed",
        "runtime_executable_changed",
        "runtime_exited",
        "runtime_restarted",
    }
)
_ERROR_REASON_CODES = {
    "application_identity_mismatch": "model_advertisement_changed",
    "application_not_ready": "health_probe_failed",
    "binding_changed": "binding_changed",
    "binding_not_verified": "binding_changed",
    "cleanup_failed": "cleanup_unverified",
    "cleanup_unverified": "cleanup_unverified",
    "endpoint_ambiguous": "ownership_unknown",
    "endpoint_in_use": "endpoint_already_occupied",
    "endpoint_observation_failed": "ownership_unknown",
    "endpoint_owner_mismatch": "port_substituted",
    "endpoint_ownership_lost": "port_substituted",
    "executable_identity_mismatch": "runtime_executable_changed",
    "launch_failed": "runtime_exited",
    "launch_invalid": "runtime_exited",
    "process_contract_mismatch": "process_tree_changed",
    "process_exited": "runtime_exited",
    "process_identity_changed": "pid_reused",
    "process_observation_failed": "ownership_unknown",
    "runtime_config_invalid": "binding_changed",
    "runtime_executable_mismatch": "runtime_executable_changed",
    "runtime_plan_invalid": "binding_changed",
    "startup_timeout": "health_probe_failed",
}


class RuntimeSupervisorError(RuntimeError):
    """Stable lifecycle failure without private paths or response bodies."""

    def __init__(
        self,
        code: str,
        detail: str = "Runtime supervision failed safely.",
        *,
        reason_codes: Sequence[str] = (),
    ) -> None:
        self.code = str(code)
        normalized = tuple(sorted(set(str(item) for item in reason_codes)))
        self.reason_codes = normalized
        super().__init__(detail)


class RuntimeAdapter(Protocol):
    @property
    def owns_process(self) -> bool: ...

    @property
    def cleanup_unknown(self) -> bool: ...

    def start(
        self,
        *,
        on_process_launched: Callable[[], None] | None = None,
        on_process_observed: Callable[[ProcessTreeEvidence], None] | None = None,
    ) -> LlamaCppRuntimeEvidence: ...

    def inspect(self) -> LlamaCppRuntimeEvidence: ...

    def stop(self) -> None: ...


Resolver = Callable[[str | Path], ResolvedCellRuntimeLaunch]
AdapterFactory = Callable[[LlamaCppRuntimeSpec], RuntimeAdapter]


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeSupervisorError("receipt_invalid", f"{label} is invalid.")
    return value


@dataclass(frozen=True)
class RuntimeSupervisorStartResult:
    """Sanitizable ready result; private process data stays in nested evidence."""

    receipt: RuntimeSupervisorLeaseReceipt
    evidence: LlamaCppRuntimeEvidence
    state: str = "ready"

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "contract": "mymoe-runtime-supervisor-start/v1",
                "state": self.state,
                "lease_receipt_sha256": self.receipt.digest,
                "runtime_evidence_sha256": self.evidence.digest,
            }
        )


@dataclass(frozen=True)
class RuntimeSupervisorInspectionResult:
    """One sampled continuity result with no launch paths or response bodies."""

    evidence: LlamaCppRuntimeEvidence
    binding_manifest_sha256: str
    state: str = "ready"

    def __post_init__(self) -> None:
        _require_sha256(self.binding_manifest_sha256, "binding manifest digest")

    @property
    def digest(self) -> str:
        return _sha256_json(
            {
                "contract": "mymoe-runtime-supervisor-inspection/v1",
                "state": self.state,
                "binding_manifest_sha256": self.binding_manifest_sha256,
                "runtime_evidence_sha256": self.evidence.digest,
            }
        )


@dataclass(frozen=True)
class RuntimeSupervisorSessionReceipt:
    """Final digest-only account of the owned diagnostic lifecycle."""

    state: str
    binding_sha256: str
    lease_receipt_sha256_chain: tuple[str, ...]
    process_tree_sha256: str | None
    endpoint_evidence_sha256: str | None
    application_evidence_sha256: str | None
    reason_codes: tuple[str, ...]
    lifecycle_operations: int
    process_mutations: bool
    model_invocations: int = 0
    supervisor_remote_egress: bool = False
    runtime_egress_attestation: str = "not_observed"
    offline_launch_profile: bool = True
    authorizes_inference: bool = False
    diagnostic_only: bool = True
    digest: str = ""
    contract: str = "mymoe-runtime-supervisor-session-receipt/v1"

    def __post_init__(self) -> None:
        if self.state not in {"stopped", "unknown_blocking"}:
            raise RuntimeSupervisorError("receipt_invalid", "Final state is invalid.")
        _require_sha256(self.binding_sha256, "binding digest")
        chain = tuple(self.lease_receipt_sha256_chain)
        if not chain:
            raise RuntimeSupervisorError("receipt_invalid", "Lease receipt chain is empty.")
        for item in chain:
            _require_sha256(item, "lease receipt digest")
        if len(chain) != len(set(chain)):
            raise RuntimeSupervisorError("receipt_invalid", "Lease receipt chain repeats.")
        object.__setattr__(self, "lease_receipt_sha256_chain", chain)
        for name in (
            "process_tree_sha256",
            "endpoint_evidence_sha256",
            "application_evidence_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)
        reasons = tuple(sorted(set(self.reason_codes)))
        if not set(reasons).issubset(_ALLOWED_FINAL_REASONS):
            raise RuntimeSupervisorError("receipt_invalid", "Reason codes are invalid.")
        if self.state == "unknown_blocking" and "cleanup_unverified" not in reasons:
            raise RuntimeSupervisorError(
                "receipt_invalid", "Unknown cleanup requires cleanup_unverified."
            )
        object.__setattr__(self, "reason_codes", reasons)
        if type(self.lifecycle_operations) is not int or not 0 <= self.lifecycle_operations <= 2:
            raise RuntimeSupervisorError(
                "receipt_invalid", "Lifecycle operation count is invalid."
            )
        if self.process_mutations is not (self.lifecycle_operations > 0):
            raise RuntimeSupervisorError(
                "receipt_invalid", "Process mutation declaration is inconsistent."
            )
        if self.model_invocations != 0:
            raise RuntimeSupervisorError("receipt_invalid", "Model invocations must remain zero.")
        if (
            self.supervisor_remote_egress
            or self.runtime_egress_attestation != "not_observed"
            or not self.offline_launch_profile
            or self.authorizes_inference
            or not self.diagnostic_only
        ):
            raise RuntimeSupervisorError("receipt_invalid", "Receipt authority widened.")
        expected = _sha256_json(self.content_payload())
        if self.digest and self.digest != expected:
            raise RuntimeSupervisorError("receipt_invalid", "Receipt digest is invalid.")
        object.__setattr__(self, "digest", expected)

    def content_payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "state": self.state,
            "binding_sha256": self.binding_sha256,
            "lease_receipt_sha256_chain": list(self.lease_receipt_sha256_chain),
            "process_tree_sha256": self.process_tree_sha256,
            "endpoint_evidence_sha256": self.endpoint_evidence_sha256,
            "application_evidence_sha256": self.application_evidence_sha256,
            "reason_codes": list(self.reason_codes),
            "lifecycle_operations": self.lifecycle_operations,
            "process_mutations": self.process_mutations,
            "model_invocations": self.model_invocations,
            "supervisor_remote_egress": self.supervisor_remote_egress,
            "runtime_egress_attestation": self.runtime_egress_attestation,
            "offline_launch_profile": self.offline_launch_profile,
            "authorizes_inference": self.authorizes_inference,
            "diagnostic_only": self.diagnostic_only,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _LaunchFingerprint:
    executable: _FileFingerprint
    model: _FileFingerprint


def _file_fingerprint(path: Path) -> _FileFingerprint:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeSupervisorError("binding_changed") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeSupervisorError("binding_changed")
    return _FileFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _launch_fingerprint(resolved: ResolvedCellRuntimeLaunch) -> _LaunchFingerprint:
    return _LaunchFingerprint(
        executable=_file_fingerprint(resolved.runtime_executable_path),
        model=_file_fingerprint(resolved.model_artifact_path),
    )


def _positive_int(value: str, label: str) -> int:
    try:
        rendered = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeSupervisorError("runtime_plan_invalid", f"{label} is invalid.") from exc
    if rendered < 1 or str(rendered) != value:
        raise RuntimeSupervisorError("runtime_plan_invalid", f"{label} is invalid.")
    return rendered


def build_llama_cpp_runtime_spec(
    resolved: ResolvedCellRuntimeLaunch,
) -> LlamaCppRuntimeSpec:
    """Reconstruct and exactly match the hardened bound argv."""

    if resolved.backend != "llama_cpp":
        raise RuntimeSupervisorError("runtime_profile_unsupported")
    if resolved.runtime_security_profile != "process_bound_v1":
        raise RuntimeSupervisorError("runtime_profile_unsupported")
    argv = tuple(resolved.argv)
    expected_prefix = (
        "-m",
        "--alias",
        "--host",
        "--port",
        "--offline",
        "--no-ui",
        "--no-ui-mcp-proxy",
        "--no-agent",
        "--no-slots",
        "--fit",
        "--ctx-size",
        "--parallel",
    )
    if len(argv) not in {20, 22}:
        raise RuntimeSupervisorError("runtime_plan_invalid")
    if (
        argv[1] != expected_prefix[0]
        or argv[3] != expected_prefix[1]
        or argv[5] != expected_prefix[2]
        or argv[7] != expected_prefix[3]
        or argv[9:14] != expected_prefix[4:9]
        or argv[14] != expected_prefix[9]
        or argv[16] != expected_prefix[10]
        or argv[18] != expected_prefix[11]
        or argv[15] != "off"
    ):
        raise RuntimeSupervisorError("runtime_plan_invalid")
    if argv[4] != resolved.expected_model_id:
        raise RuntimeSupervisorError("runtime_plan_invalid")
    if argv[6] != resolved.endpoint_host or argv[8] != str(resolved.endpoint_port):
        raise RuntimeSupervisorError("runtime_plan_invalid")
    extra_args: tuple[str, ...] = ()
    if len(argv) == 22:
        if argv[20] != "--n-gpu-layers":
            raise RuntimeSupervisorError("runtime_plan_invalid")
        _positive_int(argv[21], "gpu layers")
        extra_args = argv[20:22]
    spec = LlamaCppRuntimeSpec(
        executable_path=str(resolved.runtime_executable_path),
        executable_sha256=resolved.runtime_executable_sha256,
        model_path=str(resolved.model_artifact_path),
        model_id=resolved.expected_model_id,
        working_directory=str(resolved.working_directory),
        host=resolved.endpoint_host,
        port=resolved.endpoint_port,
        context_size=_positive_int(argv[17], "context size"),
        parallel=_positive_int(argv[19], "parallel"),
        fit_mode="off",
        extra_args=extra_args,
    )
    normalized = list(argv)
    normalized[0] = str(resolved.runtime_executable_path)
    normalized[2] = str(resolved.model_artifact_path)
    if tuple(normalized) != spec.argv():
        raise RuntimeSupervisorError("runtime_plan_invalid")
    return spec


def _lease_binding(resolved: ResolvedCellRuntimeLaunch) -> RuntimeSupervisorLeaseBinding:
    manifest = resolved.bundle.manifest
    return RuntimeSupervisorLeaseBinding(
        binding_request_sha256=resolved.request_sha256,
        binding_manifest_sha256=resolved.binding_manifest_sha256,
        launch_plan_sha256=resolved.launch_plan_sha256,
        config_source_sha256=manifest.config_source_sha256,
        runtime_config_sha256=manifest.runtime_config_sha256,
        runtime_identity_sha256=manifest.runtime_identity_sha256,
        model_identity_sha256=resolved.model_identity_sha256,
        endpoint_authority_sha256=resolved.endpoint_authority_sha256,
        adapter_id=_ADAPTER_ID,
        runtime_backend=resolved.backend,
    )


def _same_resolution(
    expected: ResolvedCellRuntimeLaunch,
    observed: ResolvedCellRuntimeLaunch,
) -> bool:
    names = (
        "backend",
        "runtime_security_profile",
        "expert_id",
        "endpoint_host",
        "endpoint_port",
        "expected_model_id",
        "request_sha256",
        "binding_manifest_sha256",
        "launch_plan_sha256",
        "endpoint_authority_sha256",
        "runtime_executable_sha256",
        "model_identity_sha256",
    )
    return (
        all(getattr(expected, name) == getattr(observed, name) for name in names)
        and expected.request_path == observed.request_path
        and expected.working_directory == observed.working_directory
        and expected.runtime_executable_path == observed.runtime_executable_path
        and expected.model_artifact_path == observed.model_artifact_path
        and expected.argv == observed.argv
    )


def _reason_for_error(error: BaseException) -> str:
    raw_code = getattr(error, "code", "")
    code = str(raw_code) if isinstance(raw_code, str) else ""
    return _ERROR_REASON_CODES.get(code, "ownership_unknown")


class ProcessBoundRuntimeSession:
    """Foreground owner for one acquired exact endpoint lease."""

    def __init__(
        self,
        *,
        request_path: Path,
        resolved: ResolvedCellRuntimeLaunch,
        lease_store: SQLiteRuntimeSupervisorLeaseStore,
        acquisition: RuntimeSupervisorLeaseAcquisition,
        adapter: RuntimeAdapter,
        resolver: Resolver,
        static_recheck_interval_seconds: float,
        monotonic: Callable[[], float],
    ) -> None:
        if (
            isinstance(static_recheck_interval_seconds, bool)
            or not isinstance(static_recheck_interval_seconds, (int, float))
            or not math.isfinite(static_recheck_interval_seconds)
            or static_recheck_interval_seconds <= 0
        ):
            raise RuntimeSupervisorError("runtime_policy_invalid")
        self._request_path = request_path
        self._resolved = resolved
        self._binding = _lease_binding(resolved)
        self._store = lease_store
        self._handle: RuntimeSupervisorLeaseHandle = acquisition.handle
        self._adapter = adapter
        self._resolver = resolver
        self._static_recheck_interval_seconds = float(static_recheck_interval_seconds)
        self._monotonic = monotonic
        self._last_full_resolution_at = monotonic()
        self._fingerprint = _launch_fingerprint(resolved)
        self._lease_chain: list[RuntimeSupervisorLeaseReceipt] = [acquisition.receipt]
        self._state = "prepared"
        self._runtime_evidence: LlamaCppRuntimeEvidence | None = None
        self._final_receipt: RuntimeSupervisorSessionReceipt | None = None
        self._process_started = False
        self._process_stopped = False
        self._ledger_failed = False
        self._reasons: set[str] = set()
        self._lock = threading.RLock()

    @property
    def cleanup_unknown(self) -> bool:
        with self._lock:
            return (
                self._state == "unknown_blocking"
                or self._ledger_failed
                or self._adapter.cleanup_unknown
            )

    def start(self) -> RuntimeSupervisorStartResult:
        with self._lock:
            if self._state != "prepared":
                raise RuntimeSupervisorError("already_started")
            try:
                self._full_static_recheck()

                def on_process_launched() -> None:
                    self._process_started = True

                def on_process_observed(evidence: ProcessTreeEvidence) -> None:
                    receipt = self._store.transition(
                        self._handle,
                        "starting",
                        runtime_pid=evidence.root_pid,
                        runtime_create_time_ns=evidence.create_time_ns,
                        runtime_executable_sha256=evidence.root_executable_sha256,
                    )
                    self._append_transition(receipt)

                evidence = self._adapter.start(
                    on_process_launched=on_process_launched,
                    on_process_observed=on_process_observed
                )
                if self._state != "starting" or not self._process_started:
                    raise RuntimeSupervisorError("ownership_unknown")
                self._runtime_evidence = evidence
                self._full_static_recheck()
                evidence = self._adapter.inspect()
                self._assert_runtime_continuity(evidence)
                self._runtime_evidence = evidence
                receipt = self._store.transition(
                    self._handle,
                    "ready",
                    process_tree_sha256=evidence.process.digest,
                    endpoint_evidence_sha256=evidence.endpoint.digest,
                )
                self._append_transition(receipt)
                return RuntimeSupervisorStartResult(
                    receipt=receipt,
                    evidence=evidence,
                )
            except Exception as exc:
                self._record_failure(exc)
                if isinstance(exc, RuntimeSupervisorError):
                    raise
                code = str(getattr(exc, "code", "lifecycle_failed"))
                raise RuntimeSupervisorError(
                    code,
                    reason_codes=(_reason_for_error(exc),),
                ) from exc

    def inspect(self) -> RuntimeSupervisorInspectionResult:
        with self._lock:
            if self._state != "ready":
                raise RuntimeSupervisorError("runtime_not_ready")
            try:
                evidence = self._adapter.inspect()
                self._assert_runtime_continuity(evidence)
                current_fingerprint = _launch_fingerprint(self._resolved)
                now = self._monotonic()
                rehashed = False
                if current_fingerprint != self._fingerprint:
                    self._full_static_recheck(now=now)
                    rehashed = True
                elif now - self._last_full_resolution_at >= self._static_recheck_interval_seconds:
                    self._full_static_recheck(now=now)
                    rehashed = True
                if rehashed:
                    evidence = self._adapter.inspect()
                    self._assert_runtime_continuity(evidence)
                self._runtime_evidence = evidence
                return RuntimeSupervisorInspectionResult(
                    evidence=evidence,
                    binding_manifest_sha256=self._resolved.binding_manifest_sha256,
                )
            except Exception as exc:
                self._record_failure(exc)
                if isinstance(exc, RuntimeSupervisorError):
                    raise
                code = str(getattr(exc, "code", "lifecycle_failed"))
                raise RuntimeSupervisorError(
                    code,
                    reason_codes=(_reason_for_error(exc),),
                ) from exc

    def stop(self) -> RuntimeSupervisorSessionReceipt:
        with self._lock:
            if self._final_receipt is not None:
                return self._final_receipt
            if self._adapter.cleanup_unknown:
                self._ensure_unknown("cleanup_unverified")
                self._final_receipt = self._build_final_receipt("unknown_blocking")
                return self._final_receipt
            ledger_error: BaseException | None = None
            try:
                if self._state != "stopping":
                    try:
                        receipt = self._store.transition(
                            self._handle,
                            "stopping",
                            reason_codes=tuple(sorted(self._reasons)),
                        )
                        self._append_transition(receipt)
                    except Exception as exc:
                        self._ledger_failed = True
                        ledger_error = exc
                self._adapter.stop()
                if self._adapter.cleanup_unknown:
                    raise RuntimeSupervisorError("cleanup_unverified")
                if self._process_started:
                    self._process_stopped = True
                try:
                    self._full_static_recheck()
                except Exception:
                    self._reasons.add("binding_changed")
                if ledger_error is not None:
                    self._reasons.add("cleanup_unverified")
                    self._ensure_unknown("cleanup_unverified")
                    self._final_receipt = self._build_final_receipt(
                        "unknown_blocking"
                    )
                    return self._final_receipt
                receipt = self._store.transition(
                    self._handle,
                    "stopped",
                    reason_codes=tuple(sorted(self._reasons)),
                )
                self._append_transition(receipt)
                self._final_receipt = self._build_final_receipt("stopped")
                return self._final_receipt
            except Exception as exc:
                self._reasons.add("cleanup_unverified")
                self._ensure_unknown("cleanup_unverified")
                if self._ledger_failed:
                    raise RuntimeSupervisorError("cleanup_unverified") from exc
                self._final_receipt = self._build_final_receipt("unknown_blocking")
                return self._final_receipt

    def _append_transition(self, receipt: RuntimeSupervisorLeaseReceipt) -> None:
        self._lease_chain.append(receipt)
        self._state = receipt.state

    def _full_static_recheck(self, *, now: float | None = None) -> None:
        try:
            observed = self._resolver(self._request_path)
        except Exception as exc:
            raise RuntimeSupervisorError("binding_changed") from exc
        if not _same_resolution(self._resolved, observed):
            raise RuntimeSupervisorError("binding_changed")
        if build_llama_cpp_runtime_spec(observed).argv() != build_llama_cpp_runtime_spec(
            self._resolved
        ).argv():
            raise RuntimeSupervisorError("binding_changed")
        self._fingerprint = _launch_fingerprint(observed)
        self._last_full_resolution_at = self._monotonic() if now is None else now

    def _assert_runtime_continuity(self, evidence: LlamaCppRuntimeEvidence) -> None:
        previous = self._runtime_evidence
        if previous is None:
            raise RuntimeSupervisorError("ownership_unknown")
        if (
            evidence.process.root_pid != previous.process.root_pid
            or evidence.process.create_time_ns != previous.process.create_time_ns
            or evidence.process.root_executable_sha256
            != previous.process.root_executable_sha256
        ):
            raise RuntimeSupervisorError("process_identity_changed")
        if (
            evidence.endpoint.host != previous.endpoint.host
            or evidence.endpoint.port != previous.endpoint.port
            or evidence.endpoint.listener_pids != previous.endpoint.listener_pids
        ):
            raise RuntimeSupervisorError("endpoint_ownership_lost")
        if evidence.application.model_ids != previous.application.model_ids:
            raise RuntimeSupervisorError("application_identity_mismatch")

    def _record_failure(self, error: BaseException) -> None:
        reason = _reason_for_error(error)
        self._reasons.add(reason)
        if self._adapter.cleanup_unknown or reason == "cleanup_unverified":
            self._ensure_unknown("cleanup_unverified")
            return
        if self._state in {"prepared", "starting", "ready"}:
            try:
                receipt = self._store.transition(
                    self._handle,
                    "revoked",
                    reason_codes=(reason,),
                )
                self._append_transition(receipt)
            except Exception:
                self._ledger_failed = True

    def _ensure_unknown(self, reason: str) -> None:
        self._reasons.add(reason)
        if self._state == "unknown_blocking":
            return
        if self._state == "stopped":
            self._ledger_failed = True
            return
        try:
            receipt = self._store.transition(
                self._handle,
                "unknown_blocking",
                reason_codes=tuple(sorted(self._reasons)),
            )
            self._append_transition(receipt)
        except Exception:
            self._ledger_failed = True

    def _build_final_receipt(self, state: str) -> RuntimeSupervisorSessionReceipt:
        evidence = self._runtime_evidence
        operations = int(self._process_started) + int(self._process_stopped)
        return RuntimeSupervisorSessionReceipt(
            state=state,
            binding_sha256=self._binding.digest,
            lease_receipt_sha256_chain=tuple(
                receipt.digest for receipt in self._lease_chain
            ),
            process_tree_sha256=(
                None if evidence is None else evidence.process.digest
            ),
            endpoint_evidence_sha256=(
                None if evidence is None else evidence.endpoint.digest
            ),
            application_evidence_sha256=(
                None if evidence is None else evidence.application.digest
            ),
            reason_codes=tuple(sorted(self._reasons)),
            lifecycle_operations=operations,
            process_mutations=operations > 0,
        )


def build_runtime_supervisor(
    request_path: str | Path,
    *,
    lease_store: SQLiteRuntimeSupervisorLeaseStore,
    observer: RuntimeProcessObserver | None = None,
    resolver: Resolver = resolve_verified_cell_runtime_launch,
    adapter_factory: AdapterFactory | None = None,
    static_recheck_interval_seconds: float = _DEFAULT_STATIC_RECHECK_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
) -> ProcessBoundRuntimeSession:
    """Resolve, reserve, and construct one non-started foreground session."""

    request = Path(request_path).resolve()
    try:
        resolved = resolver(request)
        spec = build_llama_cpp_runtime_spec(resolved)
        binding = _lease_binding(resolved)
        acquisition = lease_store.acquire(binding)
    except RuntimeSupervisorError:
        raise
    except (RuntimeBindingInspectionError, RuntimeSupervisorLeaseStoreError) as exc:
        code = str(getattr(exc, "code", "lifecycle_failed"))
        raise RuntimeSupervisorError(code) from exc
    factory = adapter_factory
    if factory is None:
        process_observer = observer or PsutilRuntimeProcessObserver()
        factory = lambda runtime_spec: LlamaCppRuntimeSupervisor(
            runtime_spec,
            observer=process_observer,
        )
    try:
        adapter = factory(spec)
        return ProcessBoundRuntimeSession(
            request_path=request,
            resolved=resolved,
            lease_store=lease_store,
            acquisition=acquisition,
            adapter=adapter,
            resolver=resolver,
            static_recheck_interval_seconds=static_recheck_interval_seconds,
            monotonic=monotonic,
        )
    except Exception as exc:
        try:
            lease_store.transition(acquisition.handle, "stopping")
            lease_store.transition(acquisition.handle, "stopped")
        except Exception as cleanup_error:
            raise RuntimeSupervisorError("cleanup_unverified") from cleanup_error
        if isinstance(exc, RuntimeSupervisorError):
            raise
        raise RuntimeSupervisorError("lifecycle_failed") from exc


def create_runtime_supervisor_session(
    request_path: str | Path,
    *,
    state_directory: str | Path | None = None,
) -> ProcessBoundRuntimeSession:
    """Create the production session used by the installed foreground CLI."""

    if state_directory is None:
        store = SQLiteRuntimeSupervisorLeaseStore()
    else:
        root = Path(state_directory).resolve()
        store = SQLiteRuntimeSupervisorLeaseStore(
            root / "leases.sqlite3",
            sentinel_root=root / "owners",
        )
    return build_runtime_supervisor(request_path, lease_store=store)


__all__ = [
    "ProcessBoundRuntimeSession",
    "RuntimeSupervisorError",
    "RuntimeSupervisorInspectionResult",
    "RuntimeSupervisorSessionReceipt",
    "RuntimeSupervisorStartResult",
    "build_llama_cpp_runtime_spec",
    "build_runtime_supervisor",
    "create_runtime_supervisor_session",
]
