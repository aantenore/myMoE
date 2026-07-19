from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import stat
import threading
from types import MappingProxyType
from typing import Any, Mapping

from filelock import FileLock, Timeout

from .route_signals import TaskSignals
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    DIFFICULTIES,
    EVIDENCE_STRENGTHS,
    OUTCOME_STATUSES,
    ROUTE_PLANS,
    VerifiedRoutingError,
    canonical_json,
    now_utc,
    reject_unknown,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


_RECORD_FIELDS = {
    "schema_version",
    "record_id",
    "created_at",
    "route_receipt_id",
    "route_receipt_sha256",
    "task_fingerprint",
    "config_sha256",
    "signal_provider_config_sha256",
    "runtime_plan_sha256",
    "profile",
    "planned_route",
    "final_provider",
    "capabilities",
    "difficulty",
    "confidence",
    "source",
    "abstained",
    "outcome",
    "evidence_strength",
    "evidence_sha256",
    "failure_class",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "premium_calls",
    "remote_payload_chars",
    "estimated_cost_usd",
    "provider_runtime_sha256",
    "model",
}
_RECORD_OPTIONAL_FIELDS = {
    "route_canary",
    "paired_run",
    "paired_cost",
    "paired_evidence",
}
_RECORD_ALLOWED_FIELDS = _RECORD_FIELDS | _RECORD_OPTIONAL_FIELDS
_BRIDGE_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "code",
    "route_receipt",
    "verification",
    "commands",
    "capsule",
    "final_provider",
    "premium_calls_used",
    "privacy",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "contract",
    "receipt_id",
    "task",
    "route",
    "local_provider",
    "premium_provider",
    "local_gaps",
    "premium_gaps",
    "remote_allowed",
    "premium_call_budget",
    "rationale_codes",
    "expected_flow",
    "config_sha256",
    "workspace",
    "local_runtime",
    "premium_runtime",
    "route_canary",
}
_RECEIPT_REQUIRED_FIELDS = _RECEIPT_FIELDS.difference({"route_canary"})
_TASK_FIELDS = {
    "task_id",
    "objective_sha256",
    "task_fingerprint",
    "objective_chars",
    "profile",
    "capability_demand",
    "constraint_count",
    "no_change_expected",
    "required_verifier_ids",
    "allow_remote",
    "allow_remote_workspace",
    "max_premium_calls",
}
_EVIDENCE_FIELDS = {
    "id",
    "verifier",
    "kind",
    "passed",
    "code",
    "artifact_sha256",
    "observed_chars",
    "evidence_ref",
    "task_fingerprint",
    "workspace_fingerprint",
    "verifier_spec_sha256",
}
_COMMAND_FIELDS = {
    "provider_id",
    "status",
    "code",
    "returncode",
    "duration_ms",
    "output_sha256",
    "output_chars",
    "stdout_sha256",
    "stdout_bytes",
    "stderr_sha256",
    "stderr_bytes",
    "command_sha256",
    "usage",
}
_CAPSULE_FIELDS = {
    "capsule_id",
    "sha256",
    "characters",
    "objective_sha256",
    "constraint_count",
    "verification_count",
    "failure_codes",
    "diff_sha256",
    "redaction_count",
    "residual_assured",
    "residual_detector",
    "truncated",
    "content_in_metadata",
}
_OUTCOME_ID = re.compile(r"^outcome-[0-9a-f]{64}$")
_OUTCOME_STORE_LOCK_TIMEOUT_SECONDS = 10.0
_MAX_OUTCOME_RECORD_BYTES = 2 * 1024 * 1024
_MAX_OUTCOME_STORE_BYTES = 128 * 1024 * 1024
_MAX_OUTCOME_RECORDS = 100_000


@dataclass(frozen=True)
class VerifiedOutcomeRecord:
    schema_version: str
    record_id: str
    created_at: str
    route_receipt_id: str
    route_receipt_sha256: str
    task_fingerprint: str
    config_sha256: str
    signal_provider_config_sha256: str
    runtime_plan_sha256: str
    profile: str
    planned_route: str
    final_provider: str | None
    capabilities: tuple[str, ...]
    difficulty: str
    confidence: float
    source: str
    abstained: bool
    outcome: str
    evidence_strength: str
    evidence_sha256: str
    failure_class: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    premium_calls: int
    remote_payload_chars: int
    estimated_cost_usd: float | None = None
    provider_runtime_sha256: str | None = None
    model: str | None = None
    route_canary: Mapping[str, object] | None = None
    paired_run: Mapping[str, object] | None = None
    paired_cost: Mapping[str, object] | None = None
    paired_evidence: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported outcome schema_version.")
        normalized_time = require_utc_timestamp(self.created_at, "created_at")
        if normalized_time != self.created_at:
            raise VerifiedRoutingError("created_at must use canonical UTC form.")
        require_safe_id(self.route_receipt_id, "route_receipt_id")
        require_sha256(self.route_receipt_sha256, "route_receipt_sha256")
        require_sha256(self.task_fingerprint, "task_fingerprint")
        require_sha256(self.config_sha256, "config_sha256")
        require_sha256(
            self.signal_provider_config_sha256,
            "signal_provider_config_sha256",
        )
        require_sha256(self.runtime_plan_sha256, "runtime_plan_sha256")
        require_safe_id(self.profile, "profile")
        if self.planned_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("planned_route is unsupported.")
        if self.final_provider is not None:
            require_safe_id(self.final_provider, "final_provider")
        capabilities = require_identifier_tuple(self.capabilities, "capabilities")
        object.__setattr__(self, "capabilities", capabilities)
        if self.difficulty not in DIFFICULTIES:
            raise VerifiedRoutingError("difficulty is unsupported.")
        confidence = require_finite_number(
            self.confidence, "confidence", minimum=0.0, maximum=1.0
        )
        object.__setattr__(self, "confidence", confidence)
        require_safe_id(self.source, "source")
        if not isinstance(self.abstained, bool):
            raise VerifiedRoutingError("abstained must be boolean.")
        if self.outcome not in OUTCOME_STATUSES:
            raise VerifiedRoutingError("outcome is unsupported.")
        if self.evidence_strength not in EVIDENCE_STRENGTHS:
            raise VerifiedRoutingError("evidence_strength is unsupported.")
        require_sha256(self.evidence_sha256, "evidence_sha256")
        require_safe_id(self.failure_class, "failure_class")
        for name in (
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "premium_calls",
            "remote_payload_chars",
        ):
            object.__setattr__(
                self,
                name,
                require_non_negative_int(getattr(self, name), name),
            )
        if self.estimated_cost_usd is not None:
            object.__setattr__(
                self,
                "estimated_cost_usd",
                require_finite_number(
                    self.estimated_cost_usd,
                    "estimated_cost_usd",
                    minimum=0.0,
                ),
            )
        if self.provider_runtime_sha256 is not None:
            require_sha256(
                self.provider_runtime_sha256, "provider_runtime_sha256"
            )
        if self.model is not None:
            require_safe_id(self.model, "model")
        if self.route_canary is not None:
            from .route_canary import CanaryRouteDecision

            canary = CanaryRouteDecision.from_payload(self.route_canary)
            if (
                canary.effective_route != self.planned_route
                or canary.bridge_config_sha256 != self.config_sha256
                or canary.task_fingerprint != self.task_fingerprint
                or canary.profile != self.profile
                or canary.capabilities != tuple(sorted(self.capabilities))
                or canary.difficulty != self.difficulty
                or (
                    canary.signal_provider_config_sha256
                    != self.signal_provider_config_sha256
                )
                or canary.runtime_plan_sha256 != self.runtime_plan_sha256
            ):
                raise VerifiedRoutingError(
                    "Outcome route canary binding is invalid."
                )
            object.__setattr__(
                self,
                "route_canary",
                _freeze_json(canary.payload()),
            )
        if self.paired_run is not None:
            from .paired_execution_contracts import PairedOutcomeBinding

            binding = PairedOutcomeBinding.from_payload(self.paired_run)
            if (
                binding.task_fingerprint != self.task_fingerprint
                or binding.bridge_config_sha256 != self.config_sha256
                or binding.route != self.planned_route
            ):
                raise VerifiedRoutingError(
                    "Outcome paired-run binding is invalid."
                )
            object.__setattr__(
                self,
                "paired_run",
                _freeze_json(binding.payload()),
            )
        if self.paired_cost is not None:
            from decimal import Decimal

            from .paired_execution_pricing import PairedCostEvidence

            cost = PairedCostEvidence.from_payload(self.paired_cost)
            if self.paired_run is None:
                raise VerifiedRoutingError(
                    "Paired cost evidence requires paired-run lineage."
                )
            pricing_sha256 = str(self.paired_run["pricing_sha256"])
            if (
                cost.pricing_sha256 != pricing_sha256
                or self.estimated_cost_usd is None
                # `estimated_cost_usd` is the legacy numeric projection used by
                # aggregate scorecards. The paired payload remains the exact
                # canonical Decimal source of truth and must reproduce that
                # projection, rather than being rounded to match it.
                or self.estimated_cost_usd
                != float(Decimal(cost.total_cost_usd))
            ):
                raise VerifiedRoutingError(
                    "Outcome paired cost evidence is invalid."
                )
            object.__setattr__(
                self,
                "paired_cost",
                _freeze_json(cost.payload()),
            )
        if self.paired_evidence is not None:
            from .assistant_bridge_two_phase_contracts import ArtifactDescriptor
            from .paired_evidence import PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE

            if self.paired_run is None:
                raise VerifiedRoutingError(
                    "Paired attestation evidence requires paired-run lineage."
                )
            try:
                descriptor = ArtifactDescriptor.from_payload(self.paired_evidence)
            except ValueError as exc:
                raise VerifiedRoutingError(str(exc)) from exc
            if (
                descriptor.media_type != PAIRED_ATTESTATION_RECEIPT_MEDIA_TYPE
                or descriptor.size_bytes <= 0
            ):
                raise VerifiedRoutingError(
                    "Paired attestation receipt descriptor is invalid."
                )
            object.__setattr__(
                self,
                "paired_evidence",
                _freeze_json(descriptor.payload()),
            )
        if self.final_provider is None and (
            self.provider_runtime_sha256 is not None or self.model is not None
        ):
            raise VerifiedRoutingError(
                "Provider runtime metadata requires final_provider."
            )
        if self.outcome == "passed" and self.failure_class != "none":
            raise VerifiedRoutingError("Passed outcomes cannot carry a failure class.")
        if self.outcome == "failed" and self.failure_class == "none":
            raise VerifiedRoutingError("Failed outcomes require a failure class.")
        expected_id = f"outcome-{sha256_json(self._unsigned_payload())}"
        if _OUTCOME_ID.fullmatch(self.record_id) is None or self.record_id != expected_id:
            raise VerifiedRoutingError("record_id does not match record content.")

    def _unsigned_payload(self) -> dict[str, object]:
        payload = self.payload()
        payload.pop("record_id")
        return payload

    def payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "created_at": self.created_at,
            "route_receipt_id": self.route_receipt_id,
            "route_receipt_sha256": self.route_receipt_sha256,
            "task_fingerprint": self.task_fingerprint,
            "config_sha256": self.config_sha256,
            "signal_provider_config_sha256": self.signal_provider_config_sha256,
            "runtime_plan_sha256": self.runtime_plan_sha256,
            "profile": self.profile,
            "planned_route": self.planned_route,
            "final_provider": self.final_provider,
            "capabilities": list(self.capabilities),
            "difficulty": self.difficulty,
            "confidence": self.confidence,
            "source": self.source,
            "abstained": self.abstained,
            "outcome": self.outcome,
            "evidence_strength": self.evidence_strength,
            "evidence_sha256": self.evidence_sha256,
            "failure_class": self.failure_class,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "premium_calls": self.premium_calls,
            "remote_payload_chars": self.remote_payload_chars,
            "estimated_cost_usd": self.estimated_cost_usd,
            "provider_runtime_sha256": self.provider_runtime_sha256,
            "model": self.model,
        }
        if self.route_canary is not None:
            payload["route_canary"] = _thaw_json(self.route_canary)
        if self.paired_run is not None:
            payload["paired_run"] = _thaw_json(self.paired_run)
        if self.paired_cost is not None:
            payload["paired_cost"] = _thaw_json(self.paired_cost)
        if self.paired_evidence is not None:
            payload["paired_evidence"] = _thaw_json(self.paired_evidence)
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "VerifiedOutcomeRecord":
        payload = _mapping(raw, "outcome record")
        reject_unknown(payload, _RECORD_ALLOWED_FIELDS, "outcome record")
        missing = sorted(_RECORD_FIELDS.difference(payload))
        if missing:
            raise VerifiedRoutingError(
                f"Missing outcome record fields: {', '.join(missing)}."
            )
        return cls(**payload)  # type: ignore[arg-type]


def build_verified_outcome(
    bridge_metadata: Mapping[str, object],
    signals: TaskSignals | Mapping[str, object],
    *,
    estimated_cost_usd: float | None = None,
    created_at: str | None = None,
    paired_run: Mapping[str, object] | None = None,
    paired_cost: Mapping[str, object] | None = None,
    paired_evidence: Mapping[str, object] | None = None,
) -> VerifiedOutcomeRecord:
    if estimated_cost_usd is not None:
        estimated_cost_usd = require_finite_number(
            estimated_cost_usd,
            "estimated_cost_usd",
            minimum=0.0,
        )
    bridge = _mapping(bridge_metadata, "bridge metadata")
    _require_exact_fields(bridge, _BRIDGE_FIELDS, "bridge metadata")
    if (
        bridge["schema_version"] != "2.0"
        or bridge["mode"] != "assistant_bridge"
        or bridge["privacy"] != "metadata_only"
    ):
        raise VerifiedRoutingError("Bridge metadata contract is unsupported.")
    require_safe_id(bridge["status"], "bridge status")
    require_safe_id(bridge["code"], "bridge code")

    receipt = _mapping(bridge["route_receipt"], "route receipt")
    reject_unknown(receipt, _RECEIPT_FIELDS, "route receipt")
    missing_receipt_fields = sorted(_RECEIPT_REQUIRED_FIELDS.difference(receipt))
    if missing_receipt_fields:
        raise VerifiedRoutingError(
            "Missing route receipt fields: "
            + ", ".join(missing_receipt_fields)
            + "."
        )
    if receipt["schema_version"] != "2.0" or receipt["contract"] != "RouteDecisionReceipt":
        raise VerifiedRoutingError("Route receipt contract is unsupported.")
    route_receipt_id = require_safe_id(receipt["receipt_id"], "route_receipt_id")
    planned_route = str(receipt["route"])
    if planned_route not in ROUTE_PLANS:
        raise VerifiedRoutingError("Only executable route plans can record outcomes.")
    config_sha256 = require_sha256(receipt["config_sha256"], "config_sha256")
    runtime_plan_digest = runtime_plan_sha256(receipt)
    route_canary: dict[str, object] | None = None
    if "route_canary" in receipt:
        from .route_canary import validate_canary_receipt_binding

        canary = validate_canary_receipt_binding(
            _mapping(receipt["route_canary"], "route canary"),
            receipt,
        )
        route_canary = canary.payload()

    task = _mapping(receipt["task"], "route task")
    _require_exact_fields(task, _TASK_FIELDS, "route task")
    task_fingerprint = require_sha256(
        task["task_fingerprint"], "task_fingerprint"
    )
    require_sha256(task["objective_sha256"], "objective_sha256")
    required_verifier_ids = require_identifier_tuple(
        task["required_verifier_ids"], "required_verifier_ids"
    )
    profile = require_safe_id(task["profile"], "profile")
    demand = _mapping(task["capability_demand"], "capability demand")
    _require_exact_fields(
        demand, {"required", "tools", "risk_class"}, "capability demand"
    )
    required_capabilities = require_identifier_tuple(
        demand["required"], "required capabilities"
    )
    require_identifier_tuple(demand["tools"], "required tools")
    require_safe_id(demand["risk_class"], "risk class")

    signal = signals if isinstance(signals, TaskSignals) else TaskSignals.from_payload(signals)
    if signal.request_fingerprint != task_fingerprint:
        raise VerifiedRoutingError("Signals do not belong to the routed task.")
    if tuple(signal.capabilities) != tuple(sorted(required_capabilities)):
        raise VerifiedRoutingError("Signals capabilities do not match the route receipt.")
    if route_canary is not None and (
        route_canary["difficulty"] != signal.difficulty
        or route_canary["signal_provider_config_sha256"]
        != signal.provider_config_sha256
    ):
        raise VerifiedRoutingError(
            "Signals do not match the route canary decision."
        )

    verification = _mapping(bridge["verification"], "verification")
    _require_exact_fields(verification, {"prior", "final"}, "verification")
    prior_evidence = _validate_evidence_list(
        verification["prior"], task_fingerprint, "prior evidence"
    )
    final_evidence = _validate_evidence_list(
        verification["final"], task_fingerprint, "final evidence"
    )
    outcome, failure_class, evidence_strength = _classify_outcome(
        bridge_status=str(bridge["status"]),
        bridge_code=str(bridge["code"]),
        prior_evidence=prior_evidence,
        final_evidence=final_evidence,
        required_verifier_ids=required_verifier_ids,
    )
    all_evidence = {"prior": prior_evidence, "final": final_evidence}

    commands = bridge["commands"]
    if not isinstance(commands, list):
        raise VerifiedRoutingError("commands must be a list.")
    latency_ms = 0
    prompt_tokens = 0
    completion_tokens = 0
    for index, raw_command in enumerate(commands):
        command = _mapping(raw_command, f"commands[{index}]")
        _require_exact_fields(command, _COMMAND_FIELDS, f"commands[{index}]")
        require_safe_id(command["provider_id"], f"commands[{index}].provider_id")
        require_safe_id(command["status"], f"commands[{index}].status")
        require_safe_id(command["code"], f"commands[{index}].code")
        latency_ms += require_non_negative_int(
            command["duration_ms"], f"commands[{index}].duration_ms"
        )
        usage = _mapping(command["usage"], f"commands[{index}].usage")
        _require_exact_fields(
            usage,
            {"prompt_tokens", "completion_tokens", "cost", "cost_status"},
            f"commands[{index}].usage",
        )
        prompt_tokens += _optional_non_negative_int(
            usage["prompt_tokens"], f"commands[{index}].prompt_tokens"
        )
        completion_tokens += _optional_non_negative_int(
            usage["completion_tokens"], f"commands[{index}].completion_tokens"
        )

    capsule = bridge["capsule"]
    remote_payload_chars = 0
    if capsule is not None:
        capsule_payload = _mapping(capsule, "capsule")
        _require_exact_fields(capsule_payload, _CAPSULE_FIELDS, "capsule")
        if capsule_payload["content_in_metadata"] is not False:
            raise VerifiedRoutingError("Capsule metadata must not contain content.")
        remote_payload_chars = require_non_negative_int(
            capsule_payload["characters"], "capsule.characters"
        )

    final_provider = bridge["final_provider"]
    if final_provider is not None:
        final_provider = require_safe_id(final_provider, "final_provider")
    provider_runtime_sha256, model = _selected_runtime(receipt, final_provider)
    premium_calls = require_non_negative_int(
        bridge["premium_calls_used"], "premium_calls_used"
    )

    unsigned = {
        "schema_version": CONTRACT_VERSION,
        "created_at": created_at or now_utc(),
        "route_receipt_id": route_receipt_id,
        "route_receipt_sha256": sha256_json(receipt),
        "task_fingerprint": task_fingerprint,
        "config_sha256": config_sha256,
        "signal_provider_config_sha256": signal.provider_config_sha256,
        "runtime_plan_sha256": runtime_plan_digest,
        "profile": profile,
        "planned_route": planned_route,
        "final_provider": final_provider,
        "capabilities": list(signal.capabilities),
        "difficulty": signal.difficulty,
        "confidence": signal.confidence,
        "source": signal.source,
        "abstained": signal.abstained,
        "outcome": outcome,
        "evidence_strength": evidence_strength,
        "evidence_sha256": sha256_json(all_evidence),
        "failure_class": failure_class,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "premium_calls": premium_calls,
        "remote_payload_chars": remote_payload_chars,
        "estimated_cost_usd": estimated_cost_usd,
        "provider_runtime_sha256": provider_runtime_sha256,
        "model": model,
    }
    if route_canary is not None:
        unsigned["route_canary"] = route_canary
    if paired_run is not None:
        from .paired_execution_contracts import PairedOutcomeBinding

        unsigned["paired_run"] = PairedOutcomeBinding.from_payload(
            paired_run
        ).payload()
    if paired_cost is not None:
        from .paired_execution_pricing import PairedCostEvidence

        unsigned["paired_cost"] = PairedCostEvidence.from_payload(
            paired_cost
        ).payload()
    if paired_evidence is not None:
        from .assistant_bridge_two_phase_contracts import ArtifactDescriptor

        try:
            unsigned["paired_evidence"] = ArtifactDescriptor.from_payload(
                paired_evidence
            ).payload()
        except ValueError as exc:
            raise VerifiedRoutingError(str(exc)) from exc
    payload = dict(unsigned)
    payload["record_id"] = f"outcome-{sha256_json(unsigned)}"
    return VerifiedOutcomeRecord.from_payload(payload)


class OutcomeStore:
    """Fail-closed append-only JSONL storage for metadata-only outcomes."""

    def __init__(
        self,
        path: str | Path,
        *,
        lock_timeout_seconds: float = _OUTCOME_STORE_LOCK_TIMEOUT_SECONDS,
    ):
        declared = Path(path).expanduser()
        try:
            # Resolve only the parent so an existing final-component symlink is
            # still visible to the no-follow validation below.  Pinning the
            # parent also prevents an ancestor symlink from redirecting later
            # reads or appends after construction.
            parent = declared.parent.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise VerifiedRoutingError(
                "Outcome store parent cannot be resolved safely."
            ) from exc
        self.path = parent / declared.name
        timeout = require_finite_number(
            lock_timeout_seconds,
            "outcome store lock_timeout_seconds",
            minimum=0.0,
        )
        if timeout == 0.0:
            raise VerifiedRoutingError(
                "outcome store lock_timeout_seconds must be positive."
            )
        self.lock_timeout_seconds = timeout
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._thread_lock = threading.Lock()
        self._process_lock = FileLock(
            str(self.lock_path),
            timeout=self.lock_timeout_seconds,
            mode=0o600,
        )

    def append(self, record: VerifiedOutcomeRecord) -> bool:
        if not isinstance(record, VerifiedOutcomeRecord):
            raise TypeError("record must be a VerifiedOutcomeRecord.")
        with self._locked():
            existing = self._read_unlocked()
            if any(item.record_id == record.record_id for item in existing):
                return False
            encoded = (canonical_json(record.payload()) + "\n").encode("utf-8")
            if len(encoded) > _MAX_OUTCOME_RECORD_BYTES:
                raise VerifiedRoutingError("Outcome record exceeds its size limit.")
            _append_secure_outcome_file(self.path, encoded)
        return True

    def list_records(self) -> tuple[VerifiedOutcomeRecord, ...]:
        with self._locked():
            return self._read_unlocked()

    def read_records(self) -> tuple[VerifiedOutcomeRecord, ...]:
        return self.list_records()

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            acquired = False
            try:
                _ensure_private_outcome_directory(self.path.parent)
                lock_before = _ensure_secure_outcome_lock(self.lock_path)
                parent_before = _validate_private_outcome_directory(
                    self.path.parent
                )
                self._process_lock.acquire(timeout=self.lock_timeout_seconds)
                acquired = True
            except Timeout as exc:
                raise VerifiedRoutingError(
                    "Outcome store lock acquisition timed out."
                ) from exc
            except OSError as exc:
                raise VerifiedRoutingError(
                    "Outcome store lock acquisition failed."
                ) from exc
            try:
                parent_after = _validate_private_outcome_directory(
                    self.path.parent
                )
                lock_after = _validate_outcome_regular_file(
                    self.lock_path,
                    "outcome store lock",
                    maximum_bytes=4096,
                )
                if (
                    _outcome_directory_identity(parent_before)
                    != _outcome_directory_identity(parent_after)
                    or _outcome_file_identity(lock_before)
                    != _outcome_file_identity(lock_after)
                ):
                    raise VerifiedRoutingError(
                        "Outcome store lock or parent changed during acquisition."
                    )
                yield
            finally:
                if acquired:
                    try:
                        self._process_lock.release()
                    except OSError as exc:
                        raise VerifiedRoutingError(
                            "Outcome store lock release failed."
                        ) from exc

    def _read_unlocked(self) -> tuple[VerifiedOutcomeRecord, ...]:
        encoded = _read_secure_outcome_file(self.path)
        if encoded is None or not encoded:
            return ()
        try:
            rendered = encoded.decode("utf-8")
        except UnicodeError as exc:
            raise VerifiedRoutingError("Outcome store is not valid UTF-8.") from exc
        if not rendered.endswith("\n"):
            raise VerifiedRoutingError("Outcome store is missing its final newline.")
        lines = rendered.splitlines()
        if len(lines) > _MAX_OUTCOME_RECORDS:
            raise VerifiedRoutingError("Outcome store contains too many records.")
        records: list[VerifiedOutcomeRecord] = []
        seen: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise VerifiedRoutingError(
                    f"Outcome store contains an empty line at {line_number}."
                )
            if len(line.encode("utf-8")) > _MAX_OUTCOME_RECORD_BYTES:
                raise VerifiedRoutingError(
                    f"Outcome store record exceeds its size limit at line {line_number}."
                )
            try:
                raw = _strict_json_loads(line)
                record = VerifiedOutcomeRecord.from_payload(
                    _mapping(raw, f"outcome line {line_number}")
                )
            except (UnicodeError, ValueError) as exc:
                raise VerifiedRoutingError(
                    f"Outcome store is corrupt at line {line_number}: {exc}"
                ) from exc
            if record.record_id in seen:
                raise VerifiedRoutingError(
                    f"Outcome store contains duplicate record_id at line {line_number}."
                )
            if line != canonical_json(record.payload()):
                raise VerifiedRoutingError(
                    f"Outcome store is not canonical at line {line_number}."
                )
            seen.add(record.record_id)
            records.append(record)
        return tuple(records)


def _ensure_private_outcome_directory(path: Path) -> os.stat_result:
    missing: list[Path] = []
    cursor = path
    while _outcome_lstat_optional(cursor) is None:
        if cursor == cursor.parent:
            raise VerifiedRoutingError(
                "Outcome store parent cannot be created securely."
            )
        missing.append(cursor)
        cursor = cursor.parent
    _validate_outcome_directory_kind(cursor, "outcome store ancestor")
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
            if os.name != "nt":
                os.chmod(directory, 0o700, follow_symlinks=False)
            _fsync_outcome_directory(directory.parent)
        except FileExistsError:
            pass
        except OSError as exc:
            raise VerifiedRoutingError(
                "Outcome store parent cannot be created securely."
            ) from exc
        _validate_private_outcome_directory(directory)
    return _validate_private_outcome_directory(path)


def _validate_outcome_directory_kind(
    path: Path,
    label: str,
    metadata: os.stat_result | None = None,
) -> os.stat_result:
    inspected = (
        _outcome_lstat_required(path, label) if metadata is None else metadata
    )
    if stat.S_ISLNK(inspected.st_mode) or not stat.S_ISDIR(inspected.st_mode):
        raise VerifiedRoutingError(f"{label} must be a non-link directory.")
    return inspected


def _validate_private_outcome_directory(
    path: Path,
    metadata: os.stat_result | None = None,
) -> os.stat_result:
    inspected = _validate_outcome_directory_kind(
        path,
        "outcome store parent",
        metadata,
    )
    if os.name != "nt" and stat.S_IMODE(inspected.st_mode) != 0o700:
        raise VerifiedRoutingError("Outcome store parent permissions must be 0700.")
    return inspected


def _ensure_secure_outcome_lock(path: Path) -> os.stat_result:
    metadata = _outcome_lstat_optional(path)
    if metadata is None:
        _create_secure_outcome_file(path, "outcome store lock")
    return _validate_outcome_regular_file(
        path,
        "outcome store lock",
        maximum_bytes=4096,
    )


def _create_secure_outcome_file(path: Path, label: str) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    opened: os.stat_result | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        _validate_outcome_file_metadata(opened, label, maximum_bytes=4096)
        os.fsync(descriptor)
    except FileExistsError:
        return
    except OSError as exc:
        raise VerifiedRoutingError(f"{label} cannot be created securely.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    current = _validate_outcome_regular_file(path, label, maximum_bytes=4096)
    if opened is None or _outcome_file_identity(opened) != _outcome_file_identity(
        current
    ):
        raise VerifiedRoutingError(f"{label} changed during creation.")
    _fsync_outcome_directory(path.parent)


def _append_secure_outcome_file(path: Path, encoded: bytes) -> None:
    before = _outcome_lstat_optional(path)
    created = before is None
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if created:
        flags |= os.O_CREAT | os.O_EXCL
    else:
        before = _validate_outcome_regular_file(
            path,
            "outcome store",
            before,
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        if created and os.name != "nt":
            os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        _validate_outcome_file_metadata(
            opened,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
        current = _validate_outcome_regular_file(
            path,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
        if _outcome_file_identity(opened) != _outcome_file_identity(current):
            raise VerifiedRoutingError("Outcome store changed before append.")
        if before is not None and (
            _outcome_file_snapshot(before) != _outcome_file_snapshot(opened)
            or _outcome_file_snapshot(before) != _outcome_file_snapshot(current)
        ):
            raise VerifiedRoutingError("Outcome store changed before append.")
        expected_size = opened.st_size + len(encoded)
        if expected_size > _MAX_OUTCOME_STORE_BYTES:
            raise VerifiedRoutingError("Outcome store exceeds its size limit.")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Outcome append did not make progress.")
            view = view[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        _validate_outcome_file_metadata(
            after,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
        installed = _validate_outcome_regular_file(
            path,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
        if (
            _outcome_file_identity(opened) != _outcome_file_identity(after)
            or _outcome_file_identity(opened) != _outcome_file_identity(installed)
            or after.st_size != expected_size
            or installed.st_size != expected_size
        ):
            raise VerifiedRoutingError("Outcome store changed during append.")
    except VerifiedRoutingError:
        raise
    except FileExistsError as exc:
        raise VerifiedRoutingError("Outcome store creation lost a race.") from exc
    except OSError as exc:
        raise VerifiedRoutingError("Outcome store append failed securely.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_outcome_directory(path.parent)


def _read_secure_outcome_file(path: Path) -> bytes | None:
    before = _outcome_lstat_optional(path)
    if before is None:
        return None
    before = _validate_outcome_regular_file(
        path,
        "outcome store",
        before,
        maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _validate_outcome_file_metadata(
            opened,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
        current = _validate_outcome_regular_file(
            path,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
        if (
            _outcome_file_snapshot(before) != _outcome_file_snapshot(opened)
            or _outcome_file_snapshot(before) != _outcome_file_snapshot(current)
        ):
            raise VerifiedRoutingError("Outcome store changed before read.")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_OUTCOME_STORE_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_OUTCOME_STORE_BYTES:
                raise VerifiedRoutingError("Outcome store exceeds its size limit.")
        after = os.fstat(descriptor)
        _validate_outcome_file_metadata(
            after,
            "outcome store",
            maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
        )
    except VerifiedRoutingError:
        raise
    except OSError as exc:
        raise VerifiedRoutingError("Outcome store cannot be read safely.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    installed = _validate_outcome_regular_file(
        path,
        "outcome store",
        maximum_bytes=_MAX_OUTCOME_STORE_BYTES,
    )
    if not (
        _outcome_file_snapshot(before)
        == _outcome_file_snapshot(opened)
        == _outcome_file_snapshot(after)
        == _outcome_file_snapshot(installed)
    ):
        raise VerifiedRoutingError("Outcome store changed while it was read.")
    return b"".join(chunks)


def _validate_outcome_regular_file(
    path: Path,
    label: str,
    metadata: os.stat_result | None = None,
    *,
    maximum_bytes: int,
) -> os.stat_result:
    inspected = (
        _outcome_lstat_required(path, label) if metadata is None else metadata
    )
    _validate_outcome_file_metadata(
        inspected,
        label,
        maximum_bytes=maximum_bytes,
    )
    return inspected


def _validate_outcome_file_metadata(
    metadata: os.stat_result,
    label: str,
    *,
    maximum_bytes: int,
) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VerifiedRoutingError(f"{label} must be a regular non-link file.")
    if metadata.st_nlink != 1:
        raise VerifiedRoutingError(f"{label} must have exactly one hard link.")
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise VerifiedRoutingError(f"{label} permissions must be 0600.")
    if metadata.st_size < 0 or metadata.st_size > maximum_bytes:
        raise VerifiedRoutingError(f"{label} exceeds its size limit.")


def _outcome_lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise VerifiedRoutingError("Outcome store path cannot be inspected.") from exc


def _outcome_lstat_required(path: Path, label: str) -> os.stat_result:
    metadata = _outcome_lstat_optional(path)
    if metadata is None:
        raise VerifiedRoutingError(f"{label} is missing.")
    return metadata


def _outcome_directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _outcome_file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _outcome_file_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _fsync_outcome_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EINVAL, errno.EPERM}:
            raise VerifiedRoutingError(
                "Outcome store directory could not be synchronized."
            ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _classify_outcome(
    *,
    bridge_status: str,
    bridge_code: str,
    prior_evidence: list[dict[str, object]],
    final_evidence: list[dict[str, object]],
    required_verifier_ids: tuple[str, ...],
) -> tuple[str, str, str]:
    successful_bridge_codes = {
        "completed",
        "local_candidate_generated",
        "local_verification_passed",
        "premium_candidate_generated",
        "premium_verification_passed",
    }
    if bridge_status != "completed" or bridge_code not in successful_bridge_codes:
        return "failed", bridge_code, "deterministic"

    final_by_id = {str(item["id"]): item for item in final_evidence}
    missing = [item for item in required_verifier_ids if item not in final_by_id]
    if missing:
        return "failed", "required_verifier_missing", "deterministic"
    failed_required = [
        final_by_id[item]
        for item in required_verifier_ids
        if final_by_id[item]["passed"] is not True
    ]
    if failed_required:
        return (
            "failed",
            _failure_class(failed_required),
            _evidence_strength(failed_required),
        )

    failed_final = [item for item in final_evidence if item["passed"] is False]
    if failed_final:
        return "failed", _failure_class(failed_final), _evidence_strength(failed_final)
    if final_evidence and all(item["passed"] is True for item in final_evidence):
        return "passed", "none", _evidence_strength(final_evidence)

    failed_prior = [item for item in prior_evidence if item["passed"] is False]
    if failed_prior:
        return "failed", _failure_class(failed_prior), _evidence_strength(failed_prior)
    return "inconclusive", "verification_missing", "implicit"


def _failure_class(evidence: list[dict[str, object]]) -> str:
    codes = sorted({str(item["code"]) for item in evidence})
    return codes[0] if len(codes) == 1 else "multiple_verification_failures"


def _evidence_strength(evidence: list[dict[str, object]]) -> str:
    if any(item["kind"] == "external" for item in evidence):
        return "independent"
    return "deterministic" if evidence else "implicit"


def _validate_evidence_list(
    raw: object, task_fingerprint: str, label: str
) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        raise VerifiedRoutingError(f"{label} must be a list.")
    parsed: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw):
        item = _mapping(value, f"{label}[{index}]")
        _require_exact_fields(item, _EVIDENCE_FIELDS, f"{label}[{index}]")
        for field in ("id", "verifier", "kind", "code"):
            require_safe_id(item[field], f"{label}[{index}].{field}")
        evidence_id = str(item["id"])
        if evidence_id in seen_ids:
            raise VerifiedRoutingError(f"{label} contains duplicate evidence ids.")
        seen_ids.add(evidence_id)
        if not isinstance(item["passed"], bool):
            raise VerifiedRoutingError(f"{label}[{index}].passed must be boolean.")
        for field in (
            "artifact_sha256",
            "task_fingerprint",
            "workspace_fingerprint",
            "verifier_spec_sha256",
        ):
            require_sha256(item[field], f"{label}[{index}].{field}")
        if item["task_fingerprint"] != task_fingerprint:
            raise VerifiedRoutingError(f"{label}[{index}] belongs to another task.")
        require_non_negative_int(item["observed_chars"], f"{label}[{index}].observed_chars")
        if item["evidence_ref"] is not None and not isinstance(item["evidence_ref"], str):
            raise VerifiedRoutingError(f"{label}[{index}].evidence_ref must be a string or null.")
        parsed.append(item)
    return parsed


def runtime_plan_sha256(receipt: Mapping[str, object] | object) -> str:
    """Digest the complete attested local and premium runtime plan."""

    if isinstance(receipt, Mapping):
        local_raw = receipt.get("local_runtime")
        premium_raw = receipt.get("premium_runtime")
    else:
        local_raw = getattr(receipt, "local_runtime", None)
        premium_raw = getattr(receipt, "premium_runtime", None)
    local_runtime = _validated_runtime_descriptor(local_raw, "local runtime")
    premium_runtime = _validated_runtime_descriptor(
        premium_raw, "premium runtime"
    )
    return sha256_json(
        {
            "local_runtime": local_runtime,
            "premium_runtime": premium_runtime,
        }
    )


def _selected_runtime(
    receipt: dict[str, Any], final_provider: str | None
) -> tuple[str | None, str | None]:
    if final_provider is None:
        return None, None
    local_provider = require_safe_id(receipt["local_provider"], "local_provider")
    premium_provider = receipt["premium_provider"]
    if premium_provider is not None:
        premium_provider = require_safe_id(premium_provider, "premium_provider")
    if final_provider == local_provider:
        runtime = _validated_runtime_descriptor(
            receipt["local_runtime"], "local runtime"
        )
    elif final_provider == premium_provider:
        runtime = _validated_runtime_descriptor(
            receipt["premium_runtime"], "premium runtime"
        )
    else:
        raise VerifiedRoutingError("final_provider is not bound to the route receipt.")
    model = runtime.get("model")
    digest = require_sha256(
        runtime["runtime_sha256"], "provider_runtime_sha256"
    )
    return digest, require_safe_id(model, "runtime model") if model else None


def _validated_runtime_descriptor(value: object, label: str) -> dict[str, Any]:
    runtime = _mapping(value, label)
    digest = require_sha256(runtime.get("runtime_sha256"), f"{label} digest")
    unsigned = dict(runtime)
    unsigned.pop("runtime_sha256", None)
    if sha256_json(unsigned) != digest:
        raise VerifiedRoutingError(f"{label} digest is invalid.")
    return runtime


def _optional_non_negative_int(value: object, label: str) -> int:
    return 0 if value is None else require_non_negative_int(value, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerifiedRoutingError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise VerifiedRoutingError(f"{label} keys must be strings.")
    return dict(value)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_fields(raw: dict[str, Any], fields: set[str], label: str) -> None:
    reject_unknown(raw, fields, label)
    missing = sorted(fields.difference(raw))
    if missing:
        raise VerifiedRoutingError(
            f"Missing {label} fields: {', '.join(missing)}."
        )


def _strict_json_loads(value: str) -> object:
    def reject_constant(token: str) -> object:
        raise VerifiedRoutingError(f"Non-finite JSON number {token!r} is forbidden.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise VerifiedRoutingError(f"Duplicate JSON key {key!r} is forbidden.")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
