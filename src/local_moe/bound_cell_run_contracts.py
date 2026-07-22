from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Mapping

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


RUN_POLICY_CONTRACT = "BoundCellRunPolicy"
RUN_RECEIPT_CONTRACT = "BoundCellRunReceipt"
RUN_STATUSES = frozenset({"completed", "blocked", "failed", "invalidated"})
DELIVERY_STATUSES = frozenset(
    {"not_attempted", "attempted_unknown", "response_received"}
)
RUN_REASON_CODES = frozenset(
    {
        "adaptive_admission_blocked",
        "adaptive_preview_invalid",
        "binding_changed",
        "binding_inspection_failed",
        "binding_not_verified",
        "clock_invalid",
        "config_changed",
        "confirmation_required",
        "declaration_mismatch",
        "endpoint_not_loopback",
        "expert_mismatch",
        "expected_model_missing",
        "execution_interrupted",
        "model_identity_changed",
        "model_probe_failed",
        "passport_mismatch",
        "post_binding_inspection_failed",
        "response_invalid",
        "response_too_large",
        "risk_class_blocked",
        "runtime_config_mismatch",
        "selected_cell_mismatch",
        "task_invalid",
        "tool_surface_blocked",
        "transport_blocked",
        "transport_failed",
    }
)


class BoundCellRunContractError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _call(function, *args):
    try:
        return function(*args)
    except BoundCellRunContractError:
        raise
    except (VerifiedRoutingError, OverflowError, TypeError, ValueError) as exc:
        raise BoundCellRunContractError("contract_invalid", str(exc)) from exc


def _sha(value: object, label: str) -> str:
    return _call(require_sha256, value, label)


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _optional_id(value: object, label: str) -> str | None:
    return None if value is None else _call(require_safe_id, value, label)


def _integer(value: object, label: str) -> int:
    return _call(require_non_negative_int, value, label)


def _digest(value: object, content: Mapping[str, object], label: str) -> str:
    expected = sha256_json(content)
    if value not in (None, "") and _sha(value, label) != expected:
        raise BoundCellRunContractError(
            "digest_mismatch", f"{label} does not match content."
        )
    return expected


@dataclass(frozen=True)
class BoundCellRunPolicy:
    timeout_seconds: float = 60.0
    max_task_bytes: int = 256 * 1024
    max_response_bytes: int = 4 * 1024 * 1024
    max_probe_bytes: int = 1024 * 1024
    max_output_tokens: int = 2048
    max_models: int = 1024
    max_attempts: int = 1
    risk_classes: tuple[str, ...] = ("compute_only",)
    max_tool_surfaces: int = 0
    require_loopback: bool = True
    digest: str = ""
    contract: str = RUN_POLICY_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != CONTRACT_VERSION
            or self.contract != RUN_POLICY_CONTRACT
        ):
            raise BoundCellRunContractError(
                "policy_invalid", "Unsupported run policy contract."
            )
        if isinstance(self.timeout_seconds, bool):
            raise BoundCellRunContractError(
                "policy_invalid", "timeout_seconds must be numeric."
            )
        try:
            timeout = float(self.timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise BoundCellRunContractError(
                "policy_invalid", "timeout_seconds must be numeric."
            ) from exc
        if not 0 < timeout <= 120:
            raise BoundCellRunContractError(
                "policy_invalid", "timeout_seconds exceeds the v1 bound."
            )
        limits = {
            "max_task_bytes": (1, 1024 * 1024),
            "max_response_bytes": (1, 16 * 1024 * 1024),
            "max_probe_bytes": (1, 4 * 1024 * 1024),
            "max_output_tokens": (1, 32768),
            "max_models": (1, 4096),
        }
        for name, (minimum, maximum) in limits.items():
            value = _integer(getattr(self, name), name)
            if not minimum <= value <= maximum:
                raise BoundCellRunContractError(
                    "policy_invalid", f"{name} exceeds the v1 bound."
                )
            object.__setattr__(self, name, value)
        if _integer(self.max_attempts, "max_attempts") != 1:
            raise BoundCellRunContractError(
                "policy_invalid", "V1 permits exactly one attempt."
            )
        risks = _call(require_identifier_tuple, self.risk_classes, "risk_classes")
        if risks != ("compute_only",):
            raise BoundCellRunContractError(
                "policy_invalid", "V1 permits compute_only only."
            )
        if _integer(self.max_tool_surfaces, "max_tool_surfaces") != 0:
            raise BoundCellRunContractError("policy_invalid", "V1 permits no tools.")
        if self.require_loopback is not True:
            raise BoundCellRunContractError("policy_invalid", "V1 requires loopback.")
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "max_attempts", 1)
        object.__setattr__(self, "max_tool_surfaces", 0)
        object.__setattr__(self, "risk_classes", risks)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "policy digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "timeout_seconds": self.timeout_seconds,
            "max_task_bytes": self.max_task_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_probe_bytes": self.max_probe_bytes,
            "max_output_tokens": self.max_output_tokens,
            "max_models": self.max_models,
            "max_attempts": self.max_attempts,
            "risk_classes": list(self.risk_classes),
            "max_tool_surfaces": self.max_tool_surfaces,
            "require_loopback": self.require_loopback,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class BoundCellRunReceipt:
    policy_sha256: str
    status: str
    reason_codes: tuple[str, ...]
    started_at: str
    completed_at: str
    confirmed: bool
    task_sha256: str
    task_bytes: int
    preview_sha256: str | None = None
    selected_cell_id: str | None = None
    passport_sha256: str | None = None
    declaration_sha256: str | None = None
    expert_id: str | None = None
    pre_binding_bundle_sha256: str | None = None
    pre_binding_request_sha256: str | None = None
    pre_binding_manifest_sha256: str | None = None
    pre_inspection_receipt_sha256: str | None = None
    post_binding_bundle_sha256: str | None = None
    post_binding_request_sha256: str | None = None
    post_binding_manifest_sha256: str | None = None
    post_inspection_receipt_sha256: str | None = None
    pre_config_source_sha256: str | None = None
    post_config_source_sha256: str | None = None
    pre_model_identity_set_sha256: str | None = None
    post_model_identity_set_sha256: str | None = None
    response_sha256: str | None = None
    response_bytes: int | None = None
    response_chars: int | None = None
    invocation_attempts: int = 0
    endpoint_probe_requests: int = 0
    delivery_status: str = "not_attempted"
    elapsed_ms: int = 0
    retries: int = 0
    tools_invoked: int = 0
    risk_class: str = "compute_only"
    tool_surfaces: tuple[str, ...] = ()
    execution_scope: str = "device_only"
    execution_transport: str = "direct_local"
    process_mutations: bool = False
    lifecycle_operations: int = 0
    remote_egress: bool = False
    binding_continuity: str = "sampled_pre_post"
    endpoint_process_identity_verified: bool = False
    semantic_outcome_verified: bool = False
    authorizes_future_execution: bool = False
    digest: str = ""
    contract: str = RUN_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != CONTRACT_VERSION
            or self.contract != RUN_RECEIPT_CONTRACT
        ):
            raise BoundCellRunContractError(
                "receipt_invalid", "Unsupported run receipt contract."
            )
        object.__setattr__(
            self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256")
        )
        object.__setattr__(self, "task_sha256", _sha(self.task_sha256, "task_sha256"))
        for name in (
            "preview_sha256",
            "passport_sha256",
            "declaration_sha256",
            "pre_binding_bundle_sha256",
            "pre_binding_request_sha256",
            "pre_binding_manifest_sha256",
            "pre_inspection_receipt_sha256",
            "post_binding_bundle_sha256",
            "post_binding_request_sha256",
            "post_binding_manifest_sha256",
            "post_inspection_receipt_sha256",
            "pre_config_source_sha256",
            "post_config_source_sha256",
            "pre_model_identity_set_sha256",
            "post_model_identity_set_sha256",
            "response_sha256",
        ):
            object.__setattr__(self, name, _optional_sha(getattr(self, name), name))
        for name in ("selected_cell_id", "expert_id"):
            object.__setattr__(self, name, _optional_id(getattr(self, name), name))
        object.__setattr__(
            self,
            "started_at",
            _call(require_utc_timestamp, self.started_at, "started_at"),
        )
        object.__setattr__(
            self,
            "completed_at",
            _call(require_utc_timestamp, self.completed_at, "completed_at"),
        )
        if datetime.fromisoformat(self.completed_at) < datetime.fromisoformat(
            self.started_at
        ):
            raise BoundCellRunContractError(
                "receipt_invalid", "completed_at precedes started_at."
            )
        if self.status not in RUN_STATUSES:
            raise BoundCellRunContractError(
                "receipt_invalid", "Unsupported run status."
            )
        reasons = tuple(
            sorted(_call(require_identifier_tuple, self.reason_codes, "reason_codes"))
        )
        if not set(reasons).issubset(RUN_REASON_CODES):
            raise BoundCellRunContractError(
                "receipt_invalid", "Unsupported run reason code."
            )
        if self.status == "completed" and reasons:
            raise BoundCellRunContractError(
                "receipt_invalid", "Completed runs cannot contain blockers."
            )
        if self.status != "completed" and not reasons:
            raise BoundCellRunContractError(
                "receipt_invalid", "Non-completed runs require a reason."
            )
        if type(self.confirmed) is not bool:
            raise BoundCellRunContractError(
                "receipt_invalid", "confirmed must be boolean."
            )
        task_bytes = _integer(self.task_bytes, "task_bytes")
        attempts = _integer(self.invocation_attempts, "invocation_attempts")
        probes = _integer(self.endpoint_probe_requests, "endpoint_probe_requests")
        lifecycle = _integer(self.lifecycle_operations, "lifecycle_operations")
        elapsed = _integer(self.elapsed_ms, "elapsed_ms")
        retries = _integer(self.retries, "retries")
        tools = _integer(self.tools_invoked, "tools_invoked")
        if attempts > 1 or probes > 2 or lifecycle != 0 or retries != 0 or tools != 0:
            raise BoundCellRunContractError(
                "receipt_invalid", "Run counters exceed the v1 boundary."
            )
        if self.delivery_status not in DELIVERY_STATUSES:
            raise BoundCellRunContractError(
                "receipt_invalid", "Unsupported delivery status."
            )
        if self.status == "blocked" and attempts != 0:
            raise BoundCellRunContractError(
                "receipt_invalid", "Blocked runs cannot attempt inference."
            )
        if self.status in {"completed", "failed", "invalidated"} and attempts != 1:
            raise BoundCellRunContractError(
                "receipt_invalid", "Attempted runs require exactly one attempt."
            )
        if attempts == 0 and self.delivery_status != "not_attempted":
            raise BoundCellRunContractError(
                "receipt_invalid", "Unattempted runs cannot report delivery."
            )
        if attempts == 1 and self.delivery_status == "not_attempted":
            raise BoundCellRunContractError(
                "receipt_invalid", "Attempted runs require delivery state."
            )
        if attempts == 1 and not self.confirmed:
            raise BoundCellRunContractError(
                "receipt_invalid", "An inference attempt requires confirmation."
            )
        attempted_required = (
            self.preview_sha256,
            self.selected_cell_id,
            self.passport_sha256,
            self.declaration_sha256,
            self.expert_id,
            self.pre_binding_bundle_sha256,
            self.pre_binding_request_sha256,
            self.pre_binding_manifest_sha256,
            self.pre_inspection_receipt_sha256,
            self.pre_config_source_sha256,
            self.pre_model_identity_set_sha256,
        )
        if attempts == 1 and (
            task_bytes < 1
            or probes != 2
            or any(item is None for item in attempted_required)
        ):
            raise BoundCellRunContractError(
                "receipt_invalid", "Attempted run evidence is incomplete."
            )
        if attempts == 0 and probes > 1:
            raise BoundCellRunContractError(
                "receipt_invalid", "Unattempted runs cannot contain a post probe."
            )
        if not self.confirmed and "confirmation_required" not in reasons:
            raise BoundCellRunContractError(
                "receipt_invalid", "Unconfirmed runs require confirmation_required."
            )
        if self.status == "completed":
            required = (
                self.preview_sha256,
                self.selected_cell_id,
                self.passport_sha256,
                self.declaration_sha256,
                self.expert_id,
                self.pre_binding_bundle_sha256,
                self.post_binding_bundle_sha256,
                self.pre_binding_request_sha256,
                self.post_binding_request_sha256,
                self.pre_binding_manifest_sha256,
                self.post_binding_manifest_sha256,
                self.pre_inspection_receipt_sha256,
                self.post_inspection_receipt_sha256,
                self.pre_config_source_sha256,
                self.post_config_source_sha256,
                self.pre_model_identity_set_sha256,
                self.post_model_identity_set_sha256,
                self.response_sha256,
            )
            if (
                not self.confirmed
                or probes != 2
                or any(item is None for item in required)
            ):
                raise BoundCellRunContractError(
                    "receipt_invalid", "Completed run evidence is incomplete."
                )
            if (
                task_bytes < 1
                or self.pre_binding_manifest_sha256 != self.post_binding_manifest_sha256
                or self.pre_binding_request_sha256 != self.post_binding_request_sha256
                or self.pre_config_source_sha256 != self.post_config_source_sha256
                or self.pre_model_identity_set_sha256
                != self.post_model_identity_set_sha256
            ):
                raise BoundCellRunContractError(
                    "receipt_invalid", "Completed run evidence changed."
                )
        if self.response_sha256 is None:
            if self.response_bytes is not None or self.response_chars is not None:
                raise BoundCellRunContractError(
                    "receipt_invalid", "response size requires its digest."
                )
        else:
            response_bytes = _integer(self.response_bytes, "response_bytes")
            response_chars = _integer(self.response_chars, "response_chars")
            if response_bytes < 1:
                raise BoundCellRunContractError(
                    "receipt_invalid", "response_bytes must be positive."
                )
            if self.delivery_status != "response_received":
                raise BoundCellRunContractError(
                    "receipt_invalid", "Response evidence requires response_received."
                )
            object.__setattr__(self, "response_bytes", response_bytes)
            object.__setattr__(self, "response_chars", response_chars)
        if (
            self.risk_class != "compute_only"
            or tuple(self.tool_surfaces) != ()
            or self.execution_scope != "device_only"
            or self.execution_transport != "direct_local"
            or self.process_mutations is not False
            or self.remote_egress is not False
            or self.binding_continuity != "sampled_pre_post"
            or self.endpoint_process_identity_verified is not False
            or self.semantic_outcome_verified is not False
            or self.authorizes_future_execution is not False
        ):
            raise BoundCellRunContractError(
                "receipt_invalid", "Run authority exceeds the v1 contract."
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "task_bytes", task_bytes)
        object.__setattr__(self, "invocation_attempts", attempts)
        object.__setattr__(self, "endpoint_probe_requests", probes)
        object.__setattr__(self, "lifecycle_operations", lifecycle)
        object.__setattr__(self, "elapsed_ms", elapsed)
        object.__setattr__(self, "retries", retries)
        object.__setattr__(self, "tools_invoked", tools)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "run receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "policy_sha256": self.policy_sha256,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "confirmed": self.confirmed,
            "task_sha256": self.task_sha256,
            "task_bytes": self.task_bytes,
            "preview_sha256": self.preview_sha256,
            "selected_cell_id": self.selected_cell_id,
            "passport_sha256": self.passport_sha256,
            "declaration_sha256": self.declaration_sha256,
            "expert_id": self.expert_id,
            "pre_binding_bundle_sha256": self.pre_binding_bundle_sha256,
            "pre_binding_request_sha256": self.pre_binding_request_sha256,
            "pre_binding_manifest_sha256": self.pre_binding_manifest_sha256,
            "pre_inspection_receipt_sha256": self.pre_inspection_receipt_sha256,
            "post_binding_bundle_sha256": self.post_binding_bundle_sha256,
            "post_binding_request_sha256": self.post_binding_request_sha256,
            "post_binding_manifest_sha256": self.post_binding_manifest_sha256,
            "post_inspection_receipt_sha256": self.post_inspection_receipt_sha256,
            "pre_config_source_sha256": self.pre_config_source_sha256,
            "post_config_source_sha256": self.post_config_source_sha256,
            "pre_model_identity_set_sha256": self.pre_model_identity_set_sha256,
            "post_model_identity_set_sha256": self.post_model_identity_set_sha256,
            "response_sha256": self.response_sha256,
            "response_bytes": self.response_bytes,
            "response_chars": self.response_chars,
            "invocation_attempts": self.invocation_attempts,
            "endpoint_probe_requests": self.endpoint_probe_requests,
            "delivery_status": self.delivery_status,
            "elapsed_ms": self.elapsed_ms,
            "retries": self.retries,
            "tools_invoked": self.tools_invoked,
            "risk_class": self.risk_class,
            "tool_surfaces": list(self.tool_surfaces),
            "execution_scope": self.execution_scope,
            "execution_transport": self.execution_transport,
            "process_mutations": self.process_mutations,
            "lifecycle_operations": self.lifecycle_operations,
            "remote_egress": self.remote_egress,
            "binding_continuity": self.binding_continuity,
            "endpoint_process_identity_verified": self.endpoint_process_identity_verified,
            "semantic_outcome_verified": self.semantic_outcome_verified,
            "authorizes_future_execution": self.authorizes_future_execution,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def _strict_dataclass_payload(
    raw: Mapping[str, object],
    contract_type: type[BoundCellRunPolicy] | type[BoundCellRunReceipt],
    *,
    tuple_fields: tuple[str, ...],
    label: str,
) -> BoundCellRunPolicy | BoundCellRunReceipt:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise BoundCellRunContractError(
            "contract_invalid", f"{label} must be an object."
        )
    expected = {item.name for item in fields(contract_type)}
    supplied = set(raw)
    if supplied != expected:
        raise BoundCellRunContractError(
            "contract_invalid",
            f"{label} fields do not match the supported contract.",
        )
    values = dict(raw)
    for name in tuple_fields:
        value = values[name]
        if not isinstance(value, (list, tuple)):
            raise BoundCellRunContractError(
                "contract_invalid", f"{label} {name} must be an array."
            )
        values[name] = tuple(value)
    try:
        return contract_type(**values)
    except BoundCellRunContractError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise BoundCellRunContractError(
            "contract_invalid", f"{label} is invalid."
        ) from exc


def bound_cell_run_policy_from_payload(
    raw: Mapping[str, object],
) -> BoundCellRunPolicy:
    result = _strict_dataclass_payload(
        raw,
        BoundCellRunPolicy,
        tuple_fields=("risk_classes",),
        label="Bound cell run policy",
    )
    if not isinstance(result, BoundCellRunPolicy):  # pragma: no cover - type narrowing
        raise AssertionError("Unexpected contract type.")
    return result


def bound_cell_run_receipt_from_payload(
    raw: Mapping[str, object],
) -> BoundCellRunReceipt:
    result = _strict_dataclass_payload(
        raw,
        BoundCellRunReceipt,
        tuple_fields=("reason_codes", "tool_surfaces"),
        label="Bound cell run receipt",
    )
    if not isinstance(result, BoundCellRunReceipt):  # pragma: no cover - type narrowing
        raise AssertionError("Unexpected contract type.")
    return result


__all__ = [
    "BoundCellRunContractError",
    "BoundCellRunPolicy",
    "BoundCellRunReceipt",
    "DELIVERY_STATUSES",
    "RUN_REASON_CODES",
    "RUN_STATUSES",
    "bound_cell_run_policy_from_payload",
    "bound_cell_run_receipt_from_payload",
]
