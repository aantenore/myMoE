"""Strict metadata contracts for one process-bound local runtime lease.

These values describe lifecycle evidence only.  They do not start, stop, adopt,
contact, or authorize a runtime and the raw lease capability is never part of a
serializable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, TypeVar

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_identifier_tuple,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


RUNTIME_SUPERVISOR_POLICY_CONTRACT = "RuntimeSupervisorLeasePolicy"
RUNTIME_SUPERVISOR_BINDING_CONTRACT = "RuntimeSupervisorLeaseBinding"
RUNTIME_SUPERVISOR_RECEIPT_CONTRACT = "RuntimeSupervisorLeaseReceipt"

RUNTIME_SUPERVISOR_STATES = frozenset(
    {
        "prepared",
        "starting",
        "ready",
        "stopping",
        "stopped",
        "revoked",
        "unknown_blocking",
    }
)
ACTIVE_RUNTIME_SUPERVISOR_STATES = RUNTIME_SUPERVISOR_STATES - {"stopped"}
RUNTIME_SUPERVISOR_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "prepared": frozenset(
        {"starting", "stopping", "stopped", "revoked", "unknown_blocking"}
    ),
    "starting": frozenset(
        {"ready", "stopping", "revoked", "unknown_blocking"}
    ),
    "ready": frozenset({"stopping", "revoked", "unknown_blocking"}),
    "revoked": frozenset({"stopping", "unknown_blocking"}),
    "stopping": frozenset({"stopped", "unknown_blocking"}),
    # Unknown ownership is sticky.  A future, separately confirmed reconcile
    # operation may remove a proven-dead row; ordinary transitions cannot.
    "unknown_blocking": frozenset({"unknown_blocking"}),
    "stopped": frozenset(),
}
RUNTIME_SUPERVISOR_REASON_CODES = frozenset(
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
MAX_SQLITE_INTEGER = (1 << 63) - 1


class RuntimeSupervisorContractError(VerifiedRoutingError):
    """Raised when process-bound runtime metadata is malformed or unbound."""


def _call(function, *args):
    try:
        return function(*args)
    except RuntimeSupervisorContractError:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise RuntimeSupervisorContractError(str(exc)) from exc


def _schema(value: object, label: str) -> str:
    if value != CONTRACT_VERSION:
        raise RuntimeSupervisorContractError(
            f"Unsupported {label} schema version."
        )
    return CONTRACT_VERSION


def _safe(value: object, label: str) -> str:
    return _call(require_safe_id, value, label)


def _sha(value: object, label: str) -> str:
    return _call(require_sha256, value, label)


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_SQLITE_INTEGER:
        raise RuntimeSupervisorContractError(
            f"{label} must be an integer between {minimum} and "
            f"{MAX_SQLITE_INTEGER}."
        )
    return value


def _optional_integer(
    value: object, label: str, *, minimum: int = 0
) -> int | None:
    return None if value is None else _integer(value, label, minimum=minimum)


def _fixed_boolean(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        rendered = "true" if expected else "false"
        raise RuntimeSupervisorContractError(f"{label} must remain {rendered}.")
    return value


def _digest(value: object, payload: Mapping[str, object], label: str) -> str:
    expected = _call(sha256_json, payload)
    if value not in (None, "") and _sha(value, label) != expected:
        raise RuntimeSupervisorContractError(f"{label} does not match its content.")
    return expected


@dataclass(frozen=True)
class RuntimeSupervisorLeasePolicy:
    """Durability limits and fixed authority boundary for the lease ledger."""

    busy_timeout_ms: int = 5_000
    max_active_leases: int = 64
    metadata_only: bool = True
    process_mutations: bool = False
    raw_tokens_persisted: bool = False
    adoption_allowed: bool = False
    automatic_restart: bool = False
    digest: str = ""
    contract: str = RUNTIME_SUPERVISOR_POLICY_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "runtime supervisor policy")
        if self.contract != RUNTIME_SUPERVISOR_POLICY_CONTRACT:
            raise RuntimeSupervisorContractError(
                "Unsupported runtime supervisor policy contract."
            )
        object.__setattr__(
            self,
            "busy_timeout_ms",
            _integer(self.busy_timeout_ms, "busy_timeout_ms", minimum=1),
        )
        object.__setattr__(
            self,
            "max_active_leases",
            _integer(self.max_active_leases, "max_active_leases", minimum=1),
        )
        _fixed_boolean(self.metadata_only, True, "metadata_only")
        _fixed_boolean(self.process_mutations, False, "process_mutations")
        _fixed_boolean(self.raw_tokens_persisted, False, "raw_tokens_persisted")
        _fixed_boolean(self.adoption_allowed, False, "adoption_allowed")
        _fixed_boolean(self.automatic_restart, False, "automatic_restart")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "policy digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "busy_timeout_ms": self.busy_timeout_ms,
            "max_active_leases": self.max_active_leases,
            "metadata_only": self.metadata_only,
            "process_mutations": self.process_mutations,
            "raw_tokens_persisted": self.raw_tokens_persisted,
            "adoption_allowed": self.adoption_allowed,
            "automatic_restart": self.automatic_restart,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class RuntimeSupervisorLeaseBinding:
    """Digest-only link to one exact, separately verified runtime launch."""

    binding_request_sha256: str
    binding_manifest_sha256: str
    launch_plan_sha256: str
    config_source_sha256: str
    runtime_config_sha256: str
    runtime_identity_sha256: str
    model_identity_sha256: str
    endpoint_authority_sha256: str
    adapter_id: str
    runtime_backend: str
    execution_scope: str = "device_only"
    transport: str = "direct_local"
    digest: str = ""
    contract: str = RUNTIME_SUPERVISOR_BINDING_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "runtime supervisor binding")
        if self.contract != RUNTIME_SUPERVISOR_BINDING_CONTRACT:
            raise RuntimeSupervisorContractError(
                "Unsupported runtime supervisor binding contract."
            )
        for name in (
            "binding_request_sha256",
            "binding_manifest_sha256",
            "launch_plan_sha256",
            "config_source_sha256",
            "runtime_config_sha256",
            "runtime_identity_sha256",
            "model_identity_sha256",
            "endpoint_authority_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("adapter_id", "runtime_backend"):
            object.__setattr__(self, name, _safe(getattr(self, name), name))
        if self.execution_scope != "device_only" or self.transport != "direct_local":
            raise RuntimeSupervisorContractError(
                "Runtime supervisor binding must remain device-only and direct-local."
            )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "binding digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "binding_request_sha256": self.binding_request_sha256,
            "binding_manifest_sha256": self.binding_manifest_sha256,
            "launch_plan_sha256": self.launch_plan_sha256,
            "config_source_sha256": self.config_source_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "endpoint_authority_sha256": self.endpoint_authority_sha256,
            "adapter_id": self.adapter_id,
            "runtime_backend": self.runtime_backend,
            "execution_scope": self.execution_scope,
            "transport": self.transport,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class RuntimeSupervisorLeaseReceipt:
    """One content-addressed lease state; never an inference authorization."""

    policy_sha256: str
    binding_sha256: str
    endpoint_authority_sha256: str
    coordination_domain_sha256: str
    lease_id: str
    lease_token_sha256: str
    owner_pid: int
    state: str
    reason_codes: tuple[str, ...]
    transition_index: int
    previous_receipt_sha256: str | None
    runtime_pid: int | None
    runtime_create_time_ns: int | None
    runtime_executable_sha256: str | None
    process_tree_sha256: str | None
    endpoint_evidence_sha256: str | None
    updated_at: str
    metadata_only: bool = True
    process_mutations: bool = False
    authorizes_inference: bool = False
    raw_token_serialized: bool = False
    digest: str = ""
    contract: str = RUNTIME_SUPERVISOR_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "runtime supervisor receipt")
        if self.contract != RUNTIME_SUPERVISOR_RECEIPT_CONTRACT:
            raise RuntimeSupervisorContractError(
                "Unsupported runtime supervisor receipt contract."
            )
        for name in (
            "policy_sha256",
            "binding_sha256",
            "endpoint_authority_sha256",
            "coordination_domain_sha256",
            "lease_token_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "lease_id", _safe(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "owner_pid", _integer(self.owner_pid, "owner_pid", minimum=1)
        )
        if self.state not in RUNTIME_SUPERVISOR_STATES:
            raise RuntimeSupervisorContractError(
                "Runtime supervisor state is unsupported."
            )
        reasons = tuple(
            sorted(_call(require_identifier_tuple, self.reason_codes, "reason_codes"))
        )
        if not set(reasons).issubset(RUNTIME_SUPERVISOR_REASON_CODES):
            raise RuntimeSupervisorContractError(
                "Runtime supervisor reason_codes are unsupported."
            )
        if self.state in {"revoked", "unknown_blocking"} and not reasons:
            raise RuntimeSupervisorContractError(
                "Revoked and unknown-blocking receipts require a reason code."
            )
        if self.state == "ready" and reasons:
            raise RuntimeSupervisorContractError(
                "Ready runtime supervisor receipts cannot carry failure reasons."
            )
        object.__setattr__(self, "reason_codes", reasons)
        transition_index = _integer(self.transition_index, "transition_index")
        previous = _optional_sha(
            self.previous_receipt_sha256, "previous_receipt_sha256"
        )
        if (transition_index == 0) != (previous is None):
            raise RuntimeSupervisorContractError(
                "Only the initial receipt may omit previous_receipt_sha256."
            )
        if transition_index == 0 and self.state != "prepared":
            raise RuntimeSupervisorContractError(
                "The initial runtime supervisor receipt must be prepared."
            )
        object.__setattr__(self, "transition_index", transition_index)
        object.__setattr__(self, "previous_receipt_sha256", previous)
        runtime_pid = _optional_integer(self.runtime_pid, "runtime_pid", minimum=1)
        runtime_create_time_ns = _optional_integer(
            self.runtime_create_time_ns, "runtime_create_time_ns", minimum=1
        )
        runtime_executable = _optional_sha(
            self.runtime_executable_sha256, "runtime_executable_sha256"
        )
        process_tree = _optional_sha(self.process_tree_sha256, "process_tree_sha256")
        endpoint_evidence = _optional_sha(
            self.endpoint_evidence_sha256, "endpoint_evidence_sha256"
        )
        identity_shape = (
            runtime_pid is not None,
            runtime_create_time_ns is not None,
            runtime_executable is not None,
        )
        if len(set(identity_shape)) != 1:
            raise RuntimeSupervisorContractError(
                "Runtime PID, create time, and executable identity must be supplied together."
            )
        if self.state == "prepared" and any(
            value is not None
            for value in (
                runtime_pid,
                runtime_create_time_ns,
                runtime_executable,
                process_tree,
                endpoint_evidence,
            )
        ):
            raise RuntimeSupervisorContractError(
                "Prepared leases cannot claim observed runtime identity."
            )
        if self.state == "starting" and runtime_pid is None:
            raise RuntimeSupervisorContractError(
                "Starting leases require an observed root-process identity."
            )
        if process_tree is not None and runtime_pid is None:
            raise RuntimeSupervisorContractError(
                "Process-tree evidence requires an observed root-process identity."
            )
        if endpoint_evidence is not None and process_tree is None:
            raise RuntimeSupervisorContractError(
                "Endpoint evidence requires process-tree evidence."
            )
        if self.state == "ready" and any(
            value is None
            for value in (
                runtime_pid,
                runtime_create_time_ns,
                runtime_executable,
                process_tree,
                endpoint_evidence,
            )
        ):
            raise RuntimeSupervisorContractError(
                "Ready leases require complete process-tree and endpoint evidence."
            )
        object.__setattr__(self, "runtime_pid", runtime_pid)
        object.__setattr__(
            self, "runtime_create_time_ns", runtime_create_time_ns
        )
        object.__setattr__(
            self, "runtime_executable_sha256", runtime_executable
        )
        object.__setattr__(self, "process_tree_sha256", process_tree)
        object.__setattr__(self, "endpoint_evidence_sha256", endpoint_evidence)
        object.__setattr__(
            self, "updated_at", _call(require_utc_timestamp, self.updated_at, "updated_at")
        )
        _fixed_boolean(self.metadata_only, True, "metadata_only")
        _fixed_boolean(self.process_mutations, False, "process_mutations")
        _fixed_boolean(self.authorizes_inference, False, "authorizes_inference")
        _fixed_boolean(self.raw_token_serialized, False, "raw_token_serialized")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "policy_sha256": self.policy_sha256,
            "binding_sha256": self.binding_sha256,
            "endpoint_authority_sha256": self.endpoint_authority_sha256,
            "coordination_domain_sha256": self.coordination_domain_sha256,
            "lease_id": self.lease_id,
            "lease_token_sha256": self.lease_token_sha256,
            "owner_pid": self.owner_pid,
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "transition_index": self.transition_index,
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "runtime_pid": self.runtime_pid,
            "runtime_create_time_ns": self.runtime_create_time_ns,
            "runtime_executable_sha256": self.runtime_executable_sha256,
            "process_tree_sha256": self.process_tree_sha256,
            "endpoint_evidence_sha256": self.endpoint_evidence_sha256,
            "updated_at": self.updated_at,
            "metadata_only": self.metadata_only,
            "process_mutations": self.process_mutations,
            "authorizes_inference": self.authorizes_inference,
            "raw_token_serialized": self.raw_token_serialized,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


_ContractT = TypeVar(
    "_ContractT",
    RuntimeSupervisorLeasePolicy,
    RuntimeSupervisorLeaseBinding,
    RuntimeSupervisorLeaseReceipt,
)


def _strict_payload(
    raw: object,
    contract_type: type[_ContractT],
    *,
    tuple_fields: tuple[str, ...] = (),
) -> _ContractT:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise RuntimeSupervisorContractError("Contract payload must be an object.")
    expected = {item.name for item in fields(contract_type)}
    if set(raw) != expected:
        raise RuntimeSupervisorContractError(
            "Contract fields do not match the supported schema."
        )
    values = dict(raw)
    for name in tuple_fields:
        value = values[name]
        if not isinstance(value, (list, tuple)):
            raise RuntimeSupervisorContractError(f"{name} must be an array.")
        values[name] = tuple(value)
    try:
        return contract_type(**values)
    except RuntimeSupervisorContractError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeSupervisorContractError("Contract payload is invalid.") from exc


def runtime_supervisor_policy_from_payload(
    raw: object,
) -> RuntimeSupervisorLeasePolicy:
    return _strict_payload(raw, RuntimeSupervisorLeasePolicy)


def runtime_supervisor_binding_from_payload(
    raw: object,
) -> RuntimeSupervisorLeaseBinding:
    return _strict_payload(raw, RuntimeSupervisorLeaseBinding)


def runtime_supervisor_receipt_from_payload(
    raw: object,
) -> RuntimeSupervisorLeaseReceipt:
    return _strict_payload(
        raw, RuntimeSupervisorLeaseReceipt, tuple_fields=("reason_codes",)
    )


def runtime_supervisor_transition_allowed(current: str, target: str) -> bool:
    if (
        current not in RUNTIME_SUPERVISOR_TRANSITIONS
        or target not in RUNTIME_SUPERVISOR_STATES
    ):
        return False
    return target in RUNTIME_SUPERVISOR_TRANSITIONS[current]


__all__ = [
    "ACTIVE_RUNTIME_SUPERVISOR_STATES",
    "MAX_SQLITE_INTEGER",
    "RUNTIME_SUPERVISOR_BINDING_CONTRACT",
    "RUNTIME_SUPERVISOR_POLICY_CONTRACT",
    "RUNTIME_SUPERVISOR_REASON_CODES",
    "RUNTIME_SUPERVISOR_RECEIPT_CONTRACT",
    "RUNTIME_SUPERVISOR_STATES",
    "RUNTIME_SUPERVISOR_TRANSITIONS",
    "RuntimeSupervisorContractError",
    "RuntimeSupervisorLeaseBinding",
    "RuntimeSupervisorLeasePolicy",
    "RuntimeSupervisorLeaseReceipt",
    "runtime_supervisor_binding_from_payload",
    "runtime_supervisor_policy_from_payload",
    "runtime_supervisor_receipt_from_payload",
    "runtime_supervisor_transition_allowed",
]
