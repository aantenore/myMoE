from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_identifier_tuple,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


LEASE_POLICY_CONTRACT = "CooperativeResourceLeasePolicy"
LEASE_CLAIM_CONTRACT = "CooperativeResourceLeaseClaim"
LEASE_ADMISSION_RECEIPT_CONTRACT = "CooperativeResourceLeaseAdmissionReceipt"
LEASE_TRANSITION_RECEIPT_CONTRACT = "CooperativeResourceLeaseTransitionReceipt"
LEASE_RELEASE_RECEIPT_CONTRACT = "CooperativeResourceLeaseReleaseReceipt"
CLAIM_BASIS = "conservative_peak"
POOL_KINDS = frozenset({"system", "unified", "discrete"})
ADMISSION_STATUSES = frozenset({"acquired", "denied", "unknown_blocking"})
RELEASE_STATUSES = frozenset(
    {
        "released",
        "released_cleanup_deferred",
        "already_absent",
        "denied",
        "unknown_blocking",
    }
)
DELIVERY_STATUSES = frozenset(
    {"not_attempted", "attempted_unknown", "response_received"}
)
MAX_SQLITE_INTEGER = (1 << 63) - 1


class CooperativeResourceLeaseContractError(VerifiedRoutingError):
    """Raised when cooperative resource-lease metadata is not trustworthy."""


def _call(function, *args):
    try:
        return function(*args)
    except CooperativeResourceLeaseContractError:
        raise
    except (VerifiedRoutingError, OverflowError, TypeError, ValueError) as exc:
        raise CooperativeResourceLeaseContractError(str(exc)) from exc


def _sha(value: object, label: str) -> str:
    return _call(require_sha256, value, label)


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _identifier(value: object, label: str) -> str:
    return _call(require_safe_id, value, label)


def _timestamp(value: object, label: str) -> str:
    return _call(require_utc_timestamp, value, label)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > MAX_SQLITE_INTEGER:
        raise CooperativeResourceLeaseContractError(
            f"{label} must be an integer between {minimum} and {MAX_SQLITE_INTEGER}."
        )
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _boolean(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        rendered = "true" if expected else "false"
        raise CooperativeResourceLeaseContractError(f"{label} must remain {rendered}.")
    return value


def _reasons(value: object, label: str) -> tuple[str, ...]:
    items = tuple(sorted(_call(require_identifier_tuple, value, label)))
    return items


def _digest(provided: object, payload: Mapping[str, object], label: str) -> str:
    expected = sha256_json(payload)
    if provided not in (None, "") and _sha(provided, label) != expected:
        raise CooperativeResourceLeaseContractError(f"{label} does not match.")
    return expected


def _schema(value: object, label: str) -> str:
    if value != CONTRACT_VERSION:
        raise CooperativeResourceLeaseContractError(
            f"Unsupported {label} schema version."
        )
    return CONTRACT_VERSION


@dataclass(frozen=True)
class CooperativeResourceLeasePolicy:
    """Operational bounds for one local cooperative coordination domain."""

    busy_timeout_ms: int = 5_000
    max_active_leases: int = 256
    claim_basis: str = CLAIM_BASIS
    cooperative_only: bool = True
    os_memory_reserved: bool = False
    runtime_managed: bool = False
    digest: str = ""
    contract: str = LEASE_POLICY_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "lease policy")
        if self.contract != LEASE_POLICY_CONTRACT:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease policy contract."
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
        if self.claim_basis != CLAIM_BASIS:
            raise CooperativeResourceLeaseContractError(
                "Lease policy claim_basis must be conservative_peak."
            )
        _boolean(self.cooperative_only, True, "cooperative_only")
        _boolean(self.os_memory_reserved, False, "os_memory_reserved")
        _boolean(self.runtime_managed, False, "runtime_managed")
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
            "claim_basis": self.claim_basis,
            "cooperative_only": self.cooperative_only,
            "os_memory_reserved": self.os_memory_reserved,
            "runtime_managed": self.runtime_managed,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CooperativeResourceClaim:
    """Conservative concurrent-work claim; it is not incremental usage."""

    preview_sha256: str
    candidate_sha256: str
    passport_sha256: str
    resource_snapshot_sha256: str
    resource_class_sha256: str
    catalog_sha256: str
    profile_sha256: str
    pool: str
    system_claim_bytes: int
    accelerator_claim_bytes: int
    accelerator_identity_sha256: str | None
    safety_reserve_bytes: int
    claim_basis: str = CLAIM_BASIS
    digest: str = ""
    contract: str = LEASE_CLAIM_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "lease claim")
        if self.contract != LEASE_CLAIM_CONTRACT:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease claim contract."
            )
        for name in (
            "preview_sha256",
            "candidate_sha256",
            "passport_sha256",
            "resource_snapshot_sha256",
            "resource_class_sha256",
            "catalog_sha256",
            "profile_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.pool not in POOL_KINDS:
            raise CooperativeResourceLeaseContractError(
                "pool must be system, unified, or discrete."
            )
        system_bytes = _integer(
            self.system_claim_bytes, "system_claim_bytes", minimum=1
        )
        accelerator_bytes = _integer(
            self.accelerator_claim_bytes, "accelerator_claim_bytes"
        )
        accelerator_identity = _optional_sha(
            self.accelerator_identity_sha256, "accelerator_identity_sha256"
        )
        reserve = _integer(self.safety_reserve_bytes, "safety_reserve_bytes")
        if self.pool in {"system", "unified"} and (
            accelerator_bytes != 0 or accelerator_identity is not None
        ):
            raise CooperativeResourceLeaseContractError(
                "System and unified claims cannot claim a discrete accelerator pool."
            )
        if self.pool == "discrete" and (
            accelerator_bytes == 0 or accelerator_identity is None
        ):
            raise CooperativeResourceLeaseContractError(
                "Discrete claims require accelerator bytes and identity."
            )
        if self.claim_basis != CLAIM_BASIS:
            raise CooperativeResourceLeaseContractError(
                "Lease claim_basis must be conservative_peak."
            )
        object.__setattr__(self, "system_claim_bytes", system_bytes)
        object.__setattr__(self, "accelerator_claim_bytes", accelerator_bytes)
        object.__setattr__(self, "accelerator_identity_sha256", accelerator_identity)
        object.__setattr__(self, "safety_reserve_bytes", reserve)
        object.__setattr__(
            self, "digest", _digest(self.digest, self.content_payload(), "claim digest")
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "preview_sha256": self.preview_sha256,
            "candidate_sha256": self.candidate_sha256,
            "passport_sha256": self.passport_sha256,
            "resource_snapshot_sha256": self.resource_snapshot_sha256,
            "resource_class_sha256": self.resource_class_sha256,
            "catalog_sha256": self.catalog_sha256,
            "profile_sha256": self.profile_sha256,
            "pool": self.pool,
            "system_claim_bytes": self.system_claim_bytes,
            "accelerator_claim_bytes": self.accelerator_claim_bytes,
            "accelerator_identity_sha256": self.accelerator_identity_sha256,
            "safety_reserve_bytes": self.safety_reserve_bytes,
            "claim_basis": self.claim_basis,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CooperativeResourceLeaseAdmissionReceipt:
    policy_sha256: str
    claim_sha256: str
    resource_snapshot_sha256: str
    coordination_domain_sha256: str
    status: str
    reason_codes: tuple[str, ...]
    evaluated_at: str
    lease_id: str | None
    lease_token_sha256: str | None
    active_leases_before: int
    active_leases_after: int
    reaped_leases: int
    system_available_bytes: int | None
    accelerator_available_bytes: int | None
    active_system_claim_bytes: int
    active_accelerator_claim_bytes: int
    requested_system_claim_bytes: int
    requested_accelerator_claim_bytes: int
    safety_reserve_bytes: int
    applied_system_reserve_bytes: int
    applied_accelerator_reserve_bytes: int
    claim_basis: str = CLAIM_BASIS
    cooperative_only: bool = True
    os_memory_reserved: bool = False
    runtime_managed: bool = False
    digest: str = ""
    contract: str = LEASE_ADMISSION_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "lease admission receipt")
        if self.contract != LEASE_ADMISSION_RECEIPT_CONTRACT:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease admission receipt contract."
            )
        for name in (
            "policy_sha256",
            "claim_sha256",
            "resource_snapshot_sha256",
            "coordination_domain_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.status not in ADMISSION_STATUSES:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease admission status."
            )
        reasons = _reasons(self.reason_codes, "reason_codes")
        lease_id = (
            None if self.lease_id is None else _identifier(self.lease_id, "lease_id")
        )
        token_sha = _optional_sha(self.lease_token_sha256, "lease_token_sha256")
        if self.status == "acquired":
            if reasons or lease_id is None or token_sha is None:
                raise CooperativeResourceLeaseContractError(
                    "An acquired lease requires identifiers and no blockers."
                )
        elif not reasons or lease_id is not None or token_sha is not None:
            raise CooperativeResourceLeaseContractError(
                "A non-acquired lease requires blockers and no lease identifiers."
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "lease_id", lease_id)
        object.__setattr__(self, "lease_token_sha256", token_sha)
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, "evaluated_at")
        )
        for name in (
            "active_leases_before",
            "active_leases_after",
            "reaped_leases",
            "active_system_claim_bytes",
            "active_accelerator_claim_bytes",
            "requested_system_claim_bytes",
            "requested_accelerator_claim_bytes",
            "safety_reserve_bytes",
            "applied_system_reserve_bytes",
            "applied_accelerator_reserve_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(
            self,
            "system_available_bytes",
            _optional_integer(self.system_available_bytes, "system_available_bytes"),
        )
        object.__setattr__(
            self,
            "accelerator_available_bytes",
            _optional_integer(
                self.accelerator_available_bytes, "accelerator_available_bytes"
            ),
        )
        if self.status == "acquired" and self.active_leases_after != (
            self.active_leases_before + 1
        ):
            raise CooperativeResourceLeaseContractError(
                "An acquired receipt must add exactly one active lease."
            )
        if self.status != "acquired" and (
            self.active_leases_after != self.active_leases_before
        ):
            raise CooperativeResourceLeaseContractError(
                "A blocked receipt cannot add an active lease."
            )
        if self.claim_basis != CLAIM_BASIS:
            raise CooperativeResourceLeaseContractError(
                "Admission claim_basis must be conservative_peak."
            )
        _boolean(self.cooperative_only, True, "cooperative_only")
        _boolean(self.os_memory_reserved, False, "os_memory_reserved")
        _boolean(self.runtime_managed, False, "runtime_managed")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "admission receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        names = (
            "schema_version",
            "contract",
            "policy_sha256",
            "claim_sha256",
            "resource_snapshot_sha256",
            "coordination_domain_sha256",
            "status",
            "evaluated_at",
            "lease_id",
            "lease_token_sha256",
            "active_leases_before",
            "active_leases_after",
            "reaped_leases",
            "system_available_bytes",
            "accelerator_available_bytes",
            "active_system_claim_bytes",
            "active_accelerator_claim_bytes",
            "requested_system_claim_bytes",
            "requested_accelerator_claim_bytes",
            "safety_reserve_bytes",
            "applied_system_reserve_bytes",
            "applied_accelerator_reserve_bytes",
            "claim_basis",
            "cooperative_only",
            "os_memory_reserved",
            "runtime_managed",
        )
        return {name: getattr(self, name) for name in names} | {
            "reason_codes": list(self.reason_codes)
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CooperativeResourceLeaseTransitionReceipt:
    policy_sha256: str
    admission_receipt_sha256: str
    claim_sha256: str
    coordination_domain_sha256: str
    lease_id: str
    state: str
    transition_applied: bool
    reason_codes: tuple[str, ...]
    transitioned_at: str
    claim_basis: str = CLAIM_BASIS
    cooperative_only: bool = True
    os_memory_reserved: bool = False
    runtime_managed: bool = False
    digest: str = ""
    contract: str = LEASE_TRANSITION_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "lease transition receipt")
        if self.contract != LEASE_TRANSITION_RECEIPT_CONTRACT:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease transition receipt contract."
            )
        for name in (
            "policy_sha256",
            "admission_receipt_sha256",
            "claim_sha256",
            "coordination_domain_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id"))
        if self.state != "delivery_armed":
            raise CooperativeResourceLeaseContractError(
                "Transition state must be delivery_armed."
            )
        if type(self.transition_applied) is not bool:
            raise CooperativeResourceLeaseContractError(
                "transition_applied must be a boolean."
            )
        reasons = _reasons(self.reason_codes, "reason_codes")
        if self.transition_applied and reasons:
            raise CooperativeResourceLeaseContractError(
                "An applied transition cannot contain blockers."
            )
        if not self.transition_applied and not reasons:
            raise CooperativeResourceLeaseContractError(
                "A non-applied transition requires a reason."
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "transitioned_at",
            _timestamp(self.transitioned_at, "transitioned_at"),
        )
        if self.claim_basis != CLAIM_BASIS:
            raise CooperativeResourceLeaseContractError(
                "Transition claim_basis must be conservative_peak."
            )
        _boolean(self.cooperative_only, True, "cooperative_only")
        _boolean(self.os_memory_reserved, False, "os_memory_reserved")
        _boolean(self.runtime_managed, False, "runtime_managed")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "transition receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        names = (
            "schema_version",
            "contract",
            "policy_sha256",
            "admission_receipt_sha256",
            "claim_sha256",
            "coordination_domain_sha256",
            "lease_id",
            "state",
            "transition_applied",
            "transitioned_at",
            "claim_basis",
            "cooperative_only",
            "os_memory_reserved",
            "runtime_managed",
        )
        return {name: getattr(self, name) for name in names} | {
            "reason_codes": list(self.reason_codes)
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CooperativeResourceLeaseReleaseReceipt:
    policy_sha256: str
    admission_receipt_sha256: str
    claim_sha256: str
    coordination_domain_sha256: str
    lease_id: str
    status: str
    reason_codes: tuple[str, ...]
    delivery_status: str
    released_at: str
    active_leases_after: int
    claim_basis: str = CLAIM_BASIS
    cooperative_only: bool = True
    os_memory_reserved: bool = False
    runtime_managed: bool = False
    digest: str = ""
    contract: str = LEASE_RELEASE_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "lease release receipt")
        if self.contract != LEASE_RELEASE_RECEIPT_CONTRACT:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease release receipt contract."
            )
        for name in (
            "policy_sha256",
            "admission_receipt_sha256",
            "claim_sha256",
            "coordination_domain_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "lease_id", _identifier(self.lease_id, "lease_id"))
        if self.status not in RELEASE_STATUSES:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease release status."
            )
        if self.delivery_status not in DELIVERY_STATUSES:
            raise CooperativeResourceLeaseContractError(
                "Unsupported lease delivery status."
            )
        reasons = _reasons(self.reason_codes, "reason_codes")
        if self.status == "released" and reasons:
            raise CooperativeResourceLeaseContractError(
                "A released lease cannot contain blockers."
            )
        if self.status != "released" and not reasons:
            raise CooperativeResourceLeaseContractError(
                "A non-released lease requires a reason."
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self, "released_at", _timestamp(self.released_at, "released_at")
        )
        object.__setattr__(
            self,
            "active_leases_after",
            _integer(self.active_leases_after, "active_leases_after"),
        )
        if self.claim_basis != CLAIM_BASIS:
            raise CooperativeResourceLeaseContractError(
                "Release claim_basis must be conservative_peak."
            )
        _boolean(self.cooperative_only, True, "cooperative_only")
        _boolean(self.os_memory_reserved, False, "os_memory_reserved")
        _boolean(self.runtime_managed, False, "runtime_managed")
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "release receipt digest"),
        )

    def content_payload(self) -> dict[str, object]:
        names = (
            "schema_version",
            "contract",
            "policy_sha256",
            "admission_receipt_sha256",
            "claim_sha256",
            "coordination_domain_sha256",
            "lease_id",
            "status",
            "delivery_status",
            "released_at",
            "active_leases_after",
            "claim_basis",
            "cooperative_only",
            "os_memory_reserved",
            "runtime_managed",
        )
        return {name: getattr(self, name) for name in names} | {
            "reason_codes": list(self.reason_codes)
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def _strict_payload(
    raw: object,
    contract_type: type[
        CooperativeResourceLeasePolicy
        | CooperativeResourceClaim
        | CooperativeResourceLeaseAdmissionReceipt
        | CooperativeResourceLeaseTransitionReceipt
        | CooperativeResourceLeaseReleaseReceipt
    ],
    *,
    tuple_fields: tuple[str, ...] = (),
):
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise CooperativeResourceLeaseContractError(
            "Contract payload must be an object."
        )
    expected = {item.name for item in fields(contract_type)}
    if set(raw) != expected:
        raise CooperativeResourceLeaseContractError(
            "Contract fields do not match the supported schema."
        )
    values = dict(raw)
    for name in tuple_fields:
        value = values[name]
        if not isinstance(value, (list, tuple)):
            raise CooperativeResourceLeaseContractError(f"{name} must be an array.")
        values[name] = tuple(value)
    try:
        return contract_type(**values)
    except CooperativeResourceLeaseContractError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise CooperativeResourceLeaseContractError(
            "Contract payload is invalid."
        ) from exc


def cooperative_resource_lease_policy_from_payload(
    raw: object,
) -> CooperativeResourceLeasePolicy:
    return _strict_payload(raw, CooperativeResourceLeasePolicy)


def cooperative_resource_claim_from_payload(raw: object) -> CooperativeResourceClaim:
    return _strict_payload(raw, CooperativeResourceClaim)


def cooperative_resource_lease_admission_receipt_from_payload(
    raw: object,
) -> CooperativeResourceLeaseAdmissionReceipt:
    return _strict_payload(
        raw, CooperativeResourceLeaseAdmissionReceipt, tuple_fields=("reason_codes",)
    )


def cooperative_resource_lease_release_receipt_from_payload(
    raw: object,
) -> CooperativeResourceLeaseReleaseReceipt:
    return _strict_payload(
        raw, CooperativeResourceLeaseReleaseReceipt, tuple_fields=("reason_codes",)
    )


def cooperative_resource_lease_transition_receipt_from_payload(
    raw: object,
) -> CooperativeResourceLeaseTransitionReceipt:
    return _strict_payload(
        raw,
        CooperativeResourceLeaseTransitionReceipt,
        tuple_fields=("reason_codes",),
    )


__all__ = [
    "ADMISSION_STATUSES",
    "CLAIM_BASIS",
    "DELIVERY_STATUSES",
    "LEASE_ADMISSION_RECEIPT_CONTRACT",
    "LEASE_CLAIM_CONTRACT",
    "LEASE_POLICY_CONTRACT",
    "LEASE_RELEASE_RECEIPT_CONTRACT",
    "LEASE_TRANSITION_RECEIPT_CONTRACT",
    "MAX_SQLITE_INTEGER",
    "POOL_KINDS",
    "RELEASE_STATUSES",
    "CooperativeResourceClaim",
    "CooperativeResourceLeaseAdmissionReceipt",
    "CooperativeResourceLeaseContractError",
    "CooperativeResourceLeasePolicy",
    "CooperativeResourceLeaseReleaseReceipt",
    "CooperativeResourceLeaseTransitionReceipt",
    "cooperative_resource_claim_from_payload",
    "cooperative_resource_lease_admission_receipt_from_payload",
    "cooperative_resource_lease_policy_from_payload",
    "cooperative_resource_lease_release_receipt_from_payload",
    "cooperative_resource_lease_transition_receipt_from_payload",
]
