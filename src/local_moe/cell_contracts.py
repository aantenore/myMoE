from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


ADVISOR_CONTRACT = "AdaptiveCellAdvisor"
CATALOG_MODE = "advisory_offline"
UNKNOWN = "unknown"
MEMORY_POOLS = frozenset({"host", "unified", "accelerator"})
PLACEMENTS = frozenset({"cpu", "integrated_accelerator", "discrete_accelerator"})
MAX_CELLS = 256
MAX_PROFILES = 32
_AVAILABILITY = frozenset({"available", "missing", UNKNOWN})
_RESIDENCY = frozenset({"resident", "not_resident", UNKNOWN})


class CellContractError(VerifiedRoutingError):
    """Public error for every adaptive-cell contract boundary."""


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except CellContractError:
        raise
    except (
        VerifiedRoutingError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise CellContractError(str(exc)) from exc


def _safe(value: object, label: str) -> str:
    return _call(require_safe_id, value, label)


def _sha(value: object, label: str) -> str:
    return _call(require_sha256, value, label)


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _integer(value: object, label: str) -> int:
    return _call(require_non_negative_int, value, label)


def _positive(value: object, label: str) -> int:
    rendered = _integer(value, label)
    if rendered == 0:
        raise CellContractError(f"{label} must be positive.")
    return rendered


def _optional_positive(value: object, label: str) -> int | None:
    return None if value is None else _positive(value, label)


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    rendered = _call(
        require_finite_number,
        value,
        label,
        minimum=minimum,
        maximum=maximum,
    )
    return 0.0 if rendered == 0.0 else rendered


def _timestamp(value: object, label: str) -> str:
    return _call(require_utc_timestamp, value, label)


def _optional_timestamp(value: object, label: str) -> str | None:
    return None if value is None else _timestamp(value, label)


def _ids(value: object, label: str, *, non_empty: bool = False) -> tuple[str, ...]:
    items = tuple(sorted(_call(require_identifier_tuple, value, label)))
    if non_empty and not items:
        raise CellContractError(f"{label} must be non-empty.")
    return items


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CellContractError(f"{label} must be a boolean.")
    return value


def _enum(value: object, allowed: frozenset[str], label: str) -> str:
    rendered = str(value or "")
    if rendered not in allowed:
        raise CellContractError(f"{label} is not supported.")
    return rendered


def _relative_path(value: object, label: str) -> str:
    rendered = str(value or "")
    if not rendered or "\\" in rendered:
        raise CellContractError(f"{label} must be a relative POSIX path.")
    path = PurePosixPath(rendered)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CellContractError(f"{label} must stay below the catalog root.")
    return path.as_posix()


def _optional_path(value: object, label: str) -> str | None:
    return None if value is None else _relative_path(value, label)


def normalize_machine(value: object) -> str:
    rendered = str(value or "").strip()
    lowered = rendered.lower()
    if lowered in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if lowered in {"arm64", "aarch64"}:
        return "arm64"
    return _safe(rendered, "machine")


def _digest(value: object, content: Mapping[str, object], label: str) -> str:
    try:
        expected = sha256_json(content)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise CellContractError(f"Unable to canonicalize {label} content.") from exc
    if value not in (None, "") and _sha(value, label) != expected:
        raise CellContractError(f"{label} does not match its content.")
    return expected


@dataclass(frozen=True)
class WorkloadDemand:
    workload_id: str
    capabilities: tuple[str, ...]
    tool_surfaces: tuple[str, ...]
    risk_class: str
    context_tokens: int
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "workload demand")
        object.__setattr__(self, "workload_id", _safe(self.workload_id, "workload_id"))
        object.__setattr__(
            self,
            "capabilities",
            _ids(self.capabilities, "capabilities", non_empty=True),
        )
        object.__setattr__(
            self, "tool_surfaces", _ids(self.tool_surfaces, "tool_surfaces")
        )
        object.__setattr__(self, "risk_class", _safe(self.risk_class, "risk_class"))
        object.__setattr__(
            self, "context_tokens", _positive(self.context_tokens, "context_tokens")
        )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "demand digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "capabilities": list(self.capabilities),
            "tool_surfaces": list(self.tool_surfaces),
            "risk_class": self.risk_class,
            "context_tokens": self.context_tokens,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CellDeclaration:
    cell_id: str
    model: str
    quantization: str
    runtime: str
    harness: str
    capabilities: tuple[str, ...]
    tool_surfaces: tuple[str, ...]
    risk_classes: tuple[str, ...]
    supported_systems: tuple[str, ...]
    supported_machines: tuple[str, ...]
    max_context_tokens: int
    offline_capable: bool
    expected_model_sha256: str | None = None
    expected_runtime_sha256: str | None = None
    expected_harness_sha256: str | None = None
    expected_tool_contract_sha256: str | None = None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "cell declaration")
        for name in ("cell_id", "model", "quantization", "runtime", "harness"):
            object.__setattr__(self, name, _safe(getattr(self, name), name))
        for name, required in (
            ("capabilities", True),
            ("tool_surfaces", False),
            ("risk_classes", True),
            ("supported_systems", True),
        ):
            object.__setattr__(
                self, name, _ids(getattr(self, name), name, non_empty=required)
            )
        raw_machines = _ids(
            self.supported_machines, "supported_machines", non_empty=True
        )
        normalized_machines = tuple(normalize_machine(item) for item in raw_machines)
        if len(set(normalized_machines)) != len(normalized_machines):
            raise CellContractError(
                "supported_machines contains duplicate architecture aliases."
            )
        machines = tuple(sorted(normalized_machines))
        object.__setattr__(self, "supported_machines", machines)
        object.__setattr__(
            self,
            "max_context_tokens",
            _positive(self.max_context_tokens, "max_context_tokens"),
        )
        object.__setattr__(
            self, "offline_capable", _bool(self.offline_capable, "offline_capable")
        )
        for name in (
            "expected_model_sha256",
            "expected_runtime_sha256",
            "expected_harness_sha256",
            "expected_tool_contract_sha256",
        ):
            object.__setattr__(self, name, _optional_sha(getattr(self, name), name))
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "declaration digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cell_id": self.cell_id,
            "model": self.model,
            "quantization": self.quantization,
            "runtime": self.runtime,
            "harness": self.harness,
            "capabilities": list(self.capabilities),
            "tool_surfaces": list(self.tool_surfaces),
            "risk_classes": list(self.risk_classes),
            "supported_systems": list(self.supported_systems),
            "supported_machines": list(self.supported_machines),
            "max_context_tokens": self.max_context_tokens,
            "offline_capable": self.offline_capable,
            "expected_model_sha256": self.expected_model_sha256,
            "expected_runtime_sha256": self.expected_runtime_sha256,
            "expected_harness_sha256": self.expected_harness_sha256,
            "expected_tool_contract_sha256": self.expected_tool_contract_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CellObservation:
    cell_id: str
    declaration_sha256: str
    model_status: str = UNKNOWN
    runtime_status: str = UNKNOWN
    harness_status: str = UNKNOWN
    tool_contract_status: str = UNKNOWN
    residency_status: str = UNKNOWN
    observed_model_sha256: str | None = None
    observed_runtime_sha256: str | None = None
    observed_harness_sha256: str | None = None
    observed_tool_contract_sha256: str | None = None
    captured_at: str | None = None
    expires_at: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "cell observation")
        object.__setattr__(self, "cell_id", _safe(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "declaration_sha256",
            _sha(self.declaration_sha256, "declaration_sha256"),
        )
        pairs = (
            ("model_status", "observed_model_sha256"),
            ("runtime_status", "observed_runtime_sha256"),
            ("harness_status", "observed_harness_sha256"),
            ("tool_contract_status", "observed_tool_contract_sha256"),
        )
        for status_name, identity_name in pairs:
            status = _enum(getattr(self, status_name), _AVAILABILITY, status_name)
            identity = _optional_sha(getattr(self, identity_name), identity_name)
            if status == "available" and identity is None:
                raise CellContractError(
                    f"{status_name}=available requires {identity_name}."
                )
            if status != "available" and identity is not None:
                raise CellContractError(
                    f"{identity_name} is valid only for available evidence."
                )
            object.__setattr__(self, status_name, status)
            object.__setattr__(self, identity_name, identity)
        object.__setattr__(
            self,
            "residency_status",
            _enum(self.residency_status, _RESIDENCY, "residency_status"),
        )
        captured = _optional_timestamp(self.captured_at, "captured_at")
        expires = _optional_timestamp(self.expires_at, "expires_at")
        path = _optional_path(self.source_path, "source_path")
        source = _optional_sha(self.source_sha256, "source_sha256")
        all_unknown = (
            all(getattr(self, status) == UNKNOWN for status, _ in pairs)
            and self.residency_status == UNKNOWN
        )
        provenance = (captured, expires, path, source)
        if all_unknown and any(item is not None for item in provenance):
            raise CellContractError("Unknown observation must not claim provenance.")
        if not all_unknown and any(item is None for item in provenance):
            raise CellContractError(
                "Known observation requires timestamps and source provenance."
            )
        if captured is not None and expires is not None and expires <= captured:
            raise CellContractError("Observation expires_at must be after captured_at.")
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "source_path", path)
        object.__setattr__(self, "source_sha256", source)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "observation digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "cell_id",
                "declaration_sha256",
                "model_status",
                "runtime_status",
                "harness_status",
                "tool_contract_status",
                "residency_status",
                "observed_model_sha256",
                "observed_runtime_sha256",
                "observed_harness_sha256",
                "observed_tool_contract_sha256",
                "captured_at",
                "expires_at",
                "source_path",
                "source_sha256",
            )
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def _validate_resource_shape(
    memory_pool: str | None,
    placement: str | None,
    host: int | None,
    unified: int | None,
    accelerator: int | None,
    *,
    label: str,
) -> None:
    if memory_pool is None:
        if any(item is not None for item in (placement, host, unified, accelerator)):
            raise CellContractError(f"Unknown {label} must not claim resource values.")
        return
    if placement is None:
        raise CellContractError(f"Known {label} requires placement.")
    if memory_pool == "host":
        valid = (
            placement == "cpu"
            and host is not None
            and unified is None
            and accelerator is None
        )
    elif memory_pool == "unified":
        valid = (
            placement == "integrated_accelerator"
            and unified is not None
            and host is None
            and accelerator is None
        )
    else:
        valid = (
            placement == "discrete_accelerator"
            and host is not None
            and accelerator is not None
            and unified is None
        )
    if not valid:
        raise CellContractError(
            f"{label} resource values do not match memory_pool and placement."
        )


@dataclass(frozen=True)
class CellEstimate:
    cell_id: str
    declaration_sha256: str
    memory_pool: str | None = None
    placement: str | None = None
    peak_host_memory_bytes: int | None = None
    peak_unified_memory_bytes: int | None = None
    peak_accelerator_memory_bytes: int | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "cell estimate")
        object.__setattr__(self, "cell_id", _safe(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "declaration_sha256",
            _sha(self.declaration_sha256, "declaration_sha256"),
        )
        pool = (
            None
            if self.memory_pool is None
            else _enum(self.memory_pool, MEMORY_POOLS, "memory_pool")
        )
        placement = (
            None
            if self.placement is None
            else _enum(self.placement, PLACEMENTS, "placement")
        )
        host = _optional_positive(self.peak_host_memory_bytes, "peak_host_memory_bytes")
        unified = _optional_positive(
            self.peak_unified_memory_bytes, "peak_unified_memory_bytes"
        )
        accelerator = _optional_positive(
            self.peak_accelerator_memory_bytes, "peak_accelerator_memory_bytes"
        )
        path = _optional_path(self.source_path, "source_path")
        source = _optional_sha(self.source_sha256, "source_sha256")
        _validate_resource_shape(
            pool, placement, host, unified, accelerator, label="estimate"
        )
        if (pool is None) != (path is None or source is None) or (path is None) != (
            source is None
        ):
            raise CellContractError(
                "Known estimate requires source_path and source_sha256 together."
            )
        for name, value in (
            ("memory_pool", pool),
            ("placement", placement),
            ("peak_host_memory_bytes", host),
            ("peak_unified_memory_bytes", unified),
            ("peak_accelerator_memory_bytes", accelerator),
            ("source_path", path),
            ("source_sha256", source),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "estimate digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "cell_id",
                "declaration_sha256",
                "memory_pool",
                "placement",
                "peak_host_memory_bytes",
                "peak_unified_memory_bytes",
                "peak_accelerator_memory_bytes",
                "source_path",
                "source_sha256",
            )
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CellMeasurement:
    cell_id: str
    declaration_sha256: str
    sample_count: int = 0
    success_rate: float | None = None
    p95_latency_ms: float | None = None
    memory_pool: str | None = None
    placement: str | None = None
    peak_host_memory_bytes: int | None = None
    peak_unified_memory_bytes: int | None = None
    peak_accelerator_memory_bytes: int | None = None
    resource_class_sha256: str | None = None
    demand_sha256: str | None = None
    evaluation_contract_sha256: str | None = None
    measured_at: str | None = None
    expires_at: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "cell measurement")
        object.__setattr__(self, "cell_id", _safe(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "declaration_sha256",
            _sha(self.declaration_sha256, "declaration_sha256"),
        )
        samples = _integer(self.sample_count, "sample_count")
        success = (
            None
            if self.success_rate is None
            else _number(self.success_rate, "success_rate", minimum=0, maximum=1)
        )
        latency = (
            None
            if self.p95_latency_ms is None
            else _number(self.p95_latency_ms, "p95_latency_ms", minimum=0)
        )
        pool = (
            None
            if self.memory_pool is None
            else _enum(self.memory_pool, MEMORY_POOLS, "memory_pool")
        )
        placement = (
            None
            if self.placement is None
            else _enum(self.placement, PLACEMENTS, "placement")
        )
        host = _optional_positive(self.peak_host_memory_bytes, "peak_host_memory_bytes")
        unified = _optional_positive(
            self.peak_unified_memory_bytes, "peak_unified_memory_bytes"
        )
        accelerator = _optional_positive(
            self.peak_accelerator_memory_bytes, "peak_accelerator_memory_bytes"
        )
        resource_class = _optional_sha(
            self.resource_class_sha256, "resource_class_sha256"
        )
        demand = _optional_sha(self.demand_sha256, "demand_sha256")
        evaluation = _optional_sha(
            self.evaluation_contract_sha256, "evaluation_contract_sha256"
        )
        measured = _optional_timestamp(self.measured_at, "measured_at")
        expires = _optional_timestamp(self.expires_at, "expires_at")
        path = _optional_path(self.source_path, "source_path")
        source = _optional_sha(self.source_sha256, "source_sha256")
        _validate_resource_shape(
            pool, placement, host, unified, accelerator, label="measurement"
        )
        required = (
            success,
            latency,
            pool,
            placement,
            resource_class,
            demand,
            evaluation,
            measured,
            expires,
            path,
            source,
        )
        if samples == 0 and any(
            item is not None for item in required + (host, unified, accelerator)
        ):
            raise CellContractError(
                "Unknown measurement must not claim metrics or provenance."
            )
        if samples > 0 and any(item is None for item in required):
            raise CellContractError(
                "Known measurement requires exact demand, evaluation, resources, and provenance."
            )
        if measured is not None and expires is not None and expires <= measured:
            raise CellContractError("Measurement expires_at must be after measured_at.")
        for name, value in (
            ("sample_count", samples),
            ("success_rate", success),
            ("p95_latency_ms", latency),
            ("memory_pool", pool),
            ("placement", placement),
            ("peak_host_memory_bytes", host),
            ("peak_unified_memory_bytes", unified),
            ("peak_accelerator_memory_bytes", accelerator),
            ("resource_class_sha256", resource_class),
            ("demand_sha256", demand),
            ("evaluation_contract_sha256", evaluation),
            ("measured_at", measured),
            ("expires_at", expires),
            ("source_path", path),
            ("source_sha256", source),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "measurement digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "cell_id",
                "declaration_sha256",
                "sample_count",
                "success_rate",
                "p95_latency_ms",
                "memory_pool",
                "placement",
                "peak_host_memory_bytes",
                "peak_unified_memory_bytes",
                "peak_accelerator_memory_bytes",
                "resource_class_sha256",
                "demand_sha256",
                "evaluation_contract_sha256",
                "measured_at",
                "expires_at",
                "source_path",
                "source_sha256",
            )
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CellPassport:
    declaration: CellDeclaration
    observed: CellObservation
    estimated: CellEstimate
    measured: CellMeasurement
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "cell passport")
        if not isinstance(self.declaration, CellDeclaration):
            raise CellContractError("declaration must be a CellDeclaration.")
        for label, evidence, expected in (
            ("observed", self.observed, CellObservation),
            ("estimated", self.estimated, CellEstimate),
            ("measured", self.measured, CellMeasurement),
        ):
            if not isinstance(evidence, expected):
                raise CellContractError(f"{label} evidence has an invalid type.")
            if (
                evidence.cell_id != self.declaration.cell_id
                or evidence.declaration_sha256 != self.declaration.digest
            ):
                raise CellContractError(
                    f"{label} evidence is not bound to this declaration."
                )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "passport digest"),
        )

    @property
    def cell_id(self) -> str:
        return self.declaration.cell_id

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "declaration": self.declaration.payload(),
            "observed": self.observed.payload(),
            "estimated": self.estimated.payload(),
            "measured": self.measured.payload(),
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class AdvisorProfile:
    quality_weight: float
    latency_weight: float
    memory_weight: float
    min_success_rate: float
    min_samples: int
    reserve_memory_bytes: int
    latency_reference_ms: float
    memory_reference_bytes: int
    max_snapshot_age_seconds: int
    max_swap_used_bytes: int
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "advisor profile")
        for name in ("quality_weight", "latency_weight", "memory_weight"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name, minimum=0)
            )
        if not math.isfinite(self.total_weight) or self.total_weight <= 0:
            raise CellContractError(
                "Advisor profile weights must have a finite positive sum."
            )
        object.__setattr__(
            self,
            "min_success_rate",
            _number(self.min_success_rate, "min_success_rate", minimum=0, maximum=1),
        )
        object.__setattr__(
            self, "min_samples", _positive(self.min_samples, "min_samples")
        )
        object.__setattr__(
            self,
            "reserve_memory_bytes",
            _integer(self.reserve_memory_bytes, "reserve_memory_bytes"),
        )
        latency_ref = _number(
            self.latency_reference_ms, "latency_reference_ms", minimum=0
        )
        if latency_ref == 0:
            raise CellContractError("latency_reference_ms must be positive.")
        object.__setattr__(self, "latency_reference_ms", latency_ref)
        object.__setattr__(
            self,
            "memory_reference_bytes",
            _positive(self.memory_reference_bytes, "memory_reference_bytes"),
        )
        object.__setattr__(
            self,
            "max_snapshot_age_seconds",
            _positive(self.max_snapshot_age_seconds, "max_snapshot_age_seconds"),
        )
        object.__setattr__(
            self,
            "max_swap_used_bytes",
            _integer(self.max_swap_used_bytes, "max_swap_used_bytes"),
        )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "profile digest"),
        )

    @property
    def total_weight(self) -> float:
        return self.quality_weight + self.latency_weight + self.memory_weight

    def content_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "quality_weight",
                "latency_weight",
                "memory_weight",
                "min_success_rate",
                "min_samples",
                "reserve_memory_bytes",
                "latency_reference_ms",
                "memory_reference_bytes",
                "max_snapshot_age_seconds",
                "max_swap_used_bytes",
            )
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class AdaptiveCellCatalog:
    catalog_id: str
    cells: tuple[CellPassport, ...]
    profiles: Mapping[str, AdvisorProfile]
    digest: str = ""
    mode: str = CATALOG_MODE
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "adaptive cell catalog")
        if self.mode != CATALOG_MODE:
            raise CellContractError(
                "Adaptive cell catalog mode must be advisory_offline."
            )
        object.__setattr__(self, "catalog_id", _safe(self.catalog_id, "catalog_id"))
        cells = tuple(self.cells) if isinstance(self.cells, (tuple, list)) else ()
        if len(cells) > MAX_CELLS:
            raise CellContractError(
                f"Catalog cannot contain more than {MAX_CELLS} cells."
            )
        ids = [cell.cell_id for cell in cells if isinstance(cell, CellPassport)]
        if (
            len(ids) != len(cells)
            or not ids
            or ids != sorted(ids)
            or len(ids) != len(set(ids))
        ):
            raise CellContractError(
                "Catalog cells must be CellPassports, unique, and sorted by cell_id."
            )
        if not isinstance(self.profiles, Mapping) or not self.profiles:
            raise CellContractError("Catalog profiles must be a non-empty mapping.")
        if len(self.profiles) > MAX_PROFILES:
            raise CellContractError(
                f"Catalog cannot contain more than {MAX_PROFILES} profiles."
            )
        if any(not isinstance(key, str) for key in self.profiles):
            raise CellContractError("Catalog profile identifiers must be strings.")
        profiles: dict[str, AdvisorProfile] = {}
        for key in sorted(self.profiles):
            profile_id = _safe(key, "profile id")
            profile = self.profiles[key]
            if not isinstance(profile, AdvisorProfile):
                raise CellContractError(f"Profile {profile_id} is invalid.")
            profiles[profile_id] = profile
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "catalog digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "mode": self.mode,
            "cells": [cell.payload() for cell in self.cells],
            "profiles": {
                key: self.profiles[key].payload() for key in sorted(self.profiles)
            },
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def _schema(value: object, label: str) -> None:
    if value != CONTRACT_VERSION:
        raise CellContractError(f"Unsupported {label} schema_version.")


__all__ = [
    "ADVISOR_CONTRACT",
    "CATALOG_MODE",
    "MAX_CELLS",
    "MAX_PROFILES",
    "MEMORY_POOLS",
    "PLACEMENTS",
    "UNKNOWN",
    "AdaptiveCellCatalog",
    "AdvisorProfile",
    "CellContractError",
    "CellDeclaration",
    "CellEstimate",
    "CellMeasurement",
    "CellObservation",
    "CellPassport",
    "WorkloadDemand",
    "normalize_machine",
]
