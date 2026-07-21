from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .cell_contracts import (
    ADVISOR_CONTRACT,
    MEMORY_POOLS,
    PLACEMENTS,
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellContractError,
    CellPassport,
    WorkloadDemand,
    _bool,
    _digest,
    _enum,
    _ids,
    _integer,
    _number,
    _optional_sha,
    _positive,
    _safe,
    _schema,
    _sha,
    _timestamp,
    _validate_resource_shape,
)
from .resource_snapshot import ResourceSnapshot
from .verified_routing_contracts import CONTRACT_VERSION


@dataclass(frozen=True)
class AdaptiveRequest:
    exact_request_fingerprint: str
    intent_family_sha256: str | None
    demand: WorkloadDemand
    evaluation_contract_sha256: str
    profile: str
    evaluated_at: str
    offline_required: bool = True
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "adaptive request")
        object.__setattr__(
            self,
            "exact_request_fingerprint",
            _sha(self.exact_request_fingerprint, "exact_request_fingerprint"),
        )
        object.__setattr__(
            self,
            "intent_family_sha256",
            _optional_sha(self.intent_family_sha256, "intent_family_sha256"),
        )
        if not isinstance(self.demand, WorkloadDemand):
            raise CellContractError("demand must be a WorkloadDemand.")
        object.__setattr__(
            self,
            "evaluation_contract_sha256",
            _sha(self.evaluation_contract_sha256, "evaluation_contract_sha256"),
        )
        object.__setattr__(self, "profile", _safe(self.profile, "profile"))
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, "evaluated_at")
        )
        if _bool(self.offline_required, "offline_required") is not True:
            raise CellContractError(
                "Adaptive advisor v1 accepts offline requests only."
            )
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "request digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "exact_request_fingerprint": self.exact_request_fingerprint,
            "intent_family_sha256": self.intent_family_sha256,
            "demand": self.demand.payload(),
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "profile": self.profile,
            "evaluated_at": self.evaluated_at,
            "offline_required": self.offline_required,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class CandidateAssessment:
    cell_id: str
    passport_sha256: str
    hard_eligible: bool
    pareto_eligible: bool
    rejection_codes: tuple[str, ...]
    success_rate: float | None
    p95_latency_ms: float | None
    memory_pool: str | None
    placement: str | None
    effective_peak_host_memory_bytes: int | None
    effective_peak_unified_memory_bytes: int | None
    effective_peak_accelerator_memory_bytes: int | None
    utility: float | None
    digest: str = ""
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "candidate assessment")
        object.__setattr__(self, "cell_id", _safe(self.cell_id, "cell_id"))
        object.__setattr__(
            self, "passport_sha256", _sha(self.passport_sha256, "passport_sha256")
        )
        hard = _bool(self.hard_eligible, "hard_eligible")
        pareto = _bool(self.pareto_eligible, "pareto_eligible")
        reasons = _ids(self.rejection_codes, "rejection_codes")
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
        for name in (
            "effective_peak_host_memory_bytes",
            "effective_peak_unified_memory_bytes",
            "effective_peak_accelerator_memory_bytes",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self, name, None if value is None else _positive(value, name)
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
        _validate_resource_shape(
            pool,
            placement,
            self.effective_peak_host_memory_bytes,
            self.effective_peak_unified_memory_bytes,
            self.effective_peak_accelerator_memory_bytes,
            label="candidate",
        )
        utility = (
            None
            if self.utility is None
            else _number(self.utility, "utility", minimum=0, maximum=1)
        )
        if hard and (
            reasons
            or any(
                item is None
                for item in (success, latency, self.memory_pool, self.placement)
            )
        ):
            raise CellContractError(
                "Hard-eligible candidate requires complete metrics and no rejection."
            )
        if not hard and (pareto or utility is not None):
            raise CellContractError(
                "Hard-ineligible candidate cannot enter Pareto ranking."
            )
        if pareto != (utility is not None):
            raise CellContractError("Only Pareto candidates may receive utility.")
        object.__setattr__(self, "rejection_codes", reasons)
        object.__setattr__(self, "success_rate", success)
        object.__setattr__(self, "p95_latency_ms", latency)
        object.__setattr__(self, "memory_pool", pool)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "utility", utility)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "candidate digest"),
        )

    @property
    def effective_total_memory_bytes(self) -> int | None:
        values = (
            self.effective_peak_host_memory_bytes,
            self.effective_peak_unified_memory_bytes,
            self.effective_peak_accelerator_memory_bytes,
        )
        present = [item for item in values if item is not None]
        return None if not present else sum(present)

    def content_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "cell_id",
                "passport_sha256",
                "hard_eligible",
                "pareto_eligible",
                "rejection_codes",
                "success_rate",
                "p95_latency_ms",
                "memory_pool",
                "placement",
                "effective_peak_host_memory_bytes",
                "effective_peak_unified_memory_bytes",
                "effective_peak_accelerator_memory_bytes",
                "utility",
            )
        } | {"rejection_codes": list(self.rejection_codes)}

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class AdaptiveAdvice:
    catalog_sha256: str
    request_sha256: str
    resource_snapshot_sha256: str
    evaluated_at: str
    profile: str
    status: str
    selected_cell_id: str | None
    candidates: tuple[CandidateAssessment, ...]
    reason_codes: tuple[str, ...]
    applied: bool = False
    authorizes_execution: bool = False
    network_used: bool = False
    model_invocations: int = 0
    digest: str = ""
    contract: str = ADVISOR_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, "adaptive advice")
        if self.contract != ADVISOR_CONTRACT:
            raise CellContractError("Unsupported adaptive advice contract.")
        for name in ("catalog_sha256", "request_sha256", "resource_snapshot_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, "evaluated_at")
        )
        object.__setattr__(self, "profile", _safe(self.profile, "profile"))
        if self.status not in {"recommended", "abstained"}:
            raise CellContractError("Adaptive advice status is not supported.")
        candidates = tuple(self.candidates)
        ids = [
            item.cell_id for item in candidates if isinstance(item, CandidateAssessment)
        ]
        if (
            len(ids) != len(candidates)
            or ids != sorted(ids)
            or len(ids) != len(set(ids))
        ):
            raise CellContractError(
                "Candidates must be assessments, unique, and sorted."
            )
        selected = (
            None
            if self.selected_cell_id is None
            else _safe(self.selected_cell_id, "selected_cell_id")
        )
        if (
            self.status == "recommended"
            and len(
                [
                    item
                    for item in candidates
                    if item.cell_id == selected and item.pareto_eligible
                ]
            )
            != 1
        ):
            raise CellContractError(
                "Recommended advice must select one Pareto candidate."
            )
        if self.status == "abstained" and selected is not None:
            raise CellContractError("Abstained advice cannot select a cell.")
        reasons = _ids(self.reason_codes, "reason_codes", non_empty=True)
        if any(
            _bool(value, name)
            for value, name in (
                (self.applied, "applied"),
                (self.authorizes_execution, "authorizes_execution"),
                (self.network_used, "network_used"),
            )
        ):
            raise CellContractError(
                "Adaptive advice v1 is read-only and non-authorizing."
            )
        if _integer(self.model_invocations, "model_invocations") != 0:
            raise CellContractError("Adaptive advice v1 cannot invoke a model.")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_cell_id", selected)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "digest",
            _digest(self.digest, self.content_payload(), "advice digest"),
        )

    @property
    def abstained(self) -> bool:
        return self.status == "abstained"

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "catalog_sha256": self.catalog_sha256,
            "request_sha256": self.request_sha256,
            "resource_snapshot_sha256": self.resource_snapshot_sha256,
            "evaluated_at": self.evaluated_at,
            "profile": self.profile,
            "status": self.status,
            "selected_cell_id": self.selected_cell_id,
            "candidates": [item.payload() for item in self.candidates],
            "reason_codes": list(self.reason_codes),
            "applied": self.applied,
            "authorizes_execution": self.authorizes_execution,
            "network_used": self.network_used,
            "model_invocations": self.model_invocations,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def advise_cell(
    catalog: AdaptiveCellCatalog, snapshot: ResourceSnapshot, request: AdaptiveRequest
) -> AdaptiveAdvice:
    if (
        not isinstance(catalog, AdaptiveCellCatalog)
        or not isinstance(snapshot, ResourceSnapshot)
        or not isinstance(request, AdaptiveRequest)
    ):
        raise CellContractError("Advisor inputs have invalid contract types.")
    if not request.demand.capabilities or "*" in request.demand.capabilities:
        raise CellContractError(
            "Adaptive requests require at least one explicit non-wildcard capability."
        )
    profile = catalog.profiles.get(request.profile)
    if profile is None:
        raise CellContractError(f"Unknown advisor profile: {request.profile}.")
    global_rejections = _global_snapshot_rejections(snapshot, request, profile)
    if global_rejections:
        assessments = tuple(_blocked(cell, global_rejections) for cell in catalog.cells)
        return _advice(catalog, snapshot, request, assessments, None, global_rejections)

    hard = tuple(
        _hard_filter(cell, snapshot, request, profile) for cell in catalog.cells
    )
    eligible = tuple(item for item in hard if item.hard_eligible)
    by_id = {item.cell_id: item for item in hard}
    frontier: list[CandidateAssessment] = []
    for candidate in eligible:
        if any(
            _dominates(other, candidate, profile)
            for other in eligible
            if other.cell_id != candidate.cell_id
        ):
            continue
        ranked = _rank(candidate, profile)
        by_id[ranked.cell_id] = ranked
        frontier.append(ranked)
    frontier.sort(key=lambda candidate: _ranking_key(candidate, profile))
    selected = frontier[0].cell_id if frontier else None
    assessments = tuple(by_id[key] for key in sorted(by_id))
    reasons = (
        ("advisory_only", "pareto_frontier_selected")
        if selected
        else ("advisory_only", "no_eligible_cell")
    )
    return _advice(catalog, snapshot, request, assessments, selected, reasons)


def _global_snapshot_rejections(
    snapshot: ResourceSnapshot, request: AdaptiveRequest, profile: AdvisorProfile
) -> tuple[str, ...]:
    evaluated, captured = _time(request.evaluated_at), _time(snapshot.captured_at)
    reasons: set[str] = set()
    if captured > evaluated:
        reasons.add("snapshot_from_future")
    elif (evaluated - captured).total_seconds() > profile.max_snapshot_age_seconds:
        reasons.add("snapshot_stale")
    if snapshot.swap_used_bytes is None:
        reasons.add("swap_usage_unknown")
    elif snapshot.swap_used_bytes > profile.max_swap_used_bytes:
        reasons.add("swap_limit_exceeded")
    return tuple(sorted(reasons))


def _hard_filter(
    cell: CellPassport,
    snapshot: ResourceSnapshot,
    request: AdaptiveRequest,
    profile: AdvisorProfile,
) -> CandidateAssessment:
    declared, observed, estimated, measured = (
        cell.declaration,
        cell.observed,
        cell.estimated,
        cell.measured,
    )
    rejected: set[str] = set()
    if not declared.offline_capable:
        rejected.add("offline_not_supported")
    if snapshot.system not in declared.supported_systems:
        rejected.add("system_not_supported")
    if snapshot.machine not in declared.supported_machines:
        rejected.add("machine_not_supported")
    demand = request.demand
    if not set(demand.capabilities).issubset(declared.capabilities):
        rejected.add("capability_gap")
    if not set(demand.tool_surfaces).issubset(declared.tool_surfaces):
        rejected.add("tool_surface_gap")
    if demand.risk_class not in declared.risk_classes:
        rejected.add("risk_class_not_supported")
    if demand.context_tokens > declared.max_context_tokens:
        rejected.add("context_window_exceeded")
    for component, status, expected, actual in (
        (
            "model",
            observed.model_status,
            declared.expected_model_sha256,
            observed.observed_model_sha256,
        ),
        (
            "runtime",
            observed.runtime_status,
            declared.expected_runtime_sha256,
            observed.observed_runtime_sha256,
        ),
        (
            "harness",
            observed.harness_status,
            declared.expected_harness_sha256,
            observed.observed_harness_sha256,
        ),
        (
            "tool_contract",
            observed.tool_contract_status,
            declared.expected_tool_contract_sha256,
            observed.observed_tool_contract_sha256,
        ),
    ):
        if expected is None:
            rejected.add(f"{component}_expected_identity_unknown")
        if status == "unknown":
            rejected.add(f"{component}_availability_unknown")
        elif status != "available":
            rejected.add(f"{component}_unavailable")
        elif expected is not None and actual != expected:
            rejected.add(f"{component}_identity_mismatch")
    _freshness(
        observed.captured_at,
        observed.expires_at,
        request.evaluated_at,
        "observation",
        rejected,
    )
    if estimated.memory_pool is None or estimated.placement is None:
        rejected.add("resource_estimate_unknown")
    if measured.sample_count == 0:
        rejected.add("measurement_unknown")
    else:
        if measured.demand_sha256 != demand.digest:
            rejected.add("measurement_demand_mismatch")
        if measured.evaluation_contract_sha256 != request.evaluation_contract_sha256:
            rejected.add("measurement_evaluation_contract_mismatch")
        if measured.resource_class_sha256 != snapshot.resource_class_sha256:
            rejected.add("measurement_not_applicable")
        if (
            measured.memory_pool != estimated.memory_pool
            or measured.placement != estimated.placement
        ):
            rejected.add("measurement_placement_mismatch")
        if measured.sample_count < profile.min_samples:
            rejected.add("insufficient_samples")
        if (
            measured.success_rate is not None
            and measured.success_rate < profile.min_success_rate
        ):
            rejected.add("quality_floor_not_met")
        _freshness(
            measured.measured_at,
            measured.expires_at,
            request.evaluated_at,
            "measurement",
            rejected,
        )
    resources = _effective_resources(cell)
    _resource_rejections(
        resources,
        estimated.memory_pool,
        estimated.placement,
        snapshot,
        profile,
        rejected,
    )
    if (
        snapshot.system == "unknown"
        or snapshot.os_release == "unknown"
        or snapshot.cpu_count is None
        or snapshot.cpu_identity_sha256 is None
        or snapshot.total_memory_bytes is None
        or snapshot.runtime_environment_sha256 is None
        or snapshot.effective_memory_limit_bytes is None
    ):
        rejected.add("resource_class_unknown")
    if (
        estimated.placement in {"integrated_accelerator", "discrete_accelerator"}
        and snapshot.accelerator_identity_sha256 is None
    ):
        rejected.add("accelerator_identity_unknown")
    success, latency = measured.success_rate, measured.p95_latency_ms
    if not rejected and any(
        item is None
        for item in (success, latency, estimated.memory_pool, estimated.placement)
    ):
        rejected.add("ranking_metric_unknown")
    return CandidateAssessment(
        cell_id=cell.cell_id,
        passport_sha256=cell.digest,
        hard_eligible=not rejected,
        pareto_eligible=False,
        rejection_codes=tuple(rejected),
        success_rate=success,
        p95_latency_ms=latency,
        memory_pool=estimated.memory_pool,
        placement=estimated.placement,
        effective_peak_host_memory_bytes=resources[0],
        effective_peak_unified_memory_bytes=resources[1],
        effective_peak_accelerator_memory_bytes=resources[2],
        utility=None,
    )


def _freshness(
    start: str | None,
    expires: str | None,
    evaluated: str,
    label: str,
    rejected: set[str],
) -> None:
    if start is None or expires is None:
        return
    current = _time(evaluated)
    if _time(start) > current:
        rejected.add(f"{label}_from_future")
    if current >= _time(expires):
        rejected.add(f"{label}_expired")


def _effective_resources(
    cell: CellPassport,
) -> tuple[int | None, int | None, int | None]:
    estimated, measured = cell.estimated, cell.measured
    if estimated.memory_pool is None or estimated.placement is None:
        return None, None, None
    measurement_applies = (
        measured.memory_pool == estimated.memory_pool
        and measured.placement == estimated.placement
    )
    return tuple(
        _maximum(
            getattr(estimated, name),
            getattr(measured, name) if measurement_applies else None,
        )
        for name in (
            "peak_host_memory_bytes",
            "peak_unified_memory_bytes",
            "peak_accelerator_memory_bytes",
        )
    )  # type: ignore[return-value]


def _maximum(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    return left if right is None else max(left, right)


def _resource_rejections(
    resources: tuple[int | None, int | None, int | None],
    pool: str | None,
    placement: str | None,
    snapshot: ResourceSnapshot,
    profile: AdvisorProfile,
    rejected: set[str],
) -> None:
    host, unified, accelerator = resources
    system_available = None
    if (
        snapshot.available_memory_bytes is not None
        and snapshot.effective_memory_limit_bytes is not None
    ):
        system_available = min(
            snapshot.available_memory_bytes, snapshot.effective_memory_limit_bytes
        )
    if pool == "host" and placement == "cpu":
        if system_available is None:
            rejected.add("host_memory_unknown")
        elif host is None or host + profile.reserve_memory_bytes > system_available:
            rejected.add("host_memory_headroom_insufficient")
    elif pool == "unified" and placement == "integrated_accelerator":
        if (
            snapshot.memory_topology != "unified"
            or snapshot.accelerator_kind != "integrated"
        ):
            rejected.add("unified_memory_unavailable")
        if system_available is None:
            rejected.add("unified_memory_unknown")
        elif (
            unified is None or unified + profile.reserve_memory_bytes > system_available
        ):
            rejected.add("unified_memory_headroom_insufficient")
    elif pool == "accelerator" and placement == "discrete_accelerator":
        if (
            snapshot.memory_topology != "dedicated"
            or snapshot.accelerator_kind != "discrete"
            or snapshot.accelerator_memory_available_bytes is None
        ):
            rejected.add("accelerator_memory_unknown")
        elif (
            accelerator is None
            or accelerator + profile.reserve_memory_bytes
            > snapshot.accelerator_memory_available_bytes
        ):
            rejected.add("accelerator_memory_headroom_insufficient")
        if system_available is None:
            rejected.add("host_memory_unknown")
        elif host is None or host + profile.reserve_memory_bytes > system_available:
            rejected.add("host_memory_headroom_insufficient")


def _blocked(cell: CellPassport, reasons: tuple[str, ...]) -> CandidateAssessment:
    resources = _effective_resources(cell)
    return CandidateAssessment(
        cell_id=cell.cell_id,
        passport_sha256=cell.digest,
        hard_eligible=False,
        pareto_eligible=False,
        rejection_codes=reasons,
        success_rate=cell.measured.success_rate,
        p95_latency_ms=cell.measured.p95_latency_ms,
        memory_pool=cell.estimated.memory_pool,
        placement=cell.estimated.placement,
        effective_peak_host_memory_bytes=resources[0],
        effective_peak_unified_memory_bytes=resources[1],
        effective_peak_accelerator_memory_bytes=resources[2],
        utility=None,
    )


def _dominates(
    left: CandidateAssessment, right: CandidateAssessment, profile: AdvisorProfile
) -> bool:
    comparisons: list[tuple[float, float, bool]] = []
    if profile.quality_weight > 0:
        comparisons.append((float(left.success_rate), float(right.success_rate), True))
    if profile.latency_weight > 0:
        comparisons.append(
            (float(left.p95_latency_ms), float(right.p95_latency_ms), False)
        )
    if profile.memory_weight > 0:
        comparisons.append(
            (
                float(left.effective_total_memory_bytes),
                float(right.effective_total_memory_bytes),
                False,
            )
        )
    no_worse = all(a >= b if maximize else a <= b for a, b, maximize in comparisons)
    better = any(a > b if maximize else a < b for a, b, maximize in comparisons)
    return no_worse and better


def _rank(
    candidate: CandidateAssessment, profile: AdvisorProfile
) -> CandidateAssessment:
    quality = float(candidate.success_rate)
    latency = float(candidate.p95_latency_ms)
    memory = int(candidate.effective_total_memory_bytes)
    latency_score = max(0.0, 1.0 - latency / profile.latency_reference_ms)
    memory_score = max(0.0, 1.0 - memory / profile.memory_reference_bytes)
    utility = (
        profile.quality_weight * quality
        + profile.latency_weight * latency_score
        + profile.memory_weight * memory_score
    ) / profile.total_weight
    return CandidateAssessment(
        cell_id=candidate.cell_id,
        passport_sha256=candidate.passport_sha256,
        hard_eligible=True,
        pareto_eligible=True,
        rejection_codes=(),
        success_rate=quality,
        p95_latency_ms=latency,
        memory_pool=candidate.memory_pool,
        placement=candidate.placement,
        effective_peak_host_memory_bytes=candidate.effective_peak_host_memory_bytes,
        effective_peak_unified_memory_bytes=candidate.effective_peak_unified_memory_bytes,
        effective_peak_accelerator_memory_bytes=candidate.effective_peak_accelerator_memory_bytes,
        utility=round(utility, 12),
    )


def _ranking_key(
    candidate: CandidateAssessment, profile: AdvisorProfile
) -> tuple[object, ...]:
    active: list[object] = [-float(candidate.utility)]
    if profile.quality_weight > 0:
        active.append(-float(candidate.success_rate))
    if profile.latency_weight > 0:
        active.append(float(candidate.p95_latency_ms))
    if profile.memory_weight > 0:
        active.append(int(candidate.effective_total_memory_bytes))
    active.append(candidate.cell_id)
    return tuple(active)


def _advice(
    catalog: AdaptiveCellCatalog,
    snapshot: ResourceSnapshot,
    request: AdaptiveRequest,
    candidates: tuple[CandidateAssessment, ...],
    selected: str | None,
    reasons: tuple[str, ...],
) -> AdaptiveAdvice:
    return AdaptiveAdvice(
        catalog_sha256=catalog.digest,
        request_sha256=request.digest,
        resource_snapshot_sha256=snapshot.digest,
        evaluated_at=request.evaluated_at,
        profile=request.profile,
        status="recommended" if selected else "abstained",
        selected_cell_id=selected,
        candidates=candidates,
        reason_codes=tuple(sorted(set(reasons) | {"advisory_only"})),
    )


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_adaptive_request(
    *,
    exact_request_fingerprint: str,
    workload_id: str,
    required_capabilities: Iterable[str],
    required_tool_surfaces: Iterable[str] = (),
    risk_class: str,
    required_context_tokens: int,
    evaluation_contract_sha256: str,
    profile: str,
    evaluated_at: str,
    intent_family_sha256: str | None = None,
) -> AdaptiveRequest:
    demand = WorkloadDemand(
        workload_id=workload_id,
        capabilities=tuple(required_capabilities),
        tool_surfaces=tuple(required_tool_surfaces),
        risk_class=risk_class,
        context_tokens=required_context_tokens,
    )
    return AdaptiveRequest(
        exact_request_fingerprint=exact_request_fingerprint,
        intent_family_sha256=intent_family_sha256,
        demand=demand,
        evaluation_contract_sha256=evaluation_contract_sha256,
        profile=profile,
        evaluated_at=evaluated_at,
    )


__all__ = [
    "AdaptiveAdvice",
    "AdaptiveRequest",
    "CandidateAssessment",
    "advise_cell",
    "build_adaptive_request",
]
