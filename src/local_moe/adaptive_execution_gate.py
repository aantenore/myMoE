from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Mapping, TypeVar

from .adaptive_advisor_service import (
    MAX_TASK_CHARS,
    AdaptiveAdvisorReceipt,
    AdvisorServiceError,
    evaluate_advisor,
)
from .adaptive_selector import AdaptiveAdvice, AdaptiveRequest, CandidateAssessment
from .cell_contracts import MAX_CELLS, CellContractError, WorkloadDemand
from .secure_files import read_bounded_regular_file
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


EXECUTION_POLICY_CONTRACT = "AdaptiveCellExecutionPolicy"
PREVIEW_RECEIPT_CONTRACT = "AdaptiveCellExecutionPreviewReceipt"
EXECUTION_POLICY_MODE = "dry_run"
MAX_ADVISOR_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_EXECUTION_POLICY_BYTES = 64 * 1024
MAX_PREVIEW_RECEIPT_BYTES = 512 * 1024
MAX_SOURCE_RECEIPT_AGE_SECONDS = 120
PREVIEW_STATUSES = frozenset({"admission_passed", "admission_blocked"})
PREVIEW_REASON_CODES = frozenset(
    {
        "catalog_drift",
        "evaluation_contract_drift",
        "fresh_admission_blocked",
        "request_semantics_changed",
        "risk_class_blocked",
        "selected_cell_changed",
        "selected_passport_changed",
        "source_receipt_expired",
        "source_receipt_from_future",
        "source_receipt_not_recommended",
        "task_fingerprint_mismatch",
        "task_size_mismatch",
        "tool_surface_blocked",
    }
)
T = TypeVar("T")


class AdaptiveExecutionGateError(ValueError):
    """Stable failure at the dry-run execution-preview boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AdaptiveCellExecutionPolicy:
    """Fail-closed v1 policy that cannot grant execution authority."""

    max_source_receipt_age_seconds: int
    allowed_risk_classes: tuple[str, ...]
    max_tool_surfaces: int
    digest: str = ""
    mode: str = EXECUTION_POLICY_MODE
    contract: str = EXECUTION_POLICY_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise AdaptiveExecutionGateError(
                "policy_invalid", "Execution policy schema is unsupported."
            )
        if self.contract != EXECUTION_POLICY_CONTRACT:
            raise AdaptiveExecutionGateError(
                "policy_invalid", "Execution policy contract is unsupported."
            )
        if self.mode != EXECUTION_POLICY_MODE:
            raise AdaptiveExecutionGateError(
                "policy_invalid", "Execution policy mode must remain dry_run."
            )
        age = _integer(
            self.max_source_receipt_age_seconds,
            "max_source_receipt_age_seconds",
        )
        if not 0 < age <= MAX_SOURCE_RECEIPT_AGE_SECONDS:
            raise AdaptiveExecutionGateError(
                "policy_invalid",
                "Execution policy source receipt age exceeds the v1 safety bound.",
            )
        risks = _identifier_list(
            self.allowed_risk_classes,
            "allowed_risk_classes",
            non_empty=True,
        )
        if risks != ("compute_only",):
            raise AdaptiveExecutionGateError(
                "policy_invalid",
                "Execution policy v1 permits only the compute_only risk class.",
            )
        maximum_tools = _integer(self.max_tool_surfaces, "max_tool_surfaces")
        if maximum_tools != 0:
            raise AdaptiveExecutionGateError(
                "policy_invalid",
                "Execution policy v1 permits zero tool surfaces.",
            )
        object.__setattr__(self, "max_source_receipt_age_seconds", age)
        object.__setattr__(self, "allowed_risk_classes", risks)
        object.__setattr__(self, "max_tool_surfaces", maximum_tools)
        object.__setattr__(
            self,
            "digest",
            _content_digest(self.digest, self.content_payload(), "policy digest"),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "mode": self.mode,
            "max_source_receipt_age_seconds": self.max_source_receipt_age_seconds,
            "allowed_risk_classes": list(self.allowed_risk_classes),
            "max_tool_surfaces": self.max_tool_surfaces,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class AdaptiveCellExecutionPreviewReceipt:
    """Content-addressed record of a dry-run, present-tense admission check."""

    source_advisor_receipt_sha256: str
    source_request_sha256: str
    fresh_advisor_receipt_sha256: str
    fresh_request_sha256: str
    policy_sha256: str
    evaluated_at: str
    source_selected_cell_id: str | None
    fresh_selected_cell_id: str | None
    source_passport_sha256: str | None
    fresh_passport_sha256: str | None
    fresh_resource_snapshot_sha256: str
    status: str
    reason_codes: tuple[str, ...]
    task_chars: int
    applied: bool = False
    authorizes_execution: bool = False
    network_used: bool = False
    model_invocations: int = 0
    digest: str = ""
    contract: str = PREVIEW_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid", "Preview receipt schema is unsupported."
            )
        if self.contract != PREVIEW_RECEIPT_CONTRACT:
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid", "Preview receipt contract is unsupported."
            )
        for name in (
            "source_advisor_receipt_sha256",
            "source_request_sha256",
            "fresh_advisor_receipt_sha256",
            "fresh_request_sha256",
            "policy_sha256",
            "fresh_resource_snapshot_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("source_passport_sha256", "fresh_passport_sha256"):
            object.__setattr__(
                self,
                name,
                _optional_sha(getattr(self, name), name),
            )
        for name in ("source_selected_cell_id", "fresh_selected_cell_id"):
            object.__setattr__(
                self,
                name,
                _optional_identifier(getattr(self, name), name),
            )
        object.__setattr__(
            self, "evaluated_at", _timestamp(self.evaluated_at, "evaluated_at")
        )
        if self.status not in PREVIEW_STATUSES:
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid", "Preview receipt status is unsupported."
            )
        reasons = _identifier_list(self.reason_codes, "reason_codes")
        if any(reason not in PREVIEW_REASON_CODES for reason in reasons):
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid",
                "Preview receipt contains an unsupported reason code.",
            )
        if self.status == "admission_passed":
            if reasons:
                raise AdaptiveExecutionGateError(
                    "preview_receipt_invalid",
                    "A passed admission preview cannot contain blockers.",
                )
            if (
                self.source_selected_cell_id is None
                or self.source_selected_cell_id != self.fresh_selected_cell_id
                or self.source_passport_sha256 is None
                or self.source_passport_sha256 != self.fresh_passport_sha256
            ):
                raise AdaptiveExecutionGateError(
                    "preview_receipt_invalid",
                    "A passed admission preview requires one unchanged cell passport.",
                )
        elif not reasons:
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid",
                "A blocked admission preview requires at least one reason.",
            )
        task_chars = _integer(self.task_chars, "task_chars")
        if not 0 < task_chars <= MAX_TASK_CHARS:
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid", "Preview receipt task size is invalid."
            )
        for name in ("applied", "authorizes_execution", "network_used"):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise AdaptiveExecutionGateError(
                    "preview_receipt_invalid",
                    "Execution preview v1 must remain non-authorizing and read-only.",
                )
        if _integer(self.model_invocations, "model_invocations") != 0:
            raise AdaptiveExecutionGateError(
                "preview_receipt_invalid",
                "Execution preview v1 cannot invoke a model.",
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "task_chars", task_chars)
        object.__setattr__(
            self,
            "digest",
            _content_digest(
                self.digest,
                self.content_payload(),
                "preview receipt digest",
            ),
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "source_advisor_receipt_sha256": self.source_advisor_receipt_sha256,
            "source_request_sha256": self.source_request_sha256,
            "fresh_advisor_receipt_sha256": self.fresh_advisor_receipt_sha256,
            "fresh_request_sha256": self.fresh_request_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluated_at": self.evaluated_at,
            "source_selected_cell_id": self.source_selected_cell_id,
            "fresh_selected_cell_id": self.fresh_selected_cell_id,
            "source_passport_sha256": self.source_passport_sha256,
            "fresh_passport_sha256": self.fresh_passport_sha256,
            "fresh_resource_snapshot_sha256": self.fresh_resource_snapshot_sha256,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "task_chars": self.task_chars,
            "applied": self.applied,
            "authorizes_execution": self.authorizes_execution,
            "network_used": self.network_used,
            "model_invocations": self.model_invocations,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def adaptive_advisor_receipt_from_payload(
    raw: object,
) -> AdaptiveAdvisorReceipt:
    """Rebuild and verify every nested digest in a persisted advisor receipt."""

    try:
        data = _strict(raw, _field_names(AdaptiveAdvisorReceipt), "advisor receipt")
        request = _adaptive_request_from_payload(data["request"])
        advice = _adaptive_advice_from_payload(data["advice"])
        return AdaptiveAdvisorReceipt(
            request=request,
            advice=advice,
            task_chars=_integer(data["task_chars"], "task_chars"),
            display_state=_string(data["display_state"], "display_state"),
            digest=_sha(data["digest"], "advisor receipt digest"),
            contract=_string(data["contract"], "advisor receipt contract"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
    except AdaptiveExecutionGateError:
        raise
    except (AdvisorServiceError, CellContractError, VerifiedRoutingError) as exc:
        raise AdaptiveExecutionGateError(
            "source_receipt_invalid",
            "Source advisor receipt failed nested contract verification.",
        ) from exc
    except (KeyError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise AdaptiveExecutionGateError(
            "source_receipt_invalid", "Source advisor receipt is invalid."
        ) from exc


def load_adaptive_advisor_receipt(path: str | Path) -> AdaptiveAdvisorReceipt:
    """Strictly load one bounded, regular, non-link advisor receipt file."""

    raw = _load_json_object(
        path,
        maximum_bytes=MAX_ADVISOR_RECEIPT_BYTES,
        label="source advisor receipt",
        code="source_receipt_invalid",
    )
    return adaptive_advisor_receipt_from_payload(raw)


def adaptive_execution_policy_from_payload(
    raw: object,
) -> AdaptiveCellExecutionPolicy:
    try:
        data = _strict(
            raw,
            _field_names(AdaptiveCellExecutionPolicy),
            "execution policy",
        )
        return AdaptiveCellExecutionPolicy(
            max_source_receipt_age_seconds=_integer(
                data["max_source_receipt_age_seconds"],
                "max_source_receipt_age_seconds",
            ),
            allowed_risk_classes=_raw_identifier_list(
                data["allowed_risk_classes"], "allowed_risk_classes"
            ),
            max_tool_surfaces=_integer(data["max_tool_surfaces"], "max_tool_surfaces"),
            digest=_sha(data["digest"], "policy digest"),
            mode=_string(data["mode"], "policy mode"),
            contract=_string(data["contract"], "policy contract"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
    except AdaptiveExecutionGateError:
        raise
    except (KeyError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise AdaptiveExecutionGateError(
            "policy_invalid", "Execution policy is invalid."
        ) from exc


def load_adaptive_execution_policy(
    path: str | Path,
) -> AdaptiveCellExecutionPolicy:
    raw = _load_json_object(
        path,
        maximum_bytes=MAX_EXECUTION_POLICY_BYTES,
        label="adaptive execution policy",
        code="policy_invalid",
    )
    return adaptive_execution_policy_from_payload(raw)


def adaptive_cell_execution_preview_receipt_from_payload(
    raw: object,
) -> AdaptiveCellExecutionPreviewReceipt:
    try:
        data = _strict(
            raw,
            _field_names(AdaptiveCellExecutionPreviewReceipt),
            "execution preview receipt",
        )
        return AdaptiveCellExecutionPreviewReceipt(
            source_advisor_receipt_sha256=_sha(
                data["source_advisor_receipt_sha256"],
                "source_advisor_receipt_sha256",
            ),
            source_request_sha256=_sha(
                data["source_request_sha256"], "source_request_sha256"
            ),
            fresh_advisor_receipt_sha256=_sha(
                data["fresh_advisor_receipt_sha256"],
                "fresh_advisor_receipt_sha256",
            ),
            fresh_request_sha256=_sha(
                data["fresh_request_sha256"], "fresh_request_sha256"
            ),
            policy_sha256=_sha(data["policy_sha256"], "policy_sha256"),
            evaluated_at=_timestamp(data["evaluated_at"], "evaluated_at"),
            source_selected_cell_id=_optional_raw_string(
                data["source_selected_cell_id"], "source_selected_cell_id"
            ),
            fresh_selected_cell_id=_optional_raw_string(
                data["fresh_selected_cell_id"], "fresh_selected_cell_id"
            ),
            source_passport_sha256=_optional_raw_string(
                data["source_passport_sha256"], "source_passport_sha256"
            ),
            fresh_passport_sha256=_optional_raw_string(
                data["fresh_passport_sha256"], "fresh_passport_sha256"
            ),
            fresh_resource_snapshot_sha256=_sha(
                data["fresh_resource_snapshot_sha256"],
                "fresh_resource_snapshot_sha256",
            ),
            status=_string(data["status"], "status"),
            reason_codes=_raw_identifier_list(data["reason_codes"], "reason_codes"),
            task_chars=_integer(data["task_chars"], "task_chars"),
            applied=_boolean(data["applied"], "applied"),
            authorizes_execution=_boolean(
                data["authorizes_execution"], "authorizes_execution"
            ),
            network_used=_boolean(data["network_used"], "network_used"),
            model_invocations=_integer(data["model_invocations"], "model_invocations"),
            digest=_sha(data["digest"], "preview receipt digest"),
            contract=_string(data["contract"], "preview receipt contract"),
            schema_version=_string(data["schema_version"], "schema_version"),
        )
    except AdaptiveExecutionGateError:
        raise
    except (KeyError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise AdaptiveExecutionGateError(
            "preview_receipt_invalid", "Execution preview receipt is invalid."
        ) from exc


def preview_cell_execution(
    source_receipt_path: str | Path,
    task_text: str,
    catalog_path: str | Path,
    evaluation_contract_path: str | Path,
    policy_path: str | Path,
) -> AdaptiveCellExecutionPreviewReceipt:
    """Re-admit one exact advisor decision without executing or authorizing it."""

    source = load_adaptive_advisor_receipt(source_receipt_path)
    policy = load_adaptive_execution_policy(policy_path)
    try:
        fresh = evaluate_advisor(
            catalog_path=catalog_path,
            evaluation_contract_path=evaluation_contract_path,
            task_text=task_text,
            workload_id=source.request.demand.workload_id,
            required_capabilities=source.request.demand.capabilities,
            required_tool_surfaces=source.request.demand.tool_surfaces,
            risk_class=source.request.demand.risk_class,
            context_tokens=source.request.demand.context_tokens,
            profile=source.request.profile,
            intent_family_sha256=source.request.intent_family_sha256,
        )
        if not isinstance(fresh, AdaptiveAdvisorReceipt):
            raise AdaptiveExecutionGateError(
                "fresh_admission_invalid",
                "Fresh adaptive admission returned an invalid receipt.",
            )
    except AdaptiveExecutionGateError:
        raise
    except AdvisorServiceError as exc:
        raise AdaptiveExecutionGateError(
            exc.code, "Fresh adaptive admission could not be verified."
        ) from exc
    except Exception as exc:
        raise AdaptiveExecutionGateError(
            "fresh_admission_invalid",
            "Fresh adaptive admission could not be verified.",
        ) from exc

    reasons: set[str] = set()
    if source.advice.status != "recommended":
        reasons.add("source_receipt_not_recommended")
    source_time = _parse_timestamp(source.request.evaluated_at)
    fresh_time = _parse_timestamp(fresh.request.evaluated_at)
    if source_time > fresh_time:
        reasons.add("source_receipt_from_future")
    elif (fresh_time - source_time).total_seconds() > (
        policy.max_source_receipt_age_seconds
    ):
        reasons.add("source_receipt_expired")
    if (
        source.request.exact_request_fingerprint
        != fresh.request.exact_request_fingerprint
    ):
        reasons.add("task_fingerprint_mismatch")
    if source.task_chars != fresh.task_chars:
        reasons.add("task_size_mismatch")
    if (
        source.request.demand.digest != fresh.request.demand.digest
        or source.request.profile != fresh.request.profile
        or source.request.intent_family_sha256 != fresh.request.intent_family_sha256
        or source.request.offline_required != fresh.request.offline_required
    ):
        reasons.add("request_semantics_changed")
    if (
        source.request.demand.risk_class not in policy.allowed_risk_classes
        or fresh.request.demand.risk_class not in policy.allowed_risk_classes
    ):
        reasons.add("risk_class_blocked")
    if (
        len(source.request.demand.tool_surfaces) > policy.max_tool_surfaces
        or len(fresh.request.demand.tool_surfaces) > policy.max_tool_surfaces
    ):
        reasons.add("tool_surface_blocked")
    if source.advice.catalog_sha256 != fresh.advice.catalog_sha256:
        reasons.add("catalog_drift")
    if (
        source.request.evaluation_contract_sha256
        != fresh.request.evaluation_contract_sha256
    ):
        reasons.add("evaluation_contract_drift")

    source_cell = source.advice.selected_cell_id
    fresh_cell = fresh.advice.selected_cell_id
    source_passport = _selected_passport(source.advice)
    fresh_passport = _selected_passport(fresh.advice)
    if source_cell != fresh_cell:
        reasons.add("selected_cell_changed")
    if source_passport != fresh_passport:
        reasons.add("selected_passport_changed")
    if fresh.advice.status != "recommended":
        reasons.add("fresh_admission_blocked")

    ordered_reasons = tuple(sorted(reasons))
    return AdaptiveCellExecutionPreviewReceipt(
        source_advisor_receipt_sha256=source.digest,
        source_request_sha256=source.request.digest,
        fresh_advisor_receipt_sha256=fresh.digest,
        fresh_request_sha256=fresh.request.digest,
        policy_sha256=policy.digest,
        evaluated_at=fresh.request.evaluated_at,
        source_selected_cell_id=source_cell,
        fresh_selected_cell_id=fresh_cell,
        source_passport_sha256=source_passport,
        fresh_passport_sha256=fresh_passport,
        fresh_resource_snapshot_sha256=fresh.advice.resource_snapshot_sha256,
        status=("admission_passed" if not ordered_reasons else "admission_blocked"),
        reason_codes=ordered_reasons,
        task_chars=fresh.task_chars,
    )


def _adaptive_request_from_payload(raw: object) -> AdaptiveRequest:
    data = _strict(raw, _field_names(AdaptiveRequest), "adaptive request")
    return AdaptiveRequest(
        exact_request_fingerprint=_sha(
            data["exact_request_fingerprint"], "exact_request_fingerprint"
        ),
        intent_family_sha256=_optional_raw_string(
            data["intent_family_sha256"], "intent_family_sha256"
        ),
        demand=_workload_demand_from_payload(data["demand"]),
        evaluation_contract_sha256=_sha(
            data["evaluation_contract_sha256"], "evaluation_contract_sha256"
        ),
        profile=_string(data["profile"], "profile"),
        evaluated_at=_timestamp(data["evaluated_at"], "evaluated_at"),
        offline_required=_boolean(data["offline_required"], "offline_required"),
        digest=_sha(data["digest"], "request digest"),
        schema_version=_string(data["schema_version"], "schema_version"),
    )


def _workload_demand_from_payload(raw: object) -> WorkloadDemand:
    data = _strict(raw, _field_names(WorkloadDemand), "workload demand")
    return WorkloadDemand(
        workload_id=_string(data["workload_id"], "workload_id"),
        capabilities=_raw_identifier_list(data["capabilities"], "capabilities"),
        tool_surfaces=_raw_identifier_list(data["tool_surfaces"], "tool_surfaces"),
        risk_class=_string(data["risk_class"], "risk_class"),
        context_tokens=_integer(data["context_tokens"], "context_tokens"),
        digest=_sha(data["digest"], "demand digest"),
        schema_version=_string(data["schema_version"], "schema_version"),
    )


def _adaptive_advice_from_payload(raw: object) -> AdaptiveAdvice:
    data = _strict(raw, _field_names(AdaptiveAdvice), "adaptive advice")
    candidates_raw = data["candidates"]
    if (
        not isinstance(candidates_raw, list)
        or not candidates_raw
        or len(candidates_raw) > MAX_CELLS
    ):
        raise AdaptiveExecutionGateError(
            "source_receipt_invalid",
            "Adaptive advice candidates are outside safe bounds.",
        )
    return AdaptiveAdvice(
        catalog_sha256=_sha(data["catalog_sha256"], "catalog_sha256"),
        request_sha256=_sha(data["request_sha256"], "request_sha256"),
        resource_snapshot_sha256=_sha(
            data["resource_snapshot_sha256"], "resource_snapshot_sha256"
        ),
        evaluated_at=_timestamp(data["evaluated_at"], "evaluated_at"),
        profile=_string(data["profile"], "profile"),
        status=_string(data["status"], "status"),
        selected_cell_id=_optional_raw_string(
            data["selected_cell_id"], "selected_cell_id"
        ),
        candidates=tuple(_candidate_from_payload(item) for item in candidates_raw),
        reason_codes=_raw_identifier_list(data["reason_codes"], "reason_codes"),
        applied=_boolean(data["applied"], "applied"),
        authorizes_execution=_boolean(
            data["authorizes_execution"], "authorizes_execution"
        ),
        network_used=_boolean(data["network_used"], "network_used"),
        model_invocations=_integer(data["model_invocations"], "model_invocations"),
        digest=_sha(data["digest"], "advice digest"),
        contract=_string(data["contract"], "advice contract"),
        schema_version=_string(data["schema_version"], "schema_version"),
    )


def _candidate_from_payload(raw: object) -> CandidateAssessment:
    data = _strict(raw, _field_names(CandidateAssessment), "candidate assessment")
    return CandidateAssessment(
        cell_id=_string(data["cell_id"], "cell_id"),
        passport_sha256=_sha(data["passport_sha256"], "passport_sha256"),
        hard_eligible=_boolean(data["hard_eligible"], "hard_eligible"),
        pareto_eligible=_boolean(data["pareto_eligible"], "pareto_eligible"),
        rejection_codes=_raw_identifier_list(
            data["rejection_codes"], "rejection_codes"
        ),
        success_rate=_optional_number(data["success_rate"], "success_rate"),
        p95_latency_ms=_optional_number(data["p95_latency_ms"], "p95_latency_ms"),
        memory_pool=_optional_raw_string(data["memory_pool"], "memory_pool"),
        placement=_optional_raw_string(data["placement"], "placement"),
        effective_peak_host_memory_bytes=_optional_integer(
            data["effective_peak_host_memory_bytes"],
            "effective_peak_host_memory_bytes",
        ),
        effective_peak_unified_memory_bytes=_optional_integer(
            data["effective_peak_unified_memory_bytes"],
            "effective_peak_unified_memory_bytes",
        ),
        effective_peak_accelerator_memory_bytes=_optional_integer(
            data["effective_peak_accelerator_memory_bytes"],
            "effective_peak_accelerator_memory_bytes",
        ),
        utility=_optional_number(data["utility"], "utility"),
        digest=_sha(data["digest"], "candidate digest"),
        schema_version=_string(data["schema_version"], "schema_version"),
    )


def _selected_passport(advice: AdaptiveAdvice) -> str | None:
    selected = advice.selected_cell_id
    if selected is None:
        return None
    matches = [
        item.passport_sha256 for item in advice.candidates if item.cell_id == selected
    ]
    if len(matches) != 1:
        raise AdaptiveExecutionGateError(
            "fresh_admission_invalid", "Selected cell passport could not be verified."
        )
    return matches[0]


def _load_json_object(
    path: str | Path,
    *,
    maximum_bytes: int,
    label: str,
    code: str,
) -> dict[str, object]:
    try:
        content = read_bounded_regular_file(
            path,
            maximum_bytes=maximum_bytes,
            label=label,
        )
        raw = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except AdaptiveExecutionGateError:
        raise
    except (
        CellContractError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise AdaptiveExecutionGateError(
            code, f"{label.capitalize()} is invalid."
        ) from exc
    if not isinstance(raw, dict):
        raise AdaptiveExecutionGateError(
            code, f"{label.capitalize()} must be an object."
        )
    return raw


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AdaptiveExecutionGateError(
                "json_duplicate_key", "Duplicate JSON keys are not allowed."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    del value
    raise AdaptiveExecutionGateError(
        "json_non_finite", "Non-finite JSON numbers are not allowed."
    )


def _field_names(cls: type[object]) -> set[str]:
    return {item.name for item in fields(cls)}


def _strict(raw: object, allowed: set[str], label: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label.capitalize()} must be an object."
        )
    data = dict(raw)
    if any(not isinstance(key, str) for key in data):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label.capitalize()} field names must be strings."
        )
    unknown = sorted(set(data) - allowed)
    missing = sorted(allowed - set(data))
    if unknown:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"Unknown {label} fields are not allowed."
        )
    if missing:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"Missing {label} fields are not allowed."
        )
    return data


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a string."
        )
    return value


def _optional_raw_string(value: object, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a boolean."
        )
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a non-negative integer."
        )
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _optional_number(value: object, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be numeric."
        )
    if not math.isfinite(float(value)):
        raise AdaptiveExecutionGateError("contract_invalid", f"{label} must be finite.")
    return value


def _raw_identifier_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a JSON array."
        )
    return _identifier_list(value, label)


def _identifier_list(
    value: object,
    label: str,
    *,
    non_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise AdaptiveExecutionGateError("contract_invalid", f"{label} must be a list.")
    items = tuple(_identifier(item, label) for item in value)
    if len(items) != len(set(items)):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must not contain duplicates."
        )
    if items != tuple(sorted(items)):
        raise AdaptiveExecutionGateError("contract_invalid", f"{label} must be sorted.")
    if non_empty and not items:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be non-empty."
        )
    return items


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must contain strings."
        )
    try:
        return require_safe_id(value, label)
    except VerifiedRoutingError as exc:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} contains an invalid identifier."
        ) from exc


def _optional_identifier(value: object, label: str) -> str | None:
    return None if value is None else _identifier(value, label)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a SHA-256 string."
        )
    try:
        return require_sha256(value, label)
    except VerifiedRoutingError as exc:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a lowercase SHA-256 digest."
        ) from exc


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a UTC timestamp."
        )
    try:
        normalized = require_utc_timestamp(value, label)
    except VerifiedRoutingError as exc:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must be a UTC timestamp."
        ) from exc
    if normalized != value:
        raise AdaptiveExecutionGateError(
            "contract_invalid", f"{label} must use canonical UTC form."
        )
    return normalized


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _content_digest(value: object, content: object, label: str) -> str:
    expected = sha256_json(content)
    if value not in (None, "") and _sha(value, label) != expected:
        raise AdaptiveExecutionGateError(
            "contract_digest_invalid", f"{label} does not match its content."
        )
    return expected


__all__ = [
    "EXECUTION_POLICY_CONTRACT",
    "EXECUTION_POLICY_MODE",
    "MAX_ADVISOR_RECEIPT_BYTES",
    "MAX_EXECUTION_POLICY_BYTES",
    "MAX_PREVIEW_RECEIPT_BYTES",
    "MAX_SOURCE_RECEIPT_AGE_SECONDS",
    "PREVIEW_RECEIPT_CONTRACT",
    "PREVIEW_REASON_CODES",
    "PREVIEW_STATUSES",
    "AdaptiveCellExecutionPolicy",
    "AdaptiveCellExecutionPreviewReceipt",
    "AdaptiveExecutionGateError",
    "adaptive_advisor_receipt_from_payload",
    "adaptive_cell_execution_preview_receipt_from_payload",
    "adaptive_execution_policy_from_payload",
    "load_adaptive_advisor_receipt",
    "load_adaptive_execution_policy",
    "preview_cell_execution",
]
