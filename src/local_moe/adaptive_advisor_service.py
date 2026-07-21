from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from .adaptive_selector import (
    AdaptiveAdvice,
    AdaptiveRequest,
    advise_cell,
    build_adaptive_request,
)
from .cell_contracts import CellContractError
from .cell_passport import load_cell_catalog
from .resource_snapshot import ResourceSnapshot, collect_resource_snapshot
from .secure_files import read_bounded_regular_file
from .verified_routing_contracts import CONTRACT_VERSION, now_utc, sha256_json


ADVISOR_RECEIPT_CONTRACT = "AdaptiveCellAdvisorReceipt"
MAX_EVALUATION_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_TASK_BYTES = 256 * 1024
MAX_TASK_CHARS = 128 * 1024
DISPLAY_STATES = frozenset(
    {"recommended_now", "not_available_now", "not_enough_evidence"}
)

# These codes describe a verified, present-tense incompatibility or resource
# shortage. Evidence gaps, stale evidence, unknowns, generic aggregation codes,
# and any code introduced in the future deliberately remain outside this set.
_CONCLUSIVE_BLOCKERS = frozenset(
    {
        "accelerator_memory_headroom_insufficient",
        "capability_gap",
        "context_window_exceeded",
        "harness_identity_mismatch",
        "harness_unavailable",
        "host_memory_headroom_insufficient",
        "machine_not_supported",
        "model_identity_mismatch",
        "model_unavailable",
        "offline_not_supported",
        "quality_floor_not_met",
        "risk_class_not_supported",
        "runtime_identity_mismatch",
        "runtime_unavailable",
        "swap_limit_exceeded",
        "system_not_supported",
        "tool_contract_identity_mismatch",
        "tool_contract_unavailable",
        "tool_surface_gap",
        "unified_memory_headroom_insufficient",
        "unified_memory_unavailable",
    }
)
_NON_BLOCKING_ADVICE_CODES = frozenset({"advisory_only", "no_eligible_cell"})


class AdvisorServiceError(ValueError):
    """Stable, task-safe failure at the shared advisor boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AdaptiveAdvisorReceipt:
    """Content-addressed, non-authorizing record of one advisor evaluation."""

    request: AdaptiveRequest
    advice: AdaptiveAdvice
    task_chars: int
    display_state: str
    digest: str = ""
    contract: str = ADVISOR_RECEIPT_CONTRACT
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract != ADVISOR_RECEIPT_CONTRACT:
            raise AdvisorServiceError(
                "receipt_invalid", "Advisor receipt contract is unsupported."
            )
        if self.schema_version != CONTRACT_VERSION:
            raise AdvisorServiceError(
                "receipt_invalid", "Advisor receipt schema is unsupported."
            )
        if not isinstance(self.request, AdaptiveRequest) or not isinstance(
            self.advice, AdaptiveAdvice
        ):
            raise AdvisorServiceError(
                "receipt_invalid", "Advisor receipt members are invalid."
            )
        if (
            isinstance(self.task_chars, bool)
            or not isinstance(self.task_chars, int)
            or not 0 < self.task_chars <= MAX_TASK_CHARS
        ):
            raise AdvisorServiceError(
                "receipt_invalid", "Advisor receipt task size is invalid."
            )

        expected_state = _classify_display_state(self.advice)
        if self.display_state not in DISPLAY_STATES or self.display_state != expected_state:
            raise AdvisorServiceError(
                "receipt_invalid", "Advisor receipt display state is invalid."
            )
        if self.advice.request_sha256 != self.request.digest:
            raise AdvisorServiceError(
                "receipt_binding_invalid", "Advice is not bound to its request."
            )
        if (
            self.advice.profile != self.request.profile
            or self.advice.evaluated_at != self.request.evaluated_at
            or self.advice.schema_version != self.request.schema_version
        ):
            raise AdvisorServiceError(
                "receipt_binding_invalid", "Advice metadata is not bound to its request."
            )

        expected_digest = sha256_json(self.content_payload())
        if self.digest not in (None, "") and self.digest != expected_digest:
            raise AdvisorServiceError(
                "receipt_digest_invalid", "Advisor receipt digest does not match."
            )
        object.__setattr__(self, "digest", expected_digest)

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "request": self.request.payload(),
            "advice": self.advice.payload(),
            "task_chars": self.task_chars,
            "display_state": self.display_state,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def evaluate_advisor(
    *,
    catalog_path: str | Path,
    evaluation_contract_path: str | Path,
    task_text: str,
    workload_id: str,
    required_capabilities: Iterable[str],
    required_tool_surfaces: Iterable[str],
    risk_class: str,
    context_tokens: int,
    profile: str,
    intent_family_sha256: str | None = None,
) -> AdaptiveAdvisorReceipt:
    """Evaluate one local cell request without invoking or changing a runtime."""

    task_bytes, task_chars = _validated_task(task_text)
    exact_request_fingerprint = hashlib.sha256(task_bytes).hexdigest()
    del task_bytes

    capabilities = _validated_items(
        required_capabilities, "required capabilities", non_empty=True
    )
    tool_surfaces = _validated_items(
        required_tool_surfaces, "required tool surfaces", non_empty=False
    )
    try:
        evaluation_contract = read_bounded_regular_file(
            evaluation_contract_path,
            maximum_bytes=MAX_EVALUATION_CONTRACT_BYTES,
            label="evaluation contract",
        )
    except Exception as exc:
        raise AdvisorServiceError(
            "evaluation_contract_invalid",
            "Evaluation contract could not be verified.",
        ) from exc
    if not evaluation_contract:
        raise AdvisorServiceError(
            "evaluation_contract_invalid", "Evaluation contract must not be empty."
        )
    evaluation_contract_sha256 = hashlib.sha256(evaluation_contract).hexdigest()
    del evaluation_contract

    try:
        catalog = load_cell_catalog(catalog_path)
    except Exception as exc:
        raise AdvisorServiceError(
            "catalog_invalid", "Adaptive cell catalog could not be verified."
        ) from exc

    try:
        resource_snapshot = collect_resource_snapshot()
        if not isinstance(resource_snapshot, ResourceSnapshot):
            raise CellContractError("snapshot must be a ResourceSnapshot.")
    except Exception as exc:
        raise AdvisorServiceError(
            "resource_snapshot_invalid",
            "Local resource snapshot could not be verified.",
        ) from exc

    # This timestamp is captured after the hardware probe, so the receipt
    # truthfully describes when the selection was evaluated. Deterministic
    # tests replace the clock and probe functions rather than supplying dates
    # through the public API.
    try:
        evaluation_time = now_utc()
    except Exception as exc:
        raise AdvisorServiceError(
            "evaluation_time_invalid", "Evaluation time could not be established."
        ) from exc
    try:
        request = build_adaptive_request(
            exact_request_fingerprint=exact_request_fingerprint,
            intent_family_sha256=intent_family_sha256,
            workload_id=workload_id,
            required_capabilities=capabilities,
            required_tool_surfaces=tool_surfaces,
            risk_class=risk_class,
            required_context_tokens=context_tokens,
            evaluation_contract_sha256=evaluation_contract_sha256,
            profile=profile,
            evaluated_at=evaluation_time,
        )
        advice = advise_cell(catalog, resource_snapshot, request)
        receipt = AdaptiveAdvisorReceipt(
            request=request,
            advice=advice,
            task_chars=task_chars,
            display_state=_classify_display_state(advice),
        )
    except AdvisorServiceError:
        raise
    except Exception as exc:
        raise AdvisorServiceError(
            "advisor_evaluation_invalid",
            "Adaptive advisor evaluation could not be completed.",
        ) from exc

    if advice.catalog_sha256 != catalog.digest:
        raise AdvisorServiceError(
            "receipt_binding_invalid", "Advice is not bound to its catalog."
        )
    if advice.resource_snapshot_sha256 != resource_snapshot.digest:
        raise AdvisorServiceError(
            "receipt_binding_invalid", "Advice is not bound to its resource snapshot."
        )
    return receipt


def advisor_presentation_payload(
    receipt: AdaptiveAdvisorReceipt,
) -> dict[str, object]:
    """Return a deterministic, non-technical view plus the technical receipt."""

    if not isinstance(receipt, AdaptiveAdvisorReceipt):
        raise AdvisorServiceError(
            "receipt_invalid", "Advisor presentation requires a verified receipt."
        )
    if receipt.display_state == "recommended_now":
        title = "Recommended now"
        summary = (
            f"Cell {receipt.advice.selected_cell_id} passed the declared verification "
            "boundaries for this request."
        )
        badges = [
            str(receipt.advice.selected_cell_id),
            receipt.request.profile,
            "read-only",
        ]
    elif receipt.display_state == "not_available_now":
        title = "Not available now"
        summary = (
            "Current verified boundaries rule out every declared cell for this request."
        )
        badges = [
            receipt.request.profile,
            f"{len(receipt.advice.candidates)} cells checked",
            "read-only",
        ]
    else:
        title = "Not enough evidence"
        summary = (
            "Missing, stale, or incompatible evidence prevents a safe recommendation."
        )
        badges = [
            receipt.request.profile,
            f"{len(receipt.advice.candidates)} cells checked",
            "read-only",
        ]
    return {
        "display_state": receipt.display_state,
        "title": title,
        "summary": summary,
        "badges": badges[:3],
        "receipt": receipt.payload(),
    }


def _validated_task(task_text: object) -> tuple[bytes, int]:
    if not isinstance(task_text, str) or not task_text.strip():
        raise AdvisorServiceError("task_invalid", "Task input must not be empty.")
    task_chars = len(task_text)
    if task_chars > MAX_TASK_CHARS:
        raise AdvisorServiceError(
            "task_too_large", "Task input exceeds the character limit."
        )
    try:
        encoded = task_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AdvisorServiceError(
            "task_invalid", "Task input must be valid UTF-8 text."
        ) from exc
    if len(encoded) > MAX_TASK_BYTES:
        raise AdvisorServiceError("task_too_large", "Task input exceeds the byte limit.")
    return encoded, task_chars


def _validated_items(
    values: Iterable[str], label: str, *, non_empty: bool
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AdvisorServiceError(
            "request_invalid", f"{label.capitalize()} must be a sequence."
        )
    try:
        items = tuple(values)
    except (TypeError, ValueError) as exc:
        raise AdvisorServiceError(
            "request_invalid", f"{label.capitalize()} must be a sequence."
        ) from exc
    if non_empty and not items:
        raise AdvisorServiceError(
            "request_invalid", f"{label.capitalize()} must not be empty."
        )
    return items


def _classify_display_state(advice: AdaptiveAdvice) -> str:
    if advice.status == "recommended":
        return "recommended_now"
    detailed_codes = {
        code
        for candidate in advice.candidates
        for code in candidate.rejection_codes
    }
    detailed_codes.update(
        code for code in advice.reason_codes if code not in _NON_BLOCKING_ADVICE_CODES
    )
    if detailed_codes and detailed_codes.issubset(_CONCLUSIVE_BLOCKERS):
        return "not_available_now"
    return "not_enough_evidence"


__all__ = [
    "ADVISOR_RECEIPT_CONTRACT",
    "DISPLAY_STATES",
    "MAX_EVALUATION_CONTRACT_BYTES",
    "MAX_TASK_BYTES",
    "MAX_TASK_CHARS",
    "AdaptiveAdvisorReceipt",
    "AdvisorServiceError",
    "advisor_presentation_payload",
    "evaluate_advisor",
]
