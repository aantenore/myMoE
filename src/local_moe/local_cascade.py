from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Callable, Protocol
from uuid import uuid4

from .local_cascade_contracts import (
    LocalCascadeAttemptReceiptV1,
    LocalCascadeAttemptRequestV1,
    LocalCascadeAttemptResultV1,
    LocalCascadeConfigV1,
    LocalCascadeContractError,
    LocalCascadeReceiptV1,
    LocalCascadeTaskV1,
    LocalCascadeTierV1,
    LocalCascadeTokenCountV1,
    LocalCascadeVerifierV1,
    build_local_cascade_evidence_sha256,
    build_token_totals,
    sha256_json,
    sha256_text,
)


Clock = Callable[[], float]
MAX_JSON_NESTING_DEPTH = 64
MAX_JSON_NODES = 16_384


def _new_run_id() -> str:
    return f"cascade-run-{uuid4().hex}"


class LocalCascadeAttemptPort(Protocol):
    """Injectable boundary for one offline model attempt."""

    def attempt(
        self, request: LocalCascadeAttemptRequestV1
    ) -> LocalCascadeAttemptResultV1:
        """Run exactly one tier under the request's offline constraints."""


@dataclass(frozen=True)
class LocalCascadeVerification:
    passed: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class LocalCascadeRun:
    """Keep content outside metadata; deterministic digests remain correlatable."""

    content: str | None
    receipt: LocalCascadeReceiptV1


def verify_local_cascade_content(
    content: str,
    verifier: LocalCascadeVerifierV1,
) -> LocalCascadeVerification:
    """Apply only deterministic, local checks to a candidate result."""

    if not isinstance(content, str):
        raise LocalCascadeContractError("candidate content must be a string.")

    reasons: list[str] = []
    if not content.strip():
        reasons.append("empty_content")
    if len(content) < verifier.min_characters:
        reasons.append("content_too_short")
    if len(content) > verifier.max_characters:
        reasons.append("content_too_long")

    searchable = content if verifier.case_sensitive_terms else content.casefold()
    required = (
        verifier.required_terms
        if verifier.case_sensitive_terms
        else tuple(term.casefold() for term in verifier.required_terms)
    )
    forbidden = (
        verifier.forbidden_terms
        if verifier.case_sensitive_terms
        else tuple(term.casefold() for term in verifier.forbidden_terms)
    )
    if any(term not in searchable for term in required):
        reasons.append("missing_required_term")
    if any(term in searchable for term in forbidden):
        reasons.append("forbidden_term_present")

    if verifier.output_format == "json_object":
        parsed, parse_reason = _strict_json_object(content)
        if parse_reason is not None:
            reasons.append(parse_reason)
        elif parsed is not None:
            expected = {field.name: field for field in verifier.json_fields}
            present = set(parsed)
            if any(
                field.required and field.name not in present
                for field in expected.values()
            ):
                reasons.append("missing_json_field")
            if not verifier.allow_extra_json_fields and present - set(expected):
                reasons.append("unexpected_json_field")
            type_mismatch = False
            disallowed_value = False
            for name in present & set(expected):
                field = expected[name]
                value = parsed[name]
                if not _matches_json_kind(value, field.value_kind):
                    type_mismatch = True
                    continue
                if (
                    field.allowed_string_values
                    and value not in field.allowed_string_values
                ):
                    disallowed_value = True
            if type_mismatch:
                reasons.append("json_field_type_mismatch")
            if disallowed_value:
                reasons.append("json_string_value_not_allowed")

    reason_codes = tuple(dict.fromkeys(reasons))
    return LocalCascadeVerification(
        passed=not reason_codes,
        reason_codes=reason_codes,
    )


def run_local_cascade(
    task: LocalCascadeTaskV1,
    config: LocalCascadeConfigV1,
    attempt_port: LocalCascadeAttemptPort,
    *,
    clock: Clock = time.monotonic,
) -> LocalCascadeRun:
    """Run a cheapest-first sequential cascade with deterministic escalation."""

    if not isinstance(task, LocalCascadeTaskV1):
        raise LocalCascadeContractError("task must be LocalCascadeTaskV1.")
    if not isinstance(config, LocalCascadeConfigV1):
        raise LocalCascadeContractError("config must be LocalCascadeConfigV1.")
    if task.output_format != config.verifier.output_format:
        raise LocalCascadeContractError(
            "task and verifier output_format values must match."
        )
    attempt_method = getattr(attempt_port, "attempt", None)
    if not callable(attempt_method):
        raise LocalCascadeContractError(
            "attempt_port must provide a callable attempt method."
        )

    started = clock()
    prior_reasons: tuple[str, ...] = ()
    receipts: list[LocalCascadeAttemptReceiptV1] = []
    selected_content: str | None = None
    selected_tier_id: str | None = None

    for attempt_number, tier in enumerate(config.ordered_tiers, start=1):
        request = LocalCascadeAttemptRequestV1(
            task=task,
            tier=tier,
            attempt_number=attempt_number,
            verifier_reason_codes=prior_reasons,
            requested_execution_scope=config.requested_execution_scope,
        )
        attempt_started = clock()
        try:
            candidate = attempt_method(request)
        except Exception:
            candidate = None
            attempt_status = "error"
            verification_status = "escalate"
            reason_codes = ("attempt_port_error",)
            content_digest = None
            input_tokens = LocalCascadeTokenCountV1.unknown()
            output_tokens = LocalCascadeTokenCountV1.unknown()
        else:
            if not isinstance(candidate, LocalCascadeAttemptResultV1):
                attempt_status = "error"
                verification_status = "escalate"
                reason_codes = ("attempt_result_contract_error",)
                content_digest = None
                input_tokens = LocalCascadeTokenCountV1.unknown()
                output_tokens = LocalCascadeTokenCountV1.unknown()
            else:
                attempt_status = candidate.status
                input_tokens = candidate.input_tokens
                output_tokens = candidate.output_tokens
                token_limit_reasons = _token_limit_reason_codes(candidate, tier)
                if candidate.status == "abstained":
                    verification_status = "escalate"
                    reason_codes = (
                        "attempt_abstained",
                        *token_limit_reasons,
                    )
                    content_digest = None
                else:
                    assert candidate.content is not None
                    content_digest = sha256_text(candidate.content)
                    decision = verify_local_cascade_content(
                        candidate.content,
                        config.verifier,
                    )
                    reason_codes = tuple(
                        dict.fromkeys((*decision.reason_codes, *token_limit_reasons))
                    )
                    verification_status = "passed" if not reason_codes else "escalate"
                    if not reason_codes:
                        selected_content = candidate.content
                        selected_tier_id = tier.tier_id

        duration_ms = max(0.0, (clock() - attempt_started) * 1_000.0)
        receipt = LocalCascadeAttemptReceiptV1(
            attempt_number=attempt_number,
            tier_id=tier.tier_id,
            cost_rank=tier.cost_rank,
            request_sha256=sha256_json(request.payload()),
            output_sha256=content_digest,
            attempt_status=attempt_status,
            verification_status=verification_status,
            verifier_reason_codes=reason_codes,
            duration_ms=round(duration_ms, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        receipts.append(receipt)
        if selected_content is not None:
            break
        prior_reasons = reason_codes

    attempts = tuple(receipts)
    if selected_content is not None:
        status = "passed"
    elif all(item.attempt_status == "abstained" for item in attempts):
        status = "all_abstained"
    else:
        status = "exhausted"

    total_duration_ms = max(0.0, (clock() - started) * 1_000.0)
    task_sha256 = sha256_json(task.payload())
    config_sha256 = sha256_json(config.payload())
    token_totals = build_token_totals(attempts)
    requested_execution_scope = config.requested_execution_scope
    execution_scope_attestation = "adapter_declared_unverified"
    evidence_sha256 = build_local_cascade_evidence_sha256(
        task_sha256=task_sha256,
        config_sha256=config_sha256,
        status=status,
        selected_tier_id=selected_tier_id,
        attempts=attempts,
        token_totals=token_totals,
        requested_execution_scope=requested_execution_scope,
        execution_scope_attestation=execution_scope_attestation,
    )
    receipt = LocalCascadeReceiptV1(
        run_id=_new_run_id(),
        task_sha256=task_sha256,
        config_sha256=config_sha256,
        status=status,
        selected_tier_id=selected_tier_id,
        attempt_count=len(attempts),
        total_duration_ms=round(total_duration_ms, 3),
        attempts=attempts,
        token_totals=token_totals,
        evidence_sha256=evidence_sha256,
        requested_execution_scope=requested_execution_scope,
        execution_scope_attestation=execution_scope_attestation,
    )
    return LocalCascadeRun(content=selected_content, receipt=receipt)


def _strict_json_object(content: str) -> tuple[dict[str, object] | None, str | None]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        rendered = float(value)
        if not math.isfinite(rendered):
            raise ValueError("non-finite JSON number")
        return rendered

    try:
        parsed = json.loads(
            content,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, OverflowError, RecursionError, ValueError, TypeError):
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    if not _json_within_structural_limits(parsed):
        return None, "invalid_json"
    return parsed, None


def _json_within_structural_limits(value: object) -> bool:
    """Bound JSON value nodes and container depth without recursive traversal."""

    stack: list[tuple[object, int]] = [(value, 0)]
    observed_nodes = 0
    while stack:
        node, parent_container_depth = stack.pop()
        observed_nodes += 1
        if observed_nodes > MAX_JSON_NODES:
            return False
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, list):
            children = node
        else:
            continue
        container_depth = parent_container_depth + 1
        if container_depth > MAX_JSON_NESTING_DEPTH:
            return False
        stack.extend((child, container_depth) for child in children)
    return True


def _matches_json_kind(value: object, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not isinstance(value, complex)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False


def _token_limit_reason_codes(
    candidate: LocalCascadeAttemptResultV1,
    tier: LocalCascadeTierV1,
) -> tuple[str, ...]:
    reasons: list[str] = []
    limits = (
        (candidate.input_tokens, tier.max_input_tokens, "input"),
        (candidate.output_tokens, tier.max_output_tokens, "output"),
    )
    for usage, limit, direction in limits:
        if (
            usage.source != "unknown"
            and usage.count is not None
            and usage.count > limit
        ):
            reasons.append(f"{direction}_token_limit_exceeded")
    return tuple(reasons)
