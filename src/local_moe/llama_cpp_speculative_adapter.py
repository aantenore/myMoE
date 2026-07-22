"""Bounded llama.cpp response adapter for speculative-cell evidence."""

from __future__ import annotations

from typing import Any

from .speculative_cell_contracts import (
    SpeculativeArmMeasurement,
    SpeculativeCellContractError,
)
from .verified_routing_contracts import require_non_negative_int, sha256_json


LLAMA_CPP_SPECULATIVE_ADAPTER_ID = "mymoe.llama_cpp.speculative.v1"
LLAMA_CPP_SPECULATIVE_ADAPTER_CONTRACT = {
    "schema_version": "1.0",
    "adapter_id": LLAMA_CPP_SPECULATIVE_ADAPTER_ID,
    "transport": "operator_managed_numeric_loopback",
    "request_order": "preregistered_ab_ba",
    "response_surface": "completed_openai_chat_or_completion",
    "ttft_source": "host_monotonic_first_content_event",
    "total_latency_source": "host_monotonic_complete_response",
    "memory_source": "host_observed_exact_runtime_process_tree",
    "server_metrics": [
        "predicted_n",
        "predicted_ms",
        "draft_n",
        "draft_n_accepted",
    ],
    "output_equivalence": "canonical_text_envelope_sha256",
    "content_retention": "semantic_envelope_sha256_only",
    "network_scope": "loopback_only",
    "starts_or_stops_runtime": False,
    "activation_authority": False,
}
LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256 = sha256_json(
    LLAMA_CPP_SPECULATIVE_ADAPTER_CONTRACT
)


def parse_llama_cpp_completion(
    payload: object,
    *,
    cell_sha256: str,
    ttft_ms: float,
    total_latency_ms: float,
    peak_memory_bytes: int,
) -> SpeculativeArmMeasurement:
    """Convert one llama.cpp completion into payload-free arm evidence.

    The caller owns loopback transport and host measurement. This adapter keeps
    only a digest of generated content and cross-checks llama.cpp timing counts
    against the standard usage object.
    """

    try:
        root = _record(payload, "completion response")
        choices = root.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise SpeculativeCellContractError(
                "Completion response must contain exactly one choice."
            )
        choice = _record(choices[0], "completion choice")
        output_sha256 = _completion_output_sha256(choice)
        timings = _record(root.get("timings"), "llama.cpp timings")
        usage = _record(root.get("usage"), "completion usage")
        predicted_tokens = _positive_int(timings.get("predicted_n"), "predicted_n")
        completion_tokens = _positive_int(
            usage.get("completion_tokens"), "completion_tokens"
        )
        if predicted_tokens != completion_tokens:
            raise SpeculativeCellContractError(
                "llama.cpp timings and usage token counts disagree."
            )
        predicted_ms = _positive_number(timings.get("predicted_ms"), "predicted_ms")
        draft_generated = _optional_counter(timings.get("draft_n"), "draft_n")
        draft_accepted = _optional_counter(
            timings.get("draft_n_accepted"), "draft_n_accepted"
        )
        if (draft_generated is None) != (draft_accepted is None):
            raise SpeculativeCellContractError(
                "llama.cpp draft counters must be supplied together."
            )
        return SpeculativeArmMeasurement(
            cell_sha256=cell_sha256,
            success=True,
            output_sha256=output_sha256,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            predicted_tokens=predicted_tokens,
            predicted_ms=predicted_ms,
            peak_memory_bytes=peak_memory_bytes,
            draft_generated_tokens=draft_generated,
            draft_accepted_tokens=draft_accepted,
        )
    except SpeculativeCellContractError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise SpeculativeCellContractError(
            "llama.cpp completion response is invalid."
        ) from exc


def llama_cpp_failure_measurement(
    *,
    cell_sha256: str,
    error_code: str,
) -> SpeculativeArmMeasurement:
    """Create a content-free failed arm without retaining provider errors."""

    return SpeculativeArmMeasurement(
        cell_sha256=cell_sha256,
        success=False,
        error_code=error_code,
    )


def _completion_output_sha256(choice: dict[str, Any]) -> str:
    text = choice.get("text")
    message = choice.get("message")
    if text is not None and message is not None:
        raise SpeculativeCellContractError(
            "Completion choice cannot contain both text and message."
        )
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise SpeculativeCellContractError(
            "Text-only completion requires a terminal text finish reason."
        )
    if message is not None:
        allowed_choice_keys = {"index", "message", "finish_reason", "logprobs"}
        if set(choice) - allowed_choice_keys:
            raise SpeculativeCellContractError(
                "Chat completion contains a non-text output surface."
            )
        rendered_message = _record(message, "completion message")
        if set(rendered_message) != {"role", "content"}:
            raise SpeculativeCellContractError(
                "Chat completion message must be text-only."
            )
        if rendered_message["role"] != "assistant":
            raise SpeculativeCellContractError(
                "Chat completion message role must be assistant."
            )
        content = rendered_message.get("content")
        surface = "chat"
    else:
        allowed_choice_keys = {"index", "text", "finish_reason", "logprobs"}
        if set(choice) - allowed_choice_keys:
            raise SpeculativeCellContractError(
                "Text completion contains a non-text output surface."
            )
        content = text
        surface = "completion"
    if not isinstance(content, str):
        raise SpeculativeCellContractError("Completion content must be text.")
    if len(content.encode("utf-8")) > 16 * 1024 * 1024:
        raise SpeculativeCellContractError("Completion content exceeds its bound.")
    return sha256_json(
        {
            "schema_version": "1.0",
            "surface": surface,
            "content": content,
            "finish_reason": finish_reason,
        }
    )


def _record(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpeculativeCellContractError(f"{label} must be an object.")
    return value


def _positive_int(value: object, label: str) -> int:
    rendered = require_non_negative_int(value, label)
    if rendered < 1:
        raise SpeculativeCellContractError(f"{label} must be positive.")
    return rendered


def _optional_counter(value: object, label: str) -> int | None:
    if value is None:
        return None
    return require_non_negative_int(value, label)


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SpeculativeCellContractError(f"{label} must be numeric.")
    rendered = float(value)
    if rendered <= 0 or rendered == float("inf") or rendered != rendered:
        raise SpeculativeCellContractError(f"{label} must be positive and finite.")
    return rendered
