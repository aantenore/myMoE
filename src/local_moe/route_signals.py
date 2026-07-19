from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    DIFFICULTIES,
    VerifiedRoutingError,
    reject_unknown,
    require_finite_number,
    require_identifier_tuple,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
)


_SIGNAL_FIELDS = {
    "abstained",
    "capabilities",
    "confidence",
    "constraint_count",
    "context_tokens",
    "contract",
    "difficulty",
    "objective_chars",
    "request_fingerprint",
    "schema_version",
    "source",
    "tool_count",
}
_TASK_FIELDS = {
    "allow_remote",
    "allow_remote_workspace",
    "capability_demand",
    "constraint_count",
    "max_premium_calls",
    "no_change_expected",
    "objective_chars",
    "objective_sha256",
    "profile",
    "required_verifier_ids",
    "task_fingerprint",
    "task_id",
}
_DEMAND_FIELDS = {"required", "risk_class", "tools"}
_RECEIPT_FIELDS = {
    "config_sha256",
    "contract",
    "expected_flow",
    "local_gaps",
    "local_provider",
    "local_runtime",
    "premium_call_budget",
    "premium_gaps",
    "premium_provider",
    "premium_runtime",
    "rationale_codes",
    "receipt_id",
    "remote_allowed",
    "route",
    "schema_version",
    "task",
    "workspace",
}


@dataclass(frozen=True)
class TaskSignals:
    request_fingerprint: str
    capabilities: tuple[str, ...]
    difficulty: str
    confidence: float
    abstained: bool
    source: str
    objective_chars: int = 0
    context_tokens: int = 0
    constraint_count: int = 0
    tool_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_fingerprint",
            _sha256_string(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(_identifier_tuple(self.capabilities, "capabilities"))),
        )
        if not isinstance(self.difficulty, str) or self.difficulty not in DIFFICULTIES:
            raise VerifiedRoutingError(
                f"difficulty must be one of: {', '.join(DIFFICULTIES)}."
            )
        object.__setattr__(
            self,
            "confidence",
            require_finite_number(
                self.confidence,
                "confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if not isinstance(self.abstained, bool):
            raise VerifiedRoutingError("abstained must be boolean.")
        object.__setattr__(self, "source", _safe_string(self.source, "source"))
        for name in (
            "objective_chars",
            "context_tokens",
            "constraint_count",
            "tool_count",
        ):
            object.__setattr__(
                self,
                name,
                require_non_negative_int(getattr(self, name), name),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_VERSION,
            "contract": "TaskSignals",
            "request_fingerprint": self.request_fingerprint,
            "capabilities": list(self.capabilities),
            "difficulty": self.difficulty,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "source": self.source,
            "objective_chars": self.objective_chars,
            "context_tokens": self.context_tokens,
            "constraint_count": self.constraint_count,
            "tool_count": self.tool_count,
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> TaskSignals:
        payload = _strict_mapping(raw, "TaskSignals")
        reject_unknown(payload, _SIGNAL_FIELDS, "TaskSignals")
        _require_fields(payload, _SIGNAL_FIELDS, "TaskSignals")
        if payload["contract"] != "TaskSignals":
            raise VerifiedRoutingError("TaskSignals contract is invalid.")
        if payload["schema_version"] != CONTRACT_VERSION:
            raise VerifiedRoutingError("TaskSignals schema_version is unsupported.")
        return cls(
            request_fingerprint=payload["request_fingerprint"],  # type: ignore[arg-type]
            capabilities=payload["capabilities"],  # type: ignore[arg-type]
            difficulty=payload["difficulty"],  # type: ignore[arg-type]
            confidence=payload["confidence"],  # type: ignore[arg-type]
            abstained=payload["abstained"],  # type: ignore[arg-type]
            source=payload["source"],  # type: ignore[arg-type]
            objective_chars=payload["objective_chars"],  # type: ignore[arg-type]
            context_tokens=payload["context_tokens"],  # type: ignore[arg-type]
            constraint_count=payload["constraint_count"],  # type: ignore[arg-type]
            tool_count=payload["tool_count"],  # type: ignore[arg-type]
        )


class TaskSignalProvider(Protocol):
    def signals_from_metadata(
        self,
        task_metadata: Mapping[str, object],
        *,
        context_tokens: int | None = None,
    ) -> TaskSignals: ...


@dataclass(frozen=True)
class MetadataTaskSignalProvider:
    source: str = "task-metadata-v1"
    objective_char_maxima: tuple[int, int, int] = (800, 3000, 10000)
    context_token_maxima: tuple[int, int, int] = (2048, 8192, 32768)
    constraint_maxima: tuple[int, int, int] = (1, 4, 10)
    tool_maxima: tuple[int, int, int] = (0, 2, 5)
    capability_maxima: tuple[int, int, int] = (1, 3, 6)
    risk_difficulties: tuple[tuple[str, str], ...] = (
        ("read_only", "simple"),
        ("compute_only", "medium"),
        ("write_local", "medium"),
        ("write_external", "complex"),
        ("destructive", "very_complex"),
        ("privileged", "very_complex"),
    )
    confidence_with_capabilities: float = 0.9
    confidence_without_capabilities: float = 0.35
    minimum_confidence: float = 0.6
    out_of_distribution_confidence: float = 0.0
    objective_chars_per_context_token: int = 4
    max_objective_chars: int = 200000
    max_context_tokens: int = 262144
    max_constraint_count: int = 128
    max_tool_count: int = 64
    max_capability_count: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _safe_string(self.source, "source"))
        for name in (
            "objective_char_maxima",
            "context_token_maxima",
            "constraint_maxima",
            "tool_maxima",
            "capability_maxima",
        ):
            object.__setattr__(self, name, _thresholds(getattr(self, name), name))
        risks: list[tuple[str, str]] = []
        seen: set[str] = set()
        for risk, difficulty in self.risk_difficulties:
            safe_risk = _safe_string(risk, "risk_difficulties")
            if safe_risk in seen or difficulty not in DIFFICULTIES:
                raise VerifiedRoutingError("risk_difficulties is invalid.")
            seen.add(safe_risk)
            risks.append((safe_risk, difficulty))
        object.__setattr__(self, "risk_difficulties", tuple(sorted(risks)))
        for name in (
            "confidence_with_capabilities",
            "confidence_without_capabilities",
            "minimum_confidence",
            "out_of_distribution_confidence",
        ):
            object.__setattr__(
                self,
                name,
                require_finite_number(
                    getattr(self, name), name, minimum=0.0, maximum=1.0
                ),
            )
        for name in (
            "objective_chars_per_context_token",
            "max_objective_chars",
            "max_context_tokens",
            "max_constraint_count",
            "max_tool_count",
            "max_capability_count",
        ):
            value = require_non_negative_int(getattr(self, name), name)
            if name == "objective_chars_per_context_token" and value == 0:
                raise VerifiedRoutingError(
                    "objective_chars_per_context_token must be positive."
                )
            object.__setattr__(self, name, value)

    def signals_from_metadata(
        self,
        task_metadata: Mapping[str, object],
        *,
        context_tokens: int | None = None,
    ) -> TaskSignals:
        task = _strict_mapping(task_metadata, "task metadata")
        reject_unknown(task, _TASK_FIELDS, "task metadata")
        _require_fields(
            task,
            {"task_fingerprint", "objective_chars", "capability_demand", "constraint_count"},
            "task metadata",
        )
        demand = _strict_mapping(task["capability_demand"], "capability_demand")
        reject_unknown(demand, _DEMAND_FIELDS, "capability_demand")
        _require_fields(demand, _DEMAND_FIELDS, "capability_demand")

        request_fingerprint = _sha256_string(
            task["task_fingerprint"], "task_fingerprint"
        )
        objective_chars = require_non_negative_int(
            task["objective_chars"], "objective_chars"
        )
        constraint_count = require_non_negative_int(
            task["constraint_count"], "constraint_count"
        )
        capabilities = tuple(
            sorted(_identifier_tuple(demand["required"], "capability_demand.required"))
        )
        tools = _identifier_tuple(demand["tools"], "capability_demand.tools")
        risk_class = _safe_string(demand["risk_class"], "capability_demand.risk_class")

        estimated_context = (
            objective_chars + self.objective_chars_per_context_token - 1
        ) // self.objective_chars_per_context_token
        if context_tokens is None:
            effective_context = estimated_context
        else:
            effective_context = require_non_negative_int(context_tokens, "context_tokens")
            if effective_context < estimated_context:
                raise VerifiedRoutingError(
                    "context_tokens cannot be lower than the structural estimate."
                )

        risk_map = dict(self.risk_difficulties)
        out_of_distribution = (
            risk_class not in risk_map
            or objective_chars > self.max_objective_chars
            or effective_context > self.max_context_tokens
            or constraint_count > self.max_constraint_count
            or len(tools) > self.max_tool_count
            or len(capabilities) > self.max_capability_count
        )
        levels = (
            _difficulty_index(objective_chars, self.objective_char_maxima),
            _difficulty_index(effective_context, self.context_token_maxima),
            _difficulty_index(constraint_count, self.constraint_maxima),
            _difficulty_index(len(tools), self.tool_maxima),
            _difficulty_index(len(capabilities), self.capability_maxima),
            DIFFICULTIES.index(risk_map.get(risk_class, "very_complex")),
        )
        difficulty = DIFFICULTIES[max(levels)]
        confidence = (
            self.confidence_with_capabilities
            if capabilities
            else self.confidence_without_capabilities
        )
        if out_of_distribution:
            confidence = min(confidence, self.out_of_distribution_confidence)
            difficulty = "very_complex"
        confidence = round(confidence, 6)
        return TaskSignals(
            request_fingerprint=request_fingerprint,
            capabilities=capabilities,
            difficulty=difficulty,
            confidence=confidence,
            abstained=out_of_distribution or confidence < self.minimum_confidence,
            source=self.source,
            objective_chars=objective_chars,
            context_tokens=effective_context,
            constraint_count=constraint_count,
            tool_count=len(tools),
        )


def signals_from_route_receipt(
    route_receipt: Mapping[str, object] | object,
    provider: TaskSignalProvider | None = None,
    *,
    context_tokens: int | None = None,
) -> TaskSignals:
    raw: object = route_receipt
    if not isinstance(raw, Mapping):
        payload_method = getattr(raw, "payload", None)
        if not callable(payload_method):
            raise VerifiedRoutingError(
                "route receipt must be a mapping or expose payload()."
            )
        raw = payload_method()
    receipt = _strict_mapping(raw, "route receipt")
    reject_unknown(receipt, _RECEIPT_FIELDS, "route receipt")
    if receipt.get("contract") != "RouteDecisionReceipt":
        raise VerifiedRoutingError("route receipt contract is invalid.")
    if "task" not in receipt:
        raise VerifiedRoutingError("route receipt is missing task metadata.")
    task = _strict_mapping(receipt["task"], "task metadata")
    selected = provider or MetadataTaskSignalProvider()
    signals = selected.signals_from_metadata(task, context_tokens=context_tokens)
    expected = _sha256_string(task.get("task_fingerprint"), "task_fingerprint")
    if signals.request_fingerprint != expected:
        raise VerifiedRoutingError("task signals are bound to a different request.")
    return signals


def _strict_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerifiedRoutingError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise VerifiedRoutingError(f"{label} keys must be strings.")
    return dict(value)


def _require_fields(raw: Mapping[str, object], required: set[str], label: str) -> None:
    missing = sorted(required.difference(raw))
    if missing:
        raise VerifiedRoutingError(f"Missing {label} fields: {', '.join(missing)}.")


def _safe_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise VerifiedRoutingError(f"{label} must be a string.")
    return require_safe_id(value, label)


def _sha256_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise VerifiedRoutingError(f"{label} must be a string.")
    return require_sha256(value, label)


def _identifier_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise VerifiedRoutingError(f"{label} must contain only strings.")
    return require_identifier_tuple(value, label)


def _thresholds(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise VerifiedRoutingError(f"{label} must contain three thresholds.")
    result = tuple(require_non_negative_int(item, label) for item in value)
    if result != tuple(sorted(result)):
        raise VerifiedRoutingError(f"{label} must be ordered.")
    return result  # type: ignore[return-value]


def _difficulty_index(value: int, maxima: tuple[int, int, int]) -> int:
    for index, maximum in enumerate(maxima):
        if value <= maximum:
            return index
    return len(DIFFICULTIES) - 1
