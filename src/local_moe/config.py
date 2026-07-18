from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .execution_scope import (
    ExecutionDeclaration,
    ExecutionPolicy,
    ExecutionScope,
    ExecutionTarget,
    ExecutionTransport,
    normalized_execution_declaration,
    scope_rank,
)
from .path_security import read_text_file


class ConfigError(ValueError):
    """Raised when MoE configuration is invalid."""


@dataclass(frozen=True)
class ExpertConfig:
    id: str
    provider: str
    model: str
    role: str
    weight: float = 1.0
    timeout_seconds: float = 60.0
    base_url: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    execution: ExecutionDeclaration = field(default_factory=ExecutionDeclaration)

    @property
    def execution_target(self) -> ExecutionTarget:
        endpoint = str(self.base_url) if self.base_url is not None else None
        return ExecutionTarget(
            expert_id=self.id,
            provider=self.provider,
            endpoint=endpoint,
            declaration=normalized_execution_declaration(
                provider=self.provider,
                endpoint=endpoint,
                declaration=self.execution,
            ),
        )


@dataclass(frozen=True)
class SemanticRouteExample:
    expert_id: str
    utterances: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class SemanticRoutingConfig:
    enabled: bool = False
    method: str = "char_ngrams"
    min_score: float = 0.16
    margin: float = 0.02
    weight: float = 2.4
    ngram_min: int = 3
    ngram_max: int = 5
    examples: tuple[SemanticRouteExample, ...] = ()


@dataclass(frozen=True)
class DistilledRoutingConfig:
    enabled: bool = False
    artifact_path: str = ""
    min_confidence: float = 0.12
    weight: float = 2.0


@dataclass(frozen=True)
class RoutingRule:
    expert_id: str
    keywords: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class RoutingConfig:
    top_k: int = 1
    fallback_order: tuple[str, ...] = ()
    aggregation: str = "best"
    strategy: str = "rules"
    semantic: SemanticRoutingConfig = field(default_factory=SemanticRoutingConfig)
    distilled: DistilledRoutingConfig = field(default_factory=DistilledRoutingConfig)


@dataclass(frozen=True)
class MoEConfig:
    routing: RoutingConfig
    experts: tuple[ExpertConfig, ...]
    rules: tuple[RoutingRule, ...]
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    @property
    def experts_by_id(self) -> dict[str, ExpertConfig]:
        return {expert.id: expert for expert in self.experts}


def load_config(path: str | Path) -> MoEConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_config(raw)


def load_config_within(
    path: str | Path,
    *,
    allowed_roots: tuple[str | Path, ...],
) -> MoEConfig:
    """Load a config selected by an API caller within configured profile roots."""

    _, text = read_text_file(
        path,
        allowed_roots=allowed_roots,
        label="runtime profile",
        max_bytes=2 * 1024 * 1024,
    )
    return parse_config(json.loads(text))


def parse_config(raw: dict[str, Any]) -> MoEConfig:
    execution_policy = _parse_execution_policy(raw.get("execution", {}))
    routing_raw = raw.get("routing", {})
    routing = RoutingConfig(
        top_k=int(routing_raw.get("top_k", 1)),
        fallback_order=tuple(routing_raw.get("fallback_order", [])),
        aggregation=str(routing_raw.get("aggregation", "best")),
        strategy=str(routing_raw.get("strategy", "rules")),
        semantic=_parse_semantic_routing(routing_raw.get("semantic", {})),
        distilled=_parse_distilled_routing(routing_raw.get("distilled", {})),
    )

    experts = tuple(_parse_expert(item) for item in raw.get("experts", []))
    if not experts:
        raise ConfigError("At least one expert is required.")

    expert_ids = {expert.id for expert in experts}
    if len(expert_ids) != len(experts):
        raise ConfigError("Expert ids must be unique.")

    rules = tuple(_parse_rule(item) for item in raw.get("rules", []))
    for rule in rules:
        if rule.expert_id not in expert_ids:
            raise ConfigError(f"Rule references unknown expert: {rule.expert_id}")

    for fallback in routing.fallback_order:
        if fallback not in expert_ids:
            raise ConfigError(f"Fallback references unknown expert: {fallback}")

    if routing.top_k < 1:
        raise ConfigError("routing.top_k must be >= 1.")

    if routing.top_k > len(experts):
        raise ConfigError("routing.top_k cannot exceed the number of experts.")

    supported_aggregation = {"best", "concat", "compare"}
    if routing.aggregation not in supported_aggregation:
        supported = ", ".join(sorted(supported_aggregation))
        raise ConfigError(f"Unsupported aggregation '{routing.aggregation}'. Use one of: {supported}.")

    supported_strategies = {"distilled", "hybrid", "rules"}
    if routing.strategy not in supported_strategies:
        supported = ", ".join(sorted(supported_strategies))
        raise ConfigError(f"Unsupported routing strategy '{routing.strategy}'. Use one of: {supported}.")

    if routing.semantic.enabled:
        if routing.semantic.method != "char_ngrams":
            raise ConfigError("Only semantic routing method 'char_ngrams' is currently supported.")
        if routing.semantic.ngram_min < 1 or routing.semantic.ngram_max < routing.semantic.ngram_min:
            raise ConfigError("Semantic routing ngram range is invalid.")
        for example in routing.semantic.examples:
            if example.expert_id not in expert_ids:
                raise ConfigError(f"Semantic route references unknown expert: {example.expert_id}")

    if routing.distilled.enabled:
        if not routing.distilled.artifact_path:
            raise ConfigError("routing.distilled.artifact_path is required when distilled routing is enabled.")
        if routing.distilled.min_confidence < 0 or routing.distilled.weight < 0:
            raise ConfigError("routing.distilled min_confidence and weight must be non-negative.")

    return MoEConfig(
        routing=routing,
        experts=experts,
        rules=rules,
        execution_policy=execution_policy,
    )


def _parse_expert(raw: dict[str, Any]) -> ExpertConfig:
    required = ("id", "provider", "model", "role")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ConfigError(f"Expert missing required fields: {missing}")

    endpoint = str(raw["base_url"]) if raw.get("base_url") is not None else None
    execution = _parse_execution_declaration(
        raw.get("execution", {}),
        provider=str(raw["provider"]),
        endpoint=endpoint,
    )

    return ExpertConfig(
        id=str(raw["id"]),
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        role=str(raw["role"]),
        weight=float(raw.get("weight", 1.0)),
        timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
        base_url=raw.get("base_url"),
        params=dict(raw.get("params", {})),
        execution=execution,
    )


def _parse_execution_policy(raw: object) -> ExecutionPolicy:
    if not isinstance(raw, dict):
        raise ConfigError("execution must be an object.")
    unknown = sorted(
        str(key)
        for key in raw
        if key not in {"max_scope", "allowed_scopes", "allow_scope_widening"}
    )
    if unknown:
        raise ConfigError(f"Unknown execution keys: {', '.join(unknown)}.")

    max_scope = _parse_execution_scope(
        raw.get("max_scope", ExecutionScope.DEVICE_ONLY.value),
        "execution.max_scope",
    )
    allowed_raw = raw.get("allowed_scopes")
    if allowed_raw is None:
        allowed_scopes = tuple(
            scope
            for scope in ExecutionScope
            if scope_rank(scope) <= scope_rank(max_scope)
        )
    else:
        if not isinstance(allowed_raw, list):
            raise ConfigError("execution.allowed_scopes must be a list.")
        allowed_scopes = tuple(
            _parse_execution_scope(item, "execution.allowed_scopes")
            for item in allowed_raw
        )

    allow_scope_widening = raw.get("allow_scope_widening", False)
    if not isinstance(allow_scope_widening, bool):
        raise ConfigError("execution.allow_scope_widening must be boolean.")
    try:
        return ExecutionPolicy(
            max_scope=max_scope,
            allowed_scopes=allowed_scopes,
            allow_scope_widening=allow_scope_widening,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _parse_execution_declaration(
    raw: object,
    *,
    provider: str,
    endpoint: str | None,
) -> ExecutionDeclaration:
    if not isinstance(raw, dict):
        raise ConfigError("expert.execution must be an object.")
    unknown = sorted(str(key) for key in raw if key not in {"scope", "transport"})
    if unknown:
        raise ConfigError(f"Unknown expert.execution keys: {', '.join(unknown)}.")

    scope = None
    if "scope" in raw:
        scope = _parse_execution_scope(raw["scope"], "expert.execution.scope")
    transport = None
    if "transport" in raw:
        transport = _parse_execution_transport(
            raw["transport"],
            "expert.execution.transport",
        )
    return normalized_execution_declaration(
        provider=provider,
        endpoint=endpoint,
        declaration=ExecutionDeclaration(scope=scope, transport=transport),
    )


def _parse_execution_scope(value: object, label: str) -> ExecutionScope:
    try:
        return ExecutionScope(str(value))
    except ValueError as exc:
        supported = ", ".join(scope.value for scope in ExecutionScope)
        raise ConfigError(f"{label} must be one of: {supported}.") from exc


def _parse_execution_transport(value: object, label: str) -> ExecutionTransport:
    try:
        return ExecutionTransport(str(value))
    except ValueError as exc:
        supported = ", ".join(transport.value for transport in ExecutionTransport)
        raise ConfigError(f"{label} must be one of: {supported}.") from exc


def _parse_rule(raw: dict[str, Any]) -> RoutingRule:
    keywords = tuple(str(item).lower() for item in raw.get("keywords", []))
    if not raw.get("expert_id") or not keywords:
        raise ConfigError("Routing rules require expert_id and at least one keyword.")

    return RoutingRule(
        expert_id=str(raw["expert_id"]),
        keywords=keywords,
        weight=float(raw.get("weight", 1.0)),
    )


def _parse_semantic_routing(raw: object) -> SemanticRoutingConfig:
    if not isinstance(raw, dict):
        raw = {}

    raw_examples = raw.get("examples", [])
    if not isinstance(raw_examples, list):
        raise ConfigError("routing.semantic.examples must be a list.")
    examples = tuple(_parse_semantic_example(item) for item in raw_examples)
    return SemanticRoutingConfig(
        enabled=bool(raw.get("enabled", False)),
        method=str(raw.get("method", "char_ngrams")),
        min_score=float(raw.get("min_score", 0.16)),
        margin=float(raw.get("margin", 0.02)),
        weight=float(raw.get("weight", 2.4)),
        ngram_min=int(raw.get("ngram_min", 3)),
        ngram_max=int(raw.get("ngram_max", 5)),
        examples=examples,
    )


def _parse_semantic_example(raw: object) -> SemanticRouteExample:
    if not isinstance(raw, dict):
        raise ConfigError("Semantic routing examples must be objects.")

    utterances = tuple(str(item).strip() for item in raw.get("utterances", []) if str(item).strip())
    if not raw.get("expert_id") or not utterances:
        raise ConfigError("Semantic routing examples require expert_id and at least one utterance.")

    return SemanticRouteExample(
        expert_id=str(raw["expert_id"]),
        utterances=utterances,
        weight=float(raw.get("weight", 1.0)),
    )


def _parse_distilled_routing(raw: object) -> DistilledRoutingConfig:
    if not isinstance(raw, dict):
        raw = {}
    return DistilledRoutingConfig(
        enabled=bool(raw.get("enabled", False)),
        artifact_path=str(raw.get("artifact_path", "")),
        min_confidence=float(raw.get("min_confidence", 0.12)),
        weight=float(raw.get("weight", 2.0)),
    )
