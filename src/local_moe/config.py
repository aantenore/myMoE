from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


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

    @property
    def experts_by_id(self) -> dict[str, ExpertConfig]:
        return {expert.id: expert for expert in self.experts}


def load_config(path: str | Path) -> MoEConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> MoEConfig:
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

    return MoEConfig(routing=routing, experts=experts, rules=rules)


def _parse_expert(raw: dict[str, Any]) -> ExpertConfig:
    required = ("id", "provider", "model", "role")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ConfigError(f"Expert missing required fields: {missing}")

    return ExpertConfig(
        id=str(raw["id"]),
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        role=str(raw["role"]),
        weight=float(raw.get("weight", 1.0)),
        timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
        base_url=raw.get("base_url"),
        params=dict(raw.get("params", {})),
    )


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
