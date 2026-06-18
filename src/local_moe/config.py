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
class RoutingRule:
    expert_id: str
    keywords: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class RoutingConfig:
    top_k: int = 1
    fallback_order: tuple[str, ...] = ()
    aggregation: str = "best"


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
