from __future__ import annotations

from dataclasses import dataclass

from .config import MoEConfig


@dataclass(frozen=True)
class ExpertScore:
    expert_id: str
    score: float
    matched_keywords: tuple[str, ...]


@dataclass(frozen=True)
class RouteDecision:
    selected: tuple[ExpertScore, ...]
    fallback_order: tuple[str, ...]


class RuleRouter:
    """Configurable deterministic router with no provider-specific logic."""

    def __init__(self, config: MoEConfig):
        self._config = config

    def route(self, prompt: str) -> RouteDecision:
        normalized = prompt.lower()
        matched_by_expert: dict[str, list[str]] = {
            expert.id: [] for expert in self._config.experts
        }
        scores: dict[str, float] = {
            expert.id: expert.weight for expert in self._config.experts
        }

        for rule in self._config.rules:
            matches = [kw for kw in rule.keywords if kw in normalized]
            if matches:
                scores[rule.expert_id] += rule.weight * len(matches)
                matched_by_expert[rule.expert_id].extend(matches)

        ranked = sorted(
            (
                ExpertScore(
                    expert_id=expert_id,
                    score=score,
                    matched_keywords=tuple(matched_by_expert[expert_id]),
                )
                for expert_id, score in scores.items()
            ),
            key=lambda item: (-item.score, item.expert_id),
        )

        selected = tuple(ranked[: self._config.routing.top_k])
        return RouteDecision(
            selected=selected,
            fallback_order=self._config.routing.fallback_order,
        )

