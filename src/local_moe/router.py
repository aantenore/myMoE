from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import MoEConfig, SemanticRoutingConfig
from .distilled_router import load_distilled_router_artifact
from .execution_scope import (
    ExecutionEligibility,
    ExecutionScope,
    ExecutionScopeGuard,
    ScopePolicyError,
)
from .text_features import cosine, vectorize


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
    """Configurable deterministic and semantic router with no provider-specific logic."""

    def __init__(
        self,
        config: MoEConfig,
        *,
        execution_guard: ExecutionScopeGuard | None = None,
    ):
        self._config = config
        self._execution_guard = execution_guard or ExecutionScopeGuard(
            config.execution_policy
        )
        self._semantic_profiles = _build_semantic_profiles(config.routing.semantic)
        self._distilled_artifact = load_distilled_router_artifact(config.routing.distilled)

    def route(self, prompt: str) -> RouteDecision:
        normalized = prompt.lower()
        eligibility = {
            expert.id: self._execution_guard.evaluate(expert.execution_target)
            for expert in self._config.experts
        }
        eligible_experts = tuple(
            expert
            for expert in self._config.experts
            if eligibility[expert.id].allowed
        )
        if not eligible_experts:
            blocked = "; ".join(
                f"{expert_id} ({eligibility[expert_id].detail})"
                for expert_id in sorted(eligibility)
            )
            raise ScopePolicyError(
                f"no expert satisfies the execution policy; blocked: {blocked}."
            )
        matched_by_expert: dict[str, list[str]] = {
            expert.id: [] for expert in eligible_experts
        }
        scores: dict[str, float] = {
            expert.id: expert.weight for expert in eligible_experts
        }

        for rule in self._config.rules:
            if rule.expert_id not in scores:
                continue
            matches = [kw for kw in rule.keywords if kw in normalized]
            if matches:
                scores[rule.expert_id] += rule.weight * len(matches)
                matched_by_expert[rule.expert_id].extend(matches)

        self._apply_semantic_score(prompt, scores, matched_by_expert)
        self._apply_distilled_score(prompt, scores, matched_by_expert)

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
        selected_scopes = tuple(
            scope
            for item in selected
            if (scope := eligibility[item.expert_id].scope) is not None
        )
        fallback_order = tuple(
            expert_id
            for expert_id in self._config.routing.fallback_order
            if _fallback_is_eligible(
                expert_id,
                eligibility=eligibility,
                selected_scopes=selected_scopes,
                execution_guard=self._execution_guard,
            )
        )
        return RouteDecision(
            selected=selected,
            fallback_order=fallback_order,
        )

    def _apply_semantic_score(
        self,
        prompt: str,
        scores: dict[str, float],
        matched_by_expert: dict[str, list[str]],
    ) -> None:
        semantic = self._config.routing.semantic
        if self._config.routing.strategy not in {"hybrid", "distilled"} or not semantic.enabled:
            return
        if not self._semantic_profiles:
            return

        prompt_vector = vectorize(prompt, semantic.ngram_min, semantic.ngram_max)
        if not prompt_vector:
            return

        ranked = []
        for expert_id, profile in self._semantic_profiles.items():
            if expert_id not in scores:
                continue
            best_score = 0.0
            for example_vector, example_weight in profile:
                score = cosine(prompt_vector, example_vector) * example_weight
                best_score = max(best_score, score)
            ranked.append((expert_id, best_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return

        best_expert, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < semantic.min_score:
            return
        if best_score - second_score < semantic.margin:
            return

        scores[best_expert] += semantic.weight * best_score
        matched_by_expert[best_expert].append(f"semantic:{best_score:.2f}")

    def _apply_distilled_score(
        self,
        prompt: str,
        scores: dict[str, float],
        matched_by_expert: dict[str, list[str]],
    ) -> None:
        distilled = self._config.routing.distilled
        if self._config.routing.strategy != "distilled" or not distilled.enabled:
            return
        artifact = self._distilled_artifact
        if artifact is None:
            return

        expert_id, confidence = artifact.predict(prompt)
        if expert_id is None or confidence < distilled.min_confidence:
            return
        if expert_id not in scores:
            return
        scores[expert_id] += distilled.weight * confidence
        matched_by_expert[expert_id].append(f"distilled:{confidence:.2f}")


def _build_semantic_profiles(
    semantic: SemanticRoutingConfig,
) -> dict[str, list[tuple[Counter[str], float]]]:
    profiles: dict[str, list[tuple[Counter[str], float]]] = {}
    if not semantic.enabled:
        return profiles
    for example in semantic.examples:
        for utterance in example.utterances:
            vector = vectorize(utterance, semantic.ngram_min, semantic.ngram_max)
            if vector:
                profiles.setdefault(example.expert_id, []).append((vector, example.weight))
    return profiles


def _fallback_is_eligible(
    expert_id: str,
    *,
    eligibility: dict[str, ExecutionEligibility],
    selected_scopes: tuple[ExecutionScope, ...],
    execution_guard: ExecutionScopeGuard,
) -> bool:
    candidate = eligibility[expert_id]
    if not candidate.allowed or candidate.scope is None:
        return False
    return execution_guard.permits_fallback(
        selected_scopes=selected_scopes,
        fallback_scope=candidate.scope,
    )
