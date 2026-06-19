from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

from .config import MoEConfig, SemanticRoutingConfig


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

    def __init__(self, config: MoEConfig):
        self._config = config
        self._semantic_profiles = _build_semantic_profiles(config.routing.semantic)

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

        self._apply_semantic_score(prompt, scores, matched_by_expert)

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

    def _apply_semantic_score(
        self,
        prompt: str,
        scores: dict[str, float],
        matched_by_expert: dict[str, list[str]],
    ) -> None:
        semantic = self._config.routing.semantic
        if self._config.routing.strategy != "hybrid" or not semantic.enabled:
            return
        if not self._semantic_profiles:
            return

        prompt_vector = _vectorize(prompt, semantic.ngram_min, semantic.ngram_max)
        if not prompt_vector:
            return

        ranked = []
        for expert_id, profile in self._semantic_profiles.items():
            best_score = 0.0
            for example_vector, example_weight in profile:
                score = _cosine(prompt_vector, example_vector) * example_weight
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


def _build_semantic_profiles(
    semantic: SemanticRoutingConfig,
) -> dict[str, list[tuple[Counter[str], float]]]:
    profiles: dict[str, list[tuple[Counter[str], float]]] = {}
    if not semantic.enabled:
        return profiles
    for example in semantic.examples:
        for utterance in example.utterances:
            vector = _vectorize(utterance, semantic.ngram_min, semantic.ngram_max)
            if vector:
                profiles.setdefault(example.expert_id, []).append((vector, example.weight))
    return profiles


def _vectorize(text: str, ngram_min: int, ngram_max: int) -> Counter[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return Counter()

    features: Counter[str] = Counter()
    words = normalized.split()
    for word in words:
        features[f"w:{word}"] += 2
    padded = f" {normalized} "
    for size in range(ngram_min, ngram_max + 1):
        if len(padded) < size:
            continue
        for index in range(0, len(padded) - size + 1):
            gram = padded[index : index + size]
            if gram.strip():
                features[f"c:{gram}"] += 1
    return features


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^\w\s]+", " ", without_marks, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
