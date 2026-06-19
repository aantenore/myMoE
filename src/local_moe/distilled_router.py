from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import DistilledRoutingConfig
from .text_features import cosine, vectorize


ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class RouteLabel:
    prompt_id: str
    prompt: str
    primary: str
    fallback: str = ""
    confidence: float = 1.0
    reason: str = "curated"
    risk: str = "unknown"
    teacher_source: str = "curated_eval"


@dataclass(frozen=True)
class DistilledRouterArtifact:
    method: str
    ngram_min: int
    ngram_max: int
    training_cases: int
    expert_counts: dict[str, int]
    expert_profiles: dict[str, dict[str, float]]

    def predict(self, prompt: str) -> tuple[str | None, float]:
        prompt_vector = vectorize(prompt, self.ngram_min, self.ngram_max)
        if not prompt_vector:
            return None, 0.0

        ranked = [
            (expert_id, cosine(prompt_vector, profile))
            for expert_id, profile in self.expert_profiles.items()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return None, 0.0
        best_expert, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = max(best_score - second_score, best_score * 0.5)
        return best_expert, round(confidence, 4)


def labels_from_eval_cases(raw_cases: list[dict[str, Any]], *, teacher_source: str = "curated_eval") -> list[RouteLabel]:
    labels: list[RouteLabel] = []
    for raw in raw_cases:
        labels.append(
            RouteLabel(
                prompt_id=str(raw["id"]),
                prompt=str(raw["prompt"]),
                primary=str(raw["expected_expert"]),
                fallback=str(raw.get("fallback_expert", "")),
                confidence=float(raw.get("confidence", 1.0)),
                reason=str(raw.get("complexity", "curated_eval")),
                risk=str(raw.get("risk", "unknown")),
                teacher_source=teacher_source,
            )
        )
    return labels


def load_route_labels(path: str | Path) -> list[RouteLabel]:
    labels: list[RouteLabel] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        labels.append(
            RouteLabel(
                prompt_id=str(raw["prompt_id"]),
                prompt=str(raw["prompt"]),
                primary=str(raw["primary"]),
                fallback=str(raw.get("fallback", "")),
                confidence=float(raw.get("confidence", 1.0)),
                reason=str(raw.get("reason", "curated")),
                risk=str(raw.get("risk", "unknown")),
                teacher_source=str(raw.get("teacher_source", "curated_eval")),
            )
        )
    return labels


def write_route_labels(labels: list[RouteLabel], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label.__dict__, sort_keys=True) + "\n")


def train_distilled_router_artifact(
    labels: list[RouteLabel],
    *,
    ngram_min: int = 3,
    ngram_max: int = 5,
) -> dict[str, Any]:
    profiles: dict[str, Counter[str]] = {}
    counts: dict[str, int] = {}
    for label in labels:
        vector = vectorize(label.prompt, ngram_min, ngram_max)
        if not vector:
            continue
        profile = profiles.setdefault(label.primary, Counter())
        for feature, value in vector.items():
            profile[feature] += value * max(label.confidence, 0.0)
        counts[label.primary] = counts.get(label.primary, 0) + 1

    return {
        "version": ARTIFACT_VERSION,
        "method": "char_ngram_centroid",
        "ngram_min": ngram_min,
        "ngram_max": ngram_max,
        "training_cases": sum(counts.values()),
        "expert_counts": counts,
        "expert_profiles": {
            expert_id: dict(sorted(profile.items()))
            for expert_id, profile in sorted(profiles.items())
        },
    }


def write_distilled_router_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_distilled_router_artifact(config: DistilledRoutingConfig) -> DistilledRouterArtifact | None:
    if not config.enabled:
        return None
    raw = json.loads(Path(config.artifact_path).read_text(encoding="utf-8"))
    if int(raw.get("version", 0)) != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported distilled router artifact version: {raw.get('version')}")
    if raw.get("method") != "char_ngram_centroid":
        raise ValueError(f"Unsupported distilled router method: {raw.get('method')}")
    return DistilledRouterArtifact(
        method=str(raw["method"]),
        ngram_min=int(raw["ngram_min"]),
        ngram_max=int(raw["ngram_max"]),
        training_cases=int(raw["training_cases"]),
        expert_counts={str(key): int(value) for key, value in raw.get("expert_counts", {}).items()},
        expert_profiles={
            str(expert_id): {str(feature): float(value) for feature, value in profile.items()}
            for expert_id, profile in raw.get("expert_profiles", {}).items()
        },
    )
