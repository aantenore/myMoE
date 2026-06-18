from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .config import MoEConfig
from .router import RuleRouter


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expected_expert: str
    complexity: str


@dataclass(frozen=True)
class EvalResult:
    id: str
    expected_expert: str
    selected_expert: str
    passed: bool
    complexity: str
    score: float


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            EvalCase(
                id=str(raw["id"]),
                prompt=str(raw["prompt"]),
                expected_expert=str(raw["expected_expert"]),
                complexity=str(raw.get("complexity", "unknown")),
            )
        )
    return cases


def evaluate_router(config: MoEConfig, cases: list[EvalCase]) -> dict[str, object]:
    router = RuleRouter(config)
    results: list[EvalResult] = []

    for case in cases:
        decision = router.route(case.prompt)
        selected = decision.selected[0].expert_id
        passed = selected == case.expected_expert
        results.append(
            EvalResult(
                id=case.id,
                expected_expert=case.expected_expert,
                selected_expert=selected,
                passed=passed,
                complexity=case.complexity,
                score=1.0 if passed else 0.0,
            )
        )

    accuracy = sum(item.score for item in results) / max(len(results), 1)
    by_complexity: dict[str, list[float]] = {}
    for item in results:
        by_complexity.setdefault(item.complexity, []).append(item.score)

    return {
        "accuracy": accuracy,
        "total": len(results),
        "by_complexity": {
            key: sum(values) / len(values) for key, values in by_complexity.items()
        },
        "results": [item.__dict__ for item in results],
    }

