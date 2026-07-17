from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .config import MoEConfig
from .path_security import read_text_file
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
    return _parse_eval_cases(Path(path).read_text(encoding="utf-8"))


def load_eval_cases_within(
    path: str | Path,
    *,
    allowed_roots: tuple[str | Path, ...],
) -> list[EvalCase]:
    """Load a web-requested eval set inside configured evaluation roots."""

    _, text = read_text_file(
        path,
        allowed_roots=allowed_roots,
        label="evaluation set",
        max_bytes=16 * 1024 * 1024,
    )
    return _parse_eval_cases(text)


def _parse_eval_cases(text: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in text.splitlines():
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
    passed_count = sum(1 for item in results if item.passed)
    by_complexity: dict[str, list[float]] = {}
    for item in results:
        by_complexity.setdefault(item.complexity, []).append(item.score)

    return {
        "accuracy": accuracy,
        "accuracy_ci95": _wilson_interval(passed_count, len(results)),
        "total": len(results),
        "by_complexity": {
            key: sum(values) / len(values) for key, values in by_complexity.items()
        },
        "results": [item.__dict__ for item in results],
    }


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    return {
        "lower": round(max(0.0, center - margin), 4),
        "upper": round(min(1.0, center + margin), 4),
    }
