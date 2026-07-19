from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    sha256_json,
)


_ROOT_FIELDS = {
    "schema_version",
    "contract",
    "claim_scope",
    "expected_cases",
    "calibration_bins",
    "axes",
    "outcome_templates",
    "current_baseline",
    "verified_shadow",
    "metric_model",
}
_AXIS_FIELDS = {"capabilities", "difficulties", "languages", "contexts"}
_STRATEGIES = ("local_only", "current_baseline", "verified_shadow")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic synthetic verified-routing scenarios."
    )
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fixture = _load_fixture(Path(args.fixture))
    cases = expand_cases(fixture)
    result = evaluate_cases(cases, calibration_bins=int(fixture["calibration_bins"]))
    result.update(
        {
            "schema_version": "1.0",
            "contract": "VerifiedRoutingSyntheticEvalResult",
            "claim_scope": "synthetic_deterministic_only",
            "empirical_claim": False,
            "fixture_digest": sha256_json(fixture),
            "expanded_case_digest": sha256_json({"cases": cases}),
            "cases": len(cases),
        }
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "claim_scope": result["claim_scope"],
                "out": str(destination),
            },
            sort_keys=True,
        )
    )


def expand_cases(fixture: Mapping[str, object]) -> list[dict[str, object]]:
    axes = _mapping(fixture["axes"], "axes")
    capabilities = _string_list(axes["capabilities"], "capabilities")
    difficulties = _string_list(axes["difficulties"], "difficulties")
    languages = _string_list(axes["languages"], "languages")
    contexts = _string_list(axes["contexts"], "contexts")
    templates = fixture["outcome_templates"]
    if not isinstance(templates, list) or len(templates) != 4:
        raise VerifiedRoutingError("outcome_templates must contain four objects.")
    baseline = _mapping(fixture["current_baseline"], "current_baseline")
    baseline_probabilities = _mapping(
        baseline.get("premium_probability_by_difficulty"),
        "premium_probability_by_difficulty",
    )
    baseline_threshold = _probability(
        baseline.get("premium_threshold"), "current_baseline.premium_threshold"
    )
    shadow = _mapping(fixture["verified_shadow"], "verified_shadow")
    shadow_threshold = _probability(
        shadow.get("premium_threshold"), "verified_shadow.premium_threshold"
    )
    metric_model = _mapping(fixture["metric_model"], "metric_model")
    local_model = _metric_model(metric_model.get("local"), "local")
    premium_model = _metric_model(metric_model.get("premium"), "premium")

    cases: list[dict[str, object]] = []
    for index in range(64):
        capability_index = index % 4
        difficulty_index = (index // 4) % 4
        language_index = (index // 16) % 4
        context_index = (index + difficulty_index + language_index) % 4
        template_index = (capability_index + difficulty_index + language_index) % 4
        template = _template(templates[template_index], template_index)
        difficulty = difficulties[difficulty_index]
        baseline_probability = _probability(
            baseline_probabilities.get(difficulty),
            f"baseline probability for {difficulty}",
        )
        shadow_probability = _probability(
            template["verified_shadow_probability"],
            "verified_shadow_probability",
        )
        cases.append(
            {
                "id": f"synthetic-{index + 1:03d}",
                "capability": capabilities[capability_index],
                "difficulty": difficulty,
                "language": languages[language_index],
                "context": contexts[context_index],
                "local_verified": bool(template["local_verified"]),
                "premium_verified": bool(template["premium_verified"]),
                "local": _case_metrics(
                    local_model,
                    difficulty_index=difficulty_index,
                    context_index=context_index,
                ),
                "premium": _case_metrics(
                    premium_model,
                    difficulty_index=difficulty_index,
                    context_index=context_index,
                ),
                "probabilities": {
                    "local_only": 0.0,
                    "current_baseline": baseline_probability,
                    "verified_shadow": shadow_probability,
                },
                "thresholds": {
                    "local_only": 1.0,
                    "current_baseline": baseline_threshold,
                    "verified_shadow": shadow_threshold,
                },
            }
        )
    expected = int(fixture["expected_cases"])
    if expected != 64 or len(cases) != expected:
        raise VerifiedRoutingError("Synthetic fixture must expand to exactly 64 cases.")
    return cases


def evaluate_cases(
    cases: list[dict[str, object]],
    *,
    calibration_bins: int,
) -> dict[str, object]:
    if len(cases) != 64:
        raise VerifiedRoutingError("Verified-routing evaluation requires 64 cases.")
    strategies: dict[str, object] = {}
    for strategy in _STRATEGIES:
        overall = _metrics(cases, strategy, calibration_bins)
        strata: dict[str, object] = {}
        for dimension in ("capability", "difficulty", "language", "context"):
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for case in cases:
                grouped[str(case[dimension])].append(case)
            strata[dimension] = {
                value: _metrics(group, strategy, calibration_bins)
                for value, group in sorted(grouped.items())
            }
        strategies[strategy] = {"overall": overall, "strata": strata}
    return {"strategies": strategies}


def _metrics(
    cases: list[dict[str, object]],
    strategy: str,
    calibration_bins: int,
) -> dict[str, object]:
    selected: list[tuple[dict[str, object], str, float, int]] = []
    for case in cases:
        probabilities = _mapping(case["probabilities"], "probabilities")
        thresholds = _mapping(case["thresholds"], "thresholds")
        probability = float(probabilities[strategy])
        route = "premium" if probability >= float(thresholds[strategy]) else "local"
        target = int(not bool(case["local_verified"]) and bool(case["premium_verified"]))
        selected.append((case, route, probability, target))

    true_positive = sum(1 for _, route, _, target in selected if route == "premium" and target)
    false_negative = sum(1 for _, route, _, target in selected if route == "local" and target)
    false_positive = sum(
        1
        for _, route, _, target in selected
        if route == "premium" and not target
    )
    verified = 0
    premium_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_cost = 0.0
    total_egress = 0
    latencies: list[float] = []
    predictions: list[float] = []
    targets: list[int] = []
    for case, route, probability, target in selected:
        metrics = _mapping(case[route], route)
        verified += int(bool(case[f"{route}_verified"]))
        premium_calls += int(metrics["premium_calls"])
        prompt_tokens += int(metrics["prompt_tokens"])
        completion_tokens += int(metrics["completion_tokens"])
        total_cost += float(metrics["cost_usd"])
        total_egress += int(metrics["egress_chars"])
        latencies.append(float(metrics["latency_ms"]))
        predictions.append(probability)
        targets.append(target)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "cases": len(cases),
        "verified_success": _ratio(verified, len(cases)),
        "false_local": _ratio(false_negative, recall_denominator),
        "unnecessary_premium": _ratio(false_positive, precision_denominator),
        "escalation_precision": _ratio(true_positive, precision_denominator),
        "escalation_recall": _ratio(true_positive, recall_denominator),
        "premium_calls": premium_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": round(total_cost, 12),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "egress_chars": total_egress,
        "brier_score": round(
            math.fsum((prediction - target) ** 2 for prediction, target in zip(predictions, targets))
            / len(predictions),
            12,
        ),
        "ece": _ece(predictions, targets, calibration_bins),
    }


def _ece(predictions: list[float], targets: list[int], bins: int) -> float:
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for prediction, target in zip(predictions, targets):
        index = min(bins - 1, int(prediction * bins))
        grouped[index].append((prediction, target))
    total = len(predictions)
    value = 0.0
    for items in grouped.values():
        confidence = math.fsum(item[0] for item in items) / len(items)
        frequency = math.fsum(item[1] for item in items) / len(items)
        value += len(items) / total * abs(confidence - frequency)
    return round(value, 12)


def _case_metrics(
    model: Mapping[str, object],
    *,
    difficulty_index: int,
    context_index: int,
) -> dict[str, object]:
    return {
        "latency_ms": int(model["base_latency_ms"]) + difficulty_index * 240,
        "prompt_tokens": int(model["base_prompt_tokens"]) + context_index * 160,
        "completion_tokens": int(model["base_completion_tokens"])
        + difficulty_index * 70,
        "cost_usd": round(float(model["base_cost_usd"]) * (1 + 0.2 * difficulty_index), 12),
        "egress_chars": int(model["base_egress_chars"]) + context_index * int(
            model["base_egress_chars"]
        ) // 4,
        "premium_calls": int(model["premium_calls"]),
    }


def _load_fixture(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: _reject_constant(token),
        )
    except json.JSONDecodeError as exc:
        raise VerifiedRoutingError("Invalid verified-routing eval fixture JSON.") from exc
    if not isinstance(raw, dict) or set(raw) != _ROOT_FIELDS:
        raise VerifiedRoutingError("Verified-routing eval fixture fields are invalid.")
    if raw["schema_version"] != "1.0":
        raise VerifiedRoutingError("Unsupported eval fixture schema_version.")
    if raw["contract"] != "VerifiedRoutingSyntheticEvalFixture":
        raise VerifiedRoutingError("Eval fixture contract is invalid.")
    if raw["claim_scope"] != "synthetic_deterministic_only":
        raise VerifiedRoutingError("Eval fixture must prohibit empirical claims.")
    axes = _mapping(raw["axes"], "axes")
    if set(axes) != _AXIS_FIELDS:
        raise VerifiedRoutingError("Eval fixture axes are invalid.")
    for name in sorted(_AXIS_FIELDS):
        if len(_string_list(axes[name], name)) != 4:
            raise VerifiedRoutingError(f"Axis {name} must contain four values.")
    bins = int(raw["calibration_bins"])
    if bins <= 0 or bins > 100:
        raise VerifiedRoutingError("calibration_bins must be in [1, 100].")
    return raw


def _template(raw: object, index: int) -> dict[str, object]:
    template = _mapping(raw, f"outcome_templates[{index}]")
    expected = {
        "id",
        "local_verified",
        "premium_verified",
        "verified_shadow_probability",
    }
    if set(template) != expected:
        raise VerifiedRoutingError("Outcome template fields are invalid.")
    if not isinstance(template["local_verified"], bool) or not isinstance(
        template["premium_verified"], bool
    ):
        raise VerifiedRoutingError("Outcome verification labels must be boolean.")
    return template


def _metric_model(raw: object, label: str) -> dict[str, object]:
    model = _mapping(raw, f"metric_model.{label}")
    expected = {
        "base_latency_ms",
        "base_prompt_tokens",
        "base_completion_tokens",
        "base_cost_usd",
        "base_egress_chars",
        "premium_calls",
    }
    if set(model) != expected:
        raise VerifiedRoutingError(f"metric_model.{label} fields are invalid.")
    for name in expected.difference({"base_cost_usd"}):
        value = model[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise VerifiedRoutingError(f"metric_model.{label}.{name} is invalid.")
    cost = float(model["base_cost_usd"])
    if not math.isfinite(cost) or cost < 0.0:
        raise VerifiedRoutingError(f"metric_model.{label}.base_cost_usd is invalid.")
    return model


def _mapping(raw: object, label: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise VerifiedRoutingError(f"{label} must be an object.")
    return dict(raw)


def _string_list(raw: object, label: str) -> list[str]:
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise VerifiedRoutingError(f"{label} must be a non-empty string list.")
    if len(raw) != len(set(raw)):
        raise VerifiedRoutingError(f"{label} must not contain duplicates.")
    return list(raw)


def _probability(raw: object, label: str) -> float:
    if isinstance(raw, bool):
        raise VerifiedRoutingError(f"{label} must be numeric.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise VerifiedRoutingError(f"{label} must be numeric.") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise VerifiedRoutingError(f"{label} must be in [0, 1].")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 12)


def _reject_constant(token: str) -> object:
    raise VerifiedRoutingError(f"Non-finite number {token} is forbidden.")


if __name__ == "__main__":
    main()
