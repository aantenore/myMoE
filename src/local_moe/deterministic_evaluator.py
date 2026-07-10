from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


EVALUATOR_VERSION = "deterministic-rubric-v1"
SUPPORTED_CHECKS = {
    "contains_all",
    "contains_all_groups",
    "contains_any",
    "excludes",
    "matches_regex",
    "max_words",
    "min_words",
    "nonempty",
}
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


class QualityBenchmarkError(ValueError):
    """Raised when a quality benchmark manifest or dataset is invalid."""


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    prompt: str
    category: str
    complexity: str
    task_checks: tuple[dict[str, Any], ...]
    quality_rubric: tuple[dict[str, Any], ...]


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QualityBenchmarkError(f"Invalid JSONL at line {line_number}: {exc}.") from exc
        if not isinstance(raw, dict):
            raise QualityBenchmarkError(f"Benchmark case at line {line_number} must be an object.")
        case_id = str(raw.get("id", "")).strip()
        prompt = str(raw.get("prompt", "")).strip()
        if not case_id or not prompt:
            raise QualityBenchmarkError(f"Benchmark case at line {line_number} requires id and prompt.")
        if case_id in seen:
            raise QualityBenchmarkError(f"Duplicate benchmark case id: {case_id}.")
        seen.add(case_id)
        task_checks = _validate_checks(raw.get("task_checks", []), case_id, weighted=False)
        quality_rubric = _validate_checks(raw.get("quality_rubric", []), case_id, weighted=True)
        if not task_checks:
            raise QualityBenchmarkError(f"Benchmark case {case_id} requires task_checks.")
        if not quality_rubric:
            raise QualityBenchmarkError(f"Benchmark case {case_id} requires quality_rubric.")
        cases.append(
            BenchmarkCase(
                id=case_id,
                prompt=prompt,
                category=str(raw.get("category", "unknown")),
                complexity=str(raw.get("complexity", "unknown")),
                task_checks=task_checks,
                quality_rubric=quality_rubric,
            )
        )
    if not cases:
        raise QualityBenchmarkError("Benchmark dataset must contain at least one case.")
    return cases


def evaluate_check(content: str, check: dict[str, Any]) -> dict[str, Any]:
    check_type = str(check["type"])
    normalized = content.casefold()
    word_count = len(_WORD_RE.findall(content))
    evidence: dict[str, Any]

    if check_type == "nonempty":
        passed = bool(content.strip())
        evidence = {"content_characters": len(content.strip())}
    elif check_type == "min_words":
        threshold = int(check["value"])
        passed = word_count >= threshold
        evidence = {"word_count": word_count, "minimum": threshold}
    elif check_type == "max_words":
        threshold = int(check["value"])
        passed = word_count <= threshold
        evidence = {"word_count": word_count, "maximum": threshold}
    elif check_type in {"contains_any", "contains_all", "excludes"}:
        values = [str(item) for item in check["values"]]
        matched = [value for value in values if value.casefold() in normalized]
        if check_type == "contains_any":
            passed = bool(matched)
        elif check_type == "contains_all":
            passed = len(matched) == len(values)
        else:
            passed = not matched
        evidence = {"matched": matched, "expected": values}
    elif check_type == "contains_all_groups":
        groups = [[str(item) for item in group] for group in check["groups"]]
        matches = [
            [value for value in group if value.casefold() in normalized]
            for group in groups
        ]
        passed = all(matches)
        evidence = {"group_matches": matches, "groups": groups}
    elif check_type == "matches_regex":
        pattern = str(check["pattern"])
        match = re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE)
        passed = match is not None
        evidence = {"pattern": pattern, "match": match.group(0)[:200] if match else None}
    else:  # pragma: no cover - protected by manifest validation
        raise QualityBenchmarkError(f"Unsupported deterministic check: {check_type}.")

    return {
        "id": str(check["id"]),
        "type": check_type,
        "description": str(check.get("description", "")),
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "evidence": evidence,
    }


def evaluate_case_output(
    case: BenchmarkCase,
    content: str,
    *,
    quality_pass_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_results = [evaluate_check(content, check) for check in case.task_checks]
    task_validation = {
        "status": "passed" if all(item["passed"] for item in task_results) else "failed",
        "passed": all(item["passed"] for item in task_results),
        "checks": task_results,
    }

    quality_results = []
    weighted_score = 0.0
    total_weight = 0.0
    for criterion in case.quality_rubric:
        result = evaluate_check(content, criterion)
        weight = float(criterion["weight"])
        result["weight"] = weight
        quality_results.append(result)
        weighted_score += result["score"] * weight
        total_weight += weight
    score = weighted_score / total_weight
    quality_judgment = {
        "status": "passed" if score >= quality_pass_threshold else "failed",
        "passed": score >= quality_pass_threshold,
        "score": round(score, 4),
        "threshold": quality_pass_threshold,
        "evaluator": EVALUATOR_VERSION,
        "criteria": quality_results,
    }
    return task_validation, quality_judgment


def _validate_checks(
    raw_checks: Any,
    case_id: str,
    *,
    weighted: bool,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_checks, list):
        raise QualityBenchmarkError(f"Checks for case {case_id} must be a list.")
    checks: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in raw_checks:
        if not isinstance(raw, dict):
            raise QualityBenchmarkError(f"Checks for case {case_id} must be objects.")
        check = dict(raw)
        check_id = str(check.get("id", "")).strip()
        check_type = str(check.get("type", "")).strip()
        if not check_id or check_id in ids:
            raise QualityBenchmarkError(f"Case {case_id} has a missing or duplicate check id.")
        if check_type not in SUPPORTED_CHECKS:
            raise QualityBenchmarkError(
                f"Case {case_id} check {check_id} has unsupported type {check_type!r}."
            )
        ids.add(check_id)
        _validate_check_shape(check, case_id)
        if weighted:
            weight = float(check.get("weight", 0.0))
            if weight <= 0.0 or not math.isfinite(weight):
                raise QualityBenchmarkError(
                    f"Case {case_id} quality criterion {check_id} requires a positive weight."
                )
            check["weight"] = weight
        checks.append(check)
    return tuple(checks)


def _validate_check_shape(check: dict[str, Any], case_id: str) -> None:
    check_type = str(check["type"])
    if check_type in {"min_words", "max_words"}:
        try:
            value = int(check["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QualityBenchmarkError(
                f"Case {case_id} check {check['id']} requires an integer value."
            ) from exc
        if value < 0:
            raise QualityBenchmarkError(f"Case {case_id} check {check['id']} value must be >= 0.")
    elif check_type in {"contains_any", "contains_all", "excludes"}:
        values = check.get("values")
        if not isinstance(values, list) or not values or not all(str(item) for item in values):
            raise QualityBenchmarkError(
                f"Case {case_id} check {check['id']} requires non-empty values."
            )
    elif check_type == "contains_all_groups":
        groups = check.get("groups")
        valid = (
            isinstance(groups, list)
            and bool(groups)
            and all(isinstance(group, list) and group for group in groups)
        )
        if not valid:
            raise QualityBenchmarkError(
                f"Case {case_id} check {check['id']} requires non-empty groups."
            )
    elif check_type == "matches_regex":
        try:
            re.compile(str(check["pattern"]))
        except (KeyError, re.error) as exc:
            raise QualityBenchmarkError(
                f"Case {case_id} check {check['id']} requires a valid pattern."
            ) from exc
