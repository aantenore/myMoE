from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping


def normalize_prompt(prompt: object) -> str:
    """Return a stable representation used only for evaluation provenance."""

    return " ".join(str(prompt).casefold().split())


def prompt_sha256(prompt: object) -> str:
    return sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def records_sha256(
    records: Iterable[Mapping[str, Any]],
    *,
    fields: tuple[str, ...],
) -> str:
    canonical: list[str] = []
    for record in records:
        payload = {
            field: normalize_prompt(record.get(field, ""))
            if field == "prompt"
            else record.get(field)
            for field in fields
        }
        canonical.append(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    serialized = "\n".join(sorted(canonical))
    return sha256(serialized.encode("utf-8")).hexdigest()


def analyze_route_holdout(
    training_records: list[Mapping[str, Any]],
    holdout_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    training_ids = [str(item.get("prompt_id", "")) for item in training_records]
    holdout_ids = [str(item.get("id", "")) for item in holdout_records]
    training_hashes = [prompt_sha256(item.get("prompt", "")) for item in training_records]
    holdout_hashes = [prompt_sha256(item.get("prompt", "")) for item in holdout_records]

    duplicate_training_ids = _duplicates(training_ids)
    duplicate_holdout_ids = _duplicates(holdout_ids)
    duplicate_training_prompts = _duplicates(training_hashes)
    duplicate_holdout_prompts = _duplicates(holdout_hashes)
    overlapping_ids = sorted(set(training_ids) & set(holdout_ids))
    overlapping_prompt_hashes = sorted(set(training_hashes) & set(holdout_hashes))

    training_experts = Counter(str(item.get("primary", "")) for item in training_records)
    holdout_experts = Counter(str(item.get("expected_expert", "")) for item in holdout_records)
    holdout_complexities = Counter(str(item.get("complexity", "unknown")) for item in holdout_records)
    passed = not any(
        (
            duplicate_training_ids,
            duplicate_holdout_ids,
            duplicate_training_prompts,
            duplicate_holdout_prompts,
            overlapping_ids,
            overlapping_prompt_hashes,
        )
    )

    return {
        "passed": passed,
        "training_total": len(training_records),
        "holdout_total": len(holdout_records),
        "training_data_sha256": records_sha256(
            training_records,
            fields=(
                "prompt_id",
                "prompt",
                "primary",
                "fallback",
                "confidence",
                "reason",
                "risk",
                "teacher_source",
            ),
        ),
        "holdout_data_sha256": records_sha256(
            holdout_records,
            fields=("id", "prompt", "expected_expert", "complexity"),
        ),
        "training_experts": dict(sorted(training_experts.items())),
        "holdout_experts": dict(sorted(holdout_experts.items())),
        "holdout_complexities": dict(sorted(holdout_complexities.items())),
        "duplicate_training_ids": duplicate_training_ids,
        "duplicate_holdout_ids": duplicate_holdout_ids,
        "duplicate_training_prompt_hashes": duplicate_training_prompts,
        "duplicate_holdout_prompt_hashes": duplicate_holdout_prompts,
        "overlapping_ids": overlapping_ids,
        "overlapping_prompt_hashes": overlapping_prompt_hashes,
    }


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if not value or count > 1)
