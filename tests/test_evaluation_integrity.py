from __future__ import annotations

import json
from pathlib import Path
import unittest

from local_moe.evaluation_integrity import analyze_route_holdout, records_sha256


ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class EvaluationIntegrityTests(unittest.TestCase):
    def test_detects_training_holdout_leakage(self) -> None:
        training = [
            {
                "prompt_id": "shared",
                "prompt": "Summarize this note",
                "primary": "fast_fallback",
            }
        ]
        holdout = [
            {
                "id": "shared",
                "prompt": "  summarize   THIS note ",
                "expected_expert": "fast_fallback",
                "complexity": "simple",
            }
        ]

        result = analyze_route_holdout(training, holdout)

        self.assertFalse(result["passed"])
        self.assertEqual(result["overlapping_ids"], ["shared"])
        self.assertEqual(len(result["overlapping_prompt_hashes"]), 1)

    def test_dataset_fingerprint_is_order_independent(self) -> None:
        records = [
            {"id": "a", "prompt": "One", "expected_expert": "general"},
            {"id": "b", "prompt": "Two", "expected_expert": "fast_fallback"},
        ]

        left = records_sha256(
            records,
            fields=("id", "prompt", "expected_expert"),
        )
        right = records_sha256(
            list(reversed(records)),
            fields=("id", "prompt", "expected_expert"),
        )

        self.assertEqual(left, right)

    def test_live_holdout_is_disjoint_unique_and_balanced(self) -> None:
        training = _load_jsonl(ROOT / "experiments" / "route_labels_live_general.jsonl")
        holdout = _load_jsonl(
            ROOT / "experiments" / "eval_set_live_general_holdout_v2.jsonl"
        )

        result = analyze_route_holdout(training, holdout)

        self.assertTrue(result["passed"])
        self.assertEqual(result["training_total"], 52)
        self.assertEqual(result["holdout_total"], 52)
        self.assertEqual(
            result["holdout_experts"],
            {"fast_fallback": 26, "general": 26},
        )
        self.assertEqual(
            result["holdout_complexities"],
            {"complex": 13, "medium": 13, "simple": 13, "very_complex": 13},
        )


if __name__ == "__main__":
    unittest.main()
