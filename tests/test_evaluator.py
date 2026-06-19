from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.config import load_config
from local_moe.evaluator import evaluate_router, load_eval_cases


class EvaluatorTests(unittest.TestCase):
    def test_loads_jsonl_eval_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                '{"id":"a","prompt":"Write Python","expected_expert":"coder","complexity":"simple"}\n'
                '\n'
                '{"id":"b","prompt":"Summarize","expected_expert":"general"}\n',
                encoding="utf-8",
            )

            cases = load_eval_cases(path)

        self.assertEqual([case.id for case in cases], ["a", "b"])
        self.assertEqual(cases[1].complexity, "unknown")

    def test_evaluates_accuracy_and_complexity_breakdown(self) -> None:
        config = load_config("tests/fixtures/moe.synthetic.json")
        cases = load_eval_cases("experiments/eval_set.jsonl")

        result = evaluate_router(config, cases)

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["total"], 8)
        self.assertIn("complex", result["by_complexity"])
        self.assertEqual(len(result["results"]), 8)

    def test_live_general_eval_matches_live_config_experts(self) -> None:
        config = load_config("configs/moe.live.general-mlx.example.json")
        cases = load_eval_cases("experiments/eval_set_live_general.jsonl")

        result = evaluate_router(config, cases)

        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreaterEqual(result["total"], 50)
        self.assertEqual(set(result["by_complexity"]), {"simple", "medium", "complex", "very_complex"})
        self.assertEqual({item["selected_expert"] for item in result["results"]}, {"general", "fast_fallback"})


if __name__ == "__main__":
    unittest.main()
