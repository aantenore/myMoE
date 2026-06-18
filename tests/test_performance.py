from __future__ import annotations

import unittest

from local_moe.performance import (
    load_benchmark_manifest,
    render_markdown_report,
    score_candidate,
    summarize_benchmarks,
)
from experiments.benchmark_models import _parse_json_stdout


class PerformanceTests(unittest.TestCase):
    def test_loads_benchmark_manifest(self) -> None:
        manifest = load_benchmark_manifest("configs/model-benchmark.json")

        self.assertGreaterEqual(len(manifest.candidates), 3)
        self.assertGreaterEqual(len(manifest.prompts), 2)
        self.assertEqual(manifest.hardware_budget_gb, 24.0)

    def test_gemma_e4b_uses_validated_mlx_artifact(self) -> None:
        manifest = load_benchmark_manifest("configs/model-benchmark.json")
        gemma = next(candidate for candidate in manifest.candidates if candidate.id == "gemma4-e4b-it-mlx-4bit")

        self.assertEqual(gemma.runtime, "mlx_lm")
        self.assertEqual(gemma.repo, "mlx-community/gemma-4-e4b-it-4bit")
        self.assertIn("pinned MLX profile", gemma.notes)

    def test_scores_failed_model_as_unreliable(self) -> None:
        manifest = load_benchmark_manifest("configs/model-benchmark.json")
        candidate = manifest.candidates[0]

        score = score_candidate(
            candidate,
            {"status": "failed"},
            hardware_budget_gb=manifest.hardware_budget_gb,
            weights=manifest.decision_weights,
        )

        self.assertEqual(score["overall"], 0.0)
        self.assertEqual(score["reliability"], 0.0)

    def test_summarizes_primary_and_fallback_decisions(self) -> None:
        manifest = load_benchmark_manifest("configs/model-benchmark.json")
        results = [
            {
                "candidate_id": "qwen3-4b-mlx-4bit",
                "status": "ok",
                "load_seconds": 8.0,
                "aggregate": {"generation_tps_avg": 60.0, "peak_memory_gb": 4.0},
            },
            {
                "candidate_id": "qwen3-30b-a3b-2507-mlx-4bit",
                "status": "ok",
                "load_seconds": 40.0,
                "aggregate": {"generation_tps_avg": 24.0, "peak_memory_gb": 18.0},
            },
        ]

        summary = summarize_benchmarks(manifest, results)
        report = render_markdown_report(summary)

        self.assertEqual(summary["decision"]["primary_general"]["candidate_id"], "qwen3-30b-a3b-2507-mlx-4bit")
        self.assertEqual(summary["decision"]["fast_fallback"]["candidate_id"], "qwen3-4b-mlx-4bit")
        self.assertIn("Qwen3 30B-A3B", report)

    def test_parses_benchmark_json_after_mlx_warnings(self) -> None:
        parsed = _parse_json_stdout(
            "[WARNING] noisy mlx output\n"
            "{\n"
            '  "candidate_id": "qwen",\n'
            '  "status": "ok"\n'
            "}\n"
        )

        self.assertEqual(parsed["candidate_id"], "qwen")


if __name__ == "__main__":
    unittest.main()
