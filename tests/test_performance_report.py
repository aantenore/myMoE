from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.performance_report import build_performance_report, render_performance_report_markdown


class PerformanceReportTests(unittest.TestCase):
    def test_builds_sanitized_report_from_benchmark_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark.json"
            hardware = root / "hardware.json"
            decision = root / "decision.md"
            benchmark.write_text(
                json.dumps(
                    {
                        "created_at": "2026-06-19T10:00:00+0000",
                        "manifest": "configs/model-benchmark.json",
                        "max_tokens": 16,
                        "max_kv_size": 2048,
                        "prompt_count": 1,
                        "results": [
                            {
                                "candidate_id": "qwen3-4b-mlx-4bit",
                                "status": "ok",
                                "load_seconds": 2.0,
                                "aggregate": {
                                    "generation_tps_avg": 80.0,
                                    "peak_memory_gb": 3.0,
                                },
                                "records": [{"content_excerpt": "do not expose this"}],
                            },
                            {
                                "candidate_id": "qwen3-30b-a3b-2507-mlx-4bit",
                                "status": "ok",
                                "load_seconds": 7.0,
                                "aggregate": {
                                    "generation_tps_avg": 70.0,
                                    "peak_memory_gb": 17.0,
                                },
                                "records": [{"content_excerpt": "do not expose this either"}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hardware.write_text('{"cpu_brand": "Test CPU", "machine": "test", "memory_gib": 24}', encoding="utf-8")
            decision.write_text("# Decision\n", encoding="utf-8")

            report = build_performance_report(
                benchmark_path=benchmark,
                hardware_profile_path=hardware,
                decision_markdown_path=decision,
            )
            rendered = render_performance_report_markdown(report)

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["status"], "ready_partial")
        self.assertEqual(report["decision"]["primary_general"]["candidate_id"], "qwen3-30b-a3b-2507-mlx-4bit")
        self.assertEqual(report["decision"]["fast_fallback"]["candidate_id"], "qwen3-4b-mlx-4bit")
        self.assertIn("ranked", report)
        self.assertNotIn("records", json.dumps(report))
        self.assertNotIn("content_excerpt", json.dumps(report))
        self.assertIn("myMoE Performance Report", rendered)
        self.assertIn("Qwen3 30B-A3B", rendered)

    def test_missing_benchmark_returns_actionable_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_performance_report(
                benchmark_path=Path(tmp) / "missing.json",
                hardware_profile_path=Path(tmp) / "missing-hardware.json",
                decision_markdown_path=Path(tmp) / "missing-decision.md",
            )

        self.assertEqual(report["status"], "missing")
        self.assertEqual(report["coverage"]["status"], "missing")
        self.assertIn("Run make benchmark-small", report["recommendations"][0])


if __name__ == "__main__":
    unittest.main()
