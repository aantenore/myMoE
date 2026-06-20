from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.run_log import RunLogStore
from local_moe.runtime_optimizer import (
    build_runtime_optimizer_report,
    render_runtime_optimizer_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeOptimizerTests(unittest.TestCase):
    def test_reports_high_latency_without_prompt_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root)
            app_config = load_app_config(app_config_path)
            store = RunLogStore(root / "runtime" / "runs.jsonl")
            store.record_generation(
                mode="generate",
                prompt="Private optimizer prompt",
                latency_ms=45000,
                response_payload={
                    "content": "Private optimizer answer",
                    "correlation_id": "corr-optimizer",
                    "route": {"selected": [{"expert_id": "general"}], "fallback_order": []},
                    "results": [
                        {
                            "model": "synthetic-general",
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                        }
                    ],
                    "errors": [],
                },
            )

            report = build_runtime_optimizer_report(
                config_path="tests/fixtures/moe.synthetic.json",
                app_config=app_config,
                app_config_path=str(app_config_path),
                run_log_path=store.path,
                p95_latency_attention_ms=10000,
            )
            markdown = render_runtime_optimizer_markdown(report)

        rendered = json.dumps(report)
        signal_ids = {signal["id"] for signal in report["signals"]}
        action_ids = {action["id"] for action in report["actions"]}
        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["status"], "attention")
        self.assertEqual(report["mode"], "read_only")
        self.assertIn("latency", signal_ids)
        self.assertTrue(any(signal["id"] == "latency" and signal["status"] == "attention" for signal in report["signals"]))
        self.assertIn("review_performance_report", action_ids)
        self.assertEqual(report["run_log"]["summary"]["record_count"], 1)
        self.assertEqual(report["run_log"]["diagnostics"]["skipped_records"], 0)
        self.assertIn("# myMoE Runtime Optimizer Report", markdown)
        self.assertNotIn("Private optimizer prompt", rendered)
        self.assertNotIn("Private optimizer answer", rendered)

    def test_suggests_smoke_when_no_run_observations_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root)
            app_config = load_app_config(app_config_path)

            report = build_runtime_optimizer_report(
                config_path="tests/fixtures/moe.synthetic.json",
                app_config=app_config,
                app_config_path=str(app_config_path),
            )

        self.assertEqual(report["run_log"]["summary"]["record_count"], 0)
        self.assertIn("run_generation_smoke", {action["id"] for action in report["actions"]})
        self.assertTrue(any(signal["id"] == "run_observations" for signal in report["signals"]))


def _write_temp_app_config(root: Path) -> Path:
    raw = json.loads((ROOT / "configs" / "app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = "tests/fixtures/moe.synthetic.json"
    raw["runtime"]["work_dir"] = str(root / "runtime")
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
