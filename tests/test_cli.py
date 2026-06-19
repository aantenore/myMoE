from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


class CliTests(unittest.TestCase):
    def test_eval_mode_prints_router_metrics(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--eval",
                "experiments/eval_set.jsonl",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["accuracy"], 1.0)
        self.assertEqual(payload["total"], 8)

    def test_prompt_mode_runs_synthetic_generation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--prompt",
                "Write Python tests for a class",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("[coding:synthetic-coder]", completed.stdout)
        self.assertIn('"correlation_id"', completed.stdout)
        self.assertIn('"disagreement": null', completed.stdout)

    def test_doctor_prints_runtime_and_extensions(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--doctor",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["app"]["mode"], "local_model_required")
        self.assertIn("runtime", payload)
        self.assertTrue(payload["extensions"]["tools"])

    def test_cron_status_prints_jobs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--cron-status",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertIn("jobs", payload)
        self.assertIn("memory-maintenance", {item["id"] for item in payload["jobs"]})

    def test_run_cron_dry_run_prints_due_jobs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--run-cron",
                "--cron-dry-run",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertIn("results", payload)
        self.assertTrue(all(item["status"] == "dry_run" for item in payload["results"]))

    def test_run_tool_prints_tool_result(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--run-tool",
                "mcp.search_capabilities",
                "--tool-input",
                '{"query":"filesystem"}',
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["payload"]["servers"][0]["name"], "filesystem")


if __name__ == "__main__":
    unittest.main()
