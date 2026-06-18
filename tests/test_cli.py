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
                "configs/moe.mock.json",
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

    def test_prompt_mode_runs_mock_generation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "configs/moe.mock.json",
                "--prompt",
                "Write Python tests for a class",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("[coding:mock-coder]", completed.stdout)
        self.assertIn('"correlation_id"', completed.stdout)

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


if __name__ == "__main__":
    unittest.main()
