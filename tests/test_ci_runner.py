from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_ci_checks", ROOT / "scripts" / "run_ci_checks.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/run_ci_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CiRunnerTests(unittest.TestCase):
    def test_builds_cross_platform_argv_plan(self) -> None:
        runner = _load_runner()

        steps = runner.build_check_plan("python")

        self.assertEqual(
            [step.name for step in steps],
            [
                "compile",
                "unit tests",
                "smoke eval",
                "extended smoke eval",
                "quality gate",
                "hardware report",
                "packaging smoke",
            ],
        )
        for step in steps:
            self.assertIsInstance(step.command, list)
            self.assertEqual(step.command[0], "python")
            self.assertNotIn("&&", step.command)
            self.assertNotIn("|", step.command)
            self.assertNotIn("PYTHONPATH=src", step.command)

    def test_build_env_prepends_src_with_platform_separator(self) -> None:
        runner = _load_runner()

        env = runner.build_env(ROOT)

        first = env["PYTHONPATH"].split(os.pathsep)[0]
        self.assertEqual(first, str(ROOT / "src"))

    def test_dry_run_outputs_json_plan(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_ci_checks.py",
                "--dry-run",
                "--json",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["root"], str(ROOT))
        self.assertEqual(payload["steps"][0]["name"], "compile")
        self.assertEqual(payload["steps"][-1]["command"][1], "scripts/run_packaging_smoke.py")


if __name__ == "__main__":
    unittest.main()
