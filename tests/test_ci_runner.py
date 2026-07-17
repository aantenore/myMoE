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
                "assistant bridge dependency contract",
                "unit tests",
                "smoke eval",
                "extended smoke eval",
                "live routing holdout",
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

        bridge_dependencies = next(
            step
            for step in steps
            if step.name == "assistant bridge dependency contract"
        )
        self.assertEqual(
            bridge_dependencies.command,
            ["python", "scripts/check_assistant_bridge_dependencies.py"],
        )
        unit_tests = next(step for step in steps if step.name == "unit tests")
        self.assertEqual(
            unit_tests.command,
            ["python", "-m", "unittest", "discover", "-s", "tests", "-v"],
        )

        holdout = next(step for step in steps if step.name == "live routing holdout")
        self.assertIn(
            "experiments/eval_set_live_general_holdout_v5.jsonl", holdout.command
        )
        quality_gate = next(step for step in steps if step.name == "quality gate")
        self.assertIn("configs/quality-gate-ci.json", quality_gate.command)

    def test_make_eval_holdout_uses_current_unseen_dataset(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        target = makefile.split("eval-holdout:", 1)[1].split("\n\n", 1)[0]

        self.assertIn("eval_set_live_general_holdout_v5.jsonl", target)
        self.assertNotIn("eval_set_live_general_holdout_v2.jsonl", target)

    def test_active_github_workflow_matches_documented_template(self) -> None:
        active = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        template = (ROOT / "docs" / "github-actions-ci.yml").read_text(
            encoding="utf-8"
        )

        self.assertEqual(active, template)
        self.assertIn(
            "uv run --locked --extra assistant-bridge python scripts/run_ci_checks.py",
            active,
        )
        self.assertIn('python-version: ["3.10", "3.12"]', active)
        self.assertIn("os: [ubuntu-latest, macos-latest, windows-latest]", active)

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

    def test_rejects_unsupported_python_with_actionable_message(self) -> None:
        runner = _load_runner()

        with self.assertRaisesRegex(SystemExit, "Python >= 3.10"):
            runner.require_supported_python((3, 9))


if __name__ == "__main__":
    unittest.main()
