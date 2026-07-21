from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ci_checks", ROOT / "scripts" / "run_ci_checks.py"
    )
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
                "verified routing shadow eval",
                "desktop semantic benchmark",
                "adaptive cell advisor contract benchmark",
                "adaptive cell execution gate contract benchmark",
                "bound cell attestor contract benchmark",
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
        shadow_eval = next(
            step for step in steps if step.name == "verified routing shadow eval"
        )
        self.assertIn("tests/fixtures/verified-routing-eval.json", shadow_eval.command)
        self.assertIn("outputs/verified-routing-shadow-eval.json", shadow_eval.command)
        quality_gate = next(step for step in steps if step.name == "quality gate")
        self.assertIn("configs/quality-gate-ci.json", quality_gate.command)
        desktop_benchmark = next(
            step for step in steps if step.name == "desktop semantic benchmark"
        )
        self.assertEqual(
            desktop_benchmark.command,
            [
                "python",
                "experiments/benchmark_desktop_semantic.py",
                "--out",
                "outputs/desktop-semantic-benchmark.json",
            ],
        )
        adaptive_benchmark = next(
            step
            for step in steps
            if step.name == "adaptive cell advisor contract benchmark"
        )
        self.assertEqual(
            adaptive_benchmark.command,
            [
                "python",
                "experiments/benchmark_adaptive_cell_advisor.py",
                "--out",
                "outputs/adaptive-cell-advisor-contract.json",
            ],
        )
        execution_gate_benchmark = next(
            step
            for step in steps
            if step.name == "adaptive cell execution gate contract benchmark"
        )
        self.assertEqual(
            execution_gate_benchmark.command,
            [
                "python",
                "experiments/benchmark_cell_execution_gate.py",
                "--out",
                "outputs/cell-execution-gate-contract.json",
            ],
        )
        runtime_binding_benchmark = next(
            step
            for step in steps
            if step.name == "bound cell attestor contract benchmark"
        )
        self.assertEqual(
            runtime_binding_benchmark.command,
            [
                "python",
                "experiments/benchmark_runtime_binding.py",
                "--out",
                "outputs/runtime-binding-contract.json",
            ],
        )

    def test_make_eval_holdout_uses_current_unseen_dataset(self) -> None:
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        target = makefile.split("eval-holdout:", 1)[1].split("\n\n", 1)[0]

        self.assertIn("eval_set_live_general_holdout_v5.jsonl", target)
        self.assertNotIn("eval_set_live_general_holdout_v2.jsonl", target)

    def test_active_github_workflow_matches_documented_template(self) -> None:
        active = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        template = (ROOT / "docs" / "github-actions-ci.yml").read_text(encoding="utf-8")

        self.assertEqual(active, template)
        self.assertIn(
            "uv run --locked --extra assistant-bridge python scripts/run_ci_checks.py",
            active,
        )
        self.assertIn('python-version: ["3.10", "3.12"]', active)
        self.assertIn("os: [ubuntu-latest, macos-latest, windows-latest]", active)
        browser_job_header = active.split("  browser-canary:\n", 1)[1].split(
            "    steps:\n", 1
        )[0]
        self.assertNotIn("runner.temp", browser_job_header)
        self.assertEqual(
            active.count("NPM_CONFIG_CACHE: ${{ runner.temp }}/mymoe-npm-cache"),
            2,
        )
        self.assertIn("  desktop-provider-contract:\n", active)
        self.assertIn(
            "uv run --locked --extra desktop\n"
            "          python scripts/check_desktop_provider_contract.py",
            active,
        )

    def test_build_env_removes_pythonpath_and_preserves_other_values(self) -> None:
        runner = _load_runner()

        with patch.dict(
            os.environ,
            {
                "PYTHONPATH": os.pathsep.join(["injected", "source"]),
                "MYMOE_CI_SENTINEL": "preserved",
            },
            clear=False,
        ):
            expected = {
                key: value
                for key, value in os.environ.items()
                if key.upper() != "PYTHONPATH"
            }
            env = runner.build_env(ROOT)

        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["MYMOE_CI_SENTINEL"], "preserved")
        self.assertEqual(env, expected)

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
        self.assertEqual(
            payload["steps"][-1]["command"][1], "scripts/run_packaging_smoke.py"
        )

    def test_rejects_unsupported_python_with_actionable_message(self) -> None:
        runner = _load_runner()

        with self.assertRaisesRegex(SystemExit, "Python >= 3.10"):
            runner.require_supported_python((3, 9))


if __name__ == "__main__":
    unittest.main()
