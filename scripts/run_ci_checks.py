from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the myMoE cross-platform quality gate.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned command list without running it.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Print the dry-run plan as JSON.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    steps = build_check_plan(sys.executable)
    if args.dry_run:
        payload = {"root": str(root), "steps": [step_payload(step) for step in steps]}
        if args.json_output:
            print(json.dumps(payload, indent=2))
        else:
            for step in steps:
                print(f"{step.name}: {format_command(step.command)}")
        return

    require_supported_python(sys.version_info)
    env = build_env(root)
    for step in steps:
        print(f"==> {step.name}", flush=True)
        subprocess.run(step.command, cwd=root, env=env, check=True)


class CheckStep:
    def __init__(self, name: str, command: list[str]) -> None:
        self.name = name
        self.command = command


def build_check_plan(python: str) -> list[CheckStep]:
    return [
        CheckStep("compile", [python, "-m", "compileall", "src", "tests", "experiments", "scripts"]),
        CheckStep(
            "assistant bridge dependency contract",
            [python, "scripts/check_assistant_bridge_dependencies.py"],
        ),
        CheckStep("unit tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"]),
        CheckStep(
            "smoke eval",
            [
                python,
                "experiments/run_smoke_eval.py",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--eval",
                "experiments/eval_set.jsonl",
                "--out",
                "outputs/smoke-eval.json",
            ],
        ),
        CheckStep(
            "extended smoke eval",
            [
                python,
                "experiments/run_smoke_eval.py",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--eval",
                "experiments/eval_set_extended.jsonl",
                "--out",
                "outputs/smoke-eval-extended.json",
            ],
        ),
        CheckStep(
            "live routing holdout",
            [
                python,
                "experiments/run_smoke_eval.py",
                "--config",
                "configs/moe.live.general-mlx.example.json",
                "--eval",
                "experiments/eval_set_live_general_holdout_v5.jsonl",
                "--training-labels",
                "experiments/route_labels_live_general.jsonl",
                "--out",
                "outputs/live-general-routing-holdout.json",
            ],
        ),
        CheckStep(
            "verified routing shadow eval",
            [
                python,
                "experiments/eval_verified_routing.py",
                "--fixture",
                "tests/fixtures/verified-routing-eval.json",
                "--out",
                "outputs/verified-routing-shadow-eval.json",
            ],
        ),
        CheckStep(
            "quality gate",
            [
                python,
                "experiments/run_quality_gate.py",
                "--config",
                "configs/quality-gate-ci.json",
                "--out",
                "outputs/quality-gate.json",
            ],
        ),
        CheckStep("hardware report", [python, "scripts/hardware_report.py"]),
        CheckStep("packaging smoke", [python, "scripts/run_packaging_smoke.py"]),
    ]


def require_supported_python(version_info: Any) -> None:
    major = int(version_info[0])
    minor = int(version_info[1])
    if (major, minor) < (3, 10):
        raise SystemExit(
            "myMoE requires Python >= 3.10. "
            "Run uv with Python 3.12 or use the project virtual environment."
        )


def build_env(_root: Path) -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() != "PYTHONPATH"
    }


def step_payload(step: CheckStep) -> dict[str, Any]:
    return {"name": step.name, "command": step.command}


def format_command(command: list[str]) -> str:
    return " ".join(_quote(arg) for arg in command)


def _quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    main()
