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
            "quality gate",
            [
                python,
                "experiments/run_quality_gate.py",
                "--config",
                "configs/quality-gate.json",
                "--out",
                "outputs/quality-gate.json",
            ],
        ),
        CheckStep("hardware report", [python, "scripts/hardware_report.py"]),
        CheckStep("packaging smoke", [python, "scripts/run_packaging_smoke.py"]),
    ]


def build_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    src = str(root / "src")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env


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
