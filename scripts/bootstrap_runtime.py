from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from local_moe.app_config import load_app_config
from local_moe.bootstrap import build_runtime_plan, runtime_plan_payload
from local_moe.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure local runtime and optionally download models.")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--config")
    parser.add_argument("--execute", action="store_true", help="Run safe install commands for the detected backend.")
    parser.add_argument("--download-models", action="store_true", help="Download configured models without starting servers.")
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    config = load_config(args.config or app_config.default_moe_config)
    plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
    payload = runtime_plan_payload(plan)
    print(json.dumps(payload, indent=2))

    if args.execute:
        for command in plan.install_commands:
            if command and command[0] == "install":
                print(f"Manual install required: {' '.join(command)}", file=sys.stderr)
                continue
            if tuple(command[:2]) == ("uv", "venv") and _venv_exists():
                print("Existing .venv detected; skipping uv venv.", file=sys.stderr)
                continue
            subprocess.run(command, check=True)

    if args.download_models:
        if plan.backend == "mlx_lm":
            _download_mlx_models(config)
        elif plan.backend == "ollama":
            for command in plan.model_commands:
                subprocess.run(command, check=True)
        else:
            raise SystemExit("Automatic model download is not implemented for llama.cpp fallback.")


def _download_mlx_models(config: object) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(f"huggingface_hub is required. Run bootstrap with --execute first: {exc}") from exc

    seen = set()
    for expert in config.experts:
        if expert.provider != "openai_compatible" or expert.model in seen:
            continue
        seen.add(expert.model)
        snapshot_download(expert.model)


def _venv_exists() -> bool:
    return Path(".venv/bin/python").exists() or Path(".venv/Scripts/python.exe").exists()


if __name__ == "__main__":
    main()
