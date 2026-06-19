from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from local_moe.app_config import load_app_config
from local_moe.bootstrap import build_runtime_plan, runtime_plan_payload
from local_moe.config import load_config
from local_moe.model_downloads import build_model_download_requests, validate_local_file_request


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
        _download_models(config, plan.backend)


def _download_models(config: object, default_backend: str) -> None:
    try:
        requests = build_model_download_requests(config, default_backend)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    snapshot_download = _snapshot_downloader(requests)

    for request in requests:
        if request.kind == "local_file":
            try:
                validate_local_file_request(request)
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc
            print(f"Using existing local model file: {request.model}", file=sys.stderr)
            continue

        if request.kind == "ollama_pull":
            subprocess.run(request.command, check=True)
            continue

        if request.kind == "huggingface_snapshot":
            if not request.repo_id:
                raise SystemExit(f"Missing Hugging Face repo id for model: {request.model}")
            kwargs = {}
            if request.allow_patterns:
                kwargs["allow_patterns"] = list(request.allow_patterns)
            print(f"Downloading {request.model} for {request.backend}...", file=sys.stderr)
            snapshot_download(request.repo_id, **kwargs)
            continue

        raise SystemExit(f"Unsupported model download request: {request.kind}")


def _snapshot_downloader(requests: tuple[object, ...]):
    if not any(getattr(request, "kind", "") == "huggingface_snapshot" for request in requests):
        return None
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(f"huggingface_hub is required. Run bootstrap with --execute first: {exc}") from exc
    return snapshot_download


def _venv_exists() -> bool:
    return Path(".venv/bin/python").exists() or Path(".venv/Scripts/python.exe").exists()


if __name__ == "__main__":
    main()
