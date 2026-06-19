from __future__ import annotations

import argparse
import json
import sys

from local_moe.app_config import load_app_config
from local_moe.bootstrap import build_runtime_plan, runtime_plan_payload
from local_moe.config import load_config
from local_moe.setup_runner import run_runtime_setup, setup_run_payload


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
        print("Preparing runtime install commands...", file=sys.stderr)

    if args.download_models:
        print("Preparing model assets...", file=sys.stderr)

    if args.execute or args.download_models:
        result = run_runtime_setup(
            config_path=args.config or app_config.default_moe_config,
            app_config_path=args.app_config,
            execute=args.execute,
            download_models=args.download_models,
            confirm=True,
        )
        print(json.dumps(setup_run_payload(result), indent=2))
        if result.status in {"error", "confirmation_required", "manual_required"}:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
