from __future__ import annotations

import argparse
import json
import sys

from .app_config import app_config_payload, load_app_config
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .config import load_config
from .evaluator import evaluate_router, load_eval_cases
from .extensions import create_plugin_scaffold, load_extension_registry, registry_payload
from .orchestrator import LocalMoE


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MoE orchestrator")
    parser.add_argument("--config")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--prompt")
    parser.add_argument("--eval")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--list-extensions", action="store_true")
    parser.add_argument("--create-plugin")
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    config_path = args.config or app_config.default_moe_config
    config = load_config(config_path)

    if args.doctor:
        payload = {
            "app": app_config_payload(app_config),
            "runtime": runtime_plan_payload(build_runtime_plan(config, app_config.runtime.preferred_backends)),
            "extensions": registry_payload(_registry(app_config)),
        }
        print(json.dumps(payload, indent=2))
        return

    if args.bootstrap:
        print(json.dumps(runtime_plan_payload(build_runtime_plan(config, app_config.runtime.preferred_backends)), indent=2))
        return

    if args.list_extensions:
        print(json.dumps(registry_payload(_registry(app_config)), indent=2))
        return

    if args.create_plugin:
        path = create_plugin_scaffold(args.create_plugin, root=app_config.extensions.plugins_dir)
        print(json.dumps({"created": str(path)}, indent=2))
        return

    if args.eval:
        cases = load_eval_cases(args.eval)
        print(json.dumps(evaluate_router(config, cases), indent=2))
        return

    moe = LocalMoE(config)

    if args.interactive:
        _interactive(moe, json_output=args.json_output)
        return

    if not args.prompt:
        parser.error("--prompt or --interactive is required unless --eval is provided")

    response = moe.generate(args.prompt)
    _print_response(response, json_output=args.json_output)


def _interactive(moe: LocalMoE, *, json_output: bool) -> None:
    print("myMoE interactive shell. Type /exit to quit.", file=sys.stderr)
    while True:
        try:
            prompt = input("mymoe> ")
        except EOFError:
            print(file=sys.stderr)
            return

        if prompt.strip() in {"/exit", "/quit"}:
            return
        if not prompt.strip():
            continue

        response = moe.generate(prompt)
        _print_response(response, json_output=json_output)


def _print_response(response: object, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_response_payload(response), indent=2))
        return

    print(response.content)
    print()
    print(
        json.dumps(
            _response_metadata(response),
            indent=2,
        )
    )


def _response_payload(response: object) -> dict[str, object]:
    payload = _response_metadata(response)
    payload["content"] = response.content
    payload["results"] = [item.__dict__ for item in response.results]
    return payload


def _response_metadata(response: object) -> dict[str, object]:
    return {
        "correlation_id": response.correlation_id,
        "selected": [item.__dict__ for item in response.route.selected],
        "fallback_order": list(response.route.fallback_order),
        "errors": list(response.errors),
    }


def _registry(app_config: object) -> object:
    return load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )


if __name__ == "__main__":
    main()
