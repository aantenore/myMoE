from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .evaluator import evaluate_router, load_eval_cases
from .orchestrator import LocalMoE


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MoE orchestrator")
    parser.add_argument("--config", default="configs/moe.mock.json")
    parser.add_argument("--prompt")
    parser.add_argument("--eval")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    config = load_config(args.config)

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


if __name__ == "__main__":
    main()
