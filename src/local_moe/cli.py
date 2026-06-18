from __future__ import annotations

import argparse
import json

from .config import load_config
from .evaluator import evaluate_router, load_eval_cases
from .orchestrator import LocalMoE


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MoE orchestrator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--eval")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.eval:
        cases = load_eval_cases(args.eval)
        print(json.dumps(evaluate_router(config, cases), indent=2))
        return

    if not args.prompt:
        parser.error("--prompt is required unless --eval is provided")

    moe = LocalMoE(config)
    response = moe.generate(args.prompt)
    print(response.content)
    print()
    print(
        json.dumps(
            {
                "correlation_id": response.correlation_id,
                "selected": [item.__dict__ for item in response.route.selected],
                "errors": response.errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

