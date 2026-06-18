from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_moe.config import load_config
from local_moe.evaluator import evaluate_router, load_eval_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    cases = load_eval_cases(args.eval)
    result = evaluate_router(config, cases)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"accuracy": result["accuracy"], "total": result["total"]}))


if __name__ == "__main__":
    main()

