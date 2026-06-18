from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from local_moe.config import load_config
from local_moe.evaluator import load_eval_cases
from local_moe.hardware import write_hardware_report
from local_moe.orchestrator import LocalMoE
from local_moe.runtime import LlamaServerSpec, ManagedLlamaServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/moe.live.qwen25-coder.json")
    parser.add_argument("--eval", default="experiments/eval_set.jsonl")
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--out", default="outputs/live-eval.json")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    profile = write_hardware_report("outputs/hardware-profile.json")
    config = load_config(args.config)
    cases = load_eval_cases(args.eval)[: args.limit]
    moe = LocalMoE(config)

    spec = LlamaServerSpec(
        binary=args.llama_server,
        model=args.model,
        host="127.0.0.1",
        port=args.port,
    )

    records = []
    with ManagedLlamaServer(spec, "work/runs/live-qwen25-coder.log"):
        for case in cases:
            started = time.perf_counter()
            response = moe.generate(case.prompt, correlation_id=case.id)
            elapsed = time.perf_counter() - started
            result = response.results[0]
            records.append(
                {
                    "id": case.id,
                    "expected_expert": case.expected_expert,
                    "selected_expert": response.route.selected[0].expert_id,
                    "actual_expert": result.expert_id,
                    "latency_seconds": round(elapsed, 3),
                    "completion_tokens": result.completion_tokens,
                    "predicted_tokens_per_second": result.predicted_tokens_per_second,
                    "content_excerpt": result.content[:400],
                    "errors": list(response.errors),
                }
            )

    out = {
        "hardware": profile.__dict__,
        "server": spec.__dict__,
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

