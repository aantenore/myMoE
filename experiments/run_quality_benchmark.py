from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_moe.quality_benchmark import (
    QualityBenchmarkError,
    load_benchmark_spec,
    run_quality_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a single general model with myMoE top-1 and top-2."
    )
    parser.add_argument("--manifest", default="configs/quality-benchmark.json")
    parser.add_argument("--out", default="outputs/quality-benchmark.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        spec = load_benchmark_spec(args.manifest)
        payload = run_quality_benchmark(spec, limit=args.limit)
    except (OSError, QualityBenchmarkError, ValueError) as exc:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "gate": {
                "status": "failed",
                "passed": False,
                "reason": str(exc),
            },
        }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gate": payload["gate"]["status"],
                "out": str(out_path),
            },
            indent=2,
        )
    )
    if payload["status"] == "blocked":
        raise SystemExit(2)
    if payload["status"] != "complete" or not payload["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
