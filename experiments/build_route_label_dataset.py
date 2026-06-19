from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_moe.distilled_router import labels_from_eval_cases, write_route_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build route-label JSONL from curated eval cases.")
    parser.add_argument("--eval", required=True, help="Input eval JSONL with expected_expert labels.")
    parser.add_argument("--out", required=True, help="Output route-label JSONL.")
    parser.add_argument("--teacher-source", default="curated_eval")
    args = parser.parse_args()

    raw_cases = [
        json.loads(line)
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = labels_from_eval_cases(raw_cases, teacher_source=args.teacher_source)
    write_route_labels(labels, args.out)
    print(json.dumps({"labels": len(labels), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
