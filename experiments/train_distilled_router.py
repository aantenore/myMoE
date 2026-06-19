from __future__ import annotations

import argparse
import json

from local_moe.distilled_router import (
    load_route_labels,
    train_distilled_router_artifact,
    write_distilled_router_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local distilled router artifact from route labels.")
    parser.add_argument("--labels", required=True, help="Input route-label JSONL.")
    parser.add_argument("--out", required=True, help="Output distilled router artifact JSON.")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    args = parser.parse_args()

    labels = load_route_labels(args.labels)
    artifact = train_distilled_router_artifact(
        labels,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )
    write_distilled_router_artifact(artifact, args.out)
    print(
        json.dumps(
            {
                "training_cases": artifact["training_cases"],
                "experts": sorted(artifact["expert_counts"]),
                "out": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
