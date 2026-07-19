from __future__ import annotations

import argparse
import json

from local_moe.route_scorecard import (
    build_route_scorecard,
    load_outcome_payloads,
    write_route_scorecard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a content-addressed verified-routing scorecard."
    )
    parser.add_argument("--records", required=True, help="Outcome JSON or JSONL path.")
    parser.add_argument("--out", required=True, help="Destination scorecard JSON path.")
    parser.add_argument(
        "--minimum-evidence-strength",
        default="independent",
        help="Lowest evidence strength accepted into binary aggregates.",
    )
    parser.add_argument("--ttl-seconds", type=int, default=86_400)
    parser.add_argument(
        "--generated-at",
        help="Optional canonical UTC timestamp for deterministic builds.",
    )
    args = parser.parse_args()

    records = load_outcome_payloads(args.records)
    scorecard = build_route_scorecard(
        records,
        minimum_evidence_strength=args.minimum_evidence_strength,
        generated_at=args.generated_at,
        ttl_seconds=args.ttl_seconds,
    )
    write_route_scorecard(args.out, scorecard)
    print(
        json.dumps(
            {
                "digest": scorecard.digest,
                "entries": len(scorecard.entries),
                "minimum_evidence_strength": scorecard.minimum_evidence_strength,
                "source_digest": scorecard.source_digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
