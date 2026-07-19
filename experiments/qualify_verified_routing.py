from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from local_moe.route_policy import route_policy_from_payload
from local_moe.route_promotion import (
    evaluate_route_promotion,
    load_evidence_plan,
    load_promotion_outcome_payloads,
    load_promotion_gate_policy,
    load_strict_promotion_json,
    write_content_addressed_json,
)
from local_moe.route_scorecard import route_scorecard_from_payload
from local_moe.verified_routing_contracts import VerifiedRoutingError


_EXIT_CODES = {"eligible": 0, "inconclusive": 2, "ineligible": 3}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate preregistered paired route evidence without applying a route."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gate-policy", required=True)
    parser.add_argument("--route-policy", required=True)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--training-records", required=True)
    parser.add_argument("--holdout-records", required=True)
    parser.add_argument("--evaluated-at", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        raise VerifiedRoutingError(
            "Manifest output already exists; use a new per-run path to avoid "
            "confusing stale eligibility with the current result."
        )

    route_policy_raw = load_strict_promotion_json(args.route_policy)
    scorecard_raw = load_strict_promotion_json(args.scorecard)
    if not isinstance(route_policy_raw, dict) or not isinstance(scorecard_raw, dict):
        raise VerifiedRoutingError("Route policy and scorecard must be objects.")

    report, manifest = evaluate_route_promotion(
        plan=load_evidence_plan(args.plan),
        gate_policy=load_promotion_gate_policy(args.gate_policy),
        route_policy=route_policy_from_payload(route_policy_raw),
        scorecard=route_scorecard_from_payload(
            scorecard_raw, require_fresh=False
        ),
        training_records=load_promotion_outcome_payloads(
            args.training_records
        ),
        holdout_records=load_promotion_outcome_payloads(args.holdout_records),
        evaluated_at=args.evaluated_at,
    )
    write_content_addressed_json(args.report, report)
    if manifest is not None:
        write_content_addressed_json(manifest_path, manifest)
    payload = report.payload()
    status = str(payload["status"])
    print(
        json.dumps(
            {
                "manifest_emitted": manifest is not None,
                "report_sha256": report.digest,
                "status": status,
            },
            sort_keys=True,
        )
    )
    return _EXIT_CODES[status]


if __name__ == "__main__":
    raise SystemExit(main())
