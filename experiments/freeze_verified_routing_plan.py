from __future__ import annotations

import argparse
import json
from typing import Sequence

from local_moe.route_policy import route_policy_from_payload
from local_moe.route_promotion import (
    build_evidence_plan,
    load_promotion_cases,
    load_promotion_gate_policy,
    load_strict_promotion_json,
    write_content_addressed_json,
)
from local_moe.route_scorecard import route_scorecard_from_payload
from local_moe.verified_routing_contracts import VerifiedRoutingError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a content-addressed paired routing evidence plan."
    )
    parser.add_argument("--cases", required=True, help="Planned case JSON path.")
    parser.add_argument("--route-policy", required=True)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--gate-policy", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--canary-basis-points", required=True, type=int)
    parser.add_argument("--manifest-ttl-seconds", required=True, type=int)
    parser.add_argument("--assignment-salt-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cases = load_promotion_cases(args.cases)
    route_policy_raw = load_strict_promotion_json(args.route_policy)
    scorecard_raw = load_strict_promotion_json(args.scorecard)
    if not isinstance(route_policy_raw, dict) or not isinstance(scorecard_raw, dict):
        raise VerifiedRoutingError("Route policy and scorecard must be objects.")
    route_policy = route_policy_from_payload(route_policy_raw)
    scorecard = route_scorecard_from_payload(
        scorecard_raw, require_fresh=False
    )
    gate_policy = load_promotion_gate_policy(args.gate_policy)
    plan = build_evidence_plan(
        cases,
        route_policy=route_policy,
        scorecard=scorecard,
        gate_policy=gate_policy,
        created_at=args.created_at,
        canary_basis_points=args.canary_basis_points,
        manifest_ttl_seconds=args.manifest_ttl_seconds,
        assignment_salt_sha256=args.assignment_salt_sha256,
    )
    write_content_addressed_json(args.out, plan)
    print(
        json.dumps(
            {
                "cases": len(plan.cases),
                "plan_sha256": plan.plan_sha256,
                "split_sha256": plan.split_sha256,
            },
            sort_keys=True,
        )
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
