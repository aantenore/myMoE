from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from local_moe.route_policy import load_route_policy, recommend_shadow_route
from local_moe.route_scorecard import load_route_scorecard
from local_moe.route_signals import TaskSignals
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    require_non_negative_int,
    require_sha256,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recommend, but never apply, a verified shadow route."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt")
    source.add_argument("--bridge-metadata")
    parser.add_argument("--signals", required=True)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--now", help="Optional canonical UTC evaluation time.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    source_payload = _load_object(
        Path(args.receipt or args.bridge_metadata),
        "route input",
    )
    if args.bridge_metadata:
        receipt_payload = source_payload.get("route_receipt")
        if not isinstance(receipt_payload, dict):
            raise VerifiedRoutingError(
                "Bridge metadata must contain a route_receipt object."
            )
    else:
        receipt_payload = source_payload
    signals = TaskSignals.from_payload(
        _load_object(Path(args.signals), "task signals")
    )
    policy = load_route_policy(args.policy)
    scorecard = load_route_scorecard(
        args.scorecard,
        now=args.now,
        require_fresh=True,
    )
    receipt = _receipt_view(receipt_payload)
    task = receipt.task
    profile = str(task.get("profile", ""))
    decision = recommend_shadow_route(
        receipt,
        signals,
        scorecard,
        policy,
        profile=profile,
        now=args.now,
    )
    payload = decision.payload()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


def _receipt_view(raw: dict[str, object]) -> SimpleNamespace:
    required = {
        "config_sha256",
        "local_gaps",
        "premium_call_budget",
        "premium_gaps",
        "remote_allowed",
        "route",
        "task",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise VerifiedRoutingError(
            f"Missing route receipt fields: {', '.join(missing)}."
        )
    task = raw["task"]
    if not isinstance(task, dict):
        raise VerifiedRoutingError("Route receipt task must be an object.")
    return SimpleNamespace(
        raw_payload=dict(raw),
        receipt_id=raw.get("receipt_id"),
        config_sha256=require_sha256(raw["config_sha256"], "config_sha256"),
        local_gaps=_string_tuple(raw["local_gaps"], "local_gaps"),
        premium_call_budget=require_non_negative_int(
            raw["premium_call_budget"], "premium_call_budget"
        ),
        premium_gaps=_string_tuple(raw["premium_gaps"], "premium_gaps"),
        remote_allowed=_boolean(raw["remote_allowed"], "remote_allowed"),
        route=raw["route"],
        task=task,
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise VerifiedRoutingError(f"{label} must be a string list.")
    return tuple(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise VerifiedRoutingError(f"{label} must be boolean.")
    return value


def _load_object(path: Path, label: str) -> dict[str, object]:
    def reject_constant(token: str) -> object:
        raise VerifiedRoutingError(f"Non-finite JSON number {token!r} is forbidden.")

    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise VerifiedRoutingError(f"{label} must be a JSON object.")
    return raw


if __name__ == "__main__":
    main()
