from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from local_moe.route_signals import signals_from_route_receipt
from local_moe.verified_routing_contracts import VerifiedRoutingError


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive content-free structural signals from a route receipt."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt", help="RouteDecisionReceipt JSON path.")
    source.add_argument(
        "--bridge-metadata",
        help="BridgeRunResult.metadata_payload JSON path.",
    )
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--out", required=True, help="Destination TaskSignals JSON path.")
    args = parser.parse_args(argv)

    path = Path(args.receipt or args.bridge_metadata)
    raw = _load_object(path)
    if args.bridge_metadata:
        receipt = raw.get("route_receipt")
        if not isinstance(receipt, dict):
            raise VerifiedRoutingError(
                "Bridge metadata must contain a route_receipt object."
            )
    else:
        receipt = raw
    signals = signals_from_route_receipt(
        receipt,
        context_tokens=args.context_tokens,
    )
    payload = signals.payload()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


def _load_object(path: Path) -> dict[str, object]:
    def reject_constant(token: str) -> object:
        raise VerifiedRoutingError(f"Non-finite JSON number {token!r} is forbidden.")

    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise VerifiedRoutingError("Input must be a JSON object.")
    return raw


if __name__ == "__main__":
    main()
