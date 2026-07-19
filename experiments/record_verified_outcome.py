from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from local_moe.route_outcomes import OutcomeStore, build_verified_outcome


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Record one metadata-only verified routing outcome."
    )
    parser.add_argument(
        "--bridge-metadata",
        "--bridge",
        dest="bridge_metadata",
        required=True,
        help="BridgeRunResult.metadata_payload JSON file.",
    )
    parser.add_argument("--signals", required=True, help="TaskSignals JSON file.")
    parser.add_argument(
        "--store",
        "--outcomes",
        dest="store",
        required=True,
        help="Append-only outcome JSONL file.",
    )
    parser.add_argument("--estimated-cost-usd", type=float)
    args = parser.parse_args(argv)

    bridge_metadata = _load_object(Path(args.bridge_metadata), "bridge metadata")
    signals = _load_object(Path(args.signals), "signals")
    record = build_verified_outcome(
        bridge_metadata,
        signals,
        estimated_cost_usd=args.estimated_cost_usd,
    )
    OutcomeStore(args.store).append(record)
    print(json.dumps(record.payload(), ensure_ascii=True, allow_nan=False, sort_keys=True))


def _load_object(path: Path, label: str) -> dict[str, object]:
    def reject_constant(token: str) -> object:
        raise ValueError(f"Non-finite JSON number {token!r} is forbidden.")

    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{label} must be a JSON object.")
    return raw


if __name__ == "__main__":
    main()
