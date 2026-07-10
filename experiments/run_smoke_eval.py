from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from local_moe.config import load_config
from local_moe.distilled_router import load_route_labels
from local_moe.evaluator import evaluate_router, load_eval_cases
from local_moe.evaluation_integrity import analyze_route_holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--training-labels",
        help="Optional route-label JSONL used to prove that this eval is a disjoint holdout.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cases = load_eval_cases(args.eval)
    result = evaluate_router(config, cases)
    result["provenance"] = {
        "config_path": args.config,
        "config_sha256": _file_sha256(Path(args.config)),
        "eval_path": args.eval,
        "eval_sha256": _file_sha256(Path(args.eval)),
    }
    integrity_passed = True

    if args.training_labels:
        labels = load_route_labels(args.training_labels)
        training_records = [label.__dict__ for label in labels]
        holdout_records = [
            {
                "id": case.id,
                "prompt": case.prompt,
                "expected_expert": case.expected_expert,
                "complexity": case.complexity,
            }
            for case in cases
        ]
        integrity = analyze_route_holdout(training_records, holdout_records)
        artifact_path = Path(config.routing.distilled.artifact_path)
        artifact = (
            json.loads(artifact_path.read_text(encoding="utf-8"))
            if config.routing.distilled.enabled and artifact_path.exists()
            else {}
        )
        artifact_training_data_sha256 = str(
            artifact.get("training_data_sha256", "")
        )
        integrity["artifact_matches_training"] = bool(
            artifact_training_data_sha256
            and artifact_training_data_sha256 == integrity["training_data_sha256"]
        )
        integrity["passed"] = bool(
            integrity["passed"] and integrity["artifact_matches_training"]
        )
        integrity_passed = bool(integrity["passed"])
        result["integrity"] = integrity
        result["provenance"].update({
            "holdout_data_sha256": integrity["holdout_data_sha256"],
            "training_labels_path": args.training_labels,
            "training_data_sha256": integrity["training_data_sha256"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": (
                _file_sha256(artifact_path) if artifact_path.exists() else ""
            ),
            "artifact_training_data_sha256": artifact_training_data_sha256,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "accuracy": result["accuracy"],
                "total": result["total"],
                "integrity_passed": integrity_passed,
            }
        )
    )
    if not integrity_passed:
        sys.exit(1)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
