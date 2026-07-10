from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
from typing import Any

from local_moe.config import load_config
from local_moe.evaluator import evaluate_router, load_eval_cases
from local_moe.evaluation_integrity import analyze_route_holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quality-gate.json")
    parser.add_argument("--out", default="outputs/quality-gate.json")
    args = parser.parse_args()

    gate_config = _read_json(Path(args.config))
    checks = [
        _check_required_files(gate_config.get("required_files", [])),
        _check_routing_eval(gate_config.get("routing_eval", {})),
        _check_routing_holdout(gate_config.get("routing_holdout", {})),
        _check_forbidden_listeners(gate_config.get("forbidden_listeners", [])),
    ]

    passed = all(check["passed"] for check in checks)
    report = {
        "passed": passed,
        "checks": checks,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "out": str(out)}, indent=2))

    if not passed:
        sys.exit(1)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_required_files(files: object) -> dict[str, Any]:
    missing = [path for path in files if not Path(str(path)).exists()]
    return {
        "name": "required_files",
        "passed": not missing,
        "missing": missing,
    }


def _check_routing_eval(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "name": "routing_eval",
            "passed": False,
            "error": "routing_eval config must be an object",
        }

    result_path = Path(str(raw.get("result_path", "")))
    if not result_path.exists():
        return {
            "name": "routing_eval",
            "passed": False,
            "error": f"Missing routing eval result: {result_path}",
        }

    result = _read_json(result_path)
    min_accuracy = float(raw.get("min_accuracy", 0.0))
    min_total = int(raw.get("min_total", 0))
    required_complexities = {str(item) for item in raw.get("required_complexities", [])}
    observed_complexities = set(result.get("by_complexity", {}).keys())

    accuracy = float(result.get("accuracy", 0.0))
    total = int(result.get("total", 0))
    missing_complexities = sorted(required_complexities - observed_complexities)
    failed_cases = [
        item
        for item in result.get("results", [])
        if isinstance(item, dict) and not item.get("passed")
    ]

    passed = (
        accuracy >= min_accuracy
        and total >= min_total
        and not missing_complexities
        and not failed_cases
    )

    return {
        "name": "routing_eval",
        "passed": passed,
        "accuracy": accuracy,
        "min_accuracy": min_accuracy,
        "total": total,
        "min_total": min_total,
        "missing_complexities": missing_complexities,
        "failed_case_ids": [item.get("id") for item in failed_cases],
    }


def _check_routing_holdout(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "name": "routing_holdout",
            "passed": False,
            "error": "routing_holdout config must be an object",
        }

    paths = {
        "result": Path(str(raw.get("result_path", ""))),
        "config": Path(str(raw.get("config_path", ""))),
        "holdout": Path(str(raw.get("eval_path", ""))),
        "training": Path(str(raw.get("training_labels_path", ""))),
        "artifact": Path(str(raw.get("artifact_path", ""))),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {
            "name": "routing_holdout",
            "passed": False,
            "error": f"Missing routing holdout inputs: {', '.join(sorted(missing))}",
        }

    result = _read_json(paths["result"])
    holdout_records = _read_jsonl(paths["holdout"])
    training_records = _read_jsonl(paths["training"])
    artifact = _read_json(paths["artifact"])
    moe_config = load_config(paths["config"])
    recomputed_result = evaluate_router(moe_config, load_eval_cases(paths["holdout"]))
    integrity = analyze_route_holdout(training_records, holdout_records)
    artifact_training_sha = str(artifact.get("training_data_sha256", ""))
    artifact_sha = _file_sha256(paths["artifact"])
    artifact_matches_training = bool(
        artifact_training_sha
        and artifact_training_sha == integrity["training_data_sha256"]
    )
    configured_artifact_path = Path(moe_config.routing.distilled.artifact_path)
    artifact_path_matches_config = bool(
        moe_config.routing.distilled.enabled
        and configured_artifact_path.resolve() == paths["artifact"].resolve()
    )
    provenance = result.get("provenance", {})
    provenance_matches = {
        "config": provenance.get("config_sha256") == _file_sha256(paths["config"]),
        "holdout": (
            provenance.get("holdout_data_sha256")
            == integrity["holdout_data_sha256"]
        ),
        "training": (
            provenance.get("training_data_sha256")
            == integrity["training_data_sha256"]
        ),
        "artifact": (
            provenance.get("artifact_training_data_sha256")
            == artifact_training_sha
        ),
        "artifact_content": provenance.get("artifact_sha256") == artifact_sha,
        "artifact_path": _same_path(
            provenance.get("artifact_path"), paths["artifact"]
        ),
    }
    evaluation_fields = (
        "accuracy",
        "accuracy_ci95",
        "total",
        "by_complexity",
        "results",
    )
    report_matches_recomputed = all(
        result.get(field) == recomputed_result[field] for field in evaluation_fields
    )

    min_accuracy = float(raw.get("min_accuracy", 0.0))
    min_total = int(raw.get("min_total", 0))
    required_complexities = {str(item) for item in raw.get("required_complexities", [])}
    observed_complexities = set(result.get("by_complexity", {}).keys())
    accuracy = float(result.get("accuracy", 0.0))
    total = int(result.get("total", 0))
    missing_complexities = sorted(required_complexities - observed_complexities)
    failed_cases = [
        item
        for item in result.get("results", [])
        if isinstance(item, dict) and not item.get("passed")
    ]
    passed = bool(
        integrity["passed"]
        and artifact_matches_training
        and artifact_path_matches_config
        and all(provenance_matches.values())
        and report_matches_recomputed
        and accuracy >= min_accuracy
        and total >= min_total
        and not missing_complexities
    )
    return {
        "name": "routing_holdout",
        "passed": passed,
        "accuracy": accuracy,
        "accuracy_ci95": result.get("accuracy_ci95", {}),
        "min_accuracy": min_accuracy,
        "total": total,
        "min_total": min_total,
        "missing_complexities": missing_complexities,
        "failed_case_ids": [item.get("id") for item in failed_cases],
        "integrity": integrity,
        "artifact_matches_training": artifact_matches_training,
        "artifact_path_matches_config": artifact_path_matches_config,
        "report_matches_recomputed": report_matches_recomputed,
        "provenance_matches": provenance_matches,
    }


def _check_forbidden_listeners(listeners: object) -> dict[str, Any]:
    active: list[dict[str, object]] = []
    for raw in listeners:
        if not isinstance(raw, dict):
            continue
        host = str(raw.get("host", "127.0.0.1"))
        port = int(raw["port"])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                active.append(
                    {
                        "host": host,
                        "port": port,
                        "name": raw.get("name", "unnamed"),
                    }
                )

    return {
        "name": "forbidden_listeners",
        "passed": not active,
        "active": active,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _same_path(raw: object, expected: Path) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    return Path(raw).resolve() == expected.resolve()


if __name__ == "__main__":
    main()
