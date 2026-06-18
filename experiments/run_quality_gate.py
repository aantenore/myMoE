from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quality-gate.json")
    parser.add_argument("--out", default="outputs/quality-gate.json")
    args = parser.parse_args()

    gate_config = _read_json(Path(args.config))
    checks = [
        _check_required_files(gate_config.get("required_files", [])),
        _check_routing_eval(gate_config.get("routing_eval", {})),
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


if __name__ == "__main__":
    main()
