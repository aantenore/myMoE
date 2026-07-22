"""Installable CLI for offline speculative-cell qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .llama_cpp_speculative_adapter import LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256
from .secure_files import read_bounded_regular_file
from .speculative_cell_contracts import (
    SPECULATIVE_QUALIFIER_CONTRACT_SHA256,
    SpeculativeCellContractError,
    speculative_plan_from_payload,
    speculative_trial_from_payload,
)
from .speculative_cell_qualifier import (
    qualify_speculative_cell,
    validate_speculative_plan_implementation,
)
from .verified_routing_contracts import canonical_json


EXIT_QUALIFIED = 0
EXIT_INVALID = 1
EXIT_NOT_QUALIFIED = 2
MAX_PLAN_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SpeculativeCellContractError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="mymoe-speculative",
        description=(
            "Qualify one exact llama.cpp speculative-decoding cell from "
            "preregistered, payload-free AB/BA evidence."
        ),
        epilog=(
            "A qualified receipt is host-attested, unsigned, advisory only, "
            "and never activates a runtime configuration."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init", help="Create a replaceable exact-cell plan template."
    )
    initialize.add_argument("--out", required=True, metavar="PATH")
    initialize.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    inspect = subparsers.add_parser("inspect", help="Validate a frozen plan.")
    inspect.add_argument("--plan", required=True, metavar="PATH")
    inspect.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    qualify = subparsers.add_parser(
        "qualify",
        help="Evaluate a complete JSONL trial set without contacting a model.",
    )
    qualify.add_argument("--plan", required=True, metavar="PATH")
    qualify.add_argument("--trials", required=True, metavar="PATH")
    qualify.add_argument(
        "--out",
        metavar="PATH",
        help=(
            "Create a no-clobber receipt file (mode 0600 on POSIX); existing "
            "paths are refused."
        ),
    )
    qualify.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in raw
    command = _command(raw)
    try:
        args = build_parser().parse_args(raw)
        command = str(args.command)
        json_output = bool(args.json)
        if command == "init":
            rendered = f"{canonical_json(_starter_plan_payload())}\n".encode("utf-8")
            _write_new_no_clobber_file(Path(args.out), rendered)
            payload = {
                "schema_version": "1.0",
                "command": "init",
                "status": "created",
                "adapter_contract_sha256": LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
                "qualifier_contract_sha256": (SPECULATIVE_QUALIFIER_CONTRACT_SHA256),
                "authority": "template_only",
            }
            _emit(payload, json_output=json_output)
            return EXIT_QUALIFIED

        plan = _load_plan(args.plan)
        validate_speculative_plan_implementation(plan)
        if command == "inspect":
            payload = {
                "schema_version": "1.0",
                "command": "inspect",
                "status": "valid",
                "plan_sha256": plan.digest,
                "execution_sha256": plan.execution.digest,
                "adapter_contract_sha256": (plan.execution.adapter_contract_sha256),
                "qualifier_contract_sha256": (plan.execution.qualifier_contract_sha256),
                "baseline_cell_sha256": plan.baseline.digest,
                "candidate_cell_sha256": plan.candidate.digest,
                "policy_sha256": plan.policy.digest,
                "expected_trials": plan.expected_trial_count,
                "candidate_mode": plan.candidate.speculation_mode,
                "required_regimes": list(plan.required_regimes),
                "schedule": "globally_preregistered",
                "authority": plan.authority,
            }
            _emit(payload, json_output=json_output)
            return EXIT_QUALIFIED

        trials = _load_trials(args.trials, maximum=plan.expected_trial_count)
        receipt = qualify_speculative_cell(plan, trials)
        payload = {
            "schema_version": "1.0",
            "command": "qualify",
            "status": receipt.decision,
            "receipt": receipt.payload(),
        }
        rendered = f"{canonical_json(payload)}\n".encode("utf-8")
        if args.out is not None:
            _write_new_no_clobber_file(Path(args.out), rendered)
        _emit(payload, json_output=json_output)
        return EXIT_QUALIFIED if receipt.decision == "qualified" else EXIT_NOT_QUALIFIED
    except Exception as exc:
        if not isinstance(
            exc,
            (
                SpeculativeCellContractError,
                json.JSONDecodeError,
                OSError,
                UnicodeError,
            ),
        ):
            exc = SpeculativeCellContractError("qualification failed")
        payload = {
            "schema_version": "1.0",
            "command": command,
            "status": "error",
            "error": {
                "code": "invalid_or_unavailable_evidence",
                "message": "The plan or evidence is invalid or unavailable.",
            },
        }
        _emit(payload, json_output=json_output, error=True)
        return EXIT_INVALID


def _load_plan(path: str):
    raw = read_bounded_regular_file(
        Path(path), maximum_bytes=MAX_PLAN_BYTES, label="qualification plan"
    )
    return speculative_plan_from_payload(_decode_json(raw))


def _load_trials(path: str, *, maximum: int):
    raw = read_bounded_regular_file(
        Path(path), maximum_bytes=MAX_EVIDENCE_BYTES, label="trial evidence"
    )
    text = raw.decode("utf-8", errors="strict")
    if not text.endswith("\n"):
        raise SpeculativeCellContractError("Trial JSONL requires a final newline.")
    lines = tuple(line for line in text.splitlines() if line)
    if len(lines) > maximum:
        raise SpeculativeCellContractError("Trial JSONL exceeds the frozen plan.")
    return tuple(
        speculative_trial_from_payload(_decode_json_text(line)) for line in lines
    )


def _decode_json(raw: bytes):
    return _decode_json_text(raw.decode("utf-8", errors="strict"))


def _decode_json_text(value: str):
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise SpeculativeCellContractError(
                    "JSON objects must not contain duplicate keys."
                )
            result[key] = item
        return result

    def reject_constant(_value: str) -> None:
        raise SpeculativeCellContractError("JSON numbers must be finite.")

    return json.loads(
        value,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _starter_plan_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "plan_id": "replace-with-exact-cell-plan",
        "execution": {
            "schema_version": "1.0",
            "runtime": "llama.cpp",
            "runtime_revision_sha256": "a" * 64,
            "runtime_binary_sha256": "b" * 64,
            "runtime_binding_manifest_sha256": "c" * 64,
            "hardware_sha256": "d" * 64,
            "target_model_sha256": "e" * 64,
            "shared_runtime_config_sha256": "f" * 64,
            "request_policy_sha256": "1" * 64,
            "regime_protocol_sha256": "2" * 64,
            "harness_sha256": "3" * 64,
            "collector_sha256": "4" * 64,
            "adapter_contract_sha256": LLAMA_CPP_SPECULATIVE_ADAPTER_SHA256,
            "qualifier_contract_sha256": SPECULATIVE_QUALIFIER_CONTRACT_SHA256,
            "digest": "",
        },
        "baseline": {
            "schema_version": "1.0",
            "cell_id": "replace-baseline-cell",
            "draft_model_sha256": None,
            "speculation_config_sha256": "5" * 64,
            "speculation_mode": "none",
            "digest": "",
        },
        "candidate": {
            "schema_version": "1.0",
            "cell_id": "replace-speculative-cell",
            "draft_model_sha256": None,
            "speculation_config_sha256": "6" * 64,
            "speculation_mode": "ngram-simple",
            "digest": "",
        },
        "workload_sha256": "7" * 64,
        "case_sha256s": ["8" * 64, "9" * 64, "a" * 64, "b" * 64],
        "order_seed_sha256": "c" * 64,
        "policy": {
            "schema_version": "1.0",
            "trials_per_case": 4,
            "minimum_median_speedup_ratio": 1.1,
            "maximum_p95_latency_ratio": 1.0,
            "maximum_p95_ttft_ratio": 1.05,
            "minimum_acceptance_rate": 0.05,
            "maximum_candidate_peak_memory_bytes": 24 * 1024**3,
            "digest": "",
        },
        "required_regimes": ["cold", "warm"],
        "authority": "advisory_only",
        "digest": "",
    }


def _write_new_no_clobber_file(path: Path, content: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    destination = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def _emit(
    payload: dict[str, object], *, json_output: bool, error: bool = False
) -> None:
    if json_output:
        print(canonical_json(payload), file=sys.stderr if error else sys.stdout)
        return
    status = str(payload["status"])
    print(
        f"mymoe-speculative {payload['command']}: {status}",
        file=sys.stderr if error else sys.stdout,
    )


def _command(argv: Sequence[str]) -> str:
    return next(
        (item for item in argv if item in {"init", "inspect", "qualify"}),
        "unknown",
    )


if __name__ == "__main__":
    raise SystemExit(main())
