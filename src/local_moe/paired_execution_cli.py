"""Installable command line entry point for verified paired execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


EXIT_SUCCESS = 0
EXIT_INDETERMINATE = 1
EXIT_CONTRACT = 2
EXIT_OPERATIONAL = 3

_SCHEMA_VERSION = "1.0"
_SIGNED_VERIFIER_MESSAGE = (
    "Paired execution requires a separately configured signed-attestation "
    "producer, public-key trust policy, and immutable evidence store."
)


class _UsageError(ValueError):
    pass


class _PairedCliFailure(RuntimeError):
    def __init__(self, *, code: str, message: str, exit_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.public_message = message
        self.exit_code = exit_code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="mymoe-paired",
        description=(
            "Inspect or execute one claim-bound AB/BA evidence case from a "
            "frozen myMoE routing plan."
        ),
        epilog=(
            "Journal and outcome metadata contains linkable stable hashes; "
            "treat it as sensitive and never publish it."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit stable machine-readable JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status",
        help="Read a paired-run journal without loading provider configuration.",
    )
    status.add_argument("--run-dir", required=True, metavar="PATH")
    status.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit stable machine-readable JSON.",
    )

    run = subparsers.add_parser(
        "run",
        help=(
            "Execute the exact frozen case when a signed independent-verifier "
            "adapter is configured."
        ),
    )
    run.add_argument(
        "--task",
        "--task-file",
        dest="task_file",
        required=True,
        metavar="PATH",
        help="AssistantTaskEnvelope JSON file.",
    )
    run.add_argument(
        "--plan",
        required=True,
        metavar="PATH",
        help="Frozen VerifiedRoutingEvidencePlan JSON file.",
    )
    run.add_argument(
        "--bridge-config",
        required=True,
        metavar="PATH",
        help="Assistant Bridge configuration used by the frozen case.",
    )
    run.add_argument(
        "--app-config",
        required=True,
        metavar="PATH",
        help="Application permission policy JSON file.",
    )
    run.add_argument(
        "--workflow-config",
        metavar="PATH",
        help=(
            "Two-phase public trust and preinitialized CAS configuration; "
            "required with --attestation-exchange-dir for execution."
        ),
    )
    run.add_argument(
        "--attestation-exchange-dir",
        metavar="PATH",
        help=(
            "Preinitialized private requests/responses exchange for the "
            "out-of-process signed verifier."
        ),
    )
    run.add_argument(
        "--workspace",
        required=True,
        metavar="PATH",
        help="Source workspace to snapshot; candidates are never applied to it.",
    )
    run.add_argument(
        "--run-dir",
        required=True,
        metavar="PATH",
        help="Metadata-only append-only paired-run journal directory.",
    )
    run.add_argument(
        "--outcome-store",
        required=True,
        metavar="PATH",
        help="Metadata-only holdout JSONL path, isolated from the workspace.",
    )
    run.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit stable machine-readable JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    json_requested = "--json" in raw
    command = _command_from(raw)
    try:
        args = build_parser().parse_args(raw)
        json_requested = bool(args.json_output)
        command = str(args.command)
        if command == "status":
            payload, exit_code = _status(args.run_dir)
        else:
            payload, exit_code = _run(args)
    except _UsageError:
        failure = _PairedCliFailure(
            code="usage_error",
            message="Command arguments are invalid; run mymoe-paired --help.",
            exit_code=EXIT_CONTRACT,
        )
        return _emit_failure(command, failure, json_output=json_requested)
    except _PairedCliFailure as failure:
        return _emit_failure(command, failure, json_output=json_requested)
    except ModuleNotFoundError:
        failure = _PairedCliFailure(
            code="runtime_dependency_missing",
            message=(
                "Paired execution dependencies are missing; install the "
                "assistant-bridge extra."
            ),
            exit_code=EXIT_CONTRACT,
        )
        return _emit_failure(command, failure, json_output=json_requested)
    except Exception:
        failure = _PairedCliFailure(
            code="operational_failure",
            message="The command failed before a safe result was available.",
            exit_code=EXIT_OPERATIONAL,
        )
        return _emit_failure(command, failure, json_output=json_requested)

    _emit_success(payload, json_output=json_requested)
    return exit_code


def _status(run_dir: str) -> tuple[dict[str, object], int]:
    try:
        store = _new_run_store(run_dir)
        status = store.status()
    except ModuleNotFoundError:
        raise
    except Exception as exc:
        if _is_contract_error(exc):
            raise _PairedCliFailure(
                code="run_store_invalid",
                message="The paired-run journal is invalid or inconsistent.",
                exit_code=EXIT_CONTRACT,
            ) from exc
        raise _PairedCliFailure(
            code="status_operational_failure",
            message="The paired-run journal could not be read safely.",
            exit_code=EXIT_OPERATIONAL,
        ) from exc
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "command": "status",
        "status": status.state,
        "run": status.payload(),
    }
    return payload, (
        EXIT_INDETERMINATE
        if status.state == "indeterminate"
        else EXIT_SUCCESS
    )


def _run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    try:
        app_config = _load_app_config(args.app_config)
        task = _load_task(args.task_file)
        plan = _load_plan(args.plan)
        case = _select_case(task, plan)
        pricing = getattr(plan, "pricing_contract", None)
        if pricing is None:
            raise _PairedCliFailure(
                code="embedded_pricing_required",
                message="The frozen evidence plan has no embedded pricing contract.",
                exit_code=EXIT_CONTRACT,
            )
        _validate_execution_authority(app_config, task, case)
        bridge_config = _load_bridge_config(args.bridge_config)
        workflow_path = args.workflow_config
        exchange_path = args.attestation_exchange_dir
        if (workflow_path is None) != (exchange_path is None):
            raise _PairedCliFailure(
                code="signed_verifier_configuration_invalid",
                message=(
                    "--workflow-config and --attestation-exchange-dir must be "
                    "provided together."
                ),
                exit_code=EXIT_CONTRACT,
            )
        if workflow_path is None:
            executor = _build_executor(bridge_config)
        else:
            workflow = _load_workflow_config(workflow_path)
            producer = _build_attestation_producer(exchange_path)
            evidence_store = _load_evidence_store(workflow.state.cas_path)
            executor = _build_executor(
                bridge_config,
                attestation_producer=producer,
                trust_config=workflow.trust,
                evidence_store=evidence_store,
            )
        _preflight_signed_verifier(executor, task)
        run_store = Path(args.run_dir)
        outcome_store = Path(args.outcome_store)
    except _PairedCliFailure:
        raise
    except ModuleNotFoundError:
        raise
    except Exception as exc:
        raise _PairedCliFailure(
            code="input_or_config_invalid",
            message="A task, plan, application policy, or Bridge config is invalid.",
            exit_code=EXIT_CONTRACT,
        ) from exc

    try:
        result = _invoke_run(
            task=task,
            plan=plan,
            case=case,
            source_workspace=Path(args.workspace),
            pricing=pricing,
            run_store=run_store,
            outcome_store=outcome_store,
            executor=executor,
        )
    except Exception as exc:
        run_state = _run_store_state(run_store)
        if _is_indeterminate_error(exc) or run_state in {
            "indeterminate",
            "running",
        }:
            raise _PairedCliFailure(
                code="run_indeterminate",
                message=(
                    "A provider invocation may have occurred without a durable "
                    "checkpoint; automatic retry is forbidden."
                ),
                exit_code=EXIT_INDETERMINATE,
            ) from exc
        if run_state in {"missing", "unknown"}:
            raise _PairedCliFailure(
                code="run_state_unknown",
                message=(
                    "The paired-run journal cannot prove that retry is safe; "
                    "automatic retry is forbidden."
                ),
                exit_code=EXIT_INDETERMINATE,
            ) from exc
        if _is_contract_error(exc):
            raise _PairedCliFailure(
                code="paired_contract_invalid",
                message="The frozen paired-execution contract does not match.",
                exit_code=EXIT_CONTRACT,
            ) from exc
        raise _PairedCliFailure(
            code="pre_provider_operational_failure",
            message=(
                "Paired execution failed while no ambiguous provider claim "
                "was present."
            ),
            exit_code=EXIT_OPERATIONAL,
        ) from exc

    return (
        {
            "schema_version": _SCHEMA_VERSION,
            "command": "run",
            "status": result.state,
            "run": result.metadata_payload(),
        },
        EXIT_SUCCESS,
    )


def _select_case(task: object, plan: object) -> object:
    task_fingerprint = getattr(task, "task_fingerprint", None)
    cases = tuple(getattr(plan, "cases", ()))
    matches = tuple(
        case
        for case in cases
        if getattr(case, "task_fingerprint", None) == task_fingerprint
    )
    if len(matches) != 1:
        raise _PairedCliFailure(
            code="task_case_not_found",
            message="The task must match exactly one case in the frozen plan.",
            exit_code=EXIT_CONTRACT,
        )
    return matches[0]


def _validate_execution_authority(
    app_config: object,
    task: object,
    case: object,
) -> None:
    permissions = getattr(app_config, "permissions", None)
    policy = str(
        getattr(permissions, "assistant_bridge_execution_policy", "disabled")
    ).strip().lower()
    if policy == "disabled":
        raise _PairedCliFailure(
            code="bridge_execution_disabled",
            message="Application policy disables Assistant Bridge execution.",
            exit_code=EXIT_CONTRACT,
        )
    if policy not in {"local_only", "hybrid_receipt_confirmation"}:
        raise _PairedCliFailure(
            code="bridge_policy_invalid",
            message="Application Assistant Bridge execution policy is invalid.",
            exit_code=EXIT_CONTRACT,
        )
    routes = {
        str(getattr(case, "baseline_route", "")),
        str(getattr(case, "candidate_route", "")),
    }
    if policy == "local_only" and routes.intersection(
        {"local_then_verify", "premium"}
    ):
        raise _PairedCliFailure(
            code="bridge_route_forbidden",
            message="Application local-only policy forbids a planned route.",
            exit_code=EXIT_CONTRACT,
        )

    demand = getattr(task, "capability_demand", None)
    risk_class = str(getattr(demand, "risk_class", "read_only")).strip().lower()
    write_policy = str(
        getattr(permissions, "default_write_policy", "approval_required")
    ).strip().lower()
    if risk_class == "write_local" and write_policy in {
        "deny",
        "denied",
        "disabled",
        "forbidden",
    }:
        raise _PairedCliFailure(
            code="local_write_forbidden",
            message="Application policy forbids local write tasks.",
            exit_code=EXIT_CONTRACT,
        )
    if risk_class in {"write_external", "destructive", "privileged"}:
        raise _PairedCliFailure(
            code="task_authority_forbidden",
            message="Paired execution cannot grant external or privileged authority.",
            exit_code=EXIT_CONTRACT,
        )


def _preflight_signed_verifier(executor: object, task: object) -> None:
    preflight = getattr(executor, "preflight", None)
    if not callable(preflight):
        raise _PairedCliFailure(
            code="signed_verifier_required",
            message=_SIGNED_VERIFIER_MESSAGE,
            exit_code=EXIT_CONTRACT,
        )
    try:
        preflight(task)
    except Exception as exc:
        if (
            getattr(exc, "code", None) == "signed_verifier_required"
            or "signed_verifier_required" in str(exc)
        ):
            raise _PairedCliFailure(
                code="signed_verifier_required",
                message=_SIGNED_VERIFIER_MESSAGE,
                exit_code=EXIT_CONTRACT,
            ) from exc
        raise


def _load_app_config(path: str) -> object:
    from .app_config import load_app_config

    return load_app_config(path)


def _load_task(path: str) -> object:
    from .assistant_bridge import load_assistant_task

    return load_assistant_task(path)


def _load_plan(path: str) -> object:
    from .route_promotion import load_evidence_plan

    return load_evidence_plan(path)


def _load_bridge_config(path: str) -> object:
    from .assistant_bridge import load_assistant_bridge_config

    return load_assistant_bridge_config(path)


def _load_workflow_config(path: str) -> object:
    from .assistant_bridge_two_phase_config import (
        load_two_phase_lifecycle_config,
    )

    return load_two_phase_lifecycle_config(path)


def _build_attestation_producer(path: str) -> object:
    from .paired_attestation_directory import DirectoryPairedAttestationProducer

    return DirectoryPairedAttestationProducer(path)


def _load_evidence_store(path: Path) -> object:
    from .assistant_bridge_cas import ContentAddressedStore

    return ContentAddressedStore(path, create_if_missing=False)


def _build_executor(
    bridge_config: object,
    **components: object,
) -> object:
    from .assistant_bridge import AssistantBridgeRunner
    from .paired_execution_bridge import AssistantBridgePairedArmExecutor

    return AssistantBridgePairedArmExecutor(
        AssistantBridgeRunner(bridge_config),
        **components,
    )


def _new_run_store(path: str) -> object:
    from .paired_execution_store import PairedExecutionStore

    return PairedExecutionStore(path)


def _invoke_run(**kwargs: object) -> object:
    from .paired_execution import run_paired_case

    return run_paired_case(**kwargs)  # type: ignore[arg-type]


def _is_indeterminate_error(exc: Exception) -> bool:
    try:
        from .paired_execution_store import PairedRunIndeterminateError
    except ModuleNotFoundError:
        return False
    return isinstance(exc, PairedRunIndeterminateError)


def _is_contract_error(exc: Exception) -> bool:
    try:
        from .verified_routing_contracts import VerifiedRoutingError
    except ModuleNotFoundError:
        return isinstance(exc, (TypeError, ValueError))
    return isinstance(exc, (VerifiedRoutingError, TypeError, ValueError))


def _run_store_state(run_dir: Path) -> str:
    try:
        run_dir.lstat()
    except FileNotFoundError:
        return "missing"
    except Exception:
        return "unknown"
    try:
        return str(_new_run_store(str(run_dir)).status().state)
    except Exception:
        return "unknown"


def _command_from(argv: Sequence[str]) -> str:
    for item in argv:
        if item in {"status", "run"}:
            return item
    return "unknown"


def _emit_success(payload: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    command = str(payload["command"])
    state = str(payload["status"])
    run = payload.get("run")
    run_id = None
    if isinstance(run, dict):
        root = run.get("root")
        if isinstance(root, dict):
            run_id = root.get("run_id")
    suffix = "" if run_id is None else f" ({run_id})"
    print(f"mymoe-paired {command}: {state}{suffix}")


def _emit_failure(
    command: str,
    failure: _PairedCliFailure,
    *,
    json_output: bool,
) -> int:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "command": command,
        "status": "error",
        "error": {
            "code": failure.code,
            "message": failure.public_message,
        },
    }
    if json_output:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        print(
            f"mymoe-paired {command}: {failure.public_message}",
            file=sys.stderr,
        )
    return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
