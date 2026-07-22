from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tempfile
from typing import Any, Iterator
from unittest import mock
import urllib.request

if __package__:
    from .benchmark_runtime_binding import (
        CAPTURED_AT,
        _Fixture,
        _inspect,
        _observed_identities,
    )
else:
    from benchmark_runtime_binding import (  # type: ignore[no-redef]
        CAPTURED_AT,
        _Fixture,
        _inspect,
        _observed_identities,
    )

from local_moe.adaptive_execution_gate import (
    AdaptiveCellExecutionPreviewReceipt,
)
from local_moe.bound_cell_run import (
    BoundCellRunResult,
    BoundCellRunTransportError,
    ModelIdentityProbe,
    resolve_bound_cell_target,
    run_bound_cell,
)
from local_moe.bound_cell_run_contracts import BoundCellRunPolicy
from local_moe.runtime_binding_inspector import CellBindingInspectionBundle


TASK_BODY = "BOUND-CELL-RUN-TASK-BODY-MUST-NOT-APPEAR"
RESPONSE_BODY = "BOUND-CELL-RUN-RESPONSE-BODY-MUST-NOT-APPEAR"
FAILED_RESPONSE_BODY = "BOUND-CELL-RUN-FAILED-BODY-MUST-NOT-APPEAR"
RUN_AT = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
GUARDED_SIDE_EFFECT_SURFACES = (
    "model_server_start",
    "network_socket",
    "process_spawn",
    "url_fetch",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _preview(
    *,
    cell_id: str,
    passport_sha256: str,
    admitted: bool,
) -> AdaptiveCellExecutionPreviewReceipt:
    return AdaptiveCellExecutionPreviewReceipt(
        source_advisor_receipt_sha256=_sha("source-advisor"),
        source_request_sha256=_sha("source-request"),
        fresh_advisor_receipt_sha256=_sha("fresh-advisor"),
        fresh_request_sha256=_sha("fresh-request"),
        policy_sha256=_sha("adaptive-policy"),
        evaluated_at=RUN_AT.isoformat(),
        source_selected_cell_id=cell_id,
        fresh_selected_cell_id=cell_id,
        source_passport_sha256=passport_sha256,
        fresh_passport_sha256=passport_sha256,
        fresh_resource_snapshot_sha256=_sha("fresh-resource-snapshot"),
        status="admission_passed" if admitted else "admission_blocked",
        reason_codes=() if admitted else ("fresh_admission_blocked",),
        task_chars=len(TASK_BODY),
    )


def _binding_drift(
    source: CellBindingInspectionBundle,
) -> CellBindingInspectionBundle:
    manifest = replace(
        source.manifest,
        producer_code_sha256=_sha("post-binding-producer-code"),
        digest="",
    )
    receipt = replace(
        source.receipt,
        binding_manifest_sha256=manifest.digest,
        digest="",
    )
    return CellBindingInspectionBundle(
        request_sha256=source.request_sha256,
        manifest=manifest,
        receipt=receipt,
        publication_protected_roots=source.publication_protected_roots,
    )


class _Inspector:
    def __init__(self, bundles: tuple[CellBindingInspectionBundle, ...]) -> None:
        self._bundles = bundles
        self.calls = 0

    def __call__(
        self,
        _path: str | Path,
        *,
        now: datetime,
        publication_path: str | Path | None = None,
    ):
        del now, publication_path
        index = self.calls
        self.calls += 1
        if index >= len(self._bundles):
            raise AssertionError("Unexpected extra binding inspection.")
        return self._bundles[index]


class _Transport:
    def __init__(
        self,
        *,
        model_sets: tuple[tuple[str, ...], ...],
        response: str = RESPONSE_BODY,
        failure: bool = False,
    ) -> None:
        self._model_sets = model_sets
        self._response = response
        self._failure = failure
        self.probe_requests = 0
        self.post_requests = 0

    def probe_models(self, **_kwargs: object) -> ModelIdentityProbe:
        index = self.probe_requests
        self.probe_requests += 1
        if index >= len(self._model_sets):
            raise AssertionError("Unexpected extra model probe.")
        return ModelIdentityProbe.from_ids(
            self._model_sets[index],
            maximum=16,
        )

    def invoke(self, **_kwargs: object) -> str:
        self.post_requests += 1
        if self.post_requests > 1:
            raise AssertionError("Bound Cell Run retried the model invocation.")
        if self._failure:
            raise BoundCellRunTransportError(
                "transport_failed",
                "Synthetic transport failure.",
            )
        return self._response


@contextmanager
def _side_effect_guard() -> Iterator[None]:
    blocked = AssertionError(
        "The deterministic Bound Cell Run benchmark attempted a real side effect."
    )
    with (
        mock.patch.object(socket, "socket", side_effect=blocked),
        mock.patch.object(socket, "create_connection", side_effect=blocked),
        mock.patch.object(subprocess, "Popen", side_effect=blocked),
        mock.patch.object(urllib.request, "urlopen", side_effect=blocked),
        mock.patch(
            "local_moe.model_servers.ModelServerManager.start",
            side_effect=blocked,
        ),
    ):
        yield


def _run_scenario(
    *,
    target: object,
    preview: AdaptiveCellExecutionPreviewReceipt,
    bundles: tuple[CellBindingInspectionBundle, ...],
    model_sets: tuple[tuple[str, ...], ...],
    response: str = RESPONSE_BODY,
    failure: bool = False,
) -> tuple[BoundCellRunResult, dict[str, int]]:
    inspector = _Inspector(bundles)
    transport = _Transport(
        model_sets=model_sets,
        response=response,
        failure=failure,
    )

    def resolver(_path: str | Path):
        return target

    def previewer(*_args: object):
        return preview

    def clock() -> datetime:
        return RUN_AT

    with _side_effect_guard():
        result = run_bound_cell(
            "synthetic-advisor-receipt.json",
            TASK_BODY,
            "synthetic-catalog.json",
            "synthetic-evaluation-contract.json",
            "synthetic-adaptive-policy.json",
            "synthetic-binding-request.json",
            confirmed=True,
            policy=BoundCellRunPolicy(),
            transport=transport,
            previewer=previewer,
            inspector=inspector,
            resolver=resolver,
            clock=clock,
            monotonic_clock=lambda: 0.0,
        )
    return result, {
        "binding_inspections": inspector.calls,
        "probe_requests": transport.probe_requests,
        "post_requests": transport.post_requests,
        "retries": max(0, transport.post_requests - 1),
    }


def _scenario(
    result: BoundCellRunResult,
    counters: dict[str, int],
) -> dict[str, Any]:
    return {
        "receipt": result.receipt.payload(),
        "transport_counters": counters,
        "response_returned": result.response_text is not None,
    }


def render_report(report: dict[str, Any]) -> str:
    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def run_benchmark() -> dict[str, Any]:
    """Exercise Bound Cell Run state transitions without network or processes."""

    read_requests: list[int] = []
    model_read_requests: list[int] = []
    with tempfile.TemporaryDirectory(prefix="mymoe-bound-cell-run-") as temp:
        fixture = _Fixture(Path(temp))
        initial = _inspect(
            fixture,
            read_requests,
            model_read_requests,
            now=CAPTURED_AT,
        )
        fixture.write_catalog(_observed_identities(initial))
        verified = _inspect(
            fixture,
            read_requests,
            model_read_requests,
            now=CAPTURED_AT,
        )
        target = resolve_bound_cell_target(fixture.request_path)
        drifted_binding = _binding_drift(verified)

        admitted = _preview(
            cell_id=target.request.cell_id,
            passport_sha256=target.passport.digest,
            admitted=True,
        )
        blocked = _preview(
            cell_id=target.request.cell_id,
            passport_sha256=target.passport.digest,
            admitted=False,
        )
        expected_model = target.expert.model
        stable_models = ((expected_model,), (expected_model,))

        completed, completed_counts = _run_scenario(
            target=target,
            preview=admitted,
            bundles=(verified, verified),
            model_sets=stable_models,
        )
        precondition_blocked, blocked_counts = _run_scenario(
            target=target,
            preview=blocked,
            bundles=(verified,),
            model_sets=(),
        )
        transport_failure, failure_counts = _run_scenario(
            target=target,
            preview=admitted,
            bundles=(verified, verified),
            model_sets=stable_models,
            response=FAILED_RESPONSE_BODY,
            failure=True,
        )
        post_binding_drift, binding_drift_counts = _run_scenario(
            target=target,
            preview=admitted,
            bundles=(verified, drifted_binding),
            model_sets=stable_models,
        )
        post_model_drift, model_drift_counts = _run_scenario(
            target=target,
            preview=admitted,
            bundles=(verified, verified),
            model_sets=((expected_model,), (expected_model, "synthetic-extra-model")),
        )

    scenarios = {
        "completed": _scenario(completed, completed_counts),
        "precondition_blocked": _scenario(
            precondition_blocked,
            blocked_counts,
        ),
        "transport_failure": _scenario(transport_failure, failure_counts),
        "post_binding_drift": _scenario(
            post_binding_drift,
            binding_drift_counts,
        ),
        "post_model_identity_drift": _scenario(
            post_model_drift,
            model_drift_counts,
        ),
    }
    receipts = [item["receipt"] for item in scenarios.values()]
    body_free_payload = json.dumps(scenarios, allow_nan=False, sort_keys=True)
    criteria = {
        "completed_run_is_evidence_bound": (
            completed.receipt.status == "completed"
            and completed.receipt.reason_codes == ()
            and completed.receipt.invocation_attempts == 1
            and completed.receipt.endpoint_probe_requests == 2
            and completed.receipt.delivery_status == "response_received"
            and completed_counts
            == {
                "binding_inspections": 2,
                "probe_requests": 2,
                "post_requests": 1,
                "retries": 0,
            }
        ),
        "blocked_precondition_performs_no_probe_or_post": (
            precondition_blocked.receipt.status == "blocked"
            and precondition_blocked.receipt.invocation_attempts == 0
            and precondition_blocked.receipt.endpoint_probe_requests == 0
            and blocked_counts["probe_requests"] == 0
            and blocked_counts["post_requests"] == 0
        ),
        "transport_failure_is_one_attempt_without_retry": (
            transport_failure.receipt.status == "failed"
            and transport_failure.receipt.reason_codes == ("transport_failed",)
            and transport_failure.receipt.invocation_attempts == 1
            and transport_failure.receipt.delivery_status == "attempted_unknown"
            and failure_counts["post_requests"] == 1
            and failure_counts["retries"] == 0
        ),
        "post_binding_drift_invalidates_the_response": (
            post_binding_drift.receipt.status == "invalidated"
            and "binding_changed" in post_binding_drift.receipt.reason_codes
            and post_binding_drift.response_text == RESPONSE_BODY
        ),
        "post_model_identity_drift_invalidates_the_response": (
            post_model_drift.receipt.status == "invalidated"
            and "model_identity_changed" in post_model_drift.receipt.reason_codes
            and post_model_drift.response_text == RESPONSE_BODY
        ),
        "task_and_response_bodies_are_absent": all(
            marker not in body_free_payload
            for marker in (TASK_BODY, RESPONSE_BODY, FAILED_RESPONSE_BODY)
        ),
        "all_receipts_report_zero_process_and_lifecycle_mutation": all(
            not receipt["process_mutations"]
            and receipt["lifecycle_operations"] == 0
            and not receipt["endpoint_process_identity_verified"]
            and not receipt["authorizes_future_execution"]
            for receipt in receipts
        ),
        "all_attempted_runs_are_single_attempt": all(
            receipt["invocation_attempts"] <= 1 for receipt in receipts
        ),
        "all_receipts_report_zero_retries_and_tools": all(
            receipt["retries"] == 0 and receipt["tools_invoked"] == 0
            for receipt in receipts
        ),
        "all_receipts_report_no_remote_egress_or_semantic_claim": all(
            not receipt["remote_egress"] and not receipt["semantic_outcome_verified"]
            for receipt in receipts
        ),
        "completed_run_binds_one_unchanged_inspection_request": (
            completed.receipt.pre_binding_request_sha256 is not None
            and completed.receipt.pre_binding_request_sha256
            == completed.receipt.post_binding_request_sha256
        ),
    }
    return {
        "schema_version": "1.0",
        "benchmark": "bound_cell_run_contract",
        "fixture": {
            "kind": "deterministic_synthetic_contract_fixture",
            "task_sha256": _sha(TASK_BODY),
            "task_bytes": len(TASK_BODY.encode("utf-8")),
            "task_body_included": False,
            "response_bodies_included": False,
            "policy_sha256": BoundCellRunPolicy().digest,
            "guarded_side_effect_surfaces": list(GUARDED_SIDE_EFFECT_SURFACES),
        },
        "scenarios": scenarios,
        "criteria": criteria,
        "contract_checks_passed": all(criteria.values()),
        "limits": [
            "synthetic_contract_fixture_only",
            "does_not_measure_model_quality",
            "does_not_measure_latency_throughput_or_memory",
            "does_not_open_network_connections_or_spawn_processes",
            "does_not_start_load_unload_or_stop_models",
            "does_not_run_tools",
            "does_not_attest_endpoint_process_identity",
            "does_not_prove_model_residency",
            "full_binding_reinspection_cost_is_not_benchmarked",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Bound Cell Run contract benchmark."
    )
    parser.add_argument(
        "--out",
        help="Optional path for the machine-readable JSON report.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    rendered = render_report(report)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not report["contract_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
