from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

if __package__:
    from .benchmark_adaptive_cell_advisor import _catalog, _request, _snapshot
else:
    from benchmark_adaptive_cell_advisor import _catalog, _request, _snapshot
from local_moe.adaptive_advisor_service import (
    AdaptiveAdvisorReceipt,
    evaluate_advisor,
)
from local_moe.adaptive_execution_gate import (
    AdaptiveCellExecutionPolicy,
    AdaptiveCellExecutionPreviewReceipt,
    preview_cell_execution,
)
from local_moe.cell_contracts import AdaptiveCellCatalog


SOURCE_TASK = "Summarize this local design note."
CHANGED_TASK = "Summarize this local design note!"
SOURCE_EVALUATED_AT = "2026-07-21T10:00:00+00:00"
FRESH_EVALUATED_AT = "2026-07-21T10:00:30+00:00"
EXPIRED_EVALUATED_AT = "2026-07-21T10:01:01+00:00"
FRESH_CAPTURED_AT = "2026-07-21T10:00:20+00:00"
EXPIRED_CAPTURED_AT = "2026-07-21T10:00:51+00:00"
EVALUATION_CONTRACT = b"synthetic adaptive advisor contract fixture v1"


def _receipt(
    *,
    task_text: str,
    evaluated_at: str,
    snapshot: object,
    catalog: AdaptiveCellCatalog,
    evaluation_contract_path: Path,
) -> AdaptiveAdvisorReceipt:
    with (
        patch(
            "local_moe.adaptive_advisor_service.load_cell_catalog",
            return_value=catalog,
        ),
        patch(
            "local_moe.adaptive_advisor_service.collect_resource_snapshot",
            return_value=snapshot,
        ),
        patch(
            "local_moe.adaptive_advisor_service.now_utc",
            return_value=evaluated_at,
        ),
    ):
        return evaluate_advisor(
            catalog_path="synthetic-catalog.json",
            evaluation_contract_path=evaluation_contract_path,
            task_text=task_text,
            workload_id="local-summary",
            required_capabilities=("summarization",),
            required_tool_surfaces=(),
            risk_class="compute_only",
            context_tokens=4_096,
            profile="efficiency",
        )


def _preview(
    *,
    source: AdaptiveAdvisorReceipt,
    fresh: AdaptiveAdvisorReceipt,
    task_text: str,
    policy: AdaptiveCellExecutionPolicy,
    workspace: Path,
    label: str,
) -> AdaptiveCellExecutionPreviewReceipt:
    receipt_path = workspace / f"{label}-source-receipt.json"
    policy_path = workspace / f"{label}-policy.json"
    receipt_path.write_text(
        json.dumps(source.payload(), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    policy_path.write_text(
        json.dumps(policy.payload(), allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    with patch(
        "local_moe.adaptive_execution_gate.evaluate_advisor",
        return_value=fresh,
    ):
        return preview_cell_execution(
            source_receipt_path=receipt_path,
            task_text=task_text,
            catalog_path="synthetic-catalog.json",
            evaluation_contract_path="synthetic-evaluation-contract.json",
            policy_path=policy_path,
        )


def _scenario(receipt: AdaptiveCellExecutionPreviewReceipt) -> dict[str, Any]:
    return {
        "status": receipt.status,
        "reason_codes": list(receipt.reason_codes),
        "source_selected_cell_id": receipt.source_selected_cell_id,
        "fresh_selected_cell_id": receipt.fresh_selected_cell_id,
        "source_passport_sha256": receipt.source_passport_sha256,
        "fresh_passport_sha256": receipt.fresh_passport_sha256,
        "fresh_resource_snapshot_sha256": (receipt.fresh_resource_snapshot_sha256),
        "applied": receipt.applied,
        "authorizes_execution": receipt.authorizes_execution,
        "network_used": receipt.network_used,
        "model_invocations": receipt.model_invocations,
        "receipt_sha256": receipt.digest,
    }


def run_benchmark() -> dict[str, Any]:
    """Exercise the exact receipt gate with deterministic synthetic evidence."""

    source_snapshot = _snapshot()
    demand = _request(SOURCE_TASK, profile="efficiency").demand
    catalog = _catalog(source_snapshot, demand)
    drifted_catalog = AdaptiveCellCatalog(
        catalog_id="synthetic-contract-fixture-drifted",
        cells=catalog.cells,
        profiles=catalog.profiles,
    )
    policy = AdaptiveCellExecutionPolicy(
        max_source_receipt_age_seconds=60,
        allowed_risk_classes=("compute_only",),
        max_tool_surfaces=0,
    )

    with tempfile.TemporaryDirectory(prefix="mymoe-cell-execution-gate-") as tmp:
        workspace = Path(tmp)
        evaluation_contract_path = workspace / "evaluation-contract.json"
        evaluation_contract_path.write_bytes(EVALUATION_CONTRACT)

        source = _receipt(
            task_text=SOURCE_TASK,
            evaluated_at=SOURCE_EVALUATED_AT,
            snapshot=source_snapshot,
            catalog=catalog,
            evaluation_contract_path=evaluation_contract_path,
        )
        fresh = _receipt(
            task_text=SOURCE_TASK,
            evaluated_at=FRESH_EVALUATED_AT,
            snapshot=_snapshot(captured_at=FRESH_CAPTURED_AT),
            catalog=catalog,
            evaluation_contract_path=evaluation_contract_path,
        )
        changed_task = _receipt(
            task_text=CHANGED_TASK,
            evaluated_at=FRESH_EVALUATED_AT,
            snapshot=_snapshot(captured_at=FRESH_CAPTURED_AT),
            catalog=catalog,
            evaluation_contract_path=evaluation_contract_path,
        )
        expired = _receipt(
            task_text=SOURCE_TASK,
            evaluated_at=EXPIRED_EVALUATED_AT,
            snapshot=_snapshot(captured_at=EXPIRED_CAPTURED_AT),
            catalog=catalog,
            evaluation_contract_path=evaluation_contract_path,
        )
        drifted = _receipt(
            task_text=SOURCE_TASK,
            evaluated_at=FRESH_EVALUATED_AT,
            snapshot=_snapshot(captured_at=FRESH_CAPTURED_AT),
            catalog=drifted_catalog,
            evaluation_contract_path=evaluation_contract_path,
        )
        pressured = _receipt(
            task_text=SOURCE_TASK,
            evaluated_at=FRESH_EVALUATED_AT,
            snapshot=_snapshot(
                captured_at=FRESH_CAPTURED_AT,
                available_memory_bytes=4 * 1024**3,
            ),
            catalog=catalog,
            evaluation_contract_path=evaluation_contract_path,
        )

        unchanged_preview = _preview(
            source=source,
            fresh=fresh,
            task_text=SOURCE_TASK,
            policy=policy,
            workspace=workspace,
            label="unchanged",
        )
        changed_task_preview = _preview(
            source=source,
            fresh=changed_task,
            task_text=CHANGED_TASK,
            policy=policy,
            workspace=workspace,
            label="changed-task",
        )
        expired_preview = _preview(
            source=source,
            fresh=expired,
            task_text=SOURCE_TASK,
            policy=policy,
            workspace=workspace,
            label="expired",
        )
        drifted_preview = _preview(
            source=source,
            fresh=drifted,
            task_text=SOURCE_TASK,
            policy=policy,
            workspace=workspace,
            label="drifted",
        )
        pressured_preview = _preview(
            source=source,
            fresh=pressured,
            task_text=SOURCE_TASK,
            policy=policy,
            workspace=workspace,
            label="pressured",
        )

    previews = (
        unchanged_preview,
        changed_task_preview,
        expired_preview,
        drifted_preview,
        pressured_preview,
    )
    criteria = {
        "unchanged_exact_cell_passes_admission": (
            unchanged_preview.status == "admission_passed"
            and unchanged_preview.reason_codes == ()
        ),
        "fresh_snapshot_is_bound_separately": (
            unchanged_preview.fresh_resource_snapshot_sha256
            != source.advice.resource_snapshot_sha256
        ),
        "exact_task_drift_is_blocked": (
            changed_task_preview.status == "admission_blocked"
            and changed_task_preview.reason_codes == ("task_fingerprint_mismatch",)
        ),
        "expired_source_receipt_is_blocked": (
            expired_preview.status == "admission_blocked"
            and "source_receipt_expired" in expired_preview.reason_codes
        ),
        "catalog_drift_is_blocked": (
            drifted_preview.status == "admission_blocked"
            and drifted_preview.reason_codes == ("catalog_drift",)
        ),
        "fresh_resource_pressure_is_blocked": (
            pressured_preview.status == "admission_blocked"
            and "fresh_admission_blocked" in pressured_preview.reason_codes
        ),
        "every_preview_remains_non_authorizing_and_model_free": all(
            not receipt.applied
            and not receipt.authorizes_execution
            and not receipt.network_used
            and receipt.model_invocations == 0
            for receipt in previews
        ),
        "source_receipt_binding_is_preserved": all(
            receipt.source_advisor_receipt_sha256 == source.digest
            and receipt.source_request_sha256 == source.request.digest
            for receipt in previews
        ),
    }
    return {
        "schema_version": "1.0",
        "benchmark": "adaptive_cell_execution_gate_contract",
        "fixture": {
            "kind": "deterministic_synthetic_contract_fixture",
            "task_text_interpreted": False,
            "source_task_chars": len(SOURCE_TASK),
            "policy_sha256": policy.digest,
            "policy_mode": policy.mode,
            "configured_cell_count": len(catalog.cells),
        },
        "scenarios": {
            "unchanged_exact_cell": _scenario(unchanged_preview),
            "exact_task_drift": _scenario(changed_task_preview),
            "expired_source_receipt": _scenario(expired_preview),
            "catalog_drift": _scenario(drifted_preview),
            "fresh_resource_pressure": _scenario(pressured_preview),
        },
        "criteria": criteria,
        "contract_checks_passed": all(criteria.values()),
        "limits": [
            "synthetic_contract_fixture_only",
            "does_not_measure_model_quality",
            "does_not_measure_real_latency_or_memory",
            "does_not_start_stop_or_call_a_model",
            "does_not_run_tools",
            "does_not_reserve_resources",
            "does_not_authorize_or_apply_execution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic Adaptive Cell Execution Gate contract benchmark."
        )
    )
    parser.add_argument(
        "--out",
        help="Optional path for the machine-readable JSON report.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    rendered = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
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
