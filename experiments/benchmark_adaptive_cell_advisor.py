from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from local_moe.adaptive_selector import (
    AdaptiveAdvice,
    advise_cell,
    build_adaptive_request,
)
from local_moe.cell_contracts import (
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellDeclaration,
    CellEstimate,
    CellMeasurement,
    CellObservation,
    CellPassport,
    WorkloadDemand,
)
from local_moe.resource_snapshot import ResourceSnapshot, build_resource_snapshot


GIB = 1024**3
EVALUATED_AT = "2026-07-21T10:00:00+00:00"
FRESH_CAPTURED_AT = "2026-07-21T09:59:30+00:00"
EVALUATION_CONTRACT_SHA256 = hashlib.sha256(
    b"synthetic adaptive advisor contract fixture v1"
).hexdigest()
INTENT_FAMILY_SHA256 = hashlib.sha256(
    b"caller-declared:local-design-summary"
).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _snapshot(
    *,
    captured_at: str = FRESH_CAPTURED_AT,
    available_memory_bytes: int = 16 * GIB,
) -> ResourceSnapshot:
    return build_resource_snapshot(
        system="Linux",
        os_release="6.12",
        machine="x86_64",
        cpu_count=12,
        cpu_identity_sha256=_sha("synthetic-cpu"),
        memory_topology="system",
        total_memory_bytes=24 * GIB,
        available_memory_bytes=available_memory_bytes,
        effective_memory_limit_bytes=24 * GIB,
        swap_used_bytes=0,
        accelerator_kind="none",
        accelerator_identity_sha256=None,
        runtime_environment_sha256=_sha("synthetic-runtime-environment"),
        captured_at=captured_at,
        source={"fixture": "adaptive-cell-advisor-contract"},
    )


def _profile(
    *,
    quality_weight: float,
    latency_weight: float,
    memory_weight: float,
) -> AdvisorProfile:
    return AdvisorProfile(
        quality_weight=quality_weight,
        latency_weight=latency_weight,
        memory_weight=memory_weight,
        min_success_rate=0.8,
        min_samples=10,
        reserve_memory_bytes=2 * GIB,
        latency_reference_ms=2_000,
        memory_reference_bytes=16 * GIB,
        max_snapshot_age_seconds=60,
        max_swap_used_bytes=0,
    )


def _request(
    task_text: str,
    *,
    profile: str,
    intent_family_sha256: str | None = None,
):
    return build_adaptive_request(
        exact_request_fingerprint=hashlib.sha256(
            task_text.encode("utf-8")
        ).hexdigest(),
        intent_family_sha256=intent_family_sha256,
        workload_id="local-summary",
        required_capabilities=("summarization",),
        required_tool_surfaces=(),
        risk_class="compute_only",
        required_context_tokens=4_096,
        evaluation_contract_sha256=EVALUATION_CONTRACT_SHA256,
        profile=profile,
        evaluated_at=EVALUATED_AT,
    )


def _cell(
    *,
    cell_id: str,
    snapshot: ResourceSnapshot,
    demand: WorkloadDemand,
    success_rate: float,
    p95_latency_ms: float,
    peak_memory_bytes: int,
) -> CellPassport:
    model_sha256 = _sha(f"{cell_id}:model")
    runtime_sha256 = _sha(f"{cell_id}:runtime")
    harness_sha256 = _sha(f"{cell_id}:harness")
    tool_contract_sha256 = _sha(f"{cell_id}:tool-contract")
    declaration = CellDeclaration(
        cell_id=cell_id,
        model=f"synthetic/{cell_id}",
        quantization="int4",
        runtime="synthetic-runtime",
        harness="synthetic-harness",
        capabilities=("summarization",),
        tool_surfaces=(),
        risk_classes=("compute_only",),
        supported_systems=(snapshot.system,),
        supported_machines=(snapshot.machine,),
        max_context_tokens=16_384,
        offline_capable=True,
        expected_model_sha256=model_sha256,
        expected_runtime_sha256=runtime_sha256,
        expected_harness_sha256=harness_sha256,
        expected_tool_contract_sha256=tool_contract_sha256,
    )
    observation = CellObservation(
        cell_id=cell_id,
        declaration_sha256=declaration.digest,
        model_status="available",
        runtime_status="available",
        harness_status="available",
        tool_contract_status="available",
        residency_status="not_resident",
        observed_model_sha256=model_sha256,
        observed_runtime_sha256=runtime_sha256,
        observed_harness_sha256=harness_sha256,
        observed_tool_contract_sha256=tool_contract_sha256,
        captured_at="2026-07-21T09:00:00+00:00",
        expires_at="2026-07-22T09:00:00+00:00",
        source_path=f"evidence/{cell_id}-observation.json",
        source_sha256=_sha(f"{cell_id}:observation-source"),
    )
    estimate = CellEstimate(
        cell_id=cell_id,
        declaration_sha256=declaration.digest,
        memory_pool="host",
        placement="cpu",
        peak_host_memory_bytes=peak_memory_bytes,
        source_path=f"evidence/{cell_id}-estimate.json",
        source_sha256=_sha(f"{cell_id}:estimate-source"),
    )
    measurement = CellMeasurement(
        cell_id=cell_id,
        declaration_sha256=declaration.digest,
        sample_count=20,
        success_rate=success_rate,
        p95_latency_ms=p95_latency_ms,
        memory_pool="host",
        placement="cpu",
        peak_host_memory_bytes=peak_memory_bytes,
        resource_class_sha256=snapshot.resource_class_sha256,
        demand_sha256=demand.digest,
        evaluation_contract_sha256=EVALUATION_CONTRACT_SHA256,
        measured_at="2026-07-20T09:00:00+00:00",
        expires_at="2026-07-22T09:00:00+00:00",
        source_path=f"evidence/{cell_id}-measurement.json",
        source_sha256=_sha(f"{cell_id}:measurement-source"),
    )
    return CellPassport(declaration, observation, estimate, measurement)


def _catalog(snapshot: ResourceSnapshot, demand: WorkloadDemand) -> AdaptiveCellCatalog:
    cells = (
        _cell(
            cell_id="fast-small-cell",
            snapshot=snapshot,
            demand=demand,
            success_rate=0.86,
            p95_latency_ms=100,
            peak_memory_bytes=3 * GIB,
        ),
        _cell(
            cell_id="high-quality-cell",
            snapshot=snapshot,
            demand=demand,
            success_rate=0.97,
            p95_latency_ms=500,
            peak_memory_bytes=8 * GIB,
        ),
    )
    return AdaptiveCellCatalog(
        catalog_id="synthetic-contract-fixture",
        cells=cells,
        profiles={
            "efficiency": _profile(
                quality_weight=0.1,
                latency_weight=0.55,
                memory_weight=0.35,
            ),
            "quality": _profile(
                quality_weight=1.0,
                latency_weight=0.0,
                memory_weight=0.0,
            ),
        },
    )


def _scenario(advice: AdaptiveAdvice) -> dict[str, Any]:
    return {
        "status": advice.status,
        "selected_cell_id": advice.selected_cell_id,
        "reason_codes": list(advice.reason_codes),
        "advice_sha256": advice.digest,
        "applied": advice.applied,
        "authorizes_execution": advice.authorizes_execution,
        "network_used": advice.network_used,
        "model_invocations": advice.model_invocations,
        "candidate_rejections": {
            item.cell_id: list(item.rejection_codes) for item in advice.candidates
        },
    }


def run_benchmark() -> dict[str, Any]:
    """Exercise deterministic selector contracts without invoking a model."""

    fresh = _snapshot()
    base_request = _request("Summarize this local design note.", profile="efficiency")
    catalog = _catalog(fresh, base_request.demand)

    efficiency = advise_cell(catalog, fresh, base_request)
    quality_request = _request(
        "Summarize this local design note.",
        profile="quality",
    )
    quality = advise_cell(catalog, fresh, quality_request)
    stale = advise_cell(
        catalog,
        _snapshot(captured_at="2026-07-21T09:58:00+00:00"),
        base_request,
    )
    pressured = advise_cell(
        catalog,
        _snapshot(available_memory_bytes=4 * GIB),
        base_request,
    )

    paraphrase_a = _request(
        "Summarize this local design note.",
        profile="efficiency",
        intent_family_sha256=INTENT_FAMILY_SHA256,
    )
    paraphrase_b = _request(
        "Give me a short summary of this design note.",
        profile="efficiency",
        intent_family_sha256=INTENT_FAMILY_SHA256,
    )
    paraphrase_advice_a = advise_cell(catalog, fresh, paraphrase_a)
    paraphrase_advice_b = advise_cell(catalog, fresh, paraphrase_b)

    all_advice = (
        efficiency,
        quality,
        stale,
        pressured,
        paraphrase_advice_a,
        paraphrase_advice_b,
    )
    pressure_codes = {
        code
        for candidate in pressured.candidates
        for code in candidate.rejection_codes
    }
    criteria = {
        "efficiency_profile_selects_fast_small_cell": (
            efficiency.selected_cell_id == "fast-small-cell"
        ),
        "quality_profile_selects_high_quality_cell": (
            quality.selected_cell_id == "high-quality-cell"
        ),
        "stale_snapshot_abstains": (
            stale.abstained and "snapshot_stale" in stale.reason_codes
        ),
        "resource_pressure_abstains": (
            pressured.abstained
            and "host_memory_headroom_insufficient" in pressure_codes
        ),
        "paraphrase_exact_fingerprints_differ": (
            paraphrase_a.exact_request_fingerprint
            != paraphrase_b.exact_request_fingerprint
        ),
        "caller_intent_family_is_preserved": (
            paraphrase_a.intent_family_sha256
            == paraphrase_b.intent_family_sha256
            == INTENT_FAMILY_SHA256
        ),
        "declared_demand_is_identical": (
            paraphrase_a.demand.digest == paraphrase_b.demand.digest
        ),
        "paraphrase_advice_records_remain_distinct": (
            paraphrase_advice_a.digest != paraphrase_advice_b.digest
        ),
        "every_result_is_non_authorizing_and_model_free": all(
            not advice.applied
            and not advice.authorizes_execution
            and not advice.network_used
            and advice.model_invocations == 0
            for advice in all_advice
        ),
    }
    return {
        "schema_version": "1.0",
        "benchmark": "adaptive_cell_advisor_contract",
        "fixture": {
            "kind": "deterministic_synthetic_contract_fixture",
            "task_text_interpreted": False,
            "declared_workload_id": base_request.demand.workload_id,
            "declared_capabilities": list(base_request.demand.capabilities),
            "declared_risk_class": base_request.demand.risk_class,
            "cell_count": len(catalog.cells),
            "profile_count": len(catalog.profiles),
        },
        "scenarios": {
            "efficiency_profile": _scenario(efficiency),
            "quality_profile": _scenario(quality),
            "stale_snapshot": _scenario(stale),
            "resource_pressure": _scenario(pressured),
            "paraphrase_lineage": {
                "first_exact_request_fingerprint": (
                    paraphrase_a.exact_request_fingerprint
                ),
                "second_exact_request_fingerprint": (
                    paraphrase_b.exact_request_fingerprint
                ),
                "shared_caller_intent_family_sha256": INTENT_FAMILY_SHA256,
                "shared_declared_demand_sha256": paraphrase_a.demand.digest,
                "first_advice_sha256": paraphrase_advice_a.digest,
                "second_advice_sha256": paraphrase_advice_b.digest,
                "response_cache_lookup_performed": False,
                "response_reuse_authorized": False,
            },
        },
        "criteria": criteria,
        "contract_checks_passed": all(criteria.values()),
        "limits": [
            "synthetic_contract_fixture_only",
            "does_not_measure_model_quality",
            "does_not_measure_real_latency_or_memory",
            "does_not_compare_models_or_runtimes",
            "does_not_interpret_task_wording",
            "does_not_validate_semantic_intent_families",
            "does_not_cache_or_reuse_model_responses",
            "does_not_authorize_execution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic Adaptive Cell Advisor contract benchmark."
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
            report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
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
