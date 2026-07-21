from pathlib import Path
import os
import unittest
import json

from local_moe.adaptive_selector import (
    CandidateAssessment,
    advise_cell,
    build_adaptive_request,
)
from local_moe.cell_contracts import (
    AdaptiveCellCatalog,
    AdvisorProfile,
    CellContractError,
    CellDeclaration,
    CellEstimate,
    CellMeasurement,
    CellObservation,
    CellPassport,
    WorkloadDemand,
)
from local_moe.cell_passport import (
    _trusted_cell_catalog_from_payload,
    load_cell_catalog,
)
from local_moe.resource_snapshot import ResourceSnapshot, build_resource_snapshot
from local_moe.verified_routing_contracts import sha256_json


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
SHA_A, SHA_B, SHA_C, SHA_D = (character * 64 for character in "abcd")
CPU_SHA = sha256_json({"cpu": "fixture"})
ACCEL_SHA = sha256_json({"accelerator": "fixture"})
RUNTIME_SHA = sha256_json({"runtime": "fixture"})
EVALUATED_AT = "2026-07-21T10:00:00+00:00"
SECURE_LOADER_SUPPORTED = os.name == "nt" or (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and os.open in os.supports_dir_fd
)


def snapshot(**overrides: object) -> ResourceSnapshot:
    values: dict[str, object] = {
        "system": "Darwin",
        "os_release": "25.0.0",
        "machine": "arm64",
        "cpu_count": 12,
        "cpu_identity_sha256": CPU_SHA,
        "memory_topology": "unified",
        "total_memory_bytes": 24 * GIB,
        "available_memory_bytes": 16 * GIB,
        "effective_memory_limit_bytes": 24 * GIB,
        "swap_used_bytes": 0,
        "accelerator_kind": "integrated",
        "accelerator_identity_sha256": ACCEL_SHA,
        "runtime_environment_sha256": RUNTIME_SHA,
        "captured_at": "2026-07-21T09:59:30+00:00",
        "source": {"fixture": "snapshot"},
    }
    values.update(overrides)
    return build_resource_snapshot(**values)  # type: ignore[arg-type]


def profile(**overrides: object) -> AdvisorProfile:
    values: dict[str, object] = {
        "quality_weight": 0.4,
        "latency_weight": 0.4,
        "memory_weight": 0.2,
        "min_success_rate": 0.8,
        "min_samples": 5,
        "reserve_memory_bytes": 2 * GIB,
        "latency_reference_ms": 1000,
        "memory_reference_bytes": 16 * GIB,
        "max_snapshot_age_seconds": 60,
        "max_swap_used_bytes": 0,
    }
    values.update(overrides)
    return AdvisorProfile(**values)  # type: ignore[arg-type]


def request(
    *,
    workload_id: str = "coding.edit",
    capabilities: tuple[str, ...] = ("code",),
    profile_id: str = "balanced",
    fingerprint: str = "1" * 64,
    intent: str | None = "2" * 64,
    evaluated_at: str = EVALUATED_AT,
):
    return build_adaptive_request(
        exact_request_fingerprint=fingerprint,
        intent_family_sha256=intent,
        workload_id=workload_id,
        required_capabilities=capabilities,
        required_tool_surfaces=("workspace",),
        risk_class="low",
        required_context_tokens=4096,
        evaluation_contract_sha256=SHA_D,
        profile=profile_id,
        evaluated_at=evaluated_at,
    )


def cell(
    cell_id: str,
    host: ResourceSnapshot,
    demand: WorkloadDemand,
    *,
    quality: float = 0.9,
    latency: float = 100,
    memory: int = 4 * GIB,
    observed_model_sha256: str = SHA_A,
    measured_demand_sha256: str | None = None,
    memory_pool: str = "unified",
    placement: str = "integrated_accelerator",
) -> CellPassport:
    declared = CellDeclaration(
        cell_id=cell_id,
        model=f"local/{cell_id}",
        quantization="int4",
        runtime="runtime-a",
        harness="harness-a",
        capabilities=("analysis", "code", "summarization"),
        tool_surfaces=("workspace",),
        risk_classes=("low",),
        supported_systems=(host.system,),
        supported_machines=(host.machine,),
        max_context_tokens=16384,
        offline_capable=True,
        expected_model_sha256=SHA_A,
        expected_runtime_sha256=SHA_B,
        expected_harness_sha256=SHA_C,
        expected_tool_contract_sha256=SHA_D,
    )
    observed = CellObservation(
        cell_id=cell_id,
        declaration_sha256=declared.digest,
        model_status="available",
        runtime_status="available",
        harness_status="available",
        tool_contract_status="available",
        residency_status="not_resident",
        observed_model_sha256=observed_model_sha256,
        observed_runtime_sha256=SHA_B,
        observed_harness_sha256=SHA_C,
        observed_tool_contract_sha256=SHA_D,
        captured_at="2026-07-21T09:00:00+00:00",
        expires_at="2026-07-22T09:00:00+00:00",
        source_path="evidence/observed.json",
        source_sha256=SHA_A,
    )
    resource_values: dict[str, object] = {
        "memory_pool": memory_pool,
        "placement": placement,
        "source_path": "evidence/resources.json",
        "source_sha256": SHA_B,
    }
    if memory_pool == "unified":
        resource_values["peak_unified_memory_bytes"] = memory
    elif memory_pool == "host":
        resource_values["peak_host_memory_bytes"] = memory
    else:
        resource_values["peak_host_memory_bytes"] = GIB
        resource_values["peak_accelerator_memory_bytes"] = memory
    estimated = CellEstimate(
        cell_id=cell_id,
        declaration_sha256=declared.digest,
        **resource_values,
    )  # type: ignore[arg-type]
    measured_values = dict(resource_values)
    measured_values.pop("source_path")
    measured_values.pop("source_sha256")
    measured = CellMeasurement(
        cell_id=cell_id,
        declaration_sha256=declared.digest,
        sample_count=20,
        success_rate=quality,
        p95_latency_ms=latency,
        resource_class_sha256=host.resource_class_sha256,
        demand_sha256=measured_demand_sha256 or demand.digest,
        evaluation_contract_sha256=SHA_D,
        measured_at="2026-07-20T09:00:00+00:00",
        expires_at="2026-07-22T09:00:00+00:00",
        source_path="evidence/measured.json",
        source_sha256=SHA_C,
        **measured_values,
    )  # type: ignore[arg-type]
    return CellPassport(declared, observed, estimated, measured)


def catalog(cells: list[CellPassport]) -> AdaptiveCellCatalog:
    return AdaptiveCellCatalog(
        catalog_id="fixture",
        cells=tuple(sorted(cells, key=lambda item: item.cell_id)),
        profiles={
            "balanced": profile(),
            "quality-only": profile(
                quality_weight=1.0, latency_weight=0.0, memory_weight=0.0
            ),
        },
    )


class AdaptiveSelectorTests(unittest.TestCase):
    def test_request_keeps_exact_fingerprint_intent_family_and_demand_separate(
        self,
    ) -> None:
        first = request(fingerprint="1" * 64, intent="9" * 64)
        paraphrase = request(fingerprint="3" * 64, intent="9" * 64)
        self.assertNotEqual(first.digest, paraphrase.digest)
        self.assertEqual(first.intent_family_sha256, paraphrase.intent_family_sha256)
        self.assertEqual(first.demand.digest, paraphrase.demand.digest)
        self.assertEqual(first.demand.workload_id, "coding.edit")

    def test_request_and_selector_refuse_empty_capability_wildcard(self) -> None:
        with self.assertRaises(CellContractError):
            request(capabilities=())
        host = snapshot()
        task = request()
        configured = catalog([cell("cell-a", host, task.demand)])
        object.__setattr__(task.demand, "capabilities", ())
        with self.assertRaisesRegex(CellContractError, "explicit"):
            advise_cell(configured, host, task)
        object.__setattr__(task.demand, "capabilities", ("*",))
        with self.assertRaisesRegex(CellContractError, "non-wildcard"):
            advise_cell(configured, host, task)

    def test_stale_or_future_snapshot_causes_global_abstention(self) -> None:
        base_request = request()
        fresh = snapshot()
        configured = catalog([cell("cell-a", fresh, base_request.demand)])
        stale = snapshot(captured_at="2026-07-21T09:58:59+00:00")
        future = snapshot(captured_at="2026-07-21T10:00:01+00:00")
        stale_advice = advise_cell(configured, stale, base_request)
        future_advice = advise_cell(configured, future, base_request)
        self.assertTrue(stale_advice.abstained)
        self.assertIn("snapshot_stale", stale_advice.reason_codes)
        self.assertTrue(future_advice.abstained)
        self.assertIn("snapshot_from_future", future_advice.reason_codes)
        self.assertEqual(stale_advice.evaluated_at, EVALUATED_AT)

    def test_summary_evidence_never_qualifies_coding_demand(self) -> None:
        host = snapshot()
        coding = request()
        summary = request(workload_id="summary.short", capabilities=("summarization",))
        configured = catalog(
            [
                cell(
                    "cell-a",
                    host,
                    coding.demand,
                    measured_demand_sha256=summary.demand.digest,
                )
            ]
        )
        advice = advise_cell(configured, host, coding)
        self.assertTrue(advice.abstained)
        self.assertIn(
            "measurement_demand_mismatch", advice.candidates[0].rejection_codes
        )

    def test_evaluation_contract_mismatch_never_qualifies(self) -> None:
        host = snapshot()
        task = request()
        configured = catalog([cell("cell-a", host, task.demand)])
        mismatched = build_adaptive_request(
            exact_request_fingerprint=task.exact_request_fingerprint,
            intent_family_sha256=task.intent_family_sha256,
            workload_id=task.demand.workload_id,
            required_capabilities=task.demand.capabilities,
            required_tool_surfaces=task.demand.tool_surfaces,
            risk_class=task.demand.risk_class,
            required_context_tokens=task.demand.context_tokens,
            evaluation_contract_sha256="e" * 64,
            profile=task.profile,
            evaluated_at=task.evaluated_at,
        )
        advice = advise_cell(configured, host, mismatched)
        self.assertIn(
            "measurement_evaluation_contract_mismatch",
            advice.candidates[0].rejection_codes,
        )

    def test_identity_mismatch_and_unknown_resource_class_fail_closed(self) -> None:
        host = snapshot()
        task = request()
        mismatch = catalog(
            [cell("cell-a", host, task.demand, observed_model_sha256="f" * 64)]
        )
        mismatch_advice = advise_cell(mismatch, host, task)
        self.assertIn(
            "model_identity_mismatch", mismatch_advice.candidates[0].rejection_codes
        )
        incomplete = snapshot(cpu_identity_sha256=None)
        configured = catalog([cell("cell-a", host, task.demand)])
        incomplete_advice = advise_cell(configured, incomplete, task)
        self.assertIn(
            "resource_class_unknown", incomplete_advice.candidates[0].rejection_codes
        )

    def test_swap_unknown_or_over_profile_limit_abstains(self) -> None:
        host = snapshot()
        task = request()
        configured = catalog([cell("cell-a", host, task.demand)])
        unknown = advise_cell(configured, snapshot(swap_used_bytes=None), task)
        over = advise_cell(configured, snapshot(swap_used_bytes=1), task)
        self.assertIn("swap_usage_unknown", unknown.reason_codes)
        self.assertIn("swap_limit_exceeded", over.reason_codes)

    def test_accelerator_pool_is_checked_independently_from_host_memory(self) -> None:
        discrete = build_resource_snapshot(
            system="Linux",
            os_release="6.12",
            machine="AMD64",
            cpu_count=16,
            cpu_identity_sha256=CPU_SHA,
            memory_topology="dedicated",
            total_memory_bytes=64 * GIB,
            available_memory_bytes=48 * GIB,
            effective_memory_limit_bytes=64 * GIB,
            swap_used_bytes=0,
            accelerator_kind="discrete",
            accelerator_identity_sha256=ACCEL_SHA,
            accelerator_memory_total_bytes=8 * GIB,
            accelerator_memory_available_bytes=2 * GIB,
            runtime_environment_sha256=RUNTIME_SHA,
            captured_at="2026-07-21T09:59:30+00:00",
            source={"fixture": "discrete"},
        )
        task = request()
        configured = catalog(
            [
                cell(
                    "cell-a",
                    discrete,
                    task.demand,
                    memory=4 * GIB,
                    memory_pool="accelerator",
                    placement="discrete_accelerator",
                )
            ]
        )
        advice = advise_cell(configured, discrete, task)
        self.assertIn(
            "accelerator_memory_headroom_insufficient",
            advice.candidates[0].rejection_codes,
        )

    def test_discrete_accelerator_pool_preserves_profile_reserve(self) -> None:
        discrete = build_resource_snapshot(
            system="Linux",
            os_release="6.12",
            machine="AMD64",
            cpu_count=16,
            cpu_identity_sha256=CPU_SHA,
            memory_topology="dedicated",
            total_memory_bytes=64 * GIB,
            available_memory_bytes=48 * GIB,
            effective_memory_limit_bytes=64 * GIB,
            swap_used_bytes=0,
            accelerator_kind="discrete",
            accelerator_identity_sha256=ACCEL_SHA,
            accelerator_memory_total_bytes=8 * GIB,
            accelerator_memory_available_bytes=5 * GIB,
            runtime_environment_sha256=RUNTIME_SHA,
            captured_at="2026-07-21T09:59:30+00:00",
            source={"fixture": "discrete-reserve"},
        )
        task = request()
        configured = catalog(
            [
                cell(
                    "cell-a",
                    discrete,
                    task.demand,
                    memory=4 * GIB,
                    memory_pool="accelerator",
                    placement="discrete_accelerator",
                )
            ]
        )
        advice = advise_cell(configured, discrete, task)
        self.assertIn(
            "accelerator_memory_headroom_insufficient",
            advice.candidates[0].rejection_codes,
        )

    def test_host_and_unified_headroom_use_their_own_pool(self) -> None:
        unified = snapshot(available_memory_bytes=5 * GIB)
        task = request()
        unified_advice = advise_cell(
            catalog([cell("unified", unified, task.demand, memory=4 * GIB)]),
            unified,
            task,
        )
        self.assertIn(
            "unified_memory_headroom_insufficient",
            unified_advice.candidates[0].rejection_codes,
        )
        host = build_resource_snapshot(
            system="Linux",
            os_release="6.12",
            machine="AMD64",
            cpu_count=8,
            cpu_identity_sha256=CPU_SHA,
            memory_topology="system",
            total_memory_bytes=16 * GIB,
            available_memory_bytes=5 * GIB,
            effective_memory_limit_bytes=16 * GIB,
            swap_used_bytes=0,
            accelerator_kind="none",
            accelerator_identity_sha256=None,
            runtime_environment_sha256=RUNTIME_SHA,
            captured_at="2026-07-21T09:59:30+00:00",
            source={"fixture": "host"},
        )
        host_advice = advise_cell(
            catalog(
                [
                    cell(
                        "host",
                        host,
                        task.demand,
                        memory=4 * GIB,
                        memory_pool="host",
                        placement="cpu",
                    )
                ]
            ),
            host,
            task,
        )
        self.assertIn(
            "host_memory_headroom_insufficient",
            host_advice.candidates[0].rejection_codes,
        )

    def test_candidate_contract_rejects_invalid_enum_and_resource_shape(self) -> None:
        common = {
            "cell_id": "cell-a",
            "passport_sha256": SHA_A,
            "hard_eligible": False,
            "pareto_eligible": False,
            "rejection_codes": ("blocked",),
            "success_rate": None,
            "p95_latency_ms": None,
            "utility": None,
        }
        with self.assertRaises(CellContractError):
            CandidateAssessment(
                **common,
                memory_pool="mystery",
                placement=None,
                effective_peak_host_memory_bytes=None,
                effective_peak_unified_memory_bytes=None,
                effective_peak_accelerator_memory_bytes=None,
            )
        with self.assertRaises(CellContractError):
            CandidateAssessment(
                **common,
                memory_pool="unified",
                placement="integrated_accelerator",
                effective_peak_host_memory_bytes=GIB,
                effective_peak_unified_memory_bytes=None,
                effective_peak_accelerator_memory_bytes=None,
            )

    def test_zero_weight_metrics_do_not_change_pareto_or_tie_break(self) -> None:
        host = snapshot()
        task = request(profile_id="quality-only")
        configured = catalog(
            [
                cell(
                    "a-slow-heavy",
                    host,
                    task.demand,
                    quality=0.9,
                    latency=900,
                    memory=10 * GIB,
                ),
                cell(
                    "z-fast-light",
                    host,
                    task.demand,
                    quality=0.9,
                    latency=10,
                    memory=2 * GIB,
                ),
            ]
        )
        advice = advise_cell(configured, host, task)
        self.assertEqual(advice.selected_cell_id, "a-slow-heavy")
        self.assertTrue(all(item.pareto_eligible for item in advice.candidates))

    def test_success_receipt_remains_deterministic_and_non_authorizing(self) -> None:
        host = snapshot()
        task = request()
        configured = catalog([cell("cell-a", host, task.demand)])
        first, second = (
            advise_cell(configured, host, task),
            advise_cell(configured, host, task),
        )
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.selected_cell_id, "cell-a")
        self.assertFalse(first.applied)
        self.assertFalse(first.authorizes_execution)
        self.assertFalse(first.network_used)
        self.assertEqual(first.model_invocations, 0)

    def test_example_catalog_abstains_without_inventing_evidence(self) -> None:
        path = ROOT / "configs" / "adaptive-cells.example.json"
        example = (
            load_cell_catalog(path)
            if SECURE_LOADER_SUPPORTED
            else _trusted_cell_catalog_from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
        )
        advice = advise_cell(example, snapshot(), request())
        self.assertTrue(advice.abstained)
        self.assertTrue(
            all(
                "measurement_unknown" in item.rejection_codes
                for item in advice.candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
