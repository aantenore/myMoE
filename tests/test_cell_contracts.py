from dataclasses import FrozenInstanceError
import unittest

from local_moe.cell_contracts import (
    AdvisorProfile,
    CellContractError,
    CellDeclaration,
    CellEstimate,
    CellMeasurement,
    CellObservation,
    WorkloadDemand,
    normalize_machine,
)


SHA_A, SHA_B, SHA_C, SHA_D = (character * 64 for character in "abcd")


def declaration(**overrides: object) -> CellDeclaration:
    values: dict[str, object] = {
        "cell_id": "cell-a",
        "model": "local/model-a",
        "quantization": "int4",
        "runtime": "runtime-a",
        "harness": "harness-a",
        "capabilities": ("code", "analysis"),
        "tool_surfaces": ("workspace",),
        "risk_classes": ("low",),
        "supported_systems": ("Linux", "Darwin"),
        "supported_machines": ("AMD64", "ARM64"),
        "max_context_tokens": 8192,
        "offline_capable": True,
        "expected_model_sha256": SHA_A,
        "expected_runtime_sha256": SHA_B,
        "expected_harness_sha256": SHA_C,
        "expected_tool_contract_sha256": SHA_D,
    }
    values.update(overrides)
    return CellDeclaration(**values)  # type: ignore[arg-type]


class CellContractTests(unittest.TestCase):
    def test_declaration_is_immutable_hash_bound_and_normalizes_architectures(
        self,
    ) -> None:
        item = declaration()
        self.assertEqual(item.supported_machines, ("arm64", "x86_64"))
        self.assertEqual(normalize_machine("aarch64"), "arm64")
        self.assertEqual(normalize_machine("x86_64"), "x86_64")
        with self.assertRaises(FrozenInstanceError):
            item.cell_id = "changed"  # type: ignore[misc]
        with self.assertRaisesRegex(CellContractError, "digest"):
            declaration(model="local/changed", digest=item.digest)

    def test_available_observation_requires_all_exact_identities_and_provenance(
        self,
    ) -> None:
        item = declaration()
        with self.assertRaisesRegex(CellContractError, "observed_model_sha256"):
            CellObservation(
                cell_id=item.cell_id,
                declaration_sha256=item.digest,
                model_status="available",
                runtime_status="unknown",
            )
        observed = CellObservation(
            cell_id=item.cell_id,
            declaration_sha256=item.digest,
            model_status="available",
            runtime_status="available",
            harness_status="available",
            tool_contract_status="available",
            residency_status="not_resident",
            observed_model_sha256=SHA_A,
            observed_runtime_sha256=SHA_B,
            observed_harness_sha256=SHA_C,
            observed_tool_contract_sha256=SHA_D,
            captured_at="2026-07-21T09:00:00+00:00",
            expires_at="2026-07-22T09:00:00+00:00",
            source_path="evidence/observed.json",
            source_sha256=SHA_A,
        )
        self.assertEqual(observed.observed_harness_sha256, SHA_C)
        with self.assertRaisesRegex(CellContractError, "provenance"):
            CellObservation(
                cell_id=item.cell_id,
                declaration_sha256=item.digest,
                residency_status="resident",
            )

    def test_resource_estimate_separates_host_unified_and_accelerator_pools(
        self,
    ) -> None:
        item = declaration()
        unified = CellEstimate(
            cell_id=item.cell_id,
            declaration_sha256=item.digest,
            memory_pool="unified",
            placement="integrated_accelerator",
            peak_unified_memory_bytes=4 * 1024**3,
            source_path="evidence/estimate.json",
            source_sha256=SHA_A,
        )
        self.assertIsNone(unified.peak_host_memory_bytes)
        with self.assertRaisesRegex(CellContractError, "memory_pool"):
            CellEstimate(
                cell_id=item.cell_id,
                declaration_sha256=item.digest,
                memory_pool="accelerator",
                placement="discrete_accelerator",
                peak_accelerator_memory_bytes=4 * 1024**3,
                source_path="evidence/estimate.json",
                source_sha256=SHA_A,
            )

    def test_measurement_binds_exact_demand_and_evaluation_contract(self) -> None:
        item = declaration()
        demand = WorkloadDemand(
            workload_id="coding.edit",
            capabilities=("code",),
            tool_surfaces=("workspace",),
            risk_class="low",
            context_tokens=4096,
        )
        measured = CellMeasurement(
            cell_id=item.cell_id,
            declaration_sha256=item.digest,
            sample_count=10,
            success_rate=0.9,
            p95_latency_ms=100,
            memory_pool="unified",
            placement="integrated_accelerator",
            peak_unified_memory_bytes=4 * 1024**3,
            resource_class_sha256=SHA_A,
            demand_sha256=demand.digest,
            evaluation_contract_sha256=SHA_B,
            measured_at="2026-07-20T00:00:00+00:00",
            expires_at="2026-07-22T00:00:00+00:00",
            source_path="evidence/measurement.json",
            source_sha256=SHA_C,
        )
        self.assertEqual(measured.demand_sha256, demand.digest)
        self.assertEqual(measured.evaluation_contract_sha256, SHA_B)

    def test_workload_requires_explicit_non_wildcard_capability(self) -> None:
        for capabilities in ((), ("*",)):
            with self.assertRaises(CellContractError):
                WorkloadDemand(
                    workload_id="coding.edit",
                    capabilities=capabilities,
                    tool_surfaces=(),
                    risk_class="low",
                    context_tokens=1,
                )

    def test_profile_rejects_non_finite_aggregate_weight(self) -> None:
        with self.assertRaisesRegex(CellContractError, "finite positive sum"):
            AdvisorProfile(
                quality_weight=1e308,
                latency_weight=1e308,
                memory_weight=1e308,
                min_success_rate=0.8,
                min_samples=1,
                reserve_memory_bytes=0,
                latency_reference_ms=1000,
                memory_reference_bytes=1024,
                max_snapshot_age_seconds=60,
                max_swap_used_bytes=0,
            )

    def test_negative_zero_is_canonicalized_and_errors_are_uniform(self) -> None:
        profile = AdvisorProfile(
            quality_weight=-0.0,
            latency_weight=1.0,
            memory_weight=-0.0,
            min_success_rate=-0.0,
            min_samples=1,
            reserve_memory_bytes=0,
            latency_reference_ms=1000,
            memory_reference_bytes=1024,
            max_snapshot_age_seconds=60,
            max_swap_used_bytes=0,
        )
        self.assertEqual(str(profile.quality_weight), "0.0")
        self.assertNotIn("-0.0", str(profile.payload()))
        with self.assertRaises(CellContractError):
            declaration(cell_id="not valid")
        with self.assertRaises(CellContractError):
            WorkloadDemand(
                workload_id="coding",
                capabilities=("code", "code"),
                tool_surfaces=(),
                risk_class="low",
                context_tokens=1,
            )
        with self.assertRaises(CellContractError):
            WorkloadDemand(
                workload_id="coding",
                capabilities=("code",),
                tool_surfaces=(),
                risk_class="low",
                context_tokens=10**10000,
            )


if __name__ == "__main__":
    unittest.main()
