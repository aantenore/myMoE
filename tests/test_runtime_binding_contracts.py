from __future__ import annotations

from dataclasses import replace
import math
import unittest

from local_moe.runtime_binding_contracts import (
    BOUND_CELL_ADAPTER_CONTRACT_SHA256,
    EMPTY_TOOL_CONTRACT_SHA256,
    CellRuntimeBindingManifest,
    CellRuntimeInspectionReceipt,
    RuntimeBindingContractError,
    RuntimeComponentEvidence,
    bound_cell_adapter_contract_payload,
    model_artifact_evidence_sha256,
)
from local_moe.verified_routing_contracts import sha256_json


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _component(
    role: str,
    path: str,
    *,
    root_id: str = "runtime",
    sha256: str = SHA_A,
) -> RuntimeComponentEvidence:
    return RuntimeComponentEvidence(
        role=role,
        root_id=root_id,
        path=path,
        size_bytes=7,
        sha256=sha256,
    )


def _manifest() -> CellRuntimeBindingManifest:
    components = tuple(
        sorted(
            (
                _component("driver", "lib/driver.py", sha256=SHA_A),
                _component("harness", "lib/harness.py", sha256=SHA_D),
                _component(
                    "model_artifact",
                    f"artifact-{SHA_A}.gguf",
                    root_id="model",
                    sha256=SHA_B,
                ),
                _component("runtime_executable", "bin/runtime", sha256=SHA_C),
            ),
            key=lambda item: (item.role, item.root_id, item.path),
        )
    )
    model_evidence_sha256 = model_artifact_evidence_sha256(
        "file",
        tuple(item for item in components if item.role == "model_artifact"),
    )
    return CellRuntimeBindingManifest(
        cell_id="coder-local",
        declaration_sha256=SHA_A,
        config_source_sha256=SHA_B,
        runtime_config_sha256=SHA_C,
        expert_id="coder",
        expert_config_sha256=SHA_D,
        adapter_id="managed_direct_local_openai_v1",
        adapter_contract_sha256=BOUND_CELL_ADAPTER_CONTRACT_SHA256,
        platform_key="linux_x86_64",
        components=components,
        launch_plan_sha256=SHA_B,
        endpoint_authority_sha256=SHA_C,
        model_reference_sha256=SHA_D,
        model_artifact_kind="file",
        model_artifact_manifest_sha256=model_evidence_sha256,
        model_identity_sha256=model_evidence_sha256,
        runtime_identity_sha256=sha256_json(
            {
                "schema_version": "1.0",
                "runtime_executable_sha256": SHA_C,
                "driver_sha256": SHA_A,
            }
        ),
        harness_identity_sha256=SHA_D,
        tool_contract_identity_sha256=EMPTY_TOOL_CONTRACT_SHA256,
        model_identity_status="verified",
        execution_scope="device_only",
        transport="direct_local",
        producer_id="mymoe.bound_cell_inspector",
        producer_version="0.11.0a1",
        producer_code_sha256=SHA_B,
    )


class RuntimeBindingContractTests(unittest.TestCase):
    def test_adapter_digest_commits_the_validated_v1_boundary(self) -> None:
        payload = bound_cell_adapter_contract_payload()

        self.assertEqual(
            payload["runtime_backends"], ["llama_cpp", "mlx_lm", "mlx_vlm"]
        )
        self.assertEqual(payload["runtime_model_source"], "local")
        self.assertEqual(payload["runtime_executable_binds_launch_argv"], True)
        self.assertEqual(payload["execution_policy"]["allow_scope_widening"], False)  # type: ignore[index]
        self.assertEqual(payload["endpoint"]["scheme"], "http")  # type: ignore[index]
        self.assertEqual(payload["endpoint"]["loopback_only"], True)  # type: ignore[index]
        self.assertEqual(payload["cell_declaration"]["tool_surfaces"], [])  # type: ignore[index]
        self.assertEqual(
            payload["model_artifacts"]["public_manifest_recomputable"],  # type: ignore[index]
            True,
        )
        self.assertEqual(
            sha256_json(payload),
            BOUND_CELL_ADAPTER_CONTRACT_SHA256,
        )

    def test_component_and_manifest_are_deterministic_and_content_addressed(
        self,
    ) -> None:
        first = _manifest()
        second = _manifest()

        self.assertEqual(first, second)
        self.assertEqual(first.payload()["digest"], first.digest)
        self.assertEqual(
            [item["role"] for item in first.payload()["components"]],
            ["driver", "harness", "model_artifact", "runtime_executable"],
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "does not match"):
            replace(first, digest=SHA_D)

    def test_component_rejects_unsafe_or_non_positive_values(self) -> None:
        invalid = (
            {"path": "../escape"},
            {"path": "/absolute"},
            {"path": "windows\\path"},
            {"root_id": "unsafe root"},
            {"size_bytes": 0},
            {"size_bytes": True},
            {"size_bytes": math.inf},
            {"sha256": "not-a-digest"},
        )
        base = {
            "role": "driver",
            "root_id": "runtime",
            "path": "driver.py",
            "size_bytes": 1,
            "sha256": SHA_A,
        }
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(RuntimeBindingContractError):
                    RuntimeComponentEvidence(**{**base, **changes})

    def test_manifest_rejects_order_duplicates_and_incomplete_identity(self) -> None:
        valid = _manifest()
        with self.assertRaisesRegex(RuntimeBindingContractError, "sorted"):
            replace(valid, components=tuple(reversed(valid.components)), digest="")

        duplicate_location = _component(
            "runtime_executable",
            "lib/driver.py",
            sha256=SHA_C,
        )
        duplicated = tuple(
            sorted(
                valid.components + (duplicate_location,),
                key=lambda item: (item.role, item.root_id, item.path),
            )
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "locations"):
            replace(valid, components=duplicated, digest="")

        without_driver = tuple(
            item for item in valid.components if item.role != "driver"
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "driver"):
            replace(valid, components=without_driver, digest="")

        duplicate_driver = tuple(
            sorted(
                valid.components
                + (_component("driver", "lib/second-driver.py", sha256=SHA_B),),
                key=lambda item: (item.role, item.root_id, item.path),
            )
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "driver"):
            replace(valid, components=duplicate_driver, digest="")

        wrong_model_root = tuple(
            replace(item, root_id="runtime", digest="")
            if item.role == "model_artifact"
            else item
            for item in valid.components
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "model logical root"):
            replace(valid, components=wrong_model_root, digest="")

        with self.assertRaisesRegex(RuntimeBindingContractError, "model identity"):
            replace(valid, model_artifact_manifest_sha256=None, digest="")

        altered_model_components = tuple(
            replace(item, sha256=SHA_C, digest="")
            if item.role == "model_artifact"
            else item
            for item in valid.components
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "model evidence"):
            replace(valid, components=altered_model_components, digest="")

        second_model = _component(
            "model_artifact",
            f"artifact-{SHA_B}.gguf",
            root_id="model",
            sha256=SHA_C,
        )
        two_model_components = tuple(
            sorted(
                valid.components + (second_model,),
                key=lambda item: (item.role, item.root_id, item.path),
            )
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "one pseudonymous"):
            replace(valid, components=two_model_components, digest="")

        unredacted_model = tuple(
            replace(item, path="private-model.gguf", digest="")
            if item.role == "model_artifact"
            else item
            for item in valid.components
        )
        with self.assertRaisesRegex(RuntimeBindingContractError, "pseudonymous"):
            replace(valid, components=unredacted_model, digest="")

        with self.assertRaisesRegex(RuntimeBindingContractError, "not supported"):
            replace(valid, model_identity_status="unknown", digest="")

        for changes in (
            {"adapter_id": "other"},
            {"execution_scope": "public_mesh"},
            {"transport": "remote"},
            {"adapter_contract_sha256": SHA_A},
            {"model_artifact_kind": "archive"},
            {"runtime_identity_sha256": SHA_A},
            {"harness_identity_sha256": SHA_A},
            {"model_identity_sha256": SHA_B},
            {"tool_contract_identity_sha256": SHA_A},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(RuntimeBindingContractError):
                    replace(valid, **changes, digest="")

    def test_receipt_enforces_ttl_counts_and_non_authorizing_boundary(self) -> None:
        manifest = _manifest()
        receipt = CellRuntimeInspectionReceipt(
            binding_manifest_sha256=manifest.digest,
            status="verified",
            reason_codes=(),
            captured_at="2026-07-21T12:00:00+00:00",
            expires_at="2026-07-21T12:01:00+00:00",
            component_count=4,
            observed_component_count=4,
        )

        self.assertFalse(receipt.network_used)
        self.assertFalse(receipt.applied)
        self.assertFalse(receipt.process_mutations)
        self.assertFalse(receipt.authorizes_execution)
        self.assertEqual(receipt.processes_started, 0)
        self.assertEqual(receipt.residency_status, "unknown")
        self.assertEqual(receipt.model_invocations, 0)
        self.assertEqual(receipt.payload()["digest"], receipt.digest)

        blocked = replace(
            receipt,
            status="abstained",
            reason_codes=("model_identity_mismatch",),
            observed_component_count=4,
            digest="",
        )
        self.assertEqual(blocked.status, "abstained")

        invalid = (
            {"expires_at": receipt.captured_at},
            {"expires_at": "2026-07-21T12:02:01+00:00"},
            {"component_count": True},
            {"observed_component_count": 5},
            {"observed_component_count": 3},
            {"status": "verified", "reason_codes": ("blocked",)},
            {"status": "abstained", "reason_codes": ("blocked",)},
            {"status": "abstained", "reason_codes": ()},
            {"network_used": True},
            {"applied": True},
            {"processes_started": 1},
            {"residency_status": "resident"},
            {"process_mutations": True},
            {"authorizes_execution": True},
            {"model_invocations": 1},
            {"digest": SHA_D},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(RuntimeBindingContractError):
                    replace(receipt, **changes)


if __name__ == "__main__":
    unittest.main()
