from __future__ import annotations

import json
import unittest

from local_moe.runtime_supervisor_contracts import (
    RUNTIME_SUPERVISOR_STATES,
    RuntimeSupervisorContractError,
    RuntimeSupervisorLeaseBinding,
    RuntimeSupervisorLeasePolicy,
    RuntimeSupervisorLeaseReceipt,
    runtime_supervisor_binding_from_payload,
    runtime_supervisor_policy_from_payload,
    runtime_supervisor_receipt_from_payload,
    runtime_supervisor_transition_allowed,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
NOW = "2026-07-22T12:00:00+00:00"


def _binding(**changes) -> RuntimeSupervisorLeaseBinding:
    values = {
        "binding_request_sha256": SHA_A,
        "binding_manifest_sha256": SHA_B,
        "launch_plan_sha256": SHA_C,
        "config_source_sha256": SHA_D,
        "runtime_config_sha256": SHA_E,
        "runtime_identity_sha256": SHA_F,
        "model_identity_sha256": SHA_A,
        "endpoint_authority_sha256": SHA_B,
        "adapter_id": "mymoe.llama_cpp.direct.v1",
        "runtime_backend": "llama_cpp",
    }
    values.update(changes)
    return RuntimeSupervisorLeaseBinding(**values)


def _prepared(**changes) -> RuntimeSupervisorLeaseReceipt:
    values = {
        "policy_sha256": RuntimeSupervisorLeasePolicy().digest,
        "binding_sha256": _binding().digest,
        "endpoint_authority_sha256": SHA_B,
        "coordination_domain_sha256": SHA_C,
        "lease_id": "runtime-lease-1",
        "lease_token_sha256": SHA_D,
        "owner_pid": 123,
        "state": "prepared",
        "reason_codes": (),
        "transition_index": 0,
        "previous_receipt_sha256": None,
        "runtime_pid": None,
        "runtime_create_time_ns": None,
        "runtime_executable_sha256": None,
        "process_tree_sha256": None,
        "endpoint_evidence_sha256": None,
        "updated_at": NOW,
    }
    values.update(changes)
    return RuntimeSupervisorLeaseReceipt(**values)


class RuntimeSupervisorContractTests(unittest.TestCase):
    def test_policy_and_binding_round_trip_strictly(self) -> None:
        policy = RuntimeSupervisorLeasePolicy()
        binding = _binding()

        self.assertEqual(runtime_supervisor_policy_from_payload(policy.payload()), policy)
        self.assertEqual(
            runtime_supervisor_binding_from_payload(binding.payload()), binding
        )
        with self.assertRaises(RuntimeSupervisorContractError):
            runtime_supervisor_binding_from_payload(binding.payload() | {"extra": 1})

        tampered = binding.payload() | {"runtime_backend": "mlx_lm"}
        with self.assertRaises(RuntimeSupervisorContractError):
            runtime_supervisor_binding_from_payload(tampered)

    def test_policy_cannot_expand_metadata_authority(self) -> None:
        for name, value in (
            ("metadata_only", False),
            ("process_mutations", True),
            ("raw_tokens_persisted", True),
            ("adoption_allowed", True),
            ("automatic_restart", True),
        ):
            payload = RuntimeSupervisorLeasePolicy().payload()
            payload[name] = value
            payload["digest"] = ""
            with self.subTest(name=name), self.assertRaises(
                RuntimeSupervisorContractError
            ):
                runtime_supervisor_policy_from_payload(payload)

    def test_receipt_round_trip_is_content_addressed_and_metadata_only(self) -> None:
        receipt = _prepared()

        self.assertEqual(
            runtime_supervisor_receipt_from_payload(receipt.payload()), receipt
        )
        payload = receipt.payload()
        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("raw_token", payload)
        self.assertIn("lease_token_sha256", payload)
        self.assertNotIn("token\":", rendered)
        self.assertFalse(receipt.raw_token_serialized)
        self.assertFalse(receipt.process_mutations)
        self.assertFalse(receipt.authorizes_inference)

        tampered = receipt.payload() | {"owner_pid": 124}
        with self.assertRaises(RuntimeSupervisorContractError):
            runtime_supervisor_receipt_from_payload(tampered)

    def test_ready_receipt_requires_complete_process_and_endpoint_evidence(self) -> None:
        prepared = _prepared()
        ready = _prepared(
            state="ready",
            transition_index=2,
            previous_receipt_sha256=prepared.digest,
            runtime_pid=456,
            runtime_create_time_ns=1_234_567,
            runtime_executable_sha256=SHA_E,
            process_tree_sha256=SHA_F,
            endpoint_evidence_sha256=SHA_A,
        )
        self.assertEqual(ready.state, "ready")

        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(
                state="ready",
                reason_codes=("health_probe_failed",),
                transition_index=2,
                previous_receipt_sha256=prepared.digest,
                runtime_pid=456,
                runtime_create_time_ns=1_234_567,
                runtime_executable_sha256=SHA_E,
                process_tree_sha256=SHA_F,
                endpoint_evidence_sha256=SHA_A,
            )

        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(
                state="ready",
                transition_index=1,
                previous_receipt_sha256=prepared.digest,
                runtime_pid=456,
                runtime_create_time_ns=1_234_567,
                runtime_executable_sha256=SHA_E,
                process_tree_sha256=None,
                endpoint_evidence_sha256=SHA_A,
            )

    def test_process_identity_is_atomic_and_prepared_cannot_claim_it(self) -> None:
        prepared = _prepared()
        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(runtime_pid=1)
        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(
                state="starting",
                transition_index=1,
                previous_receipt_sha256=prepared.digest,
                runtime_pid=1,
                runtime_create_time_ns=None,
                runtime_executable_sha256=SHA_A,
            )
        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(
                state="revoked",
                reason_codes=("runtime_exited",),
                transition_index=1,
                previous_receipt_sha256=prepared.digest,
                endpoint_evidence_sha256=SHA_A,
            )

    def test_revoked_and_unknown_states_require_supported_reasons(self) -> None:
        prepared = _prepared()
        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(
                state="revoked",
                transition_index=1,
                previous_receipt_sha256=prepared.digest,
            )
        with self.assertRaises(RuntimeSupervisorContractError):
            _prepared(
                state="unknown_blocking",
                reason_codes=("not-a-contract-reason",),
                transition_index=1,
                previous_receipt_sha256=prepared.digest,
            )

    def test_state_machine_is_explicit_and_unknown_is_sticky(self) -> None:
        self.assertEqual(
            RUNTIME_SUPERVISOR_STATES,
            {
                "prepared",
                "starting",
                "ready",
                "stopping",
                "stopped",
                "revoked",
                "unknown_blocking",
            },
        )
        self.assertTrue(runtime_supervisor_transition_allowed("prepared", "starting"))
        self.assertTrue(runtime_supervisor_transition_allowed("ready", "revoked"))
        self.assertTrue(
            runtime_supervisor_transition_allowed(
                "unknown_blocking", "unknown_blocking"
            )
        )
        self.assertFalse(
            runtime_supervisor_transition_allowed("unknown_blocking", "stopped")
        )
        self.assertFalse(runtime_supervisor_transition_allowed("stopped", "ready"))

    def test_binding_cannot_widen_scope_or_transport(self) -> None:
        with self.assertRaises(RuntimeSupervisorContractError):
            _binding(execution_scope="region_allowed")
        with self.assertRaises(RuntimeSupervisorContractError):
            _binding(transport="proxy_allowed")


if __name__ == "__main__":
    unittest.main()
