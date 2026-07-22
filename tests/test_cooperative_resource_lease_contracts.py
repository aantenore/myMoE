from __future__ import annotations

import unittest

from local_moe.cooperative_resource_lease_contracts import (
    CLAIM_BASIS,
    CooperativeResourceClaim,
    CooperativeResourceLeaseAdmissionReceipt,
    CooperativeResourceLeaseContractError,
    CooperativeResourceLeasePolicy,
    CooperativeResourceLeaseReleaseReceipt,
    CooperativeResourceLeaseTransitionReceipt,
    cooperative_resource_claim_from_payload,
    cooperative_resource_lease_admission_receipt_from_payload,
    cooperative_resource_lease_policy_from_payload,
    cooperative_resource_lease_release_receipt_from_payload,
    cooperative_resource_lease_transition_receipt_from_payload,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
NOW = "2026-07-22T12:00:00+00:00"


def _claim(**changes) -> CooperativeResourceClaim:
    values = {
        "preview_sha256": SHA_A,
        "candidate_sha256": SHA_B,
        "passport_sha256": SHA_C,
        "resource_snapshot_sha256": SHA_D,
        "resource_class_sha256": SHA_E,
        "catalog_sha256": SHA_A,
        "profile_sha256": SHA_B,
        "pool": "system",
        "system_claim_bytes": 600,
        "accelerator_claim_bytes": 0,
        "accelerator_identity_sha256": None,
        "safety_reserve_bytes": 100,
    }
    values.update(changes)
    return CooperativeResourceClaim(**values)


class CooperativeResourceLeaseContractTests(unittest.TestCase):
    def test_policy_and_claim_round_trip_strictly(self) -> None:
        policy = CooperativeResourceLeasePolicy()
        claim = _claim()

        self.assertEqual(
            cooperative_resource_lease_policy_from_payload(policy.payload()), policy
        )
        self.assertEqual(
            cooperative_resource_claim_from_payload(claim.payload()), claim
        )
        self.assertEqual(claim.claim_basis, CLAIM_BASIS)

        unknown = claim.payload() | {"extra": True}
        with self.assertRaises(CooperativeResourceLeaseContractError):
            cooperative_resource_claim_from_payload(unknown)

    def test_claim_rejects_incremental_or_invalid_pool_shapes(self) -> None:
        with self.assertRaises(CooperativeResourceLeaseContractError):
            _claim(claim_basis="incremental")
        with self.assertRaises(CooperativeResourceLeaseContractError):
            _claim(accelerator_claim_bytes=1)
        with self.assertRaises(CooperativeResourceLeaseContractError):
            _claim(
                pool="discrete",
                accelerator_claim_bytes=500,
                accelerator_identity_sha256=None,
            )

    def test_discrete_claim_requires_both_physical_pools(self) -> None:
        claim = _claim(
            pool="discrete",
            system_claim_bytes=300,
            accelerator_claim_bytes=700,
            accelerator_identity_sha256=SHA_F,
        )
        self.assertEqual(claim.system_claim_bytes, 300)
        self.assertEqual(claim.accelerator_claim_bytes, 700)

    def test_admission_receipt_is_content_addressed_and_strict(self) -> None:
        receipt = CooperativeResourceLeaseAdmissionReceipt(
            policy_sha256=SHA_A,
            claim_sha256=SHA_B,
            resource_snapshot_sha256=SHA_C,
            coordination_domain_sha256=SHA_D,
            status="acquired",
            reason_codes=(),
            evaluated_at=NOW,
            lease_id="lease-1",
            lease_token_sha256=SHA_E,
            active_leases_before=0,
            active_leases_after=1,
            reaped_leases=0,
            system_available_bytes=1_000,
            accelerator_available_bytes=None,
            active_system_claim_bytes=0,
            active_accelerator_claim_bytes=0,
            requested_system_claim_bytes=600,
            requested_accelerator_claim_bytes=0,
            safety_reserve_bytes=100,
            applied_system_reserve_bytes=100,
            applied_accelerator_reserve_bytes=0,
        )
        self.assertEqual(
            cooperative_resource_lease_admission_receipt_from_payload(
                receipt.payload()
            ),
            receipt,
        )
        tampered = receipt.payload() | {"active_leases_after": 2}
        with self.assertRaises(CooperativeResourceLeaseContractError):
            cooperative_resource_lease_admission_receipt_from_payload(tampered)

    def test_transition_and_release_receipts_round_trip(self) -> None:
        transition = CooperativeResourceLeaseTransitionReceipt(
            policy_sha256=SHA_A,
            admission_receipt_sha256=SHA_B,
            claim_sha256=SHA_C,
            coordination_domain_sha256=SHA_D,
            lease_id="lease-1",
            state="delivery_armed",
            transition_applied=True,
            reason_codes=(),
            transitioned_at=NOW,
        )
        release = CooperativeResourceLeaseReleaseReceipt(
            policy_sha256=SHA_A,
            admission_receipt_sha256=SHA_B,
            claim_sha256=SHA_C,
            coordination_domain_sha256=SHA_D,
            lease_id="lease-1",
            status="released",
            reason_codes=(),
            delivery_status="response_received",
            released_at=NOW,
            active_leases_after=0,
        )
        self.assertEqual(
            cooperative_resource_lease_transition_receipt_from_payload(
                transition.payload()
            ),
            transition,
        )
        self.assertEqual(
            cooperative_resource_lease_release_receipt_from_payload(release.payload()),
            release,
        )

    def test_receipts_cannot_claim_os_or_runtime_authority(self) -> None:
        policy = CooperativeResourceLeasePolicy()
        payload = policy.payload()
        payload["os_memory_reserved"] = True
        payload["digest"] = ""
        with self.assertRaises(CooperativeResourceLeaseContractError):
            cooperative_resource_lease_policy_from_payload(payload)


if __name__ == "__main__":
    unittest.main()
