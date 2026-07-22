from __future__ import annotations

from pathlib import Path

from local_moe.adaptive_execution_gate import (
    AdaptiveCellExecutionPreviewEvaluation,
    AdaptiveCellExecutionPreviewReceipt,
)
from local_moe.cooperative_resource_lease import (
    CooperativeResourceLeaseAcquisition,
    CooperativeResourceLeaseHandle,
)
from local_moe.cooperative_resource_lease_contracts import (
    CooperativeResourceClaim,
    CooperativeResourceLeaseAdmissionReceipt,
    CooperativeResourceLeaseReleaseReceipt,
    CooperativeResourceLeaseTransitionReceipt,
)
from local_moe.resource_snapshot import ResourceSnapshot


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
NOW = "2026-07-22T12:00:00+00:00"


def resource_snapshot(*, available_bytes: int = 8_000_000_000) -> ResourceSnapshot:
    return ResourceSnapshot(
        system="Linux",
        os_release="test",
        machine="x86_64",
        cpu_count=8,
        cpu_identity_sha256=SHA_A,
        memory_topology="system",
        total_memory_bytes=16_000_000_000,
        available_memory_bytes=available_bytes,
        effective_memory_limit_bytes=16_000_000_000,
        swap_used_bytes=0,
        accelerator_kind="none",
        accelerator_identity_sha256=None,
        accelerator_memory_total_bytes=None,
        accelerator_memory_available_bytes=None,
        runtime_environment_sha256=SHA_B,
        captured_at=NOW,
        source_sha256=SHA_C,
    )


def claim_for(
    preview: AdaptiveCellExecutionPreviewReceipt,
    snapshot: ResourceSnapshot,
    *,
    passport_sha256: str,
) -> CooperativeResourceClaim:
    return CooperativeResourceClaim(
        preview_sha256=preview.digest,
        candidate_sha256=SHA_A,
        passport_sha256=passport_sha256,
        resource_snapshot_sha256=snapshot.digest,
        resource_class_sha256=snapshot.resource_class_sha256,
        catalog_sha256=SHA_B,
        profile_sha256=SHA_C,
        pool="system",
        system_claim_bytes=1_000_000_000,
        accelerator_claim_bytes=0,
        accelerator_identity_sha256=None,
        safety_reserve_bytes=100_000_000,
    )


def preview_evaluation(
    preview: AdaptiveCellExecutionPreviewReceipt,
    snapshot: ResourceSnapshot,
) -> AdaptiveCellExecutionPreviewEvaluation:
    """Build a narrow test double while keeping the public evaluation type."""

    result = object.__new__(AdaptiveCellExecutionPreviewEvaluation)
    object.__setattr__(result, "receipt", preview)
    object.__setattr__(result, "fresh_advisor_receipt", object())
    object.__setattr__(result, "resource_snapshot", snapshot)
    object.__setattr__(result, "catalog", object())
    return result


class FakeLeaseStore:
    def __init__(
        self,
        *,
        admission_status: str = "acquired",
        transition_applied: bool = True,
        events: list[str] | None = None,
    ) -> None:
        self.admission_status = admission_status
        self.transition_applied = transition_applied
        self.events = events
        self.claim: CooperativeResourceClaim | None = None
        self.admission: CooperativeResourceLeaseAdmissionReceipt | None = None
        self.handle: CooperativeResourceLeaseHandle | None = None

    def evaluate_and_acquire(self, evaluator):
        if self.events is not None:
            self.events.append("lease_evaluate")
        evaluation = evaluator()
        claim = evaluation.claim
        snapshot = evaluation.snapshot
        self.claim = claim
        acquired = self.admission_status == "acquired"
        reasons = () if acquired else ("system_capacity_insufficient",)
        self.admission = CooperativeResourceLeaseAdmissionReceipt(
            policy_sha256=SHA_A,
            claim_sha256=claim.digest,
            resource_snapshot_sha256=snapshot.digest,
            coordination_domain_sha256=SHA_D,
            status=self.admission_status,
            reason_codes=reasons,
            evaluated_at=NOW,
            lease_id="lease-a" if acquired else None,
            lease_token_sha256=SHA_B if acquired else None,
            active_leases_before=0,
            active_leases_after=1 if acquired else 0,
            reaped_leases=0,
            system_available_bytes=snapshot.available_memory_bytes,
            accelerator_available_bytes=None,
            active_system_claim_bytes=0,
            active_accelerator_claim_bytes=0,
            requested_system_claim_bytes=claim.system_claim_bytes,
            requested_accelerator_claim_bytes=claim.accelerator_claim_bytes,
            safety_reserve_bytes=claim.safety_reserve_bytes,
            applied_system_reserve_bytes=claim.safety_reserve_bytes,
            applied_accelerator_reserve_bytes=0,
        )
        self.handle = (
            CooperativeResourceLeaseHandle(
                lease_id="lease-a",
                admission_receipt_sha256=self.admission.digest,
                claim_sha256=claim.digest,
                token=b"x" * 32,
                _owner_lock=object(),  # type: ignore[arg-type]
                _sentinel_path=Path("unused-test-sentinel"),
            )
            if acquired
            else None
        )
        return CooperativeResourceLeaseAcquisition(
            receipt=self.admission,
            handle=self.handle,
            context=evaluation.context,
        )

    def arm_delivery(
        self, handle: CooperativeResourceLeaseHandle
    ) -> CooperativeResourceLeaseTransitionReceipt:
        if self.events is not None:
            self.events.append("lease_arm")
        assert self.admission is not None
        assert self.claim is not None
        assert handle is self.handle
        return CooperativeResourceLeaseTransitionReceipt(
            policy_sha256=SHA_A,
            admission_receipt_sha256=self.admission.digest,
            claim_sha256=self.claim.digest,
            coordination_domain_sha256=SHA_D,
            lease_id="lease-a",
            state="delivery_armed",
            transition_applied=self.transition_applied,
            reason_codes=() if self.transition_applied else ("lease_not_found",),
            transitioned_at=NOW,
        )

    def release(
        self,
        handle: CooperativeResourceLeaseHandle,
        *,
        delivery_status: str,
    ) -> CooperativeResourceLeaseReleaseReceipt:
        if self.events is not None:
            self.events.append(f"lease_release:{delivery_status}")
        assert self.admission is not None
        assert self.claim is not None
        assert handle is self.handle
        unknown = delivery_status == "attempted_unknown"
        return CooperativeResourceLeaseReleaseReceipt(
            policy_sha256=SHA_A,
            admission_receipt_sha256=self.admission.digest,
            claim_sha256=self.claim.digest,
            coordination_domain_sha256=SHA_D,
            lease_id="lease-a",
            status="unknown_blocking" if unknown else "released",
            reason_codes=("delivery_outcome_unknown",) if unknown else (),
            delivery_status=delivery_status,
            released_at=NOW,
            active_leases_after=1 if unknown else 0,
        )

