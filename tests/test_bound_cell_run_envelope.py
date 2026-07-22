from __future__ import annotations

import json
import unittest

from local_moe.bound_cell_run_contracts import BoundCellRunReceipt
from local_moe.bound_cell_run_envelope import (
    BOUND_CELL_RUN_ENVELOPE_V2_CONTRACT,
    BOUND_CELL_RUN_ENVELOPE_V2_SCHEMA_VERSION,
    BoundCellRunEnvelopeContractError,
    BoundCellRunEnvelopeV2,
    bound_cell_run_envelope_v2_from_payload,
)
from local_moe.cooperative_resource_lease_contracts import (
    CooperativeResourceClaim,
    CooperativeResourceLeaseAdmissionReceipt,
    CooperativeResourceLeaseReleaseReceipt,
    CooperativeResourceLeaseTransitionReceipt,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_0 = "0" * 64
SHA_1 = "1" * 64
NOW = "2026-07-22T12:00:00+00:00"
LATER = "2026-07-22T12:00:01+00:00"


def _blocked_run(**changes: object) -> BoundCellRunReceipt:
    values: dict[str, object] = {
        "policy_sha256": SHA_A,
        "status": "blocked",
        "reason_codes": ("confirmation_required",),
        "started_at": NOW,
        "completed_at": LATER,
        "confirmed": False,
        "task_sha256": SHA_B,
        "task_bytes": 12,
    }
    values.update(changes)
    return BoundCellRunReceipt(**values)


def _attempted_run(
    *, delivery_status: str = "attempted_unknown"
) -> BoundCellRunReceipt:
    return BoundCellRunReceipt(
        policy_sha256=SHA_A,
        status="failed",
        reason_codes=(
            "transport_failed"
            if delivery_status == "attempted_unknown"
            else "response_invalid",
        ),
        started_at=NOW,
        completed_at=LATER,
        confirmed=True,
        task_sha256=SHA_B,
        task_bytes=12,
        preview_sha256=SHA_C,
        selected_cell_id="cell-1",
        passport_sha256=SHA_D,
        declaration_sha256=SHA_E,
        expert_id="expert-1",
        pre_binding_bundle_sha256=SHA_F,
        pre_binding_request_sha256=SHA_0,
        pre_binding_manifest_sha256=SHA_1,
        pre_inspection_receipt_sha256=SHA_A,
        pre_config_source_sha256=SHA_B,
        pre_model_identity_set_sha256=SHA_C,
        invocation_attempts=1,
        endpoint_probe_requests=2,
        delivery_status=delivery_status,
    )


def _claim(**changes: object) -> CooperativeResourceClaim:
    values: dict[str, object] = {
        "preview_sha256": SHA_C,
        "candidate_sha256": SHA_E,
        "passport_sha256": SHA_D,
        "resource_snapshot_sha256": SHA_F,
        "resource_class_sha256": SHA_0,
        "catalog_sha256": SHA_1,
        "profile_sha256": SHA_A,
        "pool": "system",
        "system_claim_bytes": 600,
        "accelerator_claim_bytes": 0,
        "accelerator_identity_sha256": None,
        "safety_reserve_bytes": 100,
    }
    values.update(changes)
    return CooperativeResourceClaim(**values)


def _admission(
    claim: CooperativeResourceClaim,
    *,
    status: str = "acquired",
    **changes: object,
) -> CooperativeResourceLeaseAdmissionReceipt:
    acquired = status == "acquired"
    values: dict[str, object] = {
        "policy_sha256": SHA_A,
        "claim_sha256": claim.digest,
        "resource_snapshot_sha256": claim.resource_snapshot_sha256,
        "coordination_domain_sha256": SHA_B,
        "status": status,
        "reason_codes": () if acquired else ("capacity_exceeded",),
        "evaluated_at": NOW,
        "lease_id": "lease-1" if acquired else None,
        "lease_token_sha256": SHA_E if acquired else None,
        "active_leases_before": 0,
        "active_leases_after": 1 if acquired else 0,
        "reaped_leases": 0,
        "system_available_bytes": 1_000,
        "accelerator_available_bytes": None,
        "active_system_claim_bytes": 0,
        "active_accelerator_claim_bytes": 0,
        "requested_system_claim_bytes": claim.system_claim_bytes,
        "requested_accelerator_claim_bytes": claim.accelerator_claim_bytes,
        "safety_reserve_bytes": claim.safety_reserve_bytes,
        "applied_system_reserve_bytes": claim.safety_reserve_bytes,
        "applied_accelerator_reserve_bytes": 0,
    }
    values.update(changes)
    return CooperativeResourceLeaseAdmissionReceipt(**values)


def _transition(
    claim: CooperativeResourceClaim,
    admission: CooperativeResourceLeaseAdmissionReceipt,
    *,
    applied: bool = True,
    **changes: object,
) -> CooperativeResourceLeaseTransitionReceipt:
    values: dict[str, object] = {
        "policy_sha256": admission.policy_sha256,
        "admission_receipt_sha256": admission.digest,
        "claim_sha256": claim.digest,
        "coordination_domain_sha256": admission.coordination_domain_sha256,
        "lease_id": admission.lease_id,
        "state": "delivery_armed",
        "transition_applied": applied,
        "reason_codes": () if applied else ("lease_state_changed",),
        "transitioned_at": NOW,
    }
    values.update(changes)
    return CooperativeResourceLeaseTransitionReceipt(**values)


def _release(
    claim: CooperativeResourceClaim,
    admission: CooperativeResourceLeaseAdmissionReceipt,
    *,
    delivery_status: str = "attempted_unknown",
    **changes: object,
) -> CooperativeResourceLeaseReleaseReceipt:
    unknown = delivery_status == "attempted_unknown"
    values: dict[str, object] = {
        "policy_sha256": admission.policy_sha256,
        "admission_receipt_sha256": admission.digest,
        "claim_sha256": claim.digest,
        "coordination_domain_sha256": admission.coordination_domain_sha256,
        "lease_id": admission.lease_id,
        "status": "unknown_blocking" if unknown else "released",
        "reason_codes": ("delivery_outcome_unknown",) if unknown else (),
        "delivery_status": delivery_status,
        "released_at": LATER,
        "active_leases_after": 1 if unknown else 0,
    }
    values.update(changes)
    return CooperativeResourceLeaseReleaseReceipt(**values)


def _attempted_envelope() -> BoundCellRunEnvelopeV2:
    run = _attempted_run()
    claim = _claim()
    admission = _admission(claim)
    transition = _transition(claim, admission)
    release = _release(claim, admission)
    return BoundCellRunEnvelopeV2(
        run_receipt=run,
        lease_claim=claim,
        lease_admission_receipt=admission,
        lease_transition_receipt=transition,
        lease_release_receipt=release,
    )


class BoundCellRunEnvelopeV2Tests(unittest.TestCase):
    def test_full_lineage_round_trips_with_exact_v2_schema(self) -> None:
        envelope = _attempted_envelope()

        self.assertEqual(
            bound_cell_run_envelope_v2_from_payload(envelope.payload()), envelope
        )
        self.assertEqual(envelope.contract, BOUND_CELL_RUN_ENVELOPE_V2_CONTRACT)
        self.assertEqual(
            envelope.schema_version, BOUND_CELL_RUN_ENVELOPE_V2_SCHEMA_VERSION
        )
        self.assertEqual(envelope.run_receipt_sha256, envelope.run_receipt.digest)
        self.assertTrue(envelope.cooperative_only)
        self.assertFalse(envelope.os_memory_reserved)
        self.assertFalse(envelope.runtime_managed)

    def test_nested_v1_payload_is_byte_equivalent(self) -> None:
        run = _attempted_run()
        envelope = _attempted_envelope()
        canonical = lambda value: json.dumps(  # noqa: E731
            value, sort_keys=True, separators=(",", ":")
        ).encode()

        self.assertEqual(
            canonical(envelope.payload()["run_receipt"]), canonical(run.payload())
        )
        self.assertEqual(envelope.run_receipt_sha256, run.digest)

    def test_parser_rejects_unknown_top_level_and_nested_fields(self) -> None:
        payload = _attempted_envelope().payload()
        with self.assertRaises(BoundCellRunEnvelopeContractError):
            bound_cell_run_envelope_v2_from_payload(payload | {"extra": True})

        nested = _attempted_envelope().payload()
        nested["run_receipt"] = dict(nested["run_receipt"]) | {"extra": True}
        nested["digest"] = ""
        with self.assertRaises(BoundCellRunEnvelopeContractError):
            bound_cell_run_envelope_v2_from_payload(nested)

        with self.assertRaisesRegex(
            BoundCellRunEnvelopeContractError, "unsupported contract type"
        ):
            BoundCellRunEnvelopeV2(
                run_receipt=_blocked_run(), lease_claim={"digest": SHA_A}
            )

    def test_content_and_nested_digest_tampering_is_rejected(self) -> None:
        payload = _attempted_envelope().payload()
        payload["lease_error_code"] = "lease_release_failed"
        with self.assertRaisesRegex(
            BoundCellRunEnvelopeContractError, "digest_mismatch"
        ):
            bound_cell_run_envelope_v2_from_payload(payload)

        nested = _attempted_envelope().payload()
        run_payload = dict(nested["run_receipt"])
        run_payload["elapsed_ms"] = 42
        nested["run_receipt"] = run_payload
        nested["digest"] = ""
        with self.assertRaises(BoundCellRunEnvelopeContractError):
            bound_cell_run_envelope_v2_from_payload(nested)

    def test_pre_lease_block_allows_all_lease_evidence_to_be_null(self) -> None:
        envelope = BoundCellRunEnvelopeV2(run_receipt=_blocked_run())
        self.assertIsNone(envelope.lease_claim)
        self.assertEqual(
            bound_cell_run_envelope_v2_from_payload(envelope.payload()), envelope
        )

        with self.assertRaisesRegex(
            BoundCellRunEnvelopeContractError, "pre-lease block"
        ):
            BoundCellRunEnvelopeV2(run_receipt=_attempted_run())

    def test_denied_admission_never_transitions_or_releases(self) -> None:
        claim = _claim()
        admission = _admission(claim, status="denied")
        run = _blocked_run(
            confirmed=True,
            reason_codes=("adaptive_admission_blocked",),
            preview_sha256=claim.preview_sha256,
            passport_sha256=claim.passport_sha256,
        )
        envelope = BoundCellRunEnvelopeV2(
            run_receipt=run,
            lease_claim=claim,
            lease_admission_receipt=admission,
        )
        self.assertEqual(envelope.lease_admission_receipt.status, "denied")

        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "cannot"):
            BoundCellRunEnvelopeV2(
                run_receipt=run,
                lease_claim=claim,
                lease_admission_receipt=admission,
                lease_transition_receipt=_transition(
                    claim, _admission(claim), applied=True
                ),
            )

    def test_acquired_unattempted_run_requires_not_attempted_release(self) -> None:
        claim = _claim()
        admission = _admission(claim)
        run = _blocked_run(
            confirmed=True,
            reason_codes=("binding_changed",),
            preview_sha256=claim.preview_sha256,
            passport_sha256=claim.passport_sha256,
        )
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "release"):
            BoundCellRunEnvelopeV2(
                run_receipt=run,
                lease_claim=claim,
                lease_admission_receipt=admission,
            )

        released = BoundCellRunEnvelopeV2(
            run_receipt=run,
            lease_claim=claim,
            lease_admission_receipt=admission,
            lease_release_receipt=_release(
                claim, admission, delivery_status="not_attempted"
            ),
        )
        self.assertEqual(
            released.lease_release_receipt.delivery_status, "not_attempted"
        )

        failed_release = BoundCellRunEnvelopeV2(
            run_receipt=run,
            lease_claim=claim,
            lease_admission_receipt=admission,
            lease_error_code="lease_release_failed",
        )
        self.assertIsNone(failed_release.lease_release_receipt)

    def test_attempt_requires_an_applied_transition(self) -> None:
        run = _attempted_run()
        claim = _claim()
        admission = _admission(claim)
        release = _release(claim, admission)

        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "transition"):
            BoundCellRunEnvelopeV2(
                run_receipt=run,
                lease_claim=claim,
                lease_admission_receipt=admission,
                lease_release_receipt=release,
            )
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "attempt"):
            BoundCellRunEnvelopeV2(
                run_receipt=run,
                lease_claim=claim,
                lease_admission_receipt=admission,
                lease_transition_receipt=_transition(
                    claim, admission, applied=False
                ),
                lease_release_receipt=release,
                lease_error_code="lease_transition_failed",
            )

    def test_delivery_and_lease_identifiers_must_match(self) -> None:
        envelope = _attempted_envelope()
        response_release = _release(
            envelope.lease_claim,
            envelope.lease_admission_receipt,
            delivery_status="response_received",
        )
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "delivery"):
            BoundCellRunEnvelopeV2(
                run_receipt=envelope.run_receipt,
                lease_claim=envelope.lease_claim,
                lease_admission_receipt=envelope.lease_admission_receipt,
                lease_transition_receipt=envelope.lease_transition_receipt,
                lease_release_receipt=response_release,
            )

        wrong_lease = _transition(
            envelope.lease_claim,
            envelope.lease_admission_receipt,
            lease_id="lease-2",
        )
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "lease id"):
            BoundCellRunEnvelopeV2(
                run_receipt=envelope.run_receipt,
                lease_claim=envelope.lease_claim,
                lease_admission_receipt=envelope.lease_admission_receipt,
                lease_transition_receipt=wrong_lease,
                lease_release_receipt=envelope.lease_release_receipt,
            )

    def test_claim_and_admission_digests_must_match(self) -> None:
        run = _attempted_run()
        claim = _claim()
        mismatched = _admission(claim, claim_sha256=SHA_0)
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "claim digest"):
            BoundCellRunEnvelopeV2(
                run_receipt=run,
                lease_claim=claim,
                lease_admission_receipt=mismatched,
                lease_transition_receipt=_transition(claim, mismatched),
                lease_release_receipt=_release(claim, mismatched),
            )

    def test_error_codes_and_authority_flags_are_closed(self) -> None:
        run = _blocked_run()
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "error_code"):
            BoundCellRunEnvelopeV2(
                run_receipt=run, lease_error_code="future_unreviewed_error"
            )
        with self.assertRaisesRegex(BoundCellRunEnvelopeContractError, "must remain"):
            BoundCellRunEnvelopeV2(run_receipt=run, os_memory_reserved=True)

    def test_payload_contains_no_raw_task_answer_or_token_secret(self) -> None:
        rendered = json.dumps(_attempted_envelope().payload(), sort_keys=True)
        for secret in (
            "RAW_TASK_super_secret_143",
            "RAW_ANSWER_super_secret_287",
            "RAW_LEASE_TOKEN_super_secret_991",
        ):
            self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
