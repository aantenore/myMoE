from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

from .bound_cell_run_contracts import (
    BoundCellRunContractError,
    BoundCellRunReceipt,
    bound_cell_run_receipt_from_payload,
)
from .cooperative_resource_lease_contracts import (
    CooperativeResourceClaim,
    CooperativeResourceLeaseAdmissionReceipt,
    CooperativeResourceLeaseContractError,
    CooperativeResourceLeaseReleaseReceipt,
    CooperativeResourceLeaseTransitionReceipt,
    cooperative_resource_claim_from_payload,
    cooperative_resource_lease_admission_receipt_from_payload,
    cooperative_resource_lease_release_receipt_from_payload,
    cooperative_resource_lease_transition_receipt_from_payload,
)
from .verified_routing_contracts import VerifiedRoutingError, require_sha256, sha256_json


BOUND_CELL_RUN_ENVELOPE_V2_CONTRACT = "BoundCellRunEnvelopeV2"
BOUND_CELL_RUN_ENVELOPE_V2_SCHEMA_VERSION = "2.0"
LEASE_ERROR_CODES = frozenset(
    {
        "adaptive_preview_blocked",
        "adaptive_preview_invalid",
        "lease_store_failed",
        "lease_transition_failed",
        "lease_release_failed",
    }
)


class BoundCellRunEnvelopeContractError(ValueError):
    """Raised when a v2 run envelope is incomplete or internally inconsistent."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _sha(value: object, label: str) -> str:
    try:
        return require_sha256(value, label)
    except (OverflowError, TypeError, ValueError, VerifiedRoutingError) as exc:
        raise BoundCellRunEnvelopeContractError("contract_invalid", str(exc)) from exc


def _digest(provided: object, content: Mapping[str, object]) -> str:
    expected = sha256_json(content)
    if provided not in (None, "") and _sha(provided, "envelope digest") != expected:
        raise BoundCellRunEnvelopeContractError(
            "digest_mismatch", "Envelope digest does not match its content."
        )
    return expected


def _fixed_boolean(value: object, *, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        rendered = "true" if expected else "false"
        raise BoundCellRunEnvelopeContractError(
            "authority_invalid", f"{label} must remain {rendered}."
        )
    return value


@dataclass(frozen=True)
class BoundCellRunEnvelopeV2:
    """Content-addressed run evidence plus cooperative lease evidence.

    The nested v1 run receipt is preserved verbatim. The additional evidence
    describes cooperative accounting only; it does not assert an operating-system
    memory reservation or runtime lifecycle authority.
    """

    run_receipt: BoundCellRunReceipt
    run_receipt_sha256: str = ""
    lease_claim: CooperativeResourceClaim | None = None
    lease_admission_receipt: CooperativeResourceLeaseAdmissionReceipt | None = None
    lease_transition_receipt: CooperativeResourceLeaseTransitionReceipt | None = None
    lease_release_receipt: CooperativeResourceLeaseReleaseReceipt | None = None
    lease_error_code: str | None = None
    cooperative_only: bool = True
    os_memory_reserved: bool = False
    runtime_managed: bool = False
    digest: str = ""
    contract: str = BOUND_CELL_RUN_ENVELOPE_V2_CONTRACT
    schema_version: str = BOUND_CELL_RUN_ENVELOPE_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run_receipt, BoundCellRunReceipt):
            raise BoundCellRunEnvelopeContractError(
                "contract_invalid", "run_receipt must be a BoundCellRunReceipt."
            )
        if (
            self.contract != BOUND_CELL_RUN_ENVELOPE_V2_CONTRACT
            or self.schema_version != BOUND_CELL_RUN_ENVELOPE_V2_SCHEMA_VERSION
        ):
            raise BoundCellRunEnvelopeContractError(
                "contract_invalid", "Unsupported bound-cell run envelope contract."
            )

        receipt_sha = self.run_receipt.digest
        if self.run_receipt_sha256 not in (None, ""):
            supplied_sha = _sha(self.run_receipt_sha256, "run_receipt_sha256")
            if supplied_sha != receipt_sha:
                raise BoundCellRunEnvelopeContractError(
                    "evidence_mismatch",
                    "run_receipt_sha256 does not match the nested v1 receipt digest.",
                )
        object.__setattr__(self, "run_receipt_sha256", receipt_sha)

        if self.lease_error_code is not None:
            if (
                not isinstance(self.lease_error_code, str)
                or self.lease_error_code not in LEASE_ERROR_CODES
            ):
                raise BoundCellRunEnvelopeContractError(
                    "contract_invalid", "Unsupported lease_error_code."
                )

        _fixed_boolean(
            self.cooperative_only, expected=True, label="cooperative_only"
        )
        _fixed_boolean(
            self.os_memory_reserved, expected=False, label="os_memory_reserved"
        )
        _fixed_boolean(self.runtime_managed, expected=False, label="runtime_managed")

        expected_evidence_types = (
            ("lease_claim", self.lease_claim, CooperativeResourceClaim),
            (
                "lease_admission_receipt",
                self.lease_admission_receipt,
                CooperativeResourceLeaseAdmissionReceipt,
            ),
            (
                "lease_transition_receipt",
                self.lease_transition_receipt,
                CooperativeResourceLeaseTransitionReceipt,
            ),
            (
                "lease_release_receipt",
                self.lease_release_receipt,
                CooperativeResourceLeaseReleaseReceipt,
            ),
        )
        for label, evidence, expected_type in expected_evidence_types:
            if evidence is not None and not isinstance(evidence, expected_type):
                raise BoundCellRunEnvelopeContractError(
                    "contract_invalid", f"{label} has an unsupported contract type."
                )

        self._validate_lease_lineage()
        object.__setattr__(self, "digest", _digest(self.digest, self.content_payload()))

    def _validate_lease_lineage(self) -> None:
        run = self.run_receipt
        claim = self.lease_claim
        admission = self.lease_admission_receipt
        transition = self.lease_transition_receipt
        release = self.lease_release_receipt
        error = self.lease_error_code

        if all(item is None for item in (claim, admission, transition, release)):
            if (
                run.status != "blocked"
                or run.invocation_attempts != 0
                or run.delivery_status != "not_attempted"
            ):
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "A run without lease evidence must be a pre-lease block.",
                )
            if error not in {
                None,
                "adaptive_preview_blocked",
                "adaptive_preview_invalid",
                "lease_store_failed",
            }:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "The lease error is incompatible with absent lease evidence.",
                )
            return

        if claim is None:
            raise BoundCellRunEnvelopeContractError(
                "lineage_invalid", "Lease evidence requires a resource claim."
            )
        if run.preview_sha256 != claim.preview_sha256:
            raise BoundCellRunEnvelopeContractError(
                "evidence_mismatch", "Run and claim preview digests differ."
            )
        if run.passport_sha256 != claim.passport_sha256:
            raise BoundCellRunEnvelopeContractError(
                "evidence_mismatch", "Run and claim passport digests differ."
            )

        if admission is None:
            if transition is not None or release is not None:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "Transition and release evidence require an admission receipt.",
                )
            if error != "lease_store_failed":
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "A claim without admission is valid only after a lease-store failure.",
                )
            if run.invocation_attempts != 0 or run.delivery_status != "not_attempted":
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid", "Lease-store failure cannot attempt inference."
                )
            return

        self._validate_claim_admission(claim, admission)

        if admission.status != "acquired":
            if transition is not None or release is not None:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "A denied admission cannot transition or release a lease.",
                )
            if (
                run.status != "blocked"
                or run.invocation_attempts != 0
                or run.delivery_status != "not_attempted"
            ):
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "A non-acquired admission requires an unattempted blocked run.",
                )
            if "adaptive_admission_blocked" not in run.reason_codes:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "A non-acquired admission must surface adaptive_admission_blocked.",
                )
            if error is not None:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "A normal non-acquired admission cannot report a lease error.",
                )
            return

        if error in {
            "adaptive_preview_blocked",
            "adaptive_preview_invalid",
            "lease_store_failed",
        }:
            raise BoundCellRunEnvelopeContractError(
                "lineage_invalid", "The lease error conflicts with acquired admission."
            )

        if transition is not None:
            self._validate_transition(admission, claim, transition)
        if release is not None:
            self._validate_release(admission, claim, release)

        if run.invocation_attempts == 1:
            if transition is None or transition.transition_applied is not True:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "An inference attempt requires an applied delivery transition.",
                )
        elif transition is not None and transition.transition_applied:
            raise BoundCellRunEnvelopeContractError(
                "lineage_invalid",
                "An applied delivery transition requires one invocation attempt.",
            )

        if release is not None and release.delivery_status != run.delivery_status:
            raise BoundCellRunEnvelopeContractError(
                "evidence_mismatch", "Run and release delivery statuses differ."
            )

        if run.invocation_attempts == 0 and error is None:
            if release is None or release.delivery_status != "not_attempted":
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid",
                    "An acquired unattempted run requires a not-attempted release.",
                )
        if run.invocation_attempts == 1 and error is None and release is None:
            raise BoundCellRunEnvelopeContractError(
                "lineage_invalid", "A completed lease lineage requires release evidence."
            )

        if error == "lease_transition_failed":
            if run.invocation_attempts != 0:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid", "A transition failure cannot attempt inference."
                )
            if transition is not None and transition.transition_applied:
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid", "A failed transition cannot be applied."
                )
        if error == "lease_release_failed":
            if release is not None and release.status == "released":
                raise BoundCellRunEnvelopeContractError(
                    "lineage_invalid", "A successful release conflicts with its error."
                )
        elif release is not None and not (
            release.status in {"released", "released_cleanup_deferred"}
            or (
                release.status == "unknown_blocking"
                and release.delivery_status == "attempted_unknown"
            )
        ):
            raise BoundCellRunEnvelopeContractError(
                "lineage_invalid",
                "A non-successful release requires lease_release_failed.",
            )

    @staticmethod
    def _validate_claim_admission(
        claim: CooperativeResourceClaim,
        admission: CooperativeResourceLeaseAdmissionReceipt,
    ) -> None:
        comparisons = {
            "claim digest": (admission.claim_sha256, claim.digest),
            "resource snapshot digest": (
                admission.resource_snapshot_sha256,
                claim.resource_snapshot_sha256,
            ),
            "system claim": (
                admission.requested_system_claim_bytes,
                claim.system_claim_bytes,
            ),
            "accelerator claim": (
                admission.requested_accelerator_claim_bytes,
                claim.accelerator_claim_bytes,
            ),
            "safety reserve": (
                admission.safety_reserve_bytes,
                claim.safety_reserve_bytes,
            ),
        }
        for label, (actual, expected) in comparisons.items():
            if actual != expected:
                raise BoundCellRunEnvelopeContractError(
                    "evidence_mismatch", f"Admission {label} differs from the claim."
                )

    @staticmethod
    def _validate_transition(
        admission: CooperativeResourceLeaseAdmissionReceipt,
        claim: CooperativeResourceClaim,
        transition: CooperativeResourceLeaseTransitionReceipt,
    ) -> None:
        comparisons = {
            "policy digest": (transition.policy_sha256, admission.policy_sha256),
            "admission digest": (
                transition.admission_receipt_sha256,
                admission.digest,
            ),
            "claim digest": (transition.claim_sha256, claim.digest),
            "coordination domain": (
                transition.coordination_domain_sha256,
                admission.coordination_domain_sha256,
            ),
            "lease id": (transition.lease_id, admission.lease_id),
        }
        for label, (actual, expected) in comparisons.items():
            if actual != expected:
                raise BoundCellRunEnvelopeContractError(
                    "evidence_mismatch", f"Transition {label} breaks lease lineage."
                )

    @staticmethod
    def _validate_release(
        admission: CooperativeResourceLeaseAdmissionReceipt,
        claim: CooperativeResourceClaim,
        release: CooperativeResourceLeaseReleaseReceipt,
    ) -> None:
        comparisons = {
            "policy digest": (release.policy_sha256, admission.policy_sha256),
            "admission digest": (
                release.admission_receipt_sha256,
                admission.digest,
            ),
            "claim digest": (release.claim_sha256, claim.digest),
            "coordination domain": (
                release.coordination_domain_sha256,
                admission.coordination_domain_sha256,
            ),
            "lease id": (release.lease_id, admission.lease_id),
        }
        for label, (actual, expected) in comparisons.items():
            if actual != expected:
                raise BoundCellRunEnvelopeContractError(
                    "evidence_mismatch", f"Release {label} breaks lease lineage."
                )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "run_receipt": self.run_receipt.payload(),
            "run_receipt_sha256": self.run_receipt_sha256,
            "lease_claim": (
                None if self.lease_claim is None else self.lease_claim.payload()
            ),
            "lease_admission_receipt": (
                None
                if self.lease_admission_receipt is None
                else self.lease_admission_receipt.payload()
            ),
            "lease_transition_receipt": (
                None
                if self.lease_transition_receipt is None
                else self.lease_transition_receipt.payload()
            ),
            "lease_release_receipt": (
                None
                if self.lease_release_receipt is None
                else self.lease_release_receipt.payload()
            ),
            "lease_error_code": self.lease_error_code,
            "cooperative_only": self.cooperative_only,
            "os_memory_reserved": self.os_memory_reserved,
            "runtime_managed": self.runtime_managed,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


def bound_cell_run_envelope_v2_from_payload(raw: object) -> BoundCellRunEnvelopeV2:
    """Parse only the exact v2 schema and all exact nested contract schemas."""

    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise BoundCellRunEnvelopeContractError(
            "contract_invalid", "Bound-cell run envelope must be an object."
        )
    expected = {item.name for item in fields(BoundCellRunEnvelopeV2)}
    if set(raw) != expected:
        raise BoundCellRunEnvelopeContractError(
            "contract_invalid",
            "Bound-cell run envelope fields do not match the v2 schema.",
        )

    values = dict(raw)
    try:
        values["run_receipt"] = bound_cell_run_receipt_from_payload(
            values["run_receipt"]
        )
        if values["lease_claim"] is not None:
            values["lease_claim"] = cooperative_resource_claim_from_payload(
                values["lease_claim"]
            )
        if values["lease_admission_receipt"] is not None:
            values["lease_admission_receipt"] = (
                cooperative_resource_lease_admission_receipt_from_payload(
                    values["lease_admission_receipt"]
                )
            )
        if values["lease_transition_receipt"] is not None:
            values["lease_transition_receipt"] = (
                cooperative_resource_lease_transition_receipt_from_payload(
                    values["lease_transition_receipt"]
                )
            )
        if values["lease_release_receipt"] is not None:
            values["lease_release_receipt"] = (
                cooperative_resource_lease_release_receipt_from_payload(
                    values["lease_release_receipt"]
                )
            )
        return BoundCellRunEnvelopeV2(**values)
    except BoundCellRunEnvelopeContractError:
        raise
    except (BoundCellRunContractError, CooperativeResourceLeaseContractError) as exc:
        raise BoundCellRunEnvelopeContractError(
            "contract_invalid", "Nested envelope evidence is invalid."
        ) from exc
    except (OverflowError, TypeError, ValueError) as exc:
        raise BoundCellRunEnvelopeContractError(
            "contract_invalid", "Bound-cell run envelope is invalid."
        ) from exc


__all__ = [
    "BOUND_CELL_RUN_ENVELOPE_V2_CONTRACT",
    "BOUND_CELL_RUN_ENVELOPE_V2_SCHEMA_VERSION",
    "LEASE_ERROR_CODES",
    "BoundCellRunEnvelopeContractError",
    "BoundCellRunEnvelopeV2",
    "bound_cell_run_envelope_v2_from_payload",
]
