from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    ROUTE_PLANS,
    VerifiedRoutingError,
    reject_unknown,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    sha256_json,
)


_ORDERS = {"AB", "BA"}
_ARMS = {"baseline", "candidate"}
_SLOTS = {"A", "B"}
_ROUTE_RANK = {"local": 0, "local_then_verify": 1, "premium": 2}
_RUN_ID = re.compile(r"^paired-run-[0-9a-f]{64}$")
_OUTCOME_ID = re.compile(r"^outcome-[0-9a-f]{64}$")
_RUN_INSTANCE_NONCE = re.compile(r"^[0-9a-f]{64}$")

_SLOT_FIELDS = {"slot", "arm", "ordinal", "route"}
_ROOT_FIELDS = {
    "schema_version",
    "contract",
    "run_id",
    "plan_sha256",
    "case_sha256",
    "task_fingerprint",
    "normalized_item_sha256",
    "source_snapshot_sha256",
    "bridge_config_sha256",
    "executor_config_sha256",
    "execution_harness_sha256",
    "lifecycle_config_sha256",
    "signals_sha256",
    "runner_sha256",
    "runner_source_sha256",
    "pricing_sha256",
    "run_instance_nonce",
    "order",
    "baseline_route",
    "candidate_route",
    "slots",
}
_CLAIM_FIELDS = {
    "schema_version",
    "contract",
    "claim_sha256",
    "run_id",
    "slot",
    "arm",
    "ordinal",
    "route",
}
_BINDING_FIELDS = {
    "schema_version",
    "contract",
    "binding_sha256",
    "run_id",
    "plan_sha256",
    "case_sha256",
    "task_fingerprint",
    "normalized_item_sha256",
    "source_snapshot_sha256",
    "bridge_config_sha256",
    "executor_config_sha256",
    "execution_harness_sha256",
    "lifecycle_config_sha256",
    "signals_sha256",
    "runner_sha256",
    "runner_source_sha256",
    "pricing_sha256",
    "run_instance_nonce",
    "order",
    "baseline_route",
    "candidate_route",
    "slot",
    "arm",
    "ordinal",
    "route",
    "claim_sha256",
    "previous_record_id",
}
_CHECKPOINT_FIELDS = {
    "schema_version",
    "contract",
    "checkpoint_sha256",
    "binding",
    "outcome_record_id",
    "route_receipt_id",
    "route_receipt_sha256",
    "evidence_sha256",
}


@dataclass(frozen=True)
class PairedRunSlot:
    """One immutable execution position in an AB/BA paired run."""

    slot: str
    arm: str
    ordinal: int
    route: str

    def __post_init__(self) -> None:
        if self.slot not in _SLOTS:
            raise VerifiedRoutingError("Paired run slot must be A or B.")
        if self.arm not in _ARMS:
            raise VerifiedRoutingError(
                "Paired run arm must be baseline or candidate."
            )
        ordinal = require_non_negative_int(self.ordinal, "slot ordinal")
        if ordinal not in (0, 1):
            raise VerifiedRoutingError("Paired run slot ordinal must be 0 or 1.")
        object.__setattr__(self, "ordinal", ordinal)
        if self.route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Paired run slot route is unsupported.")
        expected_arm = "baseline" if self.slot == "A" else "candidate"
        if self.arm != expected_arm:
            raise VerifiedRoutingError("Paired run slot and arm disagree.")

    def payload(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "arm": self.arm,
            "ordinal": self.ordinal,
            "route": self.route,
        }

    @classmethod
    def from_payload(cls, value: object) -> PairedRunSlot:
        raw = _mapping(value, "paired run slot")
        _require_exact_fields(raw, _SLOT_FIELDS, "paired run slot")
        return cls(
            slot=str(raw["slot"]),
            arm=str(raw["arm"]),
            ordinal=raw["ordinal"],  # type: ignore[arg-type]
            route=str(raw["route"]),
        )


@dataclass(frozen=True)
class PairedRunRoot:
    """Content-addressed root that freezes every shared paired-run input."""

    run_id: str
    plan_sha256: str
    case_sha256: str
    task_fingerprint: str
    normalized_item_sha256: str
    source_snapshot_sha256: str
    bridge_config_sha256: str
    executor_config_sha256: str
    execution_harness_sha256: str
    lifecycle_config_sha256: str
    signals_sha256: str
    runner_sha256: str
    runner_source_sha256: str
    pricing_sha256: str
    run_instance_nonce: str
    order: str
    baseline_route: str
    candidate_route: str
    slots: tuple[PairedRunSlot, ...]
    schema_version: str = CONTRACT_VERSION
    contract: str = "VerifiedPairedRunRoot"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported paired run schema_version.")
        if self.contract != "VerifiedPairedRunRoot":
            raise VerifiedRoutingError("Paired run root contract is unsupported.")
        for field in (
            "plan_sha256",
            "case_sha256",
            "task_fingerprint",
            "normalized_item_sha256",
            "source_snapshot_sha256",
            "bridge_config_sha256",
            "executor_config_sha256",
            "execution_harness_sha256",
            "lifecycle_config_sha256",
            "signals_sha256",
            "runner_sha256",
            "runner_source_sha256",
            "pricing_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field)
            )
        if _RUN_INSTANCE_NONCE.fullmatch(self.run_instance_nonce) is None:
            raise VerifiedRoutingError("Paired run instance nonce is invalid.")
        if self.order not in _ORDERS:
            raise VerifiedRoutingError("Paired run order must be AB or BA.")
        if self.baseline_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Paired baseline route is unsupported.")
        if self.candidate_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Paired candidate route is unsupported.")
        if (
            _ROUTE_RANK[self.candidate_route]
            >= _ROUTE_RANK[self.baseline_route]
        ):
            raise VerifiedRoutingError(
                "Paired candidate route must use less premium execution."
            )
        slots = tuple(self.slots)
        expected = _slots_for(
            self.order,
            baseline_route=self.baseline_route,
            candidate_route=self.candidate_route,
        )
        if slots != expected:
            raise VerifiedRoutingError(
                "Paired run slots do not match the declared order and routes."
            )
        object.__setattr__(self, "slots", slots)
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise VerifiedRoutingError("Paired run_id is invalid.")
        expected_id = f"paired-run-{sha256_json(self.unsigned_payload())}"
        if self.run_id != expected_id:
            raise VerifiedRoutingError("Paired run_id digest is invalid.")

    @classmethod
    def build(
        cls,
        *,
        plan_sha256: str,
        case_sha256: str,
        task_fingerprint: str,
        normalized_item_sha256: str,
        source_snapshot_sha256: str,
        bridge_config_sha256: str,
        executor_config_sha256: str,
        execution_harness_sha256: str,
        lifecycle_config_sha256: str,
        signals_sha256: str,
        runner_sha256: str,
        runner_source_sha256: str,
        pricing_sha256: str,
        run_instance_nonce: str,
        order: str,
        baseline_route: str,
        candidate_route: str,
    ) -> PairedRunRoot:
        """Create the canonical root and derive its run id from its content."""

        slots = _slots_for(
            order,
            baseline_route=baseline_route,
            candidate_route=candidate_route,
        )
        unsigned = {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedPairedRunRoot",
            "plan_sha256": plan_sha256,
            "case_sha256": case_sha256,
            "task_fingerprint": task_fingerprint,
            "normalized_item_sha256": normalized_item_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "bridge_config_sha256": bridge_config_sha256,
            "executor_config_sha256": executor_config_sha256,
            "execution_harness_sha256": execution_harness_sha256,
            "lifecycle_config_sha256": lifecycle_config_sha256,
            "signals_sha256": signals_sha256,
            "runner_sha256": runner_sha256,
            "runner_source_sha256": runner_source_sha256,
            "pricing_sha256": pricing_sha256,
            "run_instance_nonce": run_instance_nonce,
            "order": order,
            "baseline_route": baseline_route,
            "candidate_route": candidate_route,
            "slots": [slot.payload() for slot in slots],
        }
        return cls(
            run_id=f"paired-run-{sha256_json(unsigned)}",
            plan_sha256=plan_sha256,
            case_sha256=case_sha256,
            task_fingerprint=task_fingerprint,
            normalized_item_sha256=normalized_item_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            bridge_config_sha256=bridge_config_sha256,
            executor_config_sha256=executor_config_sha256,
            execution_harness_sha256=execution_harness_sha256,
            lifecycle_config_sha256=lifecycle_config_sha256,
            signals_sha256=signals_sha256,
            runner_sha256=runner_sha256,
            runner_source_sha256=runner_source_sha256,
            pricing_sha256=pricing_sha256,
            run_instance_nonce=run_instance_nonce,
            order=order,
            baseline_route=baseline_route,
            candidate_route=candidate_route,
            slots=slots,
        )

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "plan_sha256": self.plan_sha256,
            "case_sha256": self.case_sha256,
            "task_fingerprint": self.task_fingerprint,
            "normalized_item_sha256": self.normalized_item_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "bridge_config_sha256": self.bridge_config_sha256,
            "executor_config_sha256": self.executor_config_sha256,
            "execution_harness_sha256": self.execution_harness_sha256,
            "lifecycle_config_sha256": self.lifecycle_config_sha256,
            "signals_sha256": self.signals_sha256,
            "runner_sha256": self.runner_sha256,
            "runner_source_sha256": self.runner_source_sha256,
            "pricing_sha256": self.pricing_sha256,
            "run_instance_nonce": self.run_instance_nonce,
            "order": self.order,
            "baseline_route": self.baseline_route,
            "candidate_route": self.candidate_route,
            "slots": [slot.payload() for slot in self.slots],
        }

    def payload(self) -> dict[str, object]:
        payload = self.unsigned_payload()
        payload["run_id"] = self.run_id
        return payload

    @classmethod
    def from_payload(cls, value: object) -> PairedRunRoot:
        raw = _mapping(value, "paired run root")
        _require_exact_fields(raw, _ROOT_FIELDS, "paired run root")
        slots_raw = raw["slots"]
        if not isinstance(slots_raw, list):
            raise VerifiedRoutingError("Paired run root slots must be a list.")
        return cls(
            schema_version=str(raw["schema_version"]),
            contract=str(raw["contract"]),
            run_id=str(raw["run_id"]),
            plan_sha256=str(raw["plan_sha256"]),
            case_sha256=str(raw["case_sha256"]),
            task_fingerprint=str(raw["task_fingerprint"]),
            normalized_item_sha256=str(raw["normalized_item_sha256"]),
            source_snapshot_sha256=str(raw["source_snapshot_sha256"]),
            bridge_config_sha256=str(raw["bridge_config_sha256"]),
            executor_config_sha256=str(raw["executor_config_sha256"]),
            execution_harness_sha256=str(raw["execution_harness_sha256"]),
            lifecycle_config_sha256=str(raw["lifecycle_config_sha256"]),
            signals_sha256=str(raw["signals_sha256"]),
            runner_sha256=str(raw["runner_sha256"]),
            runner_source_sha256=str(raw["runner_source_sha256"]),
            pricing_sha256=str(raw["pricing_sha256"]),
            run_instance_nonce=str(raw["run_instance_nonce"]),
            order=str(raw["order"]),
            baseline_route=str(raw["baseline_route"]),
            candidate_route=str(raw["candidate_route"]),
            slots=tuple(PairedRunSlot.from_payload(item) for item in slots_raw),
        )


@dataclass(frozen=True)
class PairedRunClaim:
    """Content-addressed, pre-invocation ownership claim for one slot."""

    claim_sha256: str
    run_id: str
    slot: str
    arm: str
    ordinal: int
    route: str
    schema_version: str = CONTRACT_VERSION
    contract: str = "VerifiedPairedRunClaim"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported paired claim schema_version.")
        if self.contract != "VerifiedPairedRunClaim":
            raise VerifiedRoutingError("Paired claim contract is unsupported.")
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise VerifiedRoutingError("Paired claim run_id is invalid.")
        slot = PairedRunSlot(self.slot, self.arm, self.ordinal, self.route)
        object.__setattr__(self, "ordinal", slot.ordinal)
        require_sha256(self.claim_sha256, "claim_sha256")
        if self.claim_sha256 != sha256_json(self.unsigned_payload()):
            raise VerifiedRoutingError("Paired claim digest is invalid.")

    @classmethod
    def build(cls, root: PairedRunRoot, slot: PairedRunSlot) -> PairedRunClaim:
        if slot not in root.slots:
            raise VerifiedRoutingError("Paired claim slot is not in the run root.")
        unsigned = {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedPairedRunClaim",
            "run_id": root.run_id,
            **slot.payload(),
        }
        return cls(
            claim_sha256=sha256_json(unsigned),
            run_id=root.run_id,
            slot=slot.slot,
            arm=slot.arm,
            ordinal=slot.ordinal,
            route=slot.route,
        )

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "run_id": self.run_id,
            "slot": self.slot,
            "arm": self.arm,
            "ordinal": self.ordinal,
            "route": self.route,
        }

    def payload(self) -> dict[str, object]:
        payload = self.unsigned_payload()
        payload["claim_sha256"] = self.claim_sha256
        return payload

    @classmethod
    def from_payload(cls, value: object) -> PairedRunClaim:
        raw = _mapping(value, "paired run claim")
        _require_exact_fields(raw, _CLAIM_FIELDS, "paired run claim")
        return cls(
            schema_version=str(raw["schema_version"]),
            contract=str(raw["contract"]),
            claim_sha256=str(raw["claim_sha256"]),
            run_id=str(raw["run_id"]),
            slot=str(raw["slot"]),
            arm=str(raw["arm"]),
            ordinal=raw["ordinal"],  # type: ignore[arg-type]
            route=str(raw["route"]),
        )


@dataclass(frozen=True)
class PairedOutcomeBinding:
    """Immutable lineage embedded in the outcome produced for one arm."""

    binding_sha256: str
    run_id: str
    plan_sha256: str
    case_sha256: str
    task_fingerprint: str
    normalized_item_sha256: str
    source_snapshot_sha256: str
    bridge_config_sha256: str
    executor_config_sha256: str
    execution_harness_sha256: str
    lifecycle_config_sha256: str
    signals_sha256: str
    runner_sha256: str
    runner_source_sha256: str
    pricing_sha256: str
    run_instance_nonce: str
    order: str
    baseline_route: str
    candidate_route: str
    slot: str
    arm: str
    ordinal: int
    route: str
    claim_sha256: str
    previous_record_id: str | None
    schema_version: str = CONTRACT_VERSION
    contract: str = "VerifiedPairedOutcomeBinding"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported paired binding schema_version.")
        if self.contract != "VerifiedPairedOutcomeBinding":
            raise VerifiedRoutingError("Paired outcome binding is unsupported.")
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise VerifiedRoutingError("Paired binding run_id is invalid.")
        for field in (
            "plan_sha256",
            "case_sha256",
            "task_fingerprint",
            "normalized_item_sha256",
            "source_snapshot_sha256",
            "bridge_config_sha256",
            "executor_config_sha256",
            "execution_harness_sha256",
            "lifecycle_config_sha256",
            "signals_sha256",
            "runner_sha256",
            "runner_source_sha256",
            "pricing_sha256",
            "claim_sha256",
            "binding_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field)
            )
        if self.order not in _ORDERS:
            raise VerifiedRoutingError("Paired binding order must be AB or BA.")
        if self.baseline_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Paired binding baseline route is unsupported.")
        if self.candidate_route not in ROUTE_PLANS:
            raise VerifiedRoutingError("Paired binding candidate route is unsupported.")
        if (
            _ROUTE_RANK[self.candidate_route]
            >= _ROUTE_RANK[self.baseline_route]
        ):
            raise VerifiedRoutingError(
                "Paired binding candidate route must use less premium execution."
            )
        PairedRunRoot(
            run_id=self.run_id,
            plan_sha256=self.plan_sha256,
            case_sha256=self.case_sha256,
            task_fingerprint=self.task_fingerprint,
            normalized_item_sha256=self.normalized_item_sha256,
            source_snapshot_sha256=self.source_snapshot_sha256,
            bridge_config_sha256=self.bridge_config_sha256,
            executor_config_sha256=self.executor_config_sha256,
            execution_harness_sha256=self.execution_harness_sha256,
            lifecycle_config_sha256=self.lifecycle_config_sha256,
            signals_sha256=self.signals_sha256,
            runner_sha256=self.runner_sha256,
            runner_source_sha256=self.runner_source_sha256,
            pricing_sha256=self.pricing_sha256,
            run_instance_nonce=self.run_instance_nonce,
            order=self.order,
            baseline_route=self.baseline_route,
            candidate_route=self.candidate_route,
            slots=_slots_for(
                self.order,
                baseline_route=self.baseline_route,
                candidate_route=self.candidate_route,
            ),
        )
        slot = PairedRunSlot(self.slot, self.arm, self.ordinal, self.route)
        expected_slots = _slots_for(
            self.order,
            baseline_route=self.baseline_route,
            candidate_route=self.candidate_route,
        )
        if slot not in expected_slots:
            raise VerifiedRoutingError(
                "Paired binding slot does not match its order and routes."
            )
        object.__setattr__(self, "ordinal", slot.ordinal)
        if slot.ordinal == 0:
            if self.previous_record_id is not None:
                raise VerifiedRoutingError(
                    "The first paired arm cannot bind a previous outcome."
                )
        elif (
            not isinstance(self.previous_record_id, str)
            or _OUTCOME_ID.fullmatch(self.previous_record_id) is None
        ):
            raise VerifiedRoutingError(
                "The second paired arm must bind the first outcome record."
            )
        if self.binding_sha256 != sha256_json(self.unsigned_payload()):
            raise VerifiedRoutingError("Paired outcome binding digest is invalid.")

    @classmethod
    def build(
        cls,
        root: PairedRunRoot,
        claim: PairedRunClaim,
        *,
        previous_record_id: str | None = None,
    ) -> PairedOutcomeBinding:
        slot = _slot_for_claim(root, claim)
        unsigned = {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedPairedOutcomeBinding",
            "run_id": root.run_id,
            "plan_sha256": root.plan_sha256,
            "case_sha256": root.case_sha256,
            "task_fingerprint": root.task_fingerprint,
            "normalized_item_sha256": root.normalized_item_sha256,
            "source_snapshot_sha256": root.source_snapshot_sha256,
            "bridge_config_sha256": root.bridge_config_sha256,
            "executor_config_sha256": root.executor_config_sha256,
            "execution_harness_sha256": root.execution_harness_sha256,
            "lifecycle_config_sha256": root.lifecycle_config_sha256,
            "signals_sha256": root.signals_sha256,
            "runner_sha256": root.runner_sha256,
            "runner_source_sha256": root.runner_source_sha256,
            "pricing_sha256": root.pricing_sha256,
            "run_instance_nonce": root.run_instance_nonce,
            "order": root.order,
            "baseline_route": root.baseline_route,
            "candidate_route": root.candidate_route,
            **slot.payload(),
            "claim_sha256": claim.claim_sha256,
            "previous_record_id": previous_record_id,
        }
        return cls(
            binding_sha256=sha256_json(unsigned),
            **unsigned,  # type: ignore[arg-type]
        )

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "run_id": self.run_id,
            "plan_sha256": self.plan_sha256,
            "case_sha256": self.case_sha256,
            "task_fingerprint": self.task_fingerprint,
            "normalized_item_sha256": self.normalized_item_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "bridge_config_sha256": self.bridge_config_sha256,
            "executor_config_sha256": self.executor_config_sha256,
            "execution_harness_sha256": self.execution_harness_sha256,
            "lifecycle_config_sha256": self.lifecycle_config_sha256,
            "signals_sha256": self.signals_sha256,
            "runner_sha256": self.runner_sha256,
            "runner_source_sha256": self.runner_source_sha256,
            "pricing_sha256": self.pricing_sha256,
            "run_instance_nonce": self.run_instance_nonce,
            "order": self.order,
            "baseline_route": self.baseline_route,
            "candidate_route": self.candidate_route,
            "slot": self.slot,
            "arm": self.arm,
            "ordinal": self.ordinal,
            "route": self.route,
            "claim_sha256": self.claim_sha256,
            "previous_record_id": self.previous_record_id,
        }

    def payload(self) -> dict[str, object]:
        payload = self.unsigned_payload()
        payload["binding_sha256"] = self.binding_sha256
        return payload

    @classmethod
    def from_payload(cls, value: object) -> PairedOutcomeBinding:
        raw = _mapping(value, "paired outcome binding")
        _require_exact_fields(raw, _BINDING_FIELDS, "paired outcome binding")
        return cls(
            schema_version=str(raw["schema_version"]),
            contract=str(raw["contract"]),
            binding_sha256=str(raw["binding_sha256"]),
            run_id=str(raw["run_id"]),
            plan_sha256=str(raw["plan_sha256"]),
            case_sha256=str(raw["case_sha256"]),
            task_fingerprint=str(raw["task_fingerprint"]),
            normalized_item_sha256=str(raw["normalized_item_sha256"]),
            source_snapshot_sha256=str(raw["source_snapshot_sha256"]),
            bridge_config_sha256=str(raw["bridge_config_sha256"]),
            executor_config_sha256=str(raw["executor_config_sha256"]),
            execution_harness_sha256=str(raw["execution_harness_sha256"]),
            lifecycle_config_sha256=str(raw["lifecycle_config_sha256"]),
            signals_sha256=str(raw["signals_sha256"]),
            runner_sha256=str(raw["runner_sha256"]),
            runner_source_sha256=str(raw["runner_source_sha256"]),
            pricing_sha256=str(raw["pricing_sha256"]),
            run_instance_nonce=str(raw["run_instance_nonce"]),
            order=str(raw["order"]),
            baseline_route=str(raw["baseline_route"]),
            candidate_route=str(raw["candidate_route"]),
            slot=str(raw["slot"]),
            arm=str(raw["arm"]),
            ordinal=raw["ordinal"],  # type: ignore[arg-type]
            route=str(raw["route"]),
            claim_sha256=str(raw["claim_sha256"]),
            previous_record_id=(
                None
                if raw["previous_record_id"] is None
                else str(raw["previous_record_id"])
            ),
        )


@dataclass(frozen=True)
class PairedRunCheckpoint:
    """Content-addressed durable completion of one claimed execution slot."""

    checkpoint_sha256: str
    binding: PairedOutcomeBinding
    outcome_record_id: str
    route_receipt_id: str
    route_receipt_sha256: str
    evidence_sha256: str
    schema_version: str = CONTRACT_VERSION
    contract: str = "VerifiedPairedRunCheckpoint"

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError(
                "Unsupported paired checkpoint schema_version."
            )
        if self.contract != "VerifiedPairedRunCheckpoint":
            raise VerifiedRoutingError("Paired checkpoint contract is unsupported.")
        if not isinstance(self.binding, PairedOutcomeBinding):
            raise VerifiedRoutingError(
                "Paired checkpoint binding has the wrong type."
            )
        if _OUTCOME_ID.fullmatch(self.outcome_record_id) is None:
            raise VerifiedRoutingError(
                "Paired checkpoint outcome_record_id is invalid."
            )
        object.__setattr__(
            self,
            "route_receipt_id",
            require_safe_id(self.route_receipt_id, "route_receipt_id"),
        )
        for field in (
            "route_receipt_sha256",
            "evidence_sha256",
            "checkpoint_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field)
            )
        if self.checkpoint_sha256 != sha256_json(self.unsigned_payload()):
            raise VerifiedRoutingError("Paired checkpoint digest is invalid.")

    @classmethod
    def build(
        cls,
        binding: PairedOutcomeBinding,
        *,
        outcome_record_id: str,
        route_receipt_id: str,
        route_receipt_sha256: str,
        evidence_sha256: str,
    ) -> PairedRunCheckpoint:
        unsigned = {
            "schema_version": CONTRACT_VERSION,
            "contract": "VerifiedPairedRunCheckpoint",
            "binding": binding.payload(),
            "outcome_record_id": outcome_record_id,
            "route_receipt_id": route_receipt_id,
            "route_receipt_sha256": route_receipt_sha256,
            "evidence_sha256": evidence_sha256,
        }
        return cls(
            checkpoint_sha256=sha256_json(unsigned),
            binding=binding,
            outcome_record_id=outcome_record_id,
            route_receipt_id=route_receipt_id,
            route_receipt_sha256=route_receipt_sha256,
            evidence_sha256=evidence_sha256,
        )

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "binding": self.binding.payload(),
            "outcome_record_id": self.outcome_record_id,
            "route_receipt_id": self.route_receipt_id,
            "route_receipt_sha256": self.route_receipt_sha256,
            "evidence_sha256": self.evidence_sha256,
        }

    def payload(self) -> dict[str, object]:
        payload = self.unsigned_payload()
        payload["checkpoint_sha256"] = self.checkpoint_sha256
        return payload

    @classmethod
    def from_payload(cls, value: object) -> PairedRunCheckpoint:
        raw = _mapping(value, "paired run checkpoint")
        _require_exact_fields(raw, _CHECKPOINT_FIELDS, "paired run checkpoint")
        return cls(
            schema_version=str(raw["schema_version"]),
            contract=str(raw["contract"]),
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
            binding=PairedOutcomeBinding.from_payload(raw["binding"]),
            outcome_record_id=str(raw["outcome_record_id"]),
            route_receipt_id=str(raw["route_receipt_id"]),
            route_receipt_sha256=str(raw["route_receipt_sha256"]),
            evidence_sha256=str(raw["evidence_sha256"]),
        )


def build_paired_run_root(**kwargs: object) -> PairedRunRoot:
    """Functional constructor retained for callers that avoid class methods."""

    return PairedRunRoot.build(**kwargs)  # type: ignore[arg-type]


def _slots_for(
    order: str,
    *,
    baseline_route: str,
    candidate_route: str,
) -> tuple[PairedRunSlot, PairedRunSlot]:
    if order not in _ORDERS:
        raise VerifiedRoutingError("Paired run order must be AB or BA.")
    if baseline_route not in ROUTE_PLANS:
        raise VerifiedRoutingError("Paired baseline route is unsupported.")
    if candidate_route not in ROUTE_PLANS:
        raise VerifiedRoutingError("Paired candidate route is unsupported.")
    by_slot = {
        "A": ("baseline", baseline_route),
        "B": ("candidate", candidate_route),
    }
    return tuple(
        PairedRunSlot(slot, by_slot[slot][0], ordinal, by_slot[slot][1])
        for ordinal, slot in enumerate(order)
    )  # type: ignore[return-value]


def _slot_for_claim(root: PairedRunRoot, claim: PairedRunClaim) -> PairedRunSlot:
    if claim.run_id != root.run_id:
        raise VerifiedRoutingError("Paired claim belongs to another run.")
    matches = tuple(slot for slot in root.slots if slot.slot == claim.slot)
    if len(matches) != 1 or matches[0].payload() != {
        "slot": claim.slot,
        "arm": claim.arm,
        "ordinal": claim.ordinal,
        "route": claim.route,
    }:
        raise VerifiedRoutingError("Paired claim does not match the run root.")
    return matches[0]


def validate_binding(
    root: PairedRunRoot,
    claim: PairedRunClaim,
    binding: PairedOutcomeBinding,
) -> None:
    """Fail closed unless a binding is exactly derived from root and claim."""

    if binding != PairedOutcomeBinding.build(
        root,
        claim,
        previous_record_id=binding.previous_record_id,
    ):
        raise VerifiedRoutingError("Paired outcome binding is not exact.")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerifiedRoutingError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise VerifiedRoutingError(f"{label} keys must be strings.")
    return dict(value)


def _require_exact_fields(
    raw: dict[str, Any], fields: set[str], label: str
) -> None:
    reject_unknown(raw, fields, label)
    missing = sorted(fields.difference(raw))
    if missing:
        raise VerifiedRoutingError(
            f"Missing {label} fields: {', '.join(missing)}."
        )
