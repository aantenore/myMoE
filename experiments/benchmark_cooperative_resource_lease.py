from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from local_moe.cooperative_resource_lease import (
    CooperativeResourceLeaseAcquisition,
    SQLiteCooperativeResourceLeaseStore,
)
from local_moe.cooperative_resource_lease_contracts import (
    CLAIM_BASIS,
    CooperativeResourceClaim,
    CooperativeResourceLeasePolicy,
)
from local_moe.resource_snapshot import ResourceSnapshot, build_resource_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "outputs" / "cooperative-resource-lease-contract.json"
CAPTURED_AT = "2026-07-22T12:00:00+00:00"
MONOTONIC_TICK = 1_000_000
TASK_MARKER = "PRIVATE-LEASE-BENCHMARK-TASK-MUST-NOT-APPEAR"
ANSWER_MARKER = "PRIVATE-LEASE-BENCHMARK-ANSWER-MUST-NOT-APPEAR"
SENSITIVE_ARTIFACT_KEYS = frozenset(
    {
        "lease_id",
        "lease_token",
        "lease_token_sha256",
        "admission_receipt_sha256",
        "claim_sha256",
        "coordination_domain_sha256",
    }
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _snapshot(
    *,
    fixture: str,
    topology: str,
    available_system_bytes: int,
    available_accelerator_bytes: int | None = None,
) -> ResourceSnapshot:
    if topology == "system":
        system = "Linux"
        machine = "x86_64"
        accelerator_kind = "none"
        accelerator_identity = None
        accelerator_total = None
    elif topology == "unified":
        system = "Darwin"
        machine = "arm64"
        accelerator_kind = "integrated"
        accelerator_identity = _sha("synthetic-integrated-accelerator")
        accelerator_total = None
        available_accelerator_bytes = None
    elif topology == "dedicated":
        system = "Linux"
        machine = "x86_64"
        accelerator_kind = "discrete"
        accelerator_identity = _sha("synthetic-discrete-accelerator")
        accelerator_total = 4_000
    else:
        raise ValueError(f"Unsupported synthetic topology: {topology}")
    return build_resource_snapshot(
        system=system,
        os_release="synthetic",
        machine=machine,
        cpu_count=8,
        cpu_identity_sha256=_sha("synthetic-cpu"),
        memory_topology=topology,
        total_memory_bytes=5_000,
        available_memory_bytes=available_system_bytes,
        effective_memory_limit_bytes=5_000,
        swap_used_bytes=0,
        accelerator_kind=accelerator_kind,
        accelerator_identity_sha256=accelerator_identity,
        accelerator_memory_total_bytes=accelerator_total,
        accelerator_memory_available_bytes=available_accelerator_bytes,
        runtime_environment_sha256=_sha("synthetic-runtime"),
        captured_at=CAPTURED_AT,
        source={"fixture": fixture, "kind": "deterministic_contract_fixture"},
    )


def _claim(
    snapshot: ResourceSnapshot,
    *,
    label: str,
    pool: str,
    system_bytes: int,
    accelerator_bytes: int = 0,
    reserve_bytes: int,
) -> CooperativeResourceClaim:
    return CooperativeResourceClaim(
        preview_sha256=_sha(f"{label}:preview"),
        candidate_sha256=_sha(f"{label}:candidate"),
        passport_sha256=_sha(f"{label}:passport"),
        resource_snapshot_sha256=snapshot.digest,
        resource_class_sha256=snapshot.resource_class_sha256,
        catalog_sha256=_sha(f"{label}:catalog"),
        profile_sha256=_sha(f"{label}:profile"),
        pool=pool,
        system_claim_bytes=system_bytes,
        accelerator_claim_bytes=accelerator_bytes,
        accelerator_identity_sha256=(
            snapshot.accelerator_identity_sha256 if pool == "discrete" else None
        ),
        safety_reserve_bytes=reserve_bytes,
    )


def _store(root: Path, scenario: str) -> SQLiteCooperativeResourceLeaseStore:
    scenario_root = root / scenario
    return SQLiteCooperativeResourceLeaseStore(
        scenario_root / "leases.sqlite3",
        sentinel_root=scenario_root / "sentinels",
        policy=CooperativeResourceLeasePolicy(),
        clock=lambda: CAPTURED_AT,
        monotonic_clock=lambda: MONOTONIC_TICK,
    )


def _handle(
    acquisition: CooperativeResourceLeaseAcquisition[Any],
):
    if acquisition.handle is None:
        raise AssertionError("Synthetic fixture expected an acquired lease.")
    return acquisition.handle


def _simulate_invocation(
    acquisition: CooperativeResourceLeaseAcquisition[Any],
    counters: dict[str, int],
) -> None:
    if acquisition.handle is not None:
        counters["simulated_invocations"] += 1


def _system_capacity_scenario(root: Path) -> dict[str, Any]:
    store = _store(root, "shared-system-capacity")
    snapshot = _snapshot(
        fixture="shared-system-capacity",
        topology="system",
        available_system_bytes=1_500,
    )
    larger_reserve = _claim(
        snapshot,
        label="larger-reserve",
        pool="system",
        system_bytes=500,
        reserve_bytes=300,
    )
    smaller_reserve = _claim(
        snapshot,
        label="smaller-reserve",
        pool="system",
        system_bytes=500,
        reserve_bytes=100,
    )

    first = store.acquire(larger_reserve, snapshot)
    second = store.acquire(smaller_reserve, snapshot)
    third = store.acquire(smaller_reserve, snapshot)
    invocation_counters = {"simulated_invocations": 0}
    _simulate_invocation(third, invocation_counters)

    first_release = store.release(
        _handle(first),
        delivery_status="not_attempted",
    )
    replacement = store.acquire(smaller_reserve, snapshot)
    second_release = store.release(
        _handle(second),
        delivery_status="not_attempted",
    )
    replacement_release = store.release(
        _handle(replacement),
        delivery_status="not_attempted",
    )

    return {
        "admission_statuses": [
            first.receipt.status,
            second.receipt.status,
            third.receipt.status,
        ],
        "third_reason_codes": list(third.receipt.reason_codes),
        "reserve_accounting": {
            "available_system_bytes": third.receipt.system_available_bytes,
            "active_system_claim_bytes": (third.receipt.active_system_claim_bytes),
            "requested_system_claim_bytes": (
                third.receipt.requested_system_claim_bytes
            ),
            "incoming_reserve_bytes": third.receipt.safety_reserve_bytes,
            "applied_pool_reserve_bytes": (third.receipt.applied_system_reserve_bytes),
            "required_system_bytes": (
                third.receipt.active_system_claim_bytes
                + third.receipt.requested_system_claim_bytes
                + third.receipt.applied_system_reserve_bytes
            ),
        },
        "denied_path": invocation_counters,
        "release_status": first_release.status,
        "replacement_status": replacement.receipt.status,
        "cleanup_statuses": [second_release.status, replacement_release.status],
    }


def _unified_pool_scenario(root: Path) -> dict[str, Any]:
    store = _store(root, "unified-pool")
    snapshot = _snapshot(
        fixture="unified-pool",
        topology="unified",
        available_system_bytes=1_300,
    )
    cpu_claim = _claim(
        snapshot,
        label="unified-host-cpu",
        pool="system",
        system_bytes=600,
        reserve_bytes=100,
    )
    integrated_claim = _claim(
        snapshot,
        label="unified-integrated",
        pool="unified",
        system_bytes=600,
        reserve_bytes=100,
    )
    cpu = store.acquire(cpu_claim, snapshot)
    cpu_arm = store.arm_delivery(_handle(cpu))
    integrated = store.acquire(integrated_claim, snapshot)
    integrated_arm = store.arm_delivery(_handle(integrated))
    cpu_release = store.release(
        _handle(cpu),
        delivery_status="response_received",
    )
    integrated_release = store.release(
        _handle(integrated),
        delivery_status="response_received",
    )
    return {
        "topology": snapshot.memory_topology,
        "claim_pools": [cpu_claim.pool, integrated_claim.pool],
        "admission_statuses": [cpu.receipt.status, integrated.receipt.status],
        "armed": [cpu_arm.transition_applied, integrated_arm.transition_applied],
        "second_active_system_claim_bytes": (
            integrated.receipt.active_system_claim_bytes
        ),
        "release_statuses": [cpu_release.status, integrated_release.status],
    }


def _discrete_pool_scenario(root: Path) -> dict[str, Any]:
    store = _store(root, "discrete-pool-fixture")
    snapshot = _snapshot(
        fixture="discrete-pool-fixture",
        topology="dedicated",
        available_system_bytes=2_000,
        available_accelerator_bytes=1_300,
    )
    claim = _claim(
        snapshot,
        label="discrete-cell",
        pool="discrete",
        system_bytes=300,
        accelerator_bytes=600,
        reserve_bytes=100,
    )
    first = store.acquire(claim, snapshot)
    second = store.acquire(claim, snapshot)
    third = store.acquire(claim, snapshot)
    first_release = store.release(
        _handle(first),
        delivery_status="not_attempted",
    )
    second_release = store.release(
        _handle(second),
        delivery_status="not_attempted",
    )
    return {
        "fixture_scope": "contract_only",
        "topology": snapshot.memory_topology,
        "admission_statuses": [
            first.receipt.status,
            second.receipt.status,
            third.receipt.status,
        ],
        "third_reason_codes": list(third.receipt.reason_codes),
        "accelerator_accounting": {
            "available_accelerator_bytes": (third.receipt.accelerator_available_bytes),
            "active_accelerator_claim_bytes": (
                third.receipt.active_accelerator_claim_bytes
            ),
            "requested_accelerator_claim_bytes": (
                third.receipt.requested_accelerator_claim_bytes
            ),
            "applied_accelerator_reserve_bytes": (
                third.receipt.applied_accelerator_reserve_bytes
            ),
        },
        "cleanup_statuses": [first_release.status, second_release.status],
    }


def _sticky_unknown_scenario(root: Path) -> dict[str, Any]:
    store = _store(root, "sticky-unknown")
    snapshot = _snapshot(
        fixture="sticky-unknown",
        topology="system",
        available_system_bytes=2_000,
    )
    claim = _claim(
        snapshot,
        label="ambiguous-delivery",
        pool="system",
        system_bytes=600,
        reserve_bytes=100,
    )
    acquisition = store.acquire(claim, snapshot)
    handle = _handle(acquisition)
    transition = store.arm_delivery(handle)
    invocation_counters = {"simulated_invocations": 0}
    _simulate_invocation(acquisition, invocation_counters)
    ambiguous_release = store.release(
        handle,
        delivery_status="attempted_unknown",
    )
    blocked = store.acquire(claim, snapshot)
    scenario = {
        "initial_admission_status": acquisition.receipt.status,
        "delivery_armed": transition.transition_applied,
        "delivery_counters": invocation_counters,
        "ambiguous_release_status": ambiguous_release.status,
        "ambiguous_reason_codes": list(ambiguous_release.reason_codes),
        "subsequent_admission_status": blocked.receipt.status,
        "subsequent_reason_codes": list(blocked.receipt.reason_codes),
        "subsequent_handle_issued": blocked.handle is not None,
    }
    del blocked, handle, acquisition
    gc.collect()
    return scenario


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, dict):
        return bool(SENSITIVE_ARTIFACT_KEYS.intersection(value)) or any(
            _contains_sensitive_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def render_report(report: dict[str, Any]) -> str:
    return (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def run_benchmark() -> dict[str, Any]:
    """Exercise cooperative lease accounting without a model or network."""

    with tempfile.TemporaryDirectory(prefix="mymoe-resource-lease-") as temp:
        root = Path(temp)
        scenarios = {
            "shared_system_capacity": _system_capacity_scenario(root),
            "unified_pool": _unified_pool_scenario(root),
            "discrete_pool_contract_fixture": _discrete_pool_scenario(root),
            "armed_attempted_unknown": _sticky_unknown_scenario(root),
        }

    serialized_scenarios = json.dumps(
        scenarios,
        allow_nan=False,
        sort_keys=True,
    )
    system = scenarios["shared_system_capacity"]
    unified = scenarios["unified_pool"]
    discrete = scenarios["discrete_pool_contract_fixture"]
    unknown = scenarios["armed_attempted_unknown"]
    criteria = {
        "two_system_claims_fit_before_the_third_is_denied": (
            system["admission_statuses"] == ["acquired", "acquired", "denied"]
            and system["third_reason_codes"] == ["system_capacity_insufficient"]
        ),
        "largest_reserve_is_applied_once_to_the_system_pool": (
            system["reserve_accounting"]
            == {
                "available_system_bytes": 1_500,
                "active_system_claim_bytes": 1_000,
                "requested_system_claim_bytes": 500,
                "incoming_reserve_bytes": 100,
                "applied_pool_reserve_bytes": 300,
                "required_system_bytes": 1_800,
            }
        ),
        "denial_performs_zero_simulated_invocations": (
            system["denied_path"]["simulated_invocations"] == 0
        ),
        "release_allows_a_new_acquisition": (
            system["release_status"] == "released"
            and system["replacement_status"] == "acquired"
            and system["cleanup_statuses"] == ["released", "released"]
        ),
        "cpu_and_integrated_claims_share_unified_system_capacity": (
            unified["claim_pools"] == ["system", "unified"]
            and unified["admission_statuses"] == ["acquired", "acquired"]
            and unified["armed"] == [True, True]
            and unified["second_active_system_claim_bytes"] == 600
            and unified["release_statuses"] == ["released", "released"]
        ),
        "discrete_fixture_accounts_for_host_and_accelerator_pools": (
            discrete["admission_statuses"] == ["acquired", "acquired", "denied"]
            and discrete["third_reason_codes"] == ["accelerator_capacity_insufficient"]
            and discrete["accelerator_accounting"]
            == {
                "available_accelerator_bytes": 1_300,
                "active_accelerator_claim_bytes": 1_200,
                "requested_accelerator_claim_bytes": 600,
                "applied_accelerator_reserve_bytes": 100,
            }
            and discrete["cleanup_statuses"] == ["released", "released"]
        ),
        "attempted_unknown_remains_sticky_and_blocks_new_work": (
            unknown["initial_admission_status"] == "acquired"
            and unknown["delivery_armed"]
            and unknown["delivery_counters"]["simulated_invocations"] == 1
            and unknown["ambiguous_release_status"] == "unknown_blocking"
            and unknown["ambiguous_reason_codes"] == ["delivery_outcome_unknown"]
            and unknown["subsequent_admission_status"] == "unknown_blocking"
            and unknown["subsequent_reason_codes"] == ["lease_owner_unknown"]
            and not unknown["subsequent_handle_issued"]
        ),
        "artifact_omits_private_content_and_lease_capabilities": (
            TASK_MARKER not in serialized_scenarios
            and ANSWER_MARKER not in serialized_scenarios
            and not _contains_sensitive_key(scenarios)
        ),
    }
    pass_count = sum(criteria.values())
    return {
        "schema_version": "1.0",
        "contract": "cooperative_resource_lease",
        "benchmark": "cooperative_resource_lease_contract",
        "claim_basis": CLAIM_BASIS,
        "scenarios": scenarios,
        "criteria": criteria,
        "pass_count": pass_count,
        "check_count": len(criteria),
        "contract_checks_passed": pass_count == len(criteria),
        "limits": [
            "cooperative_participants_only",
            "does_not_reserve_ram_or_vram_at_operating_system_level",
            "does_not_start_load_unload_stop_or_evict_models",
            "does_not_coordinate_processes_that_ignore_the_ledger",
            "does_not_open_network_connections_or_invoke_models",
            "does_not_measure_latency_throughput_or_model_quality",
            "synthetic_capacity_contract_fixtures_only",
            "discrete_accelerator_path_is_contract_fixture_only",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Cooperative Resource Lease benchmark."
    )
    parser.add_argument(
        "--out",
        help="Optional report path to write or verify.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the regenerated report byte-for-byte with the artifact.",
    )
    args = parser.parse_args()

    report = run_benchmark()
    rendered = render_report(report)
    destination = Path(args.out) if args.out else DEFAULT_ARTIFACT
    if args.check:
        try:
            current = destination.read_bytes()
        except OSError as exc:
            raise SystemExit(f"Unable to read benchmark artifact: {exc}") from exc
        if current != rendered.encode("utf-8"):
            raise SystemExit(
                "Cooperative Resource Lease benchmark artifact is out of date."
            )
    elif args.out:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rendered.encode("utf-8"))
    else:
        print(rendered, end="")

    if not report["contract_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
