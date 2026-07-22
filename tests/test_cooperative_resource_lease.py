from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from local_moe.adaptive_advisor_service import AdaptiveAdvisorReceipt
from local_moe.adaptive_execution_gate import AdaptiveCellExecutionPreviewReceipt
from local_moe.adaptive_selector import advise_cell
from local_moe.cooperative_resource_lease import (
    CooperativeResourceLeaseEvaluation,
    CooperativeResourceLeaseStoreError,
    SQLiteCooperativeResourceLeaseStore,
    cooperative_resource_claim_from_preview,
)
from local_moe.cooperative_resource_lease_contracts import (
    CooperativeResourceClaim,
    CooperativeResourceLeasePolicy,
)
from local_moe.resource_snapshot import ResourceSnapshot
from tests.test_adaptive_selector import (
    catalog as advisor_catalog,
    cell as advisor_cell,
    request as advisor_request,
    snapshot as advisor_snapshot,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
NOW = "2026-07-22T12:00:00+00:00"


def _snapshot(
    *,
    available: int | None = 1_300,
    topology: str = "system",
    accelerator_available: int | None = None,
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
        accelerator_identity = SHA_C
        accelerator_total = None
        accelerator_available = None
    else:
        system = "Linux"
        machine = "x86_64"
        accelerator_kind = "discrete"
        accelerator_identity = SHA_C
        accelerator_total = 4_000
    return ResourceSnapshot(
        system=system,
        os_release="test",
        machine=machine,
        cpu_count=8,
        cpu_identity_sha256=SHA_A,
        memory_topology=topology,
        total_memory_bytes=5_000,
        available_memory_bytes=available,
        effective_memory_limit_bytes=5_000,
        swap_used_bytes=0,
        accelerator_kind=accelerator_kind,
        accelerator_identity_sha256=accelerator_identity,
        accelerator_memory_total_bytes=accelerator_total,
        accelerator_memory_available_bytes=accelerator_available,
        runtime_environment_sha256=SHA_B,
        captured_at=NOW,
        source_sha256=SHA_D,
    )


def _claim(
    snapshot: ResourceSnapshot,
    *,
    pool: str = "system",
    system_bytes: int = 600,
    accelerator_bytes: int = 0,
    reserve: int = 100,
) -> CooperativeResourceClaim:
    return CooperativeResourceClaim(
        preview_sha256=SHA_A,
        candidate_sha256=SHA_B,
        passport_sha256=SHA_C,
        resource_snapshot_sha256=snapshot.digest,
        resource_class_sha256=snapshot.resource_class_sha256,
        catalog_sha256=SHA_A,
        profile_sha256=SHA_B,
        pool=pool,
        system_claim_bytes=system_bytes,
        accelerator_claim_bytes=accelerator_bytes,
        accelerator_identity_sha256=(SHA_C if pool == "discrete" else None),
        safety_reserve_bytes=reserve,
    )


def _worker_contend(database: str, barrier, results, release_event) -> None:
    store = SQLiteCooperativeResourceLeaseStore(database)
    snapshot = _snapshot(available=1_000)
    claim = _claim(snapshot, system_bytes=600, reserve=100)
    barrier.wait(timeout=10)
    acquisition = store.acquire(claim, snapshot)
    results.put(acquisition.receipt.status)
    if acquisition.handle is not None:
        release_event.wait(timeout=10)
        store.release(acquisition.handle, delivery_status="not_attempted")


def _worker_contend_to_delivery(database: str, barrier, results, release_event) -> None:
    store = SQLiteCooperativeResourceLeaseStore(database)
    snapshot = _snapshot(available=700)
    claim = _claim(snapshot, system_bytes=600, reserve=100)
    barrier.wait(timeout=10)
    acquisition = store.acquire(claim, snapshot)
    results.put(("admission", acquisition.receipt.status))
    if acquisition.handle is None:
        return
    transition = store.arm_delivery(acquisition.handle)
    results.put(("post_authorized", transition.transition_applied))
    if not transition.transition_applied:
        store.release(acquisition.handle, delivery_status="not_attempted")
        return
    release_event.wait(timeout=10)
    store.release(acquisition.handle, delivery_status="response_received")


def _worker_crash(database: str, ready, arm_delivery: bool) -> None:
    store = SQLiteCooperativeResourceLeaseStore(database)
    snapshot = _snapshot(available=1_000)
    acquisition = store.acquire(_claim(snapshot), snapshot)
    if acquisition.handle is None:
        os._exit(4)
    if arm_delivery:
        transition = store.arm_delivery(acquisition.handle)
        if not transition.transition_applied:
            os._exit(5)
    ready.set()
    os._exit(0)


class CooperativeResourceLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "leases.sqlite3"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def store(
        self, *, busy_timeout_ms: int = 5_000
    ) -> SQLiteCooperativeResourceLeaseStore:
        return SQLiteCooperativeResourceLeaseStore(
            self.database,
            policy=CooperativeResourceLeasePolicy(busy_timeout_ms=busy_timeout_ms),
        )

    def _previewed_advisor_evidence(self, *, multiple_cells: bool = False):
        snapshot = advisor_snapshot()
        request = advisor_request()
        passports = [advisor_cell("cell-a", snapshot, request.demand)]
        if multiple_cells:
            passports.append(advisor_cell("cell-b", snapshot, request.demand))
        catalog = advisor_catalog(passports)
        advice = advise_cell(catalog, snapshot, request)
        passport = next(
            item for item in passports if item.cell_id == advice.selected_cell_id
        )
        fresh_advisor = AdaptiveAdvisorReceipt(
            request=request,
            advice=advice,
            task_chars=4,
            display_state="recommended_now",
        )
        preview = AdaptiveCellExecutionPreviewReceipt(
            source_advisor_receipt_sha256=fresh_advisor.digest,
            source_request_sha256=request.digest,
            fresh_advisor_receipt_sha256=fresh_advisor.digest,
            fresh_request_sha256=request.digest,
            policy_sha256=SHA_A,
            evaluated_at=request.evaluated_at,
            source_selected_cell_id=advice.selected_cell_id,
            fresh_selected_cell_id=advice.selected_cell_id,
            source_passport_sha256=passport.digest,
            fresh_passport_sha256=passport.digest,
            fresh_resource_snapshot_sha256=snapshot.digest,
            status="admission_passed",
            reason_codes=(),
            task_chars=4,
        )
        return preview, fresh_advisor, passport, catalog, snapshot

    def test_reserve_is_applied_once_to_the_shared_system_pool(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=1_300)
        claim = _claim(snapshot, system_bytes=600, reserve=100)

        first = store.acquire(claim, snapshot)
        second = store.acquire(claim, snapshot)
        third = store.acquire(claim, snapshot)

        self.assertEqual(first.receipt.status, "acquired")
        self.assertEqual(second.receipt.status, "acquired")
        self.assertEqual(second.receipt.active_system_claim_bytes, 600)
        self.assertEqual(third.receipt.status, "denied")
        self.assertIn("system_capacity_insufficient", third.receipt.reason_codes)
        self.assertIsNotNone(first.handle)
        self.assertIsNotNone(second.handle)
        store.release(first.handle, delivery_status="not_attempted")
        store.release(second.handle, delivery_status="not_attempted")

    def test_largest_active_reserve_remains_applied_to_the_pool(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=1_100)
        first = store.acquire(_claim(snapshot, system_bytes=600, reserve=300), snapshot)
        second = store.acquire(
            _claim(snapshot, system_bytes=300, reserve=100), snapshot
        )

        self.assertEqual(first.receipt.status, "acquired")
        self.assertEqual(second.receipt.status, "denied")
        self.assertEqual(second.receipt.applied_system_reserve_bytes, 300)
        self.assertEqual(second.receipt.applied_accelerator_reserve_bytes, 0)
        self.assertIn("system_capacity_insufficient", second.receipt.reason_codes)
        store.release(first.handle, delivery_status="not_attempted")

    def test_unified_and_system_claims_share_system_capacity(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=1_300, topology="unified")
        system_claim = _claim(snapshot, pool="system")
        unified_claim = _claim(snapshot, pool="unified")

        first = store.acquire(system_claim, snapshot)
        second = store.acquire(unified_claim, snapshot)

        self.assertEqual(first.receipt.status, "acquired")
        self.assertEqual(second.receipt.status, "acquired")
        store.release(first.handle, delivery_status="not_attempted")
        store.release(second.handle, delivery_status="not_attempted")

    def test_discrete_claim_checks_system_and_accelerator_pools(self) -> None:
        store = self.store()
        snapshot = _snapshot(
            available=2_000, topology="dedicated", accelerator_available=1_300
        )
        claim = _claim(
            snapshot,
            pool="discrete",
            system_bytes=300,
            accelerator_bytes=600,
            reserve=100,
        )

        first = store.acquire(claim, snapshot)
        second = store.acquire(claim, snapshot)
        third = store.acquire(claim, snapshot)

        self.assertEqual(first.receipt.status, "acquired")
        self.assertEqual(second.receipt.status, "acquired")
        self.assertEqual(third.receipt.status, "denied")
        self.assertIn("accelerator_capacity_insufficient", third.receipt.reason_codes)
        store.release(first.handle, delivery_status="not_attempted")
        store.release(second.handle, delivery_status="not_attempted")

    def test_unknown_capacity_blocks_without_creating_a_lease(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=None)
        result = store.acquire(_claim(snapshot), snapshot)

        self.assertEqual(result.receipt.status, "unknown_blocking")
        self.assertIn("resource_capacity_unknown", result.receipt.reason_codes)
        self.assertIsNone(result.handle)

    def test_evaluator_runs_inside_the_atomic_acquisition(self) -> None:
        store = self.store()
        snapshot = _snapshot()
        claim = _claim(snapshot)
        observed_transaction = False

        def evaluate():
            nonlocal observed_transaction
            probe = sqlite3.connect(self.database, timeout=0, isolation_level=None)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    probe.execute("BEGIN IMMEDIATE")
                observed_transaction = True
            finally:
                probe.close()
            return CooperativeResourceLeaseEvaluation(claim, snapshot, "preview")

        result = store.evaluate_and_acquire(evaluate)
        self.assertTrue(observed_transaction)
        self.assertEqual(result.context, "preview")
        self.assertEqual(result.receipt.status, "acquired")
        store.release(result.handle, delivery_status="not_attempted")

    def test_delivery_arm_is_authenticated_and_release_requires_known_outcome(
        self,
    ) -> None:
        store = self.store()
        snapshot = _snapshot()
        acquisition = store.acquire(_claim(snapshot), snapshot)
        self.assertIsNotNone(acquisition.handle)

        wrong = replace(acquisition.handle, token=b"x" * 32)
        denied = store.arm_delivery(wrong)
        self.assertFalse(denied.transition_applied)
        self.assertIn("lease_token_mismatch", denied.reason_codes)

        transition = store.arm_delivery(acquisition.handle)
        self.assertTrue(transition.transition_applied)
        ambiguous = store.release(
            acquisition.handle, delivery_status="attempted_unknown"
        )
        self.assertEqual(ambiguous.status, "unknown_blocking")
        self.assertEqual(ambiguous.active_leases_after, 1)

        released = store.release(
            acquisition.handle, delivery_status="response_received"
        )
        self.assertEqual(released.status, "unknown_blocking")
        self.assertEqual(released.active_leases_after, 1)

    def test_state_and_delivery_matrix_is_fail_closed(self) -> None:
        snapshot = _snapshot()

        reserved_store = self.store()
        reserved = reserved_store.acquire(_claim(snapshot), snapshot)
        mismatch = reserved_store.release(
            reserved.handle, delivery_status="response_received"
        )
        self.assertEqual(mismatch.status, "denied")
        self.assertIn("lease_state_outcome_mismatch", mismatch.reason_codes)
        self.assertEqual(
            reserved_store.release(
                reserved.handle, delivery_status="not_attempted"
            ).status,
            "released",
        )

        armed_database = Path(self.temporary.name) / "armed.sqlite3"
        armed_store = SQLiteCooperativeResourceLeaseStore(armed_database)
        armed = armed_store.acquire(_claim(snapshot), snapshot)
        self.assertTrue(armed_store.arm_delivery(armed.handle).transition_applied)
        mismatch = armed_store.release(armed.handle, delivery_status="not_attempted")
        self.assertEqual(mismatch.status, "denied")
        self.assertEqual(
            armed_store.release(
                armed.handle, delivery_status="response_received"
            ).status,
            "released",
        )

    def test_two_capacity_fitting_armed_leases_can_coexist(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=1_300)
        claim = _claim(snapshot, system_bytes=600, reserve=100)
        first = store.acquire(claim, snapshot)
        self.assertTrue(store.arm_delivery(first.handle).transition_applied)
        second = store.acquire(claim, snapshot)
        self.assertEqual(second.receipt.status, "acquired")
        self.assertTrue(store.arm_delivery(second.handle).transition_applied)
        third = store.acquire(claim, snapshot)
        self.assertEqual(third.receipt.status, "denied")
        self.assertEqual(
            store.release(first.handle, delivery_status="response_received").status,
            "released",
        )
        self.assertEqual(
            store.release(second.handle, delivery_status="response_received").status,
            "released",
        )

    def test_wrong_token_does_not_release_and_double_release_is_idempotent(
        self,
    ) -> None:
        store = self.store()
        snapshot = _snapshot()
        acquisition = store.acquire(_claim(snapshot), snapshot)
        wrong = replace(acquisition.handle, token=b"z" * 32)

        denied = store.release(wrong, delivery_status="not_attempted")
        self.assertEqual(denied.status, "denied")
        self.assertEqual(denied.active_leases_after, 1)
        released = store.release(acquisition.handle, delivery_status="not_attempted")
        repeated = store.release(acquisition.handle, delivery_status="not_attempted")
        self.assertEqual(released.status, "released")
        self.assertEqual(repeated.status, "already_absent")

    def test_owner_probe_error_transitions_to_permanent_unknown(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=2_000)
        first = store.acquire(_claim(snapshot), snapshot)
        second_store = self.store()
        with patch.object(second_store, "_probe_owner", return_value=("unknown", None)):
            blocked = second_store.acquire(_claim(snapshot), snapshot)
        self.assertEqual(blocked.receipt.status, "unknown_blocking")
        self.assertIn("lease_owner_unknown", blocked.receipt.reason_codes)

        # Even if the sentinel later becomes free, unknown rows are never probed/reaped.
        first.handle._owner_lock.release(force=True)
        still_blocked = second_store.acquire(_claim(snapshot), snapshot)
        self.assertEqual(still_blocked.receipt.status, "unknown_blocking")
        self.assertEqual(still_blocked.receipt.reaped_leases, 0)
        released = store.release(first.handle, delivery_status="not_attempted")
        self.assertEqual(released.status, "unknown_blocking")

    def test_receipts_and_store_never_persist_raw_token(self) -> None:
        store = self.store()
        snapshot = _snapshot()
        acquisition = store.acquire(_claim(snapshot), snapshot)
        token = acquisition.handle.token
        rendered = json.dumps(acquisition.receipt.payload(), sort_keys=True).encode()

        self.assertNotIn(token, rendered)
        self.assertNotIn(token.hex().encode(), rendered)
        persisted = b"".join(
            path.read_bytes()
            for path in self.database.parent.iterdir()
            if path.is_file()
        )
        self.assertNotIn(token, persisted)
        self.assertNotIn(token.hex().encode(), persisted)
        store.release(acquisition.handle, delivery_status="not_attempted")

    def test_busy_corrupt_and_foreign_schema_fail_closed(self) -> None:
        store = self.store(busy_timeout_ms=10)
        snapshot = _snapshot()
        blocker = sqlite3.connect(self.database, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaises(CooperativeResourceLeaseStoreError) as busy:
                store.acquire(_claim(snapshot), snapshot)
            self.assertEqual(busy.exception.code, "lease_store_busy")
        finally:
            blocker.rollback()
            blocker.close()

        corrupt = Path(self.temporary.name) / "corrupt.sqlite3"
        corrupt.write_bytes(b"not sqlite")
        with self.assertRaises(CooperativeResourceLeaseStoreError) as invalid:
            SQLiteCooperativeResourceLeaseStore(corrupt)
        self.assertEqual(invalid.exception.code, "lease_store_corrupt")

        foreign = Path(self.temporary.name) / "foreign.sqlite3"
        connection = sqlite3.connect(foreign)
        connection.execute("CREATE TABLE foreign_table (value TEXT)")
        connection.commit()
        connection.close()
        with self.assertRaises(CooperativeResourceLeaseStoreError) as schema:
            SQLiteCooperativeResourceLeaseStore(foreign)
        self.assertEqual(schema.exception.code, "lease_store_schema_invalid")

    def test_coordination_domain_binds_database_and_sentinel_root(self) -> None:
        self.store()
        different_sentinels = Path(self.temporary.name) / "different-sentinels"

        with self.assertRaises(CooperativeResourceLeaseStoreError) as mismatch:
            SQLiteCooperativeResourceLeaseStore(
                self.database, sentinel_root=different_sentinels
            )
        self.assertEqual(mismatch.exception.code, "lease_store_schema_invalid")

    def test_commit_failure_rolls_back_and_releases_owner_lock(self) -> None:
        store = self.store()
        snapshot = _snapshot()
        lease_id = "f" * 32
        sentinel = store.sentinel_root / f"{lease_id}.lock"

        @contextmanager
        def failing_transaction():
            with store._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                    connection.rollback()
                    raise CooperativeResourceLeaseStoreError(
                        "lease_store_unavailable", "Injected commit failure."
                    )
                except BaseException:
                    connection.rollback()
                    raise

        with (
            patch.object(store, "_transaction", failing_transaction),
            patch(
                "local_moe.cooperative_resource_lease.secrets.token_hex",
                return_value=lease_id,
            ),
            self.assertRaises(CooperativeResourceLeaseStoreError),
        ):
            store.acquire(_claim(snapshot), snapshot)

        self.assertFalse(sentinel.exists())
        connection = sqlite3.connect(self.database)
        try:
            rows = connection.execute("SELECT COUNT(*) FROM active_leases").fetchone()[
                0
            ]
        finally:
            connection.close()
        self.assertEqual(rows, 0)

    def test_missing_row_with_unknown_delivery_quarantines_domain(self) -> None:
        store = self.store()
        snapshot = _snapshot(available=2_000)
        acquisition = store.acquire(_claim(snapshot), snapshot)
        connection = sqlite3.connect(self.database)
        connection.execute(
            "DELETE FROM active_leases WHERE lease_id = ?",
            (acquisition.handle.lease_id,),
        )
        connection.commit()
        connection.close()

        release = store.release(acquisition.handle, delivery_status="attempted_unknown")
        self.assertEqual(release.status, "unknown_blocking")
        blocked = store.acquire(_claim(snapshot), snapshot)
        self.assertEqual(blocked.receipt.status, "unknown_blocking")
        self.assertIn("lease_owner_unknown", blocked.receipt.reason_codes)

    def test_delete_failure_keeps_row_and_unlock_failure_does_not_invent_block(
        self,
    ) -> None:
        store = self.store()
        snapshot = _snapshot(available=2_000)
        acquisition = store.acquire(_claim(snapshot), snapshot)
        connection = sqlite3.connect(self.database)
        connection.execute(
            "CREATE TRIGGER refuse_lease_delete BEFORE DELETE ON active_leases "
            "BEGIN SELECT RAISE(ABORT, 'delete refused'); END"
        )
        connection.commit()
        connection.close()
        with self.assertRaises(CooperativeResourceLeaseStoreError):
            store.release(acquisition.handle, delivery_status="not_attempted")
        connection = sqlite3.connect(self.database)
        remaining = connection.execute("SELECT COUNT(*) FROM active_leases").fetchone()[
            0
        ]
        connection.execute("DROP TRIGGER refuse_lease_delete")
        connection.commit()
        connection.close()
        self.assertEqual(remaining, 1)

        with patch(
            "local_moe.cooperative_resource_lease._release_file_lock",
            side_effect=OSError("cleanup failed"),
        ):
            release = store.release(acquisition.handle, delivery_status="not_attempted")
        self.assertEqual(release.status, "released_cleanup_deferred")
        self.assertNotEqual(release.status, "unknown_blocking")
        next_result = store.acquire(_claim(snapshot), snapshot)
        self.assertEqual(next_result.receipt.status, "acquired")
        store.release(next_result.handle, delivery_status="not_attempted")
        acquisition.handle._owner_lock.release(force=True)

    def test_preview_mapping_uses_only_the_fresh_selected_candidate(self) -> None:
        preview, fresh_advisor, passport, catalog, snapshot = (
            self._previewed_advisor_evidence()
        )
        profile = catalog.profiles[fresh_advisor.request.profile]
        advice = fresh_advisor.advice
        candidate = next(
            item
            for item in advice.candidates
            if item.cell_id == advice.selected_cell_id
        )
        claim = cooperative_resource_claim_from_preview(
            preview=preview,
            fresh_advisor=fresh_advisor,
            passport=passport,
            catalog=catalog,
            snapshot=snapshot,
        )
        self.assertEqual(claim.preview_sha256, preview.digest)
        self.assertEqual(claim.pool, "unified")
        self.assertEqual(
            claim.system_claim_bytes, candidate.effective_peak_unified_memory_bytes
        )
        self.assertEqual(claim.accelerator_claim_bytes, 0)
        self.assertEqual(claim.safety_reserve_bytes, profile.reserve_memory_bytes)
        self.assertEqual(claim.profile_sha256, profile.digest)
        self.assertEqual(claim.catalog_sha256, catalog.digest)
        self.assertEqual(claim.claim_basis, "conservative_peak")

    def test_preview_mapping_rejects_an_unselected_passport(self) -> None:
        preview, fresh_advisor, passport, catalog, snapshot = (
            self._previewed_advisor_evidence(multiple_cells=True)
        )
        other = next(item for item in catalog.cells if item.cell_id != passport.cell_id)

        with self.assertRaises(CooperativeResourceLeaseStoreError) as invalid:
            cooperative_resource_claim_from_preview(
                preview=preview,
                fresh_advisor=fresh_advisor,
                passport=other,
                catalog=catalog,
                snapshot=snapshot,
            )
        self.assertEqual(invalid.exception.code, "lease_claim_invalid")

        mismatched_preview = replace(preview, task_chars=5, digest="")
        with self.assertRaises(CooperativeResourceLeaseStoreError):
            cooperative_resource_claim_from_preview(
                preview=mismatched_preview,
                fresh_advisor=fresh_advisor,
                passport=passport,
                catalog=catalog,
                snapshot=snapshot,
            )


class CooperativeResourceLeaseMultiprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = str(Path(self.temporary.name) / "leases.sqlite3")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_two_processes_capacity_for_one_has_one_winner(self) -> None:
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        results = context.Queue()
        release_event = context.Event()
        workers = [
            context.Process(
                target=_worker_contend,
                args=(self.database, barrier, results, release_event),
            )
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        statuses = sorted(results.get(timeout=15) for _ in workers)
        self.assertEqual(statuses, ["acquired", "denied"])
        release_event.set()
        for worker in workers:
            worker.join(timeout=15)
            self.assertEqual(worker.exitcode, 0)

    def test_capacity_for_one_authorizes_exactly_one_delivery(self) -> None:
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        results = context.Queue()
        release_event = context.Event()
        workers = [
            context.Process(
                target=_worker_contend_to_delivery,
                args=(self.database, barrier, results, release_event),
            )
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        observed = [results.get(timeout=15) for _ in range(3)]
        admissions = sorted(value for kind, value in observed if kind == "admission")
        deliveries = [value for kind, value in observed if kind == "post_authorized"]
        self.assertEqual(admissions, ["acquired", "denied"])
        self.assertEqual(deliveries, [True])
        release_event.set()
        for worker in workers:
            worker.join(timeout=15)
            self.assertEqual(worker.exitcode, 0)

    def test_pre_arm_crash_is_reaped_but_post_arm_crash_becomes_sticky(self) -> None:
        context = multiprocessing.get_context("spawn")
        for arm_delivery in (False, True):
            database = str(
                Path(self.temporary.name) / f"crash-{int(arm_delivery)}.sqlite3"
            )
            ready = context.Event()
            worker = context.Process(
                target=_worker_crash,
                args=(database, ready, arm_delivery),
            )
            worker.start()
            self.assertTrue(ready.wait(timeout=15))
            worker.join(timeout=15)
            self.assertEqual(worker.exitcode, 0)

            store = SQLiteCooperativeResourceLeaseStore(database)
            snapshot = _snapshot(available=1_000)
            result = store.acquire(_claim(snapshot), snapshot)
            if arm_delivery:
                self.assertEqual(result.receipt.status, "unknown_blocking")
                self.assertEqual(result.receipt.reaped_leases, 0)
            else:
                self.assertEqual(result.receipt.status, "acquired")
                self.assertEqual(result.receipt.reaped_leases, 1)
                store.release(result.handle, delivery_status="not_attempted")


if __name__ == "__main__":
    unittest.main()
