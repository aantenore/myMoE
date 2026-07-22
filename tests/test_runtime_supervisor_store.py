from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest

from local_moe.runtime_supervisor_contracts import (
    RuntimeSupervisorLeaseBinding,
    RuntimeSupervisorLeasePolicy,
)
from local_moe.runtime_supervisor_store import (
    RuntimeSupervisorLeaseHandle,
    RuntimeSupervisorLeaseStoreError,
    SQLiteRuntimeSupervisorLeaseStore,
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


class RuntimeSupervisorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "state" / "runtime.sqlite3"
        self.sentinels = self.root / "state" / "owners"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def store(
        self, *, policy: RuntimeSupervisorLeasePolicy | None = None
    ) -> SQLiteRuntimeSupervisorLeaseStore:
        return SQLiteRuntimeSupervisorLeaseStore(
            self.database,
            sentinel_root=self.sentinels,
            policy=policy,
            clock=lambda: NOW,
        )

    def test_acquire_persists_only_token_hash_and_owner_only_files(self) -> None:
        store = self.store()
        acquisition = store.acquire(_binding())

        self.assertEqual(acquisition.receipt.state, "prepared")
        self.assertEqual(store.get(acquisition.handle.lease_id), acquisition.receipt)
        self.assertEqual(len(acquisition.handle.token), 32)
        raw_token_hex = acquisition.handle.token.hex()
        rendered = json.dumps(acquisition.receipt.payload(), sort_keys=True)
        self.assertNotIn(raw_token_hex, rendered)
        self.assertNotIn(raw_token_hex.encode("ascii"), self.database.read_bytes())
        self.assertNotIn("token=", repr(acquisition.handle))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.database.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(self.sentinels.stat().st_mode), 0o700)

    def test_full_happy_state_chain_deletes_only_stopped_active_row(self) -> None:
        store = self.store()
        acquisition = store.acquire(_binding())
        handle = acquisition.handle
        starting = store.transition(
            handle,
            "starting",
            runtime_pid=456,
            runtime_create_time_ns=123_456_789,
            runtime_executable_sha256=SHA_C,
        )
        ready = store.transition(
            handle,
            "ready",
            process_tree_sha256=SHA_D,
            endpoint_evidence_sha256=SHA_E,
        )
        stopping = store.transition(handle, "stopping")
        stopped = store.transition(handle, "stopped")

        self.assertEqual(starting.transition_index, 1)
        self.assertEqual(ready.previous_receipt_sha256, starting.digest)
        self.assertEqual(stopping.previous_receipt_sha256, ready.digest)
        self.assertEqual(stopped.previous_receipt_sha256, stopping.digest)
        self.assertIsNone(store.get(handle.lease_id))
        with self.assertRaises(RuntimeSupervisorLeaseStoreError):
            store.transition(handle, "stopped")

    def test_duplicate_endpoint_is_blocked_but_another_endpoint_is_independent(self) -> None:
        store = self.store()
        first = store.acquire(_binding())
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            store.acquire(_binding(binding_manifest_sha256=SHA_C))
        self.assertEqual(caught.exception.code, "runtime_endpoint_already_leased")

        other = store.acquire(
            _binding(
                binding_manifest_sha256=SHA_C,
                endpoint_authority_sha256=SHA_F,
            )
        )
        self.assertNotEqual(first.handle.lease_id, other.handle.lease_id)

    def test_transition_rejects_skips_and_forged_raw_token(self) -> None:
        store = self.store()
        acquisition = store.acquire(_binding())
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            store.transition(acquisition.handle, "ready")
        self.assertEqual(caught.exception.code, "runtime_lease_transition_invalid")

        forged = RuntimeSupervisorLeaseHandle(
            lease_id=acquisition.handle.lease_id,
            binding_sha256=acquisition.handle.binding_sha256,
            endpoint_authority_sha256=acquisition.handle.endpoint_authority_sha256,
            token=b"x" * 32,
            _owner_lock=acquisition.handle._owner_lock,
            _sentinel_path=acquisition.handle._sentinel_path,
        )
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            store.transition(forged, "starting", runtime_pid=1)
        self.assertEqual(caught.exception.code, "runtime_lease_token_mismatch")

    def test_revocation_and_unknown_are_fail_closed(self) -> None:
        store = self.store()
        acquisition = store.acquire(_binding())
        store.transition(
            acquisition.handle,
            "starting",
            runtime_pid=456,
            runtime_create_time_ns=123_456_789,
            runtime_executable_sha256=SHA_C,
        )
        revoked = store.transition(
            acquisition.handle,
            "revoked",
            reason_codes=("port_substituted",),
        )
        unknown = store.transition(
            acquisition.handle,
            "unknown_blocking",
            reason_codes=("cleanup_unverified",),
        )

        self.assertEqual(revoked.state, "revoked")
        self.assertEqual(unknown.state, "unknown_blocking")
        with self.assertRaises(RuntimeSupervisorLeaseStoreError):
            store.transition(acquisition.handle, "stopped")

    def test_dead_owner_becomes_sticky_unknown_and_blocks_endpoint(self) -> None:
        store = self.store()
        acquisition = store.acquire(_binding())
        # Simulate an owner that vanished without an authenticated stopped
        # transition.  Presence of the regular sentinel is not ownership.
        acquisition.handle._owner_lock.release(force=True)

        changed = store.mark_abandoned_owners()
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0].state, "unknown_blocking")
        self.assertEqual(changed[0].reason_codes, ("ownership_unknown",))
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            store.acquire(_binding())
        self.assertEqual(caught.exception.code, "runtime_endpoint_already_leased")

    def test_unknown_endpoint_does_not_exhaust_the_live_lease_cap(self) -> None:
        store = self.store(policy=RuntimeSupervisorLeasePolicy(max_active_leases=1))
        abandoned = store.acquire(_binding())
        abandoned.handle._owner_lock.release(force=True)
        self.assertEqual(
            store.mark_abandoned_owners()[0].state, "unknown_blocking"
        )

        independent = store.acquire(
            _binding(
                binding_manifest_sha256=SHA_C,
                endpoint_authority_sha256=SHA_F,
            )
        )

        self.assertEqual(independent.receipt.state, "prepared")

    def test_tampered_persisted_row_fails_closed(self) -> None:
        store = self.store()
        acquisition = store.acquire(_binding())
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE active_leases SET owner_pid = owner_pid + 1 "
                "WHERE lease_id = ?",
                (acquisition.handle.lease_id,),
            )
            connection.commit()

        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            store.get(acquisition.handle.lease_id)
        self.assertEqual(caught.exception.code, "runtime_lease_store_invalid")

    def test_foreign_schema_and_policy_identity_are_rejected(self) -> None:
        self.database.parent.mkdir(parents=True)
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE foreign_table(value TEXT)")
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            self.store()
        self.assertEqual(caught.exception.code, "runtime_lease_store_schema_invalid")

    def test_store_schema_is_independent_from_resource_lease_accounting(self) -> None:
        store = self.store()
        store.acquire(_binding())
        with sqlite3.connect(self.database) as connection:
            meta = dict(connection.execute("SELECT key, value FROM store_meta"))
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(active_leases)")
            }

        self.assertEqual(meta["contract"], "mymoe-runtime-supervisor-lease-store")
        self.assertIn("endpoint_authority_sha256", columns)
        self.assertNotIn("claim_sha256", columns)
        self.assertNotIn("system_claim_bytes", columns)
        self.assertNotIn("delivery_armed", columns)

        self.database.unlink()
        store = self.store()
        policy = replace(store.policy, max_active_leases=2, digest="")
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            self.store(policy=policy)
        self.assertEqual(caught.exception.code, "runtime_lease_store_schema_invalid")

    @unittest.skipIf(os.name == "nt", "symlink permissions differ on Windows")
    def test_database_symlink_is_rejected(self) -> None:
        target = self.root / "target.sqlite3"
        target.touch()
        self.database.parent.mkdir(parents=True)
        self.database.symlink_to(target)
        with self.assertRaises(RuntimeSupervisorLeaseStoreError) as caught:
            self.store()
        self.assertEqual(caught.exception.code, "runtime_lease_store_path_invalid")


if __name__ == "__main__":
    unittest.main()
