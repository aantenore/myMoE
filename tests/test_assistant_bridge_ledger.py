from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from local_moe.assistant_bridge_ledger import (
    LEDGER_SCHEMA_VERSION,
    BridgeLedgerError,
    BridgeStateLedger,
    budget_key,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


class AssistantBridgeLedgerTests(unittest.TestCase):
    def test_confirmation_is_binding_specific_expiring_and_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BridgeStateLedger(Path(tmp) / "state.json", namespace="test")
            ticket = ledger.issue_confirmation(DIGEST_A, ttl_seconds=10, now=100)
            self.assertNotIn(ticket.token, repr(ticket))

            with self.assertRaisesRegex(BridgeLedgerError, "binding"):
                ledger.consume_confirmation(ticket.token, DIGEST_B, now=101)
            transaction_id = ledger.consume_confirmation(
                ticket.token, DIGEST_A, now=110
            )
            self.assertEqual(transaction_id, ticket.transaction_id)
            with self.assertRaisesRegex(BridgeLedgerError, "already consumed"):
                ledger.consume_confirmation(ticket.token, DIGEST_A, now=110)

            expired = ledger.issue_confirmation(DIGEST_A, ttl_seconds=10, now=200)
            with self.assertRaisesRegex(BridgeLedgerError, "expired"):
                ledger.consume_confirmation(expired.token, DIGEST_A, now=211)
            rollback = ledger.issue_confirmation(DIGEST_A, ttl_seconds=10, now=300)
            with self.assertRaisesRegex(BridgeLedgerError, "clock"):
                ledger.consume_confirmation(rollback.token, DIGEST_A, now=299)

    def test_two_instances_atomically_share_budget_and_confirmation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            first = BridgeStateLedger(path, namespace="shared")
            second = BridgeStateLedger(path, namespace="shared")
            key = budget_key(
                namespace="shared",
                task_fingerprint=DIGEST_A,
                config_sha256=DIGEST_B,
                workspace_fingerprint=DIGEST_C,
            )
            barrier = threading.Barrier(2)
            results: list[bool] = []
            errors: list[Exception] = []
            result_lock = threading.Lock()

            def consume(instance: BridgeStateLedger) -> None:
                barrier.wait()
                try:
                    result = instance.consume_budget(key, 1)
                except Exception as exc:  # pragma: no cover - asserted below.
                    with result_lock:
                        errors.append(exc)
                    return
                with result_lock:
                    results.append(result)

            threads = [
                threading.Thread(target=consume, args=(first,)),
                threading.Thread(target=consume, args=(second,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(sorted(results), [False, True])

    def test_concurrent_budget_reservation_has_exactly_one_pending_winner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            contenders = [
                BridgeStateLedger(path, namespace="shared") for _ in range(12)
            ]
            barrier = threading.Barrier(len(contenders))
            result_lock = threading.Lock()
            leases: list[object] = []
            errors: list[Exception] = []

            def reserve(instance: BridgeStateLedger) -> None:
                barrier.wait()
                try:
                    lease = instance.reserve_budget(
                        DIGEST_A,
                        1,
                        ttl_seconds=30,
                        now=100,
                    )
                except Exception as exc:  # pragma: no cover - asserted below.
                    with result_lock:
                        errors.append(exc)
                    return
                with result_lock:
                    leases.append(lease)

            threads = [
                threading.Thread(target=reserve, args=(instance,))
                for instance in contenders
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(leases), len(contenders))
            winners = [lease for lease in leases if lease is not None]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            self.assertNotIn(winner.token, repr(winner))  # type: ignore[attr-defined]
            persisted = json.loads(path.read_text(encoding="utf-8"))
            budget = persisted["budgets"][DIGEST_A]
            self.assertEqual(budget["used"], 0)
            self.assertEqual(len(budget["pending"]), 1)

    def test_budget_lease_commit_and_popen_failure_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(path, namespace="test")
            committed = ledger.reserve_budget(DIGEST_A, 1, now=100)
            self.assertIsNotNone(committed)
            assert committed is not None

            ledger.commit_budget(committed, now=101)
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["budgets"][DIGEST_A]["used"], 1)
            self.assertEqual(persisted["budgets"][DIGEST_A]["pending"], {})
            self.assertIsNone(ledger.reserve_budget(DIGEST_A, 1, now=102))

            released = ledger.reserve_budget(DIGEST_B, 1, now=200)
            self.assertIsNotNone(released)
            assert released is not None
            ledger.release_budget_after_popen_failure(released, now=201)
            replacement = ledger.reserve_budget(DIGEST_B, 1, now=202)
            self.assertIsNotNone(replacement)

    def test_expired_pending_budget_is_promoted_to_used_fail_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(path, namespace="test")
            lease = ledger.reserve_budget(
                DIGEST_A,
                1,
                ttl_seconds=2,
                now=100,
            )
            self.assertIsNotNone(lease)
            assert lease is not None

            with self.assertRaisesRegex(BridgeLedgerError, "expired"):
                ledger.commit_budget(lease, now=102)

            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["budgets"][DIGEST_A]["used"], 1)
            self.assertEqual(persisted["budgets"][DIGEST_A]["pending"], {})
            self.assertFalse(ledger.consume_budget(DIGEST_A, 1, now=103))

    def test_pending_budget_counts_against_legacy_consume_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = BridgeStateLedger(Path(tmp) / "state.json", namespace="test")
            lease = ledger.reserve_budget(DIGEST_A, 1, now=100)
            self.assertIsNotNone(lease)

            self.assertFalse(ledger.consume_budget(DIGEST_A, 1, now=101))

    def test_simultaneous_confirmation_consumption_has_exactly_one_winner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            issuer = BridgeStateLedger(path, namespace="shared")
            ticket = issuer.issue_confirmation(
                DIGEST_A,
                ttl_seconds=60,
                now=100,
            )
            contenders = [
                BridgeStateLedger(path, namespace="shared") for _ in range(12)
            ]
            barrier = threading.Barrier(len(contenders))
            result_lock = threading.Lock()
            winners: list[str] = []
            failures: list[str] = []

            def consume(instance: BridgeStateLedger) -> None:
                barrier.wait()
                try:
                    transaction_id = instance.consume_confirmation(
                        ticket.token,
                        DIGEST_A,
                        now=101,
                    )
                except BridgeLedgerError as exc:
                    with result_lock:
                        failures.append(str(exc))
                else:
                    with result_lock:
                        winners.append(transaction_id)

            threads = [
                threading.Thread(target=consume, args=(instance,))
                for instance in contenders
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(winners, [ticket.transaction_id])
            self.assertEqual(len(failures), len(contenders) - 1)
            self.assertTrue(
                all("already consumed" in failure for failure in failures),
                failures,
            )

    def test_confirmation_capacity_prunes_expired_tickets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(
                path,
                namespace="test",
                confirmation_retention_seconds=1000,
                max_confirmation_entries=2,
            )
            ledger.issue_confirmation(DIGEST_A, ttl_seconds=1, now=100)
            ledger.issue_confirmation(DIGEST_A, ttl_seconds=1, now=100)

            current = ledger.issue_confirmation(DIGEST_A, ttl_seconds=10, now=102)

            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(persisted["confirmations"]), 2)
            self.assertIn(
                current.metadata_payload()["token_sha256"],
                persisted["confirmations"],
            )
            self.assertEqual(
                ledger.consume_confirmation(current.token, DIGEST_A, now=103),
                current.transaction_id,
            )

    def test_schema_namespace_and_timestamps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({"schema_version": "0"}), encoding="utf-8")
            ledger = BridgeStateLedger(path, namespace="test")
            with self.assertRaisesRegex(BridgeLedgerError, "keys"):
                ledger.consume_budget(DIGEST_A, 1)

            path.write_text(
                json.dumps(
                    {
                        "schema_version": LEDGER_SCHEMA_VERSION,
                        "namespace": "test",
                        "budgets": {},
                        "confirmations": {
                            DIGEST_A: {
                                "binding_sha256": DIGEST_A,
                                "transaction_id": "a" * 32,
                                "issued_at": 10,
                                "expires_at": 20,
                                "consumed_at": 21,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BridgeLedgerError, "outside"):
                ledger.consume_budget(DIGEST_A, 1)

    def test_legacy_budget_migration_never_reopens_consumed_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "namespace": "test",
                        "budgets": {DIGEST_A: 1},
                        "confirmations": {},
                    }
                ),
                encoding="utf-8",
            )
            ledger = BridgeStateLedger(path, namespace="test")

            self.assertFalse(ledger.consume_budget(DIGEST_A, 1))
            migrated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(migrated["schema_version"], LEDGER_SCHEMA_VERSION)
            self.assertEqual(migrated["budgets"][DIGEST_A]["used"], 1)
            self.assertGreater(migrated["budgets"][DIGEST_A]["updated_at"], 0)
            self.assertEqual(migrated["budgets"][DIGEST_A]["pending"], {})

    def test_v20_and_v21_clock_migrations_use_injected_time_coherently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for schema, budget, expected_updated_at in (
                ("2.0", 1, 100.0),
                ("2.1", {"used": 1, "updated_at": 50.0}, 50.0),
            ):
                with self.subTest(schema=schema):
                    path = root / f"state-{schema}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "schema_version": schema,
                                "namespace": "test",
                                "budgets": {DIGEST_A: budget},
                                "confirmations": {},
                            }
                        ),
                        encoding="utf-8",
                    )
                    ledger = BridgeStateLedger(
                        path,
                        namespace="test",
                        clock=lambda: 100.0,
                    )

                    self.assertFalse(ledger.consume_budget(DIGEST_A, 1))
                    migrated = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(migrated["schema_version"], "2.2")
                    self.assertEqual(
                        migrated["budgets"][DIGEST_A]["updated_at"],
                        expected_updated_at,
                    )
                    self.assertEqual(
                        migrated["budgets"][DIGEST_A]["pending"],
                        {},
                    )

    def test_budget_retention_is_bounded_and_prunes_oldest_epoch_safely(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(
                path,
                namespace="test",
                budget_retention_seconds=100,
                max_budget_entries=16,
            )
            keys = [f"{index:064x}" for index in range(1, 18)]
            for offset, key in enumerate(keys[:16]):
                self.assertTrue(ledger.consume_budget(key, 1, now=100 + offset))
            with self.assertRaisesRegex(BridgeLedgerError, "clock moved"):
                ledger.consume_budget(keys[0], 1, now=99)

            with self.assertRaisesRegex(BridgeLedgerError, "retention bound"):
                ledger.consume_budget(keys[16], 1, now=116)

            self.assertTrue(ledger.consume_budget(keys[16], 1, now=201))
            persisted = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn(keys[0], persisted["budgets"])
            self.assertIn(keys[1], persisted["budgets"])
            self.assertIn(keys[16], persisted["budgets"])
            self.assertEqual(len(persisted["budgets"]), 16)
            self.assertLess(path.stat().st_size, 4 * 1024 * 1024)
            descriptor = ledger.effective_descriptor()
            self.assertEqual(descriptor["budget_retention_seconds"], 100.0)
            self.assertEqual(descriptor["max_budget_entries"], 16)
            self.assertEqual(descriptor["max_confirmation_entries"], 4096)
            self.assertEqual(descriptor["budget_lease_ttl_seconds"], 60.0)

    def test_unknown_future_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "99.0",
                        "namespace": "test",
                        "budgets": {},
                        "confirmations": {},
                    }
                ),
                encoding="utf-8",
            )
            ledger = BridgeStateLedger(path, namespace="test")

            with self.assertRaisesRegex(BridgeLedgerError, "unsupported"):
                ledger.consume_budget(DIGEST_A, 1)

    def test_stale_lock_recovers_and_live_lock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(
                path,
                namespace="test",
                lock_timeout_seconds=0.1,
                stale_lock_seconds=1,
            )
            lock = path.with_suffix(".json.lock")
            lock.mkdir()
            (lock / "owner.json").write_text(
                json.dumps({"pid": 999_999_999, "created_at": 0}),
                encoding="utf-8",
            )
            old = time.time() - 10
            os.utime(lock, (old, old))
            self.assertTrue(ledger.consume_budget(DIGEST_A, 1))

            lock.mkdir()
            (lock / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created_at": time.time()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BridgeLedgerError, "busy"):
                ledger.consume_budget(DIGEST_B, 1)

    def test_owner_metadata_write_error_is_wrapped_and_lock_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(path, namespace="test")
            original_write_text = Path.write_text

            def fail_owner(path_value: Path, *args: object, **kwargs: object) -> int:
                if path_value.name == "owner.json":
                    raise OSError("sensitive filesystem diagnostic")
                return original_write_text(path_value, *args, **kwargs)  # type: ignore[arg-type]

            with patch.object(Path, "write_text", new=fail_owner):
                with self.assertRaisesRegex(
                    BridgeLedgerError,
                    "owner cannot be persisted safely",
                ) as raised:
                    ledger.consume_budget(DIGEST_A, 1)

            self.assertNotIn("sensitive filesystem diagnostic", str(raised.exception))
            self.assertFalse(path.with_suffix(".json.lock").exists())

    def test_transient_lock_permission_error_is_retried_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(path, namespace="test")
            lock = ledger.path.with_suffix(ledger.path.suffix + ".lock")
            original_mkdir = Path.mkdir
            attempts = 0

            def fail_once(
                path_value: Path,
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal attempts
                if path_value == lock and attempts == 0:
                    attempts += 1
                    raise PermissionError("sensitive transient filesystem diagnostic")
                original_mkdir(path_value, *args, **kwargs)  # type: ignore[arg-type]

            with (
                patch("local_moe.assistant_bridge_ledger._IS_WINDOWS", True),
                patch.object(Path, "mkdir", new=fail_once),
            ):
                self.assertTrue(ledger.consume_budget(DIGEST_A, 1))

            self.assertEqual(attempts, 1)
            self.assertFalse(lock.exists())

    def test_persistent_lock_permission_error_remains_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(
                path,
                namespace="test",
                lock_timeout_seconds=0.1,
            )
            lock = ledger.path.with_suffix(ledger.path.suffix + ".lock")
            original_mkdir = Path.mkdir

            def always_fail(
                path_value: Path,
                *args: object,
                **kwargs: object,
            ) -> None:
                if path_value == lock:
                    raise PermissionError("sensitive persistent filesystem diagnostic")
                original_mkdir(path_value, *args, **kwargs)  # type: ignore[arg-type]

            with (
                patch("local_moe.assistant_bridge_ledger._IS_WINDOWS", True),
                patch.object(Path, "mkdir", new=always_fail),
            ):
                with self.assertRaisesRegex(
                    BridgeLedgerError,
                    "lock could not be acquired",
                ) as raised:
                    ledger.consume_budget(DIGEST_A, 1)

            self.assertNotIn(
                "sensitive persistent filesystem diagnostic",
                str(raised.exception),
            )
            self.assertFalse(path.exists())

    @unittest.skipIf(os.name == "nt", "POSIX-specific lock behavior")
    def test_posix_permission_error_is_not_retried_without_contention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            ledger = BridgeStateLedger(path, namespace="test")
            lock = ledger.path.with_suffix(ledger.path.suffix + ".lock")
            original_mkdir = Path.mkdir
            attempts = 0

            def always_fail(
                path_value: Path,
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal attempts
                if path_value == lock:
                    attempts += 1
                    raise PermissionError("sensitive POSIX filesystem diagnostic")
                original_mkdir(path_value, *args, **kwargs)  # type: ignore[arg-type]

            with patch.object(Path, "mkdir", new=always_fail):
                with self.assertRaisesRegex(
                    BridgeLedgerError,
                    "lock could not be acquired",
                ):
                    ledger.consume_budget(DIGEST_A, 1)

            self.assertEqual(attempts, 1)
            self.assertFalse(path.exists())

    def test_symlinked_ledger_or_parent_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "state.json"
            os.symlink(target, link)
            with self.assertRaisesRegex(BridgeLedgerError, "symbolic"):
                BridgeStateLedger(link, namespace="test")

            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            os.symlink(real_parent, linked_parent)
            with self.assertRaisesRegex(BridgeLedgerError, "symbolic"):
                BridgeStateLedger(linked_parent / "state.json", namespace="test")


if __name__ == "__main__":
    unittest.main()
