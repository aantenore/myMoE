from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from local_moe.assistant_bridge_ledger import (
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

            def consume(instance: BridgeStateLedger) -> None:
                barrier.wait()
                results.append(instance.consume_budget(key, 1))

            threads = [
                threading.Thread(target=consume, args=(first,)),
                threading.Thread(target=consume, args=(second,)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(sorted(results), [False, True])

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
                        "schema_version": "2.0",
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
