from __future__ import annotations

import errno
import json
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

from local_moe import paired_execution_store as store_module
from local_moe.paired_execution_contracts import (
    PairedOutcomeBinding,
    PairedRunClaim,
    PairedRunRoot,
)
from local_moe.paired_execution_store import (
    PairedExecutionStore,
    PairedRunIndeterminateError,
)
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    canonical_json,
)


_DIGESTS = tuple(character * 64 for character in "abcdef0123456789")


class PairedExecutionContractTests(unittest.TestCase):
    def test_ab_and_ba_roots_have_canonical_ordered_slots(self) -> None:
        ab = _root(order="AB")
        ba = _root(order="BA")

        self.assertEqual(
            [(slot.slot, slot.arm, slot.ordinal, slot.route) for slot in ab.slots],
            [
                ("A", "baseline", 0, "premium"),
                ("B", "candidate", 1, "local"),
            ],
        )
        self.assertEqual(
            [(slot.slot, slot.arm, slot.ordinal, slot.route) for slot in ba.slots],
            [
                ("B", "candidate", 0, "local"),
                ("A", "baseline", 1, "premium"),
            ],
        )
        self.assertNotEqual(ab.run_id, ba.run_id)
        self.assertEqual(PairedRunRoot.from_payload(ab.payload()), ab)

    def test_root_and_binding_are_content_addressed_and_strict(self) -> None:
        root = _root()
        changed = _root(pricing_sha256=_DIGESTS[9])
        self.assertNotEqual(root.run_id, changed.run_id)

        with tempfile.TemporaryDirectory() as temporary:
            store = PairedExecutionStore(Path(temporary) / "run")
            store.prepare(root)
            claim = store.claim("A")
            binding = store.binding_for(claim)
            self.assertIsNone(binding.previous_record_id)
            self.assertEqual(
                PairedOutcomeBinding.from_payload(binding.payload()), binding
            )

            tampered = binding.payload()
            tampered["pricing_sha256"] = _DIGESTS[10]
            with self.assertRaisesRegex(VerifiedRoutingError, "digest"):
                PairedOutcomeBinding.from_payload(tampered)

            unknown = binding.payload()
            unknown["surprise"] = True
            with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
                PairedOutcomeBinding.from_payload(unknown)


class PairedExecutionStoreTests(unittest.TestCase):
    def test_status_on_empty_existing_directory_is_strictly_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir(mode=0o700)
            if os.name != "nt":
                run_dir.chmod(0o700)
            before_metadata = run_dir.stat()
            before_entries = tuple(run_dir.iterdir())

            status = PairedExecutionStore(run_dir).status()

            after_metadata = run_dir.stat()
            after_entries = tuple(run_dir.iterdir())
            self.assertEqual(status.state, "missing")
            self.assertEqual(before_entries, after_entries)
            self.assertEqual(before_metadata.st_mtime_ns, after_metadata.st_mtime_ns)
            self.assertFalse((run_dir / "run.lock").exists())

    def test_status_fails_closed_when_journal_lock_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            self.assertTrue(store.lock_path.is_file())
            store.lock_path.unlink()

            with self.assertRaisesRegex(VerifiedRoutingError, "without run.lock"):
                store.status()

            self.assertFalse(store.lock_path.exists())

    def test_prepare_is_exactly_idempotent_and_status_starts_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)

            self.assertEqual(store.status().state, "missing")
            self.assertTrue(store.prepare(_root()))
            self.assertFalse(store.prepare(_root()))
            ready = store.status()
            self.assertEqual(ready.state, "ready")
            self.assertEqual(ready.next_slot, _root().slots[0])
            with self.assertRaisesRegex(VerifiedRoutingError, "another root"):
                store.prepare(_root(task_fingerprint=_DIGESTS[11]))

            if os.name != "nt":
                self.assertEqual(run_dir.stat().st_mode & 0o777, 0o700)
                for item in ("run.json", "run.lock"):
                    self.assertEqual(
                        (run_dir / item).stat().st_mode & 0o777,
                        0o600,
                    )

    def test_out_of_order_claim_is_rejected_without_writing_an_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PairedExecutionStore(Path(temporary) / "run")
            store.prepare(_root(order="AB"))

            with self.assertRaisesRegex(VerifiedRoutingError, "out of declared order"):
                store.claim("B")

            self.assertEqual(store.status().state, "ready")
            self.assertFalse(store.events_path.exists())

    def test_uncheckpointed_claim_is_owner_running_then_indeterminate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            owner = PairedExecutionStore(run_dir)
            owner.prepare(_root())
            claim = owner.claim("A")

            self.assertEqual(owner.status().state, "running")
            recovered = PairedExecutionStore(run_dir)
            self.assertEqual(recovered.status().state, "indeterminate")
            with self.assertRaises(PairedRunIndeterminateError):
                recovered.claim("A")

            owner.abandon(claim)
            self.assertEqual(owner.status().state, "indeterminate")
            with self.assertRaises(PairedRunIndeterminateError):
                owner.binding_for(claim)

    def test_partial_run_resumes_at_second_slot_and_binds_first_outcome(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            first = PairedExecutionStore(run_dir)
            first.prepare(_root(order="BA"))
            first_record = "outcome-" + _DIGESTS[8]
            _complete(first, first.claim("B"), first_record, 0)

            resumed = PairedExecutionStore(run_dir)
            partial = resumed.status()
            self.assertEqual(partial.state, "partial")
            self.assertEqual(partial.next_slot.slot, "A")  # type: ignore[union-attr]
            with self.assertRaisesRegex(VerifiedRoutingError, "out of declared order"):
                resumed.claim("B")

            second_claim = resumed.claim("A")
            second_binding = resumed.binding_for(second_claim)
            self.assertEqual(second_binding.previous_record_id, first_record)

    def test_complete_run_and_exact_checkpoint_replay_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            first_claim = store.claim("A")
            first_binding = store.binding_for(first_claim)
            first = _complete(
                store,
                first_claim,
                "outcome-" + _DIGESTS[8],
                0,
                binding=first_binding,
            )
            replay = store.complete(
                first_binding,
                outcome_record_id=first.outcome_record_id,
                route_receipt_id=first.route_receipt_id,
                route_receipt_sha256=first.route_receipt_sha256,
                evidence_sha256=first.evidence_sha256,
            )
            self.assertEqual(replay, first)
            with self.assertRaisesRegex(VerifiedRoutingError, "different checkpoint"):
                store.complete(
                    first_binding,
                    outcome_record_id=first.outcome_record_id,
                    route_receipt_id=first.route_receipt_id,
                    route_receipt_sha256=first.route_receipt_sha256,
                    evidence_sha256=_DIGESTS[15],
                )

            second_claim = store.claim("B")
            second = _complete(
                store,
                second_claim,
                "outcome-" + _DIGESTS[9],
                1,
            )
            complete = store.status()
            self.assertEqual(complete.state, "complete")
            self.assertEqual(len(complete.checkpoints), 2)
            self.assertEqual(
                second.binding.previous_record_id, first.outcome_record_id
            )
            with self.assertRaisesRegex(VerifiedRoutingError, "already complete"):
                store.claim("A")

            reopened = PairedExecutionStore(run_dir)
            replay_reopened = reopened.complete(
                second.binding,
                outcome_record_id=second.outcome_record_id,
                route_receipt_id=second.route_receipt_id,
                route_receipt_sha256=second.route_receipt_sha256,
                evidence_sha256=second.evidence_sha256,
            )
            self.assertEqual(replay_reopened, second)

    def test_concurrent_claim_has_one_durable_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            PairedExecutionStore(run_dir).prepare(_root())
            context = multiprocessing.get_context("spawn")
            ready = context.Queue()
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_claim_worker,
                    args=(str(run_dir), ready, start, results),
                )
                for _ in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for _ in processes:
                    self.assertTrue(ready.get(timeout=15.0))
                start.set()
                observed = [results.get(timeout=20.0) for _ in processes]
            finally:
                start.set()
                for process in processes:
                    process.join(10.0)
                    if process.is_alive():
                        process.terminate()
                        process.join(5.0)
                ready.close()
                results.close()
                ready.join_thread()
                results.join_thread()

            self.assertEqual(sorted(observed), ["indeterminate", "won"])
            recovered = PairedExecutionStore(run_dir)
            self.assertEqual(recovered.status().state, "indeterminate")
            self.assertEqual(len(recovered.status().claims), 1)

    def test_provider_process_exit_leaves_an_indeterminate_immutable_claim(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            PairedExecutionStore(run_dir).prepare(_root())
            context = multiprocessing.get_context("spawn")
            process = context.Process(
                target=_crash_after_claim_worker,
                args=(str(run_dir),),
            )
            process.start()
            process.join(15.0)
            if process.is_alive():
                process.terminate()
                process.join(5.0)
            self.assertEqual(process.exitcode, 0)

            recovered = PairedExecutionStore(run_dir)
            self.assertEqual(recovered.status().state, "indeterminate")
            artifacts = tuple(recovered.events_path.iterdir())
            self.assertEqual(len(artifacts), 1)
            self.assertRegex(
                artifacts[0].name,
                r"^000000-claim-[0-9a-f]{64}\.json$",
            )
            self.assertFalse((run_dir / "events.jsonl").exists())
            if os.name != "nt":
                self.assertEqual(
                    recovered.events_path.stat().st_mode & 0o777,
                    0o700,
                )
                self.assertEqual(artifacts[0].stat().st_mode & 0o777, 0o600)

    @unittest.skipIf(os.name == "nt", "requires POSIX fork semantics")
    def test_forked_child_cannot_inherit_claim_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            owner = PairedExecutionStore(run_dir)
            owner.prepare(_root())
            claim = owner.claim("A")
            binding = owner.binding_for(claim)
            context = multiprocessing.get_context("fork")
            results = context.Queue()
            child = context.Process(
                target=_fork_authority_worker,
                args=(owner, claim, binding, results),
            )
            child.start()
            observed = results.get(timeout=15.0)
            child.join(10.0)
            if child.is_alive():
                child.terminate()
                child.join(5.0)

            self.assertEqual(child.exitcode, 0)
            self.assertEqual(observed, ("binding-rejected", "complete-rejected"))
            self.assertEqual(owner.status().state, "running")
            _complete(
                owner,
                claim,
                "outcome-" + _DIGESTS[8],
                0,
                binding=binding,
            )
            self.assertEqual(owner.status().state, "partial")
            results.close()
            results.join_thread()

    def test_duplicate_corrupt_and_out_of_order_events_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            store.claim("A")
            (store.events_path / "extra.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(VerifiedRoutingError, "extra file"):
                PairedExecutionStore(run_dir).status()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            store.claim("A")
            malformed = b'{"event":"claim","event":"claim","payload":NaN}\n'
            event_path = next(store.events_path.iterdir())
            event_path.write_bytes(malformed)
            if os.name != "nt":
                event_path.chmod(0o600)
            with self.assertRaisesRegex(VerifiedRoutingError, "corrupt"):
                store.status()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            first = _complete(
                store,
                store.claim("A"),
                "outcome-" + _DIGESTS[8],
                0,
            )
            artifacts = sorted(store.events_path.iterdir())
            self.assertEqual(len(artifacts), 2)
            claim_path, checkpoint_path = artifacts
            parked = store.events_path / "parked"
            claim_path.rename(parked)
            checkpoint_path.rename(
                store.events_path / ("000000-" + checkpoint_path.name[7:])
            )
            parked.rename(store.events_path / ("000001-" + claim_path.name[7:]))
            self.assertIsNotNone(first)
            with self.assertRaisesRegex(VerifiedRoutingError, "corrupt"):
                store.status()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            store.claim("A")
            artifact = next(store.events_path.iterdir())
            artifact.rename(store.events_path / ("000001-" + artifact.name[7:]))
            with self.assertRaisesRegex(VerifiedRoutingError, "contiguous"):
                store.status()

    @unittest.skipIf(os.name == "nt", "POSIX link and mode semantics")
    def test_symlink_hardlink_and_permissive_modes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root_dir = Path(temporary)

            symlink_run = root_dir / "symlink-run"
            symlink_store = PairedExecutionStore(symlink_run)
            symlink_store.prepare(_root())
            original = symlink_run / "root-copy.json"
            original.write_bytes(symlink_store.root_path.read_bytes())
            original.chmod(0o600)
            symlink_store.root_path.unlink()
            symlink_store.root_path.symlink_to(original)
            with self.assertRaisesRegex(VerifiedRoutingError, "non-link"):
                symlink_store.status()

            hardlink_run = root_dir / "hardlink-run"
            hardlink_store = PairedExecutionStore(hardlink_run)
            hardlink_store.prepare(_root())
            os.link(hardlink_store.root_path, hardlink_run / "alias.json")
            with self.assertRaisesRegex(VerifiedRoutingError, "hard link"):
                hardlink_store.status()

            file_mode_run = root_dir / "file-mode-run"
            file_mode_store = PairedExecutionStore(file_mode_run)
            file_mode_store.prepare(_root())
            file_mode_store.root_path.chmod(0o644)
            with self.assertRaisesRegex(VerifiedRoutingError, "0600"):
                file_mode_store.status()

            dir_mode_run = root_dir / "dir-mode-run"
            dir_mode_store = PairedExecutionStore(dir_mode_run)
            dir_mode_store.prepare(_root())
            dir_mode_run.chmod(0o755)
            with self.assertRaisesRegex(VerifiedRoutingError, "0700"):
                dir_mode_store.status()

            event_link_run = root_dir / "event-link-run"
            event_link_store = PairedExecutionStore(event_link_run)
            event_link_store.prepare(_root())
            event_link_store.claim("A")
            event_path = next(event_link_store.events_path.iterdir())
            event_copy = event_link_run / "event-copy.json"
            event_copy.write_bytes(event_path.read_bytes())
            event_copy.chmod(0o600)
            event_path.unlink()
            event_path.symlink_to(event_copy)
            with self.assertRaisesRegex(VerifiedRoutingError, "non-link"):
                event_link_store.status()

            event_hardlink_run = root_dir / "event-hardlink-run"
            event_hardlink_store = PairedExecutionStore(event_hardlink_run)
            event_hardlink_store.prepare(_root())
            event_hardlink_store.claim("A")
            event_path = next(event_hardlink_store.events_path.iterdir())
            alias = event_hardlink_store.events_path / (
                "000001-" + event_path.name[7:]
            )
            os.link(event_path, alias)
            with self.assertRaisesRegex(VerifiedRoutingError, "hard link"):
                event_hardlink_store.status()

            event_mode_run = root_dir / "event-mode-run"
            event_mode_store = PairedExecutionStore(event_mode_run)
            event_mode_store.prepare(_root())
            event_mode_store.claim("A")
            next(event_mode_store.events_path.iterdir()).chmod(0o644)
            with self.assertRaisesRegex(VerifiedRoutingError, "0600"):
                event_mode_store.status()

            event_dir_mode_run = root_dir / "event-dir-mode-run"
            event_dir_mode_store = PairedExecutionStore(event_dir_mode_run)
            event_dir_mode_store.prepare(_root())
            event_dir_mode_store.claim("A")
            event_dir_mode_store.events_path.chmod(0o755)
            with self.assertRaisesRegex(VerifiedRoutingError, "0700"):
                event_dir_mode_store.status()

    def test_directory_fsync_is_best_effort_only_for_windows_limitations(
        self,
    ) -> None:
        failure = OSError(errno.EACCES, "directory handles unsupported")
        unused = Path("unused")
        with (
            mock.patch.object(store_module.os, "name", "nt"),
            mock.patch.object(store_module.os, "open", return_value=71),
            mock.patch.object(store_module.os, "fsync", side_effect=failure),
            mock.patch.object(store_module.os, "close") as close,
        ):
            store_module._fsync_directory(unused)
            close.assert_called_once_with(71)

        with (
            mock.patch.object(store_module.os, "name", "posix"),
            mock.patch.object(store_module.os, "open", return_value=72),
            mock.patch.object(store_module.os, "fsync", side_effect=failure),
            mock.patch.object(store_module.os, "close"),
        ):
            with self.assertRaisesRegex(
                VerifiedRoutingError,
                "directory synchronization failed",
            ):
                store_module._fsync_directory(unused)

    def test_windows_best_effort_does_not_weaken_file_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "event.json"
            with (
                mock.patch.object(store_module.os, "name", "nt"),
                mock.patch.object(
                    store_module.os,
                    "fsync",
                    side_effect=OSError(errno.EACCES, "file flush failed"),
                ),
            ):
                with self.assertRaisesRegex(
                    VerifiedRoutingError,
                    "Cannot install paired store file",
                ):
                    store_module._atomic_install(target, b"{}\n")
            self.assertFalse(target.exists())

    def test_windows_lock_binds_path_to_locked_handle_without_delete_share(
        self,
    ) -> None:
        from local_moe import _win32_fs

        lock_path = Path("run.lock")
        parent_identity = _windows_identity(1, attributes=0x00000010)
        archived_parent_identity = _windows_identity(
            1,
            attributes=0x00000030,
        )
        lock_identity = _windows_identity(2)
        archived_lock_identity = _windows_identity(2, attributes=0x20)
        metadata = mock.Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
        )
        fake_msvcrt = _fake_msvcrt()
        with (
            mock.patch.object(store_module, "_is_windows", return_value=True),
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[
                    (70, parent_identity),
                    (71, lock_identity),
                    (72, parent_identity),
                    (73, lock_identity),
                ],
            ) as open_nofollow,
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                side_effect=[
                    parent_identity,
                    archived_lock_identity,
                    archived_parent_identity,
                    archived_lock_identity,
                    archived_parent_identity,
                    archived_lock_identity,
                ],
            ),
            mock.patch.object(store_module.os, "fstat", return_value=metadata),
            mock.patch.object(store_module.os, "lseek"),
            mock.patch.object(store_module.os, "close") as close,
            mock.patch.object(store_module, "_ensure_lock_file") as ensure_lock,
        ):
            with store_module._existing_file_lock(
                lock_path,
                timeout_seconds=1.0,
                exclusive=True,
                ensure_lock=True,
            ):
                self.assertEqual(
                    close.call_args_list,
                    [mock.call(73), mock.call(72)],
                )

        ensure_lock.assert_called_once_with(lock_path)
        self.assertEqual(
            open_nofollow.call_args_list,
            [
                mock.call(
                    lock_path.parent,
                    directory=True,
                    writable=False,
                    share_delete=False,
                ),
                mock.call(
                    lock_path,
                    directory=False,
                    writable=True,
                    share_delete=False,
                ),
                mock.call(
                    lock_path.parent,
                    directory=True,
                    writable=False,
                    share_delete=False,
                ),
                mock.call(
                    lock_path,
                    directory=False,
                    writable=True,
                    share_delete=False,
                ),
            ],
        )
        self.assertEqual(
            close.call_args_list,
            [mock.call(73), mock.call(72), mock.call(71), mock.call(70)],
        )
        self.assertEqual(
            fake_msvcrt.locking.call_args_list,
            [mock.call(71, 1, 1), mock.call(71, 2, 1)],
        )

    def test_windows_lock_rejects_split_brain_path_identity(self) -> None:
        from local_moe import _win32_fs

        lock_path = Path("run.lock")
        parent_identity = _windows_identity(1, attributes=0x00000010)
        locked_identity = _windows_identity(1)
        replacement_identity = _windows_identity(2)
        metadata = mock.Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
        )
        fake_msvcrt = _fake_msvcrt()
        with (
            mock.patch.object(store_module, "_is_windows", return_value=True),
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[
                    (70, parent_identity),
                    (71, locked_identity),
                    (72, parent_identity),
                    (73, replacement_identity),
                ],
            ),
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                side_effect=[
                    parent_identity,
                    locked_identity,
                    parent_identity,
                    locked_identity,
                    parent_identity,
                    replacement_identity,
                ],
            ),
            mock.patch.object(store_module.os, "fstat", return_value=metadata),
            mock.patch.object(store_module.os, "lseek"),
            mock.patch.object(store_module.os, "close") as close,
            self.assertRaisesRegex(
                VerifiedRoutingError,
                "no longer names the locked file",
            ),
        ):
            with store_module._existing_file_lock(
                lock_path,
                timeout_seconds=1.0,
                exclusive=True,
            ):
                self.fail("split-brain Windows lock must not yield")

        self.assertEqual(
            close.call_args_list,
            [mock.call(73), mock.call(72), mock.call(71), mock.call(70)],
        )
        self.assertEqual(
            fake_msvcrt.locking.call_args_list,
            [mock.call(71, 1, 1), mock.call(71, 2, 1)],
        )

    def test_windows_lock_rejects_replaced_run_directory_identity(self) -> None:
        from local_moe import _win32_fs

        lock_path = Path("run.lock")
        parent_identity = _windows_identity(1, attributes=0x00000010)
        replacement_parent = _windows_identity(2, attributes=0x00000010)
        lock_identity = _windows_identity(3)
        metadata = mock.Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
        )
        fake_msvcrt = _fake_msvcrt()
        with (
            mock.patch.object(store_module, "_is_windows", return_value=True),
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[
                    (70, parent_identity),
                    (71, lock_identity),
                    (72, replacement_parent),
                    (73, lock_identity),
                ],
            ),
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                side_effect=[
                    parent_identity,
                    lock_identity,
                    parent_identity,
                    lock_identity,
                    replacement_parent,
                    lock_identity,
                ],
            ),
            mock.patch.object(store_module.os, "fstat", return_value=metadata),
            mock.patch.object(store_module.os, "lseek"),
            mock.patch.object(store_module.os, "close") as close,
            self.assertRaisesRegex(
                VerifiedRoutingError,
                "no longer names the pinned directory",
            ),
        ):
            with store_module._existing_file_lock(
                lock_path,
                timeout_seconds=1.0,
                exclusive=True,
            ):
                self.fail("a replaced Windows run directory must not yield")

        self.assertEqual(
            close.call_args_list,
            [mock.call(73), mock.call(72), mock.call(71), mock.call(70)],
        )
        self.assertEqual(
            fake_msvcrt.locking.call_args_list,
            [mock.call(71, 1, 1), mock.call(71, 2, 1)],
        )

    def test_windows_lock_rejects_open_result_descriptor_split(self) -> None:
        from local_moe import _win32_fs

        lock_path = Path("run.lock")
        parent_identity = _windows_identity(1, attributes=0x00000010)
        path_identity = _windows_identity(1)
        descriptor_identity = _windows_identity(2)
        metadata = mock.Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=1,
        )
        fake_msvcrt = _fake_msvcrt()
        with (
            mock.patch.object(store_module, "_is_windows", return_value=True),
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[(70, parent_identity), (71, path_identity)],
            ),
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                side_effect=[parent_identity, descriptor_identity],
            ),
            mock.patch.object(store_module.os, "fstat", return_value=metadata),
            mock.patch.object(store_module.os, "close") as close,
            self.assertRaisesRegex(VerifiedRoutingError, "while it was opened"),
        ):
            with store_module._existing_file_lock(
                lock_path,
                timeout_seconds=1.0,
                exclusive=True,
            ):
                self.fail("split path/descriptor identity must not yield")

        self.assertEqual(close.call_args_list, [mock.call(71), mock.call(70)])
        fake_msvcrt.locking.assert_not_called()

    def test_windows_lock_rejects_reparse_identity_before_locking(self) -> None:
        from local_moe import _win32_fs

        lock_path = Path("run.lock")
        parent_identity = _windows_identity(1, attributes=0x00000010)
        reparse_identity = _windows_identity(
            2,
            attributes=0x00000400,
            reparse_tag=0xA000000C,
        )
        fake_msvcrt = _fake_msvcrt()
        with (
            mock.patch.object(store_module, "_is_windows", return_value=True),
            mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            mock.patch.object(
                _win32_fs,
                "open_nofollow_fd",
                side_effect=[(70, parent_identity), (71, reparse_identity)],
            ),
            mock.patch.object(
                _win32_fs,
                "identity_from_fd",
                return_value=parent_identity,
            ),
            mock.patch.object(store_module.os, "close") as close,
            self.assertRaisesRegex(VerifiedRoutingError, "reparse point"),
        ):
            with store_module._existing_file_lock(
                lock_path,
                timeout_seconds=1.0,
                exclusive=True,
            ):
                self.fail("Windows reparse lock must not yield")

        self.assertEqual(close.call_args_list, [mock.call(71), mock.call(70)])
        fake_msvcrt.locking.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "native Windows lock semantics")
    def test_native_windows_lock_path_persists_after_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PairedExecutionStore(Path(temporary) / "run")
            store.prepare(_root())
            self.assertTrue(store.lock_path.is_file())
            self.assertEqual(store.status().state, "ready")
            self.assertTrue(store.lock_path.is_file())

    @unittest.skipUnless(os.name == "nt", "native Windows reparse semantics")
    def test_native_windows_lock_rejects_reparse_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / "target.lock"
            target.touch()
            link = directory / "link.lock"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"Windows symlink unavailable: {exc}")
            with self.assertRaisesRegex(VerifiedRoutingError, "lock acquisition"):
                with store_module._existing_file_lock(
                    link,
                    timeout_seconds=1.0,
                    exclusive=True,
                ):
                    self.fail("Windows reparse lock must not yield")

    def test_json_file_descriptors_use_binary_mode_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "event.json"
            binary_flag = 1 << 29
            opened: list[tuple[Path, int]] = []
            real_open = os.open

            def tracking_open(
                path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                flags: int,
                mode: int = 0o777,
            ) -> int:
                opened.append((Path(path), flags))
                return real_open(path, flags & ~binary_flag, mode)

            with (
                mock.patch.object(
                    store_module.os,
                    "O_BINARY",
                    binary_flag,
                    create=True,
                ),
                mock.patch.object(
                    store_module.os,
                    "open",
                    side_effect=tracking_open,
                ),
            ):
                self.assertTrue(store_module._atomic_install(target, b"{}\n"))
                self.assertEqual(
                    store_module._read_secure_file(
                        target,
                        "test event",
                        maximum_bytes=1024,
                    ),
                    b"{}\n",
                )

            temporary_write = next(
                flags
                for path, flags in opened
                if path.name.startswith(".event.json.")
            )
            target_read = next(
                flags
                for path, flags in opened
                if path == target
            )
            self.assertTrue(temporary_write & binary_flag)
            self.assertTrue(target_read & binary_flag)

    def test_run_json_duplicate_keys_and_noncanonical_encoding_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            payload = _root().payload()
            rendered = canonical_json(payload)
            duplicate = rendered[:-1] + ',"run_id":"' + payload["run_id"] + '"}\n'
            store.root_path.write_text(duplicate, encoding="utf-8")
            if os.name != "nt":
                store.root_path.chmod(0o600)
            with self.assertRaisesRegex(VerifiedRoutingError, "corrupt"):
                store.status()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            store = PairedExecutionStore(run_dir)
            store.prepare(_root())
            pretty = json.dumps(_root().payload(), indent=2) + "\n"
            store.root_path.write_text(pretty, encoding="utf-8")
            if os.name != "nt":
                store.root_path.chmod(0o600)
            with self.assertRaisesRegex(VerifiedRoutingError, "not canonical"):
                store.status()


def _windows_identity(
    value: int,
    *,
    attributes: int = 0,
    reparse_tag: int = 0,
):
    from local_moe._win32_fs import Win32FileIdentity

    return Win32FileIdentity(
        volume_serial=7,
        file_id=bytes([value]) * 16,
        attributes=attributes,
        reparse_tag=reparse_tag,
    )


def _fake_msvcrt() -> types.ModuleType:
    module = types.ModuleType("msvcrt")
    module.LK_NBLCK = 1  # type: ignore[attr-defined]
    module.LK_UNLCK = 2  # type: ignore[attr-defined]
    module.locking = mock.Mock()  # type: ignore[attr-defined]
    return module


def _root(**changes: str) -> PairedRunRoot:
    values = {
        "plan_sha256": _DIGESTS[0],
        "case_sha256": _DIGESTS[1],
        "task_fingerprint": _DIGESTS[2],
        "normalized_item_sha256": _DIGESTS[3],
        "source_snapshot_sha256": _DIGESTS[4],
        "bridge_config_sha256": _DIGESTS[5],
        "executor_config_sha256": _DIGESTS[8],
        "execution_harness_sha256": _DIGESTS[14],
        "lifecycle_config_sha256": _DIGESTS[9],
        "signals_sha256": _DIGESTS[10],
        "runner_sha256": _DIGESTS[6],
        "runner_source_sha256": _DIGESTS[13],
        "pricing_sha256": _DIGESTS[7],
        "run_instance_nonce": _DIGESTS[15],
        "order": "AB",
        "baseline_route": "premium",
        "candidate_route": "local",
    }
    values.update(changes)
    return PairedRunRoot.build(**values)


def _complete(
    store: PairedExecutionStore,
    claim: object,
    outcome_record_id: str,
    index: int,
    *,
    binding: PairedOutcomeBinding | None = None,
):
    if binding is None:
        binding = store.binding_for(claim)  # type: ignore[arg-type]
    return store.complete(
        binding,
        outcome_record_id=outcome_record_id,
        route_receipt_id=f"route-{index}",
        route_receipt_sha256=_DIGESTS[10 + index],
        evidence_sha256=_DIGESTS[12 + index],
    )


def _claim_worker(
    run_dir: str,
    ready: object,
    start: object,
    results: object,
) -> None:
    store = PairedExecutionStore(run_dir)
    ready.put(True)  # type: ignore[attr-defined]
    if not start.wait(15.0):  # type: ignore[attr-defined]
        results.put("timeout")  # type: ignore[attr-defined]
        return
    try:
        store.claim("A")
    except PairedRunIndeterminateError:
        results.put("indeterminate")  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - surfaced to parent process
        results.put(f"error:{type(exc).__name__}:{exc}")  # type: ignore[attr-defined]
    else:
        results.put("won")  # type: ignore[attr-defined]


def _crash_after_claim_worker(run_dir: str) -> None:
    PairedExecutionStore(run_dir).claim("A")


def _fork_authority_worker(
    store: PairedExecutionStore,
    claim: PairedRunClaim,
    binding: PairedOutcomeBinding,
    results: object,
) -> None:
    observed: list[str] = []
    try:
        store.binding_for(claim)
    except PairedRunIndeterminateError:
        observed.append("binding-rejected")
    else:  # pragma: no cover - security regression surfaced to parent
        observed.append("binding-accepted")
    try:
        store.complete(
            binding,
            outcome_record_id="outcome-" + _DIGESTS[8],
            route_receipt_id="route-0",
            route_receipt_sha256=_DIGESTS[10],
            evidence_sha256=_DIGESTS[12],
        )
    except PairedRunIndeterminateError:
        observed.append("complete-rejected")
    else:  # pragma: no cover - security regression surfaced to parent
        observed.append("complete-accepted")
    results.put(tuple(observed))  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
