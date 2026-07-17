from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import unittest
from unittest import mock
from uuid import uuid4

import local_moe.assistant_bridge_workspace as workspace_module
from local_moe.assistant_bridge_workspace import (
    IgnoredPathRule,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    WorkspaceFile,
    apply_changeset,
    build_changeset,
    materialize_workspace,
    recover_workspace_transaction,
    snapshot_materialized,
    snapshot_workspace,
    workspace_write_capability,
)


class AssistantBridgeWorkspaceTests(unittest.TestCase):
    def test_unborn_git_and_ignored_scope_are_attested_without_real_git_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git(root, "init", "-q")
            (root / ".gitignore").write_text("private.txt\n", encoding="utf-8")
            (root / "visible.txt").write_text("visible", encoding="utf-8")
            (root / "private.txt").write_text("secret", encoding="utf-8")
            policy = WorkspaceScopePolicy()

            snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(snapshot, policy) as materialized:
                copied = {item.path for item in materialized.baseline_files}
                remotes = _git(materialized.root, "remote")

            self.assertEqual(snapshot.head_sha, "unborn")
            self.assertIn("visible.txt", copied)
            self.assertNotIn("private.txt", copied)
            self.assertEqual(remotes, "")
            self.assertTrue(snapshot.index_sha256)

    def test_declared_ignored_input_is_copied_but_cannot_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _initialize_repo(root)
            (root / ".gitignore").write_text("private.txt\n", encoding="utf-8")
            _git(root, "add", ".gitignore")
            _git(root, "commit", "-q", "-m", "ignore rule")
            (root / "private.txt").write_text("input", encoding="utf-8")
            policy = WorkspaceScopePolicy(
                ignored_paths=(IgnoredPathRule("private.txt", "input_only"),)
            )
            snapshot = snapshot_workspace(root, policy)

            with materialize_workspace(snapshot, policy) as materialized:
                (materialized.root / "private.txt").write_text(
                    "mutated", encoding="utf-8"
                )
                candidate = snapshot_materialized(materialized.root, policy)
                with self.assertRaisesRegex(WorkspaceSecurityError, "input_only"):
                    build_changeset(materialized.baseline_files, candidate)

            self.assertEqual(
                (root / "private.txt").read_text(encoding="utf-8"), "input"
            )

    def test_non_git_read_scope_is_complete_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            snapshot = snapshot_workspace(root, WorkspaceScopePolicy(max_files=2))
            self.assertFalse(snapshot.git_repository)
            self.assertEqual({item.path for item in snapshot.files}, {"a.txt", "b.txt"})
            with self.assertRaisesRegex(WorkspaceSecurityError, "file-count"):
                snapshot_workspace(root, WorkspaceScopePolicy(max_files=1))

    def test_symlink_escape_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target.txt").write_text("target", encoding="utf-8")
            os.symlink(root / "target.txt", root / "link.txt")
            with self.assertRaisesRegex(WorkspaceSecurityError, "symbolic"):
                snapshot_workspace(root, WorkspaceScopePolicy())

    def test_symlink_directory_and_workspace_root_are_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            real = root / "real"
            real.mkdir()
            (real / "file.txt").write_text("data", encoding="utf-8")
            os.symlink(real, root / "linked-directory", target_is_directory=True)
            with self.assertRaisesRegex(WorkspaceSecurityError, "symbolic"):
                snapshot_workspace(root, WorkspaceScopePolicy())

            alias = Path(tmp) / "source-alias"
            os.symlink(root, alias, target_is_directory=True)
            with self.assertRaisesRegex(WorkspaceSecurityError, "reparse"):
                snapshot_workspace(alias, WorkspaceScopePolicy())

    def test_candidate_is_reopened_and_staged_before_source_mutation(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(snapshot, policy) as materialized:
                candidate_path = materialized.root / "tracked.txt"
                candidate_path.write_text("candidate\n", encoding="utf-8")
                candidate = snapshot_materialized(materialized.root, policy)
                changes = build_changeset(materialized.baseline_files, candidate)
                outside = Path(tmp) / "outside.txt"
                outside.write_text("candidate\n", encoding="utf-8")
                candidate_path.unlink()
                os.symlink(outside, candidate_path)

                with self.assertRaisesRegex(WorkspaceSecurityError, "symbolic"):
                    apply_changeset(
                        source_snapshot=snapshot,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=Path(tmp) / "state",
                        transaction_id=uuid4().hex,
                    )

            self.assertEqual((root / "tracked.txt").read_text(), "initial\n")
            self.assertFalse(list(root.glob(".mymoe-*")))

    def test_candidate_digest_mode_and_bound_drift_precede_source_mutation(
        self,
    ) -> None:
        cases = ("digest", "mode", "bound")
        for case in cases:
            if case == "mode" and os.name == "nt":
                continue
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                _initialize_repo(root)
                policy = WorkspaceScopePolicy(max_file_bytes=64, max_total_bytes=4096)
                snapshot = snapshot_workspace(root, policy)
                with materialize_workspace(snapshot, policy) as materialized:
                    target = materialized.root / "tracked.txt"
                    target.write_text("candidate\n", encoding="utf-8")
                    candidate = snapshot_materialized(materialized.root, policy)
                    changes = build_changeset(materialized.baseline_files, candidate)
                    if case == "digest":
                        target.write_text("tampered!\n", encoding="utf-8")
                    elif case == "mode":
                        target.chmod(0o700)
                    else:
                        target.write_bytes(b"x" * 65)

                    with self.assertRaises(WorkspaceSecurityError):
                        apply_changeset(
                            source_snapshot=snapshot,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=changes,
                            policy=policy,
                            state_dir=Path(tmp) / "state",
                            transaction_id=uuid4().hex,
                        )

                self.assertEqual((root / "tracked.txt").read_text(), "initial\n")
                self.assertFalse(list(root.glob(".mymoe-*")))

    def test_changes_apply_with_cas_and_preserve_original_git_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            (root / "tracked.txt").write_text("staged\n", encoding="utf-8")
            _git(root, "add", "tracked.txt")
            index_before = _git_bytes(root, "ls-files", "--stage", "-z")
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)

            with materialize_workspace(snapshot, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n", encoding="utf-8"
                )
                (materialized.root / "new.txt").write_text("new\n", encoding="utf-8")
                candidate = snapshot_materialized(materialized.root, policy)
                changes = build_changeset(materialized.baseline_files, candidate)
                result = apply_changeset(
                    source_snapshot=snapshot,
                    candidate_root=materialized.root,
                    candidate_files=candidate,
                    changes=changes,
                    policy=policy,
                    state_dir=Path(tmp) / "state",
                    transaction_id=uuid4().hex,
                )

            self.assertEqual((root / "tracked.txt").read_text(), "candidate\n")
            self.assertEqual((root / "new.txt").read_text(), "new\n")
            self.assertEqual(
                _git_bytes(root, "ls-files", "--stage", "-z"), index_before
            )
            self.assertEqual(result.files, candidate)

    def test_source_drift_blocks_apply_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(snapshot, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n", encoding="utf-8"
                )
                candidate = snapshot_materialized(materialized.root, policy)
                changes = build_changeset(materialized.baseline_files, candidate)
                (root / "tracked.txt").write_text("concurrent\n", encoding="utf-8")
                with self.assertRaisesRegex(WorkspaceSecurityError, "changed"):
                    apply_changeset(
                        source_snapshot=snapshot,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=Path(tmp) / "state",
                        transaction_id=uuid4().hex,
                    )
            self.assertEqual((root / "tracked.txt").read_text(), "concurrent\n")

    def test_concurrent_name_reuse_is_preserved_and_requires_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex
            with materialize_workspace(snapshot, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n", encoding="utf-8"
                )
                candidate = snapshot_materialized(materialized.root, policy)
                changes = build_changeset(materialized.baseline_files, candidate)

                def concurrent_writer(target: Path) -> None:
                    target.write_text("concurrent\n", encoding="utf-8")

                with self.assertRaisesRegex(
                    WorkspaceSecurityError, "requires journal recovery"
                ):
                    apply_changeset(
                        source_snapshot=snapshot,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _test_hook_after_detach=concurrent_writer,
                    )

            self.assertEqual((root / "tracked.txt").read_text(), "concurrent\n")
            journal = state / f"transaction-{transaction_id}" / "journal.json"
            payload = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "recovery_required")

    def test_nested_candidate_directories_are_created_transactionally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(snapshot, policy) as materialized:
                nested = materialized.root / "new" / "nested"
                nested.mkdir(parents=True)
                (nested / "value.txt").write_text("value\n", encoding="utf-8")
                candidate = snapshot_materialized(materialized.root, policy)
                changes = build_changeset(materialized.baseline_files, candidate)
                apply_changeset(
                    source_snapshot=snapshot,
                    candidate_root=materialized.root,
                    candidate_files=candidate,
                    changes=changes,
                    policy=policy,
                    state_dir=Path(tmp) / "state",
                    transaction_id=uuid4().hex,
                )

            self.assertEqual(
                (root / "new" / "nested" / "value.txt").read_text(),
                "value\n",
            )

    def test_journal_flushes_file_before_replace_and_directory_after(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX fsync ordering probe")
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.json"
            events: list[str] = []
            real_fsync = os.fsync
            real_replace = os.replace

            def tracked_fsync(descriptor: int) -> None:
                metadata = os.fstat(descriptor)
                events.append(
                    "fsync-directory"
                    if stat.S_ISDIR(metadata.st_mode)
                    else "fsync-file"
                )
                real_fsync(descriptor)

            def tracked_replace(source: object, target: object) -> None:
                events.append("replace")
                real_replace(source, target)

            with (
                mock.patch.object(
                    workspace_module.os, "fsync", side_effect=tracked_fsync
                ),
                mock.patch.object(
                    workspace_module.os, "replace", side_effect=tracked_replace
                ),
            ):
                workspace_module._write_journal(journal, {"status": "prepared"})

            self.assertEqual(events[:3], ["fsync-file", "replace", "fsync-directory"])

    def test_write_capability_is_explicit_for_supported_platforms(self) -> None:
        capability = workspace_write_capability()
        if os.name in {"posix", "nt"}:
            self.assertTrue(capability.supported, capability.reason)
        self.assertTrue(capability.backend)

    def test_mode_only_drift_fails_compare_and_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            with materialize_workspace(snapshot, policy) as materialized:
                candidate = snapshot_materialized(materialized.root, policy)
                changes = build_changeset(materialized.baseline_files, candidate)
                (root / "tracked.txt").chmod(0o700)
                with self.assertRaisesRegex(WorkspaceSecurityError, "changed"):
                    apply_changeset(
                        source_snapshot=snapshot,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=Path(tmp) / "state",
                        transaction_id=uuid4().hex,
                    )

    def test_unicode_normalization_collision_is_rejected(self) -> None:
        with self.assertRaisesRegex(WorkspaceSecurityError, "collision"):
            WorkspaceScopePolicy(
                ignored_paths=(
                    IgnoredPathRule("caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt"),
                    IgnoredPathRule("cafe\N{COMBINING ACUTE ACCENT}.txt"),
                )
            )

    def test_stale_dead_lock_is_recovered_but_live_lock_is_not_stolen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            state.mkdir()
            resolved_root = root.resolve()
            lock = (
                state
                / f"workspace-{hashlib.sha256(str(resolved_root).encode()).hexdigest()[:24]}.lock"
            )
            lock.mkdir()
            (lock / "owner.json").write_text(
                json.dumps({"pid": 999_999_999, "created_at": 0}),
                encoding="utf-8",
            )
            old = time.time() - 3600
            os.utime(lock, (old, old))
            with materialize_workspace(snapshot, policy) as materialized:
                candidate = snapshot_materialized(materialized.root, policy)
                apply_changeset(
                    source_snapshot=snapshot,
                    candidate_root=materialized.root,
                    candidate_files=candidate,
                    changes=(),
                    policy=policy,
                    state_dir=state,
                    transaction_id=uuid4().hex,
                    lock_ttl_seconds=1,
                )

            lock.mkdir()
            (lock / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created_at": time.time()}),
                encoding="utf-8",
            )
            with materialize_workspace(
                snapshot_workspace(root, policy), policy
            ) as materialized:
                candidate = snapshot_materialized(materialized.root, policy)
                with self.assertRaisesRegex(WorkspaceSecurityError, "busy"):
                    apply_changeset(
                        source_snapshot=snapshot_workspace(root, policy),
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=(),
                        policy=policy,
                        state_dir=state,
                        transaction_id=uuid4().hex,
                    )

    def test_crash_journal_recovers_only_the_recorded_applied_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            target = root / "tracked.txt"
            before_data = b"before\n"
            after_data = b"after\n"
            target.write_bytes(after_data)
            transaction_id = uuid4().hex
            transaction = Path(tmp) / "state" / f"transaction-{transaction_id}"
            backups = transaction / "backups"
            backups.mkdir(parents=True)
            (backups / "00000000.bin").write_bytes(before_data)
            before = WorkspaceFile(
                "tracked.txt",
                "file",
                hashlib.sha256(before_data).hexdigest(),
                len(before_data),
                0o644,
            )
            after = WorkspaceFile(
                "tracked.txt",
                "file",
                hashlib.sha256(after_data).hexdigest(),
                len(after_data),
                0o644,
            )
            (transaction / "journal.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "transaction_id": transaction_id,
                        "source_root_sha256": hashlib.sha256(
                            str(root.resolve()).encode()
                        ).hexdigest(),
                        "source_fingerprint": "0" * 64,
                        "status": "applying",
                        "changes": [
                            {
                                "path": "tracked.txt",
                                "before": before.payload(),
                                "after": after.payload(),
                                "backup": "00000000.bin",
                                "backup_sha256": hashlib.sha256(
                                    before_data
                                ).hexdigest(),
                                "status": "applied",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            recover_workspace_transaction(
                state_dir=Path(tmp) / "state",
                transaction_id=transaction_id,
                source_root=root,
            )

            self.assertEqual(target.read_bytes(), before_data)
            self.assertFalse(transaction.exists())

    def test_fault_after_replace_or_unlink_is_recoverable_from_mutating_state(
        self,
    ) -> None:
        for delete in (False, True):
            with self.subTest(delete=delete), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                _initialize_repo(root)
                policy = WorkspaceScopePolicy()
                snapshot = snapshot_workspace(root, policy)
                transaction_id = uuid4().hex
                state = Path(tmp) / "state"
                with materialize_workspace(snapshot, policy) as materialized:
                    candidate_path = materialized.root / "tracked.txt"
                    if delete:
                        candidate_path.unlink()
                    else:
                        candidate_path.write_text("candidate\n", encoding="utf-8")
                    candidate = snapshot_materialized(materialized.root, policy)
                    changes = build_changeset(materialized.baseline_files, candidate)
                    with self.assertRaises(RuntimeError):
                        apply_changeset(
                            source_snapshot=snapshot,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=changes,
                            policy=policy,
                            state_dir=state,
                            transaction_id=transaction_id,
                            _fault_after_mutation=0,
                        )

                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                )

                self.assertEqual(
                    (root / "tracked.txt").read_text(encoding="utf-8"),
                    "initial\n",
                )


def _initialize_repo(root: Path) -> None:
    _git(root, "init", "-q")
    (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(
        root,
        "-c",
        "user.name=Antonio Antenore",
        "-c",
        "user.email=ant_ant95@hotmail.it",
        "commit",
        "-q",
        "-m",
        "initial",
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout


if __name__ == "__main__":
    unittest.main()
