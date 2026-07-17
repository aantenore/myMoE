from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from uuid import uuid4

from local_moe.assistant_bridge_workspace import (
    IgnoredPathRule,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    apply_changeset,
    build_changeset,
    materialize_workspace,
    snapshot_materialized,
    snapshot_workspace,
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
