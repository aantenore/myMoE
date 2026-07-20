from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from uuid import uuid4

import local_moe.assistant_bridge_workspace as workspace_module
import local_moe.assistant_bridge_process as process_module
from local_moe.assistant_bridge_runtime import fingerprint_environment
from local_moe.assistant_bridge_workspace import (
    GitIdentity,
    IgnoredPathRule,
    WorkspaceScopePolicy,
    WorkspaceSecurityError,
    WorkspaceFile,
    apply_changeset,
    build_changeset,
    finalize_committed_workspace_transaction,
    finalize_recovered_workspace_transaction,
    materialize_workspace,
    recover_workspace_transaction,
    snapshot_materialized,
    snapshot_workspace,
    workspace_write_capability,
)


class AssistantBridgeWorkspaceTests(unittest.TestCase):
    def test_raw_workspace_artifacts_request_binary_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "artifact.bin"
            value = b"first\r\nsecond\nthird\x1atail\r\n"
            real_open = os.open
            real_binary = getattr(os, "O_BINARY", 0)
            binary_sentinel = 1 << 29
            seen: list[int] = []

            def binary_open(
                path: object,
                flags: int,
                *args: object,
                **kwargs: object,
            ) -> int:
                seen.append(flags)
                portable_flags = (flags & ~binary_sentinel) | real_binary
                return real_open(path, portable_flags, *args, **kwargs)

            with (
                mock.patch.object(
                    workspace_module.os,
                    "O_BINARY",
                    binary_sentinel,
                    create=True,
                ),
                mock.patch.object(
                    workspace_module.os,
                    "open",
                    side_effect=binary_open,
                ),
                mock.patch.object(workspace_module, "_fsync_directory"),
            ):
                workspace_module._durable_write_new(target, value, 0o600)

            self.assertEqual(target.read_bytes(), value)
            self.assertTrue(seen)
            self.assertTrue(all(flags & binary_sentinel for flags in seen))

    def test_windows_pid_probe_never_uses_os_kill(self) -> None:
        with (
            mock.patch.object(process_module.os, "name", "nt"),
            mock.patch.object(
                process_module,
                "_windows_process_is_alive",
                side_effect=(False, True),
            ) as windows_probe,
            mock.patch.object(process_module.os, "kill") as kill,
        ):
            self.assertFalse(process_module.process_is_alive(999_999_999))
            self.assertTrue(process_module.process_is_alive(os.getpid()))

        self.assertEqual(windows_probe.call_count, 2)
        kill.assert_not_called()

    def test_windows_readonly_materialization_uses_the_original_write_handle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "readonly.bin"
            value = b"first\r\nsecond\n"
            events: list[str] = []
            real_chmod = Path.chmod
            real_fsync = workspace_module.os.fsync

            def observed_chmod(path: Path, mode: int) -> None:
                events.append("chmod")
                real_chmod(path, mode)

            def observed_fsync(descriptor: int) -> None:
                events.append("fsync")
                real_fsync(descriptor)

            try:
                with (
                    mock.patch.object(workspace_module.os, "name", "nt"),
                    mock.patch.object(
                        workspace_module.os,
                        "fchmod",
                        None,
                        create=True,
                    ),
                    mock.patch.object(Path, "chmod", new=observed_chmod),
                    mock.patch.object(
                        workspace_module.os,
                        "fsync",
                        side_effect=observed_fsync,
                    ),
                    mock.patch.object(workspace_module, "_fsync_directory"),
                    mock.patch.object(
                        workspace_module,
                        "_windows_flush_path",
                    ) as reopen_flush,
                ):
                    workspace_module._durable_write_new(target, value, 0o444)

                self.assertEqual(target.read_bytes(), value)
                reopen_flush.assert_not_called()
                self.assertLess(events.index("chmod"), events.index("fsync"))
            finally:
                if target.exists():
                    target.chmod(0o600)

    def test_deleted_candidate_restore_is_adopted_after_identity_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            target = root / "tracked.txt"
            target.write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex
            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").unlink()
                candidate = materialized.snapshot()
                with self.assertRaises(RuntimeError):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _fault_after_mutation=0,
                    )

            self.assertFalse(target.exists())
            with self.assertRaises(RuntimeError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    _fault_after_rollback_step="restore_before_identity_persist",
                )
            self.assertFalse(target.exists())
            recover_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "source\n")
            self.assertFalse((state / f"transaction-{transaction_id}").exists())
            self.assertFalse(any(root.rglob(".mymoe-*")))

    def test_install_journal_recovers_every_durable_boundary(self) -> None:
        steps = (
            "before_identity_persist",
            "during_temp_write",
            "after_temp_creation",
            "after_link",
            "after_temp_unlink",
            "after_mode_apply",
            "before_caller_identity_persist",
        )
        for step in steps:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                (root / "tracked.txt").write_text("source\n", encoding="utf-8")
                policy = WorkspaceScopePolicy()
                source = snapshot_workspace(root, policy)
                state = Path(tmp) / "state"
                transaction_id = uuid4().hex
                with materialize_workspace(source, policy) as materialized:
                    (materialized.root / "tracked.txt").write_text(
                        "candidate\n",
                        encoding="utf-8",
                    )
                    candidate = materialized.snapshot()
                    changes = build_changeset(source.files, candidate)
                    with self.assertRaises(RuntimeError):
                        apply_changeset(
                            source_snapshot=source,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=changes,
                            policy=policy,
                            state_dir=state,
                            transaction_id=transaction_id,
                            _fault_after_install_step=step,
                        )

                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                )
                self.assertEqual(
                    (root / "tracked.txt").read_text(encoding="utf-8"),
                    "source\n",
                )
                self.assertFalse(
                    any(root.rglob(".mymoe-install-*")),
                )

    def test_windows_mode_fallback_and_readonly_cleanup_preserve_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("value\n", encoding="utf-8")
            identity = workspace_module._regular_file_identity(target)
            events: list[str] = []
            real_chmod = Path.chmod
            real_fsync = workspace_module.os.fsync

            def open_writable(
                path: Path,
                *,
                directory: bool,
                writable: bool = False,
            ) -> int:
                self.assertEqual(path, target)
                self.assertFalse(directory)
                self.assertTrue(writable)
                events.append("open_writable")
                return os.open(path, os.O_WRONLY)

            def observed_chmod(path: Path, mode: int) -> None:
                events.append("chmod_readonly")
                real_chmod(path, mode)

            def observed_fsync(descriptor: int) -> None:
                events.append("flush_retained_handle")
                real_fsync(descriptor)

            def directory_flush(path: Path, *, directory: bool) -> None:
                self.assertEqual(path, target.parent)
                self.assertTrue(directory)
                events.append("flush_directory")

            with (
                mock.patch.object(workspace_module.os, "name", "nt"),
                mock.patch.object(
                    workspace_module.os,
                    "fchmod",
                    None,
                    create=True,
                ),
                mock.patch.object(
                    workspace_module,
                    "_windows_open_fd",
                    side_effect=open_writable,
                ),
                mock.patch.object(Path, "chmod", new=observed_chmod),
                mock.patch.object(
                    workspace_module.os,
                    "fsync",
                    side_effect=observed_fsync,
                ),
                mock.patch.object(
                    workspace_module,
                    "_windows_flush_path",
                    side_effect=directory_flush,
                ),
            ):
                self.assertTrue(
                    workspace_module._private_artifact_mode_is_valid(0o666)
                )
                self.assertFalse(
                    workspace_module._private_artifact_mode_is_valid(0o444)
                )
                workspace_module._set_file_mode_durable(target, 0o444, identity)
                self.assertEqual(
                    workspace_module._regular_file_identity(target),
                    identity,
                )

            self.assertLess(events.index("open_writable"), events.index("chmod_readonly"))
            self.assertLess(
                events.index("chmod_readonly"),
                events.index("flush_retained_handle"),
            )
            self.assertEqual(events[-1], "flush_directory")

            failure_target = root / "failure.txt"
            failure_target.write_text("value\n", encoding="utf-8")
            failure_identity = workspace_module._regular_file_identity(failure_target)
            original_mode = stat.S_IMODE(failure_target.stat().st_mode)
            with (
                mock.patch.object(workspace_module.os, "name", "nt"),
                mock.patch.object(
                    workspace_module.os,
                    "fchmod",
                    None,
                    create=True,
                ),
                mock.patch.object(
                    workspace_module,
                    "_windows_open_fd",
                    side_effect=PermissionError("writable handle unavailable"),
                ),
                self.assertRaises(PermissionError),
            ):
                workspace_module._set_file_mode_durable(
                    failure_target,
                    0o444,
                    failure_identity,
                )
            self.assertEqual(
                stat.S_IMODE(failure_target.stat().st_mode),
                original_mode,
            )

            with (
                mock.patch.object(workspace_module.os, "name", "nt"),
                mock.patch.object(
                    workspace_module.os,
                    "fchmod",
                    None,
                    create=True,
                ),
                mock.patch.object(
                    workspace_module,
                    "_windows_flush_path",
                ) as cleanup_flush,
            ):
                workspace_module._unlink_recovery_artifact(target, identity)

            self.assertFalse(target.exists())
            self.assertGreaterEqual(cleanup_flush.call_count, 2)

    def test_readonly_source_replace_and_delete_use_owned_cleanup(self) -> None:
        for delete in (False, True):
            with self.subTest(delete=delete), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                target = root / "tracked.txt"
                target.write_text("source\n", encoding="utf-8")
                target.chmod(0o444)
                policy = WorkspaceScopePolicy()
                source = snapshot_workspace(root, policy)
                state = Path(tmp) / "state"
                with materialize_workspace(source, policy) as materialized:
                    candidate_target = materialized.root / "tracked.txt"
                    if delete:
                        candidate_target.chmod(0o644)
                        candidate_target.unlink()
                    else:
                        candidate_target.chmod(0o644)
                        candidate_target.write_text("candidate\n", encoding="utf-8")
                    candidate = materialized.snapshot()
                    with mock.patch.object(
                        workspace_module,
                        "_unlink_recovery_artifact",
                        wraps=workspace_module._unlink_recovery_artifact,
                    ) as cleanup:
                        apply_changeset(
                            source_snapshot=source,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=build_changeset(source.files, candidate),
                            policy=policy,
                            state_dir=state,
                            transaction_id=uuid4().hex,
                        )

                self.assertTrue(
                    any(
                        Path(call.args[0]).name.startswith(".mymoe-before-")
                        for call in cleanup.call_args_list
                    )
                )
                if delete:
                    self.assertFalse(target.exists())
                else:
                    self.assertEqual(
                        target.read_text(encoding="utf-8"),
                        "candidate\n",
                    )

    def test_preinstall_detach_restore_adopts_after_second_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            target = root / "tracked.txt"
            target.write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex

            def crash_after_detach(_: Path) -> None:
                raise workspace_module._SimulatedTransactionCrash(
                    "simulated crash after quarantine detach"
                )

            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
                with self.assertRaises(RuntimeError):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _test_hook_after_detach=crash_after_detach,
                    )

            self.assertFalse(target.exists())
            with self.assertRaises(RuntimeError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    _fault_after_rollback_step="restore_before_identity_persist",
                )
            recover_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "source\n")
            self.assertFalse(any(root.rglob(".mymoe-*")))

    def test_deletion_detach_restore_adopts_after_second_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            target = root / "tracked.txt"
            target.write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex

            def crash_after_detach(_: Path) -> None:
                raise workspace_module._SimulatedTransactionCrash(
                    "simulated crash after quarantine detach"
                )

            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").unlink()
                candidate = materialized.snapshot()
                with self.assertRaises(RuntimeError):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _test_hook_after_detach=crash_after_detach,
                    )

            self.assertFalse(target.exists())
            with self.assertRaises(RuntimeError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    _fault_after_rollback_step="restore_before_identity_persist",
                )
            recover_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "source\n")
            self.assertFalse(any(root.rglob(".mymoe-*")))

    def test_deletion_restore_adoption_rejects_ambiguous_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            target = root / "tracked.txt"
            target.write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex

            def crash_after_detach(_: Path) -> None:
                raise workspace_module._SimulatedTransactionCrash(
                    "simulated crash after quarantine detach"
                )

            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").unlink()
                candidate = materialized.snapshot()
                with self.assertRaises(RuntimeError):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _test_hook_after_detach=crash_after_detach,
                    )
            with self.assertRaises(RuntimeError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    _fault_after_rollback_step="restore_before_identity_persist",
                )

            rollback = root / f".mymoe-rollback-{transaction_id}-00000000"
            rollback.write_text("ambiguous\n", encoding="utf-8")
            with self.assertRaises(WorkspaceSecurityError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                )
            self.assertEqual(rollback.read_text(encoding="utf-8"), "ambiguous\n")
            self.assertFalse(target.exists())

    def test_install_link_failure_and_readonly_cleanup_are_recoverable(self) -> None:
        for failure in ("link", "unlink"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                (root / "tracked.txt").write_text("source\n", encoding="utf-8")
                policy = WorkspaceScopePolicy()
                source = snapshot_workspace(root, policy)
                with materialize_workspace(source, policy) as materialized:
                    (materialized.root / "tracked.txt").write_text(
                        "candidate\n",
                        encoding="utf-8",
                    )
                    candidate = materialized.snapshot()
                    changes = build_changeset(source.files, candidate)
                    if failure == "link":
                        original_link = workspace_module.os.link
                        link_failures = 0

                        def fail_install_link_once(
                            source_path: Path,
                            target_path: Path,
                            *args: object,
                            **kwargs: object,
                        ) -> None:
                            nonlocal link_failures
                            if (
                                Path(source_path).name.startswith(".mymoe-install-")
                                and link_failures == 0
                            ):
                                link_failures += 1
                                raise OSError(errno.EXDEV, "cross-device")
                            original_link(source_path, target_path, *args, **kwargs)

                        patcher = mock.patch.object(
                            workspace_module.os,
                            "link",
                            side_effect=fail_install_link_once,
                        )
                        capability = mock.patch.object(
                            workspace_module,
                            "_require_secure_apply_capabilities",
                            return_value=None,
                        )
                    else:
                        original = workspace_module._unlink_recovery_artifact
                        calls = 0

                        def fail_once(path: Path, identity: dict[str, int]) -> None:
                            nonlocal calls
                            calls += 1
                            if calls == 1:
                                raise PermissionError("simulated readonly")
                            original(path, identity)

                        patcher = mock.patch.object(
                            workspace_module,
                            "_unlink_recovery_artifact",
                            side_effect=fail_once,
                        )
                        capability = mock.patch.object(
                            workspace_module,
                            "_require_secure_apply_capabilities",
                            wraps=workspace_module._require_secure_apply_capabilities,
                        )
                    with (
                        patcher,
                        capability,
                        self.assertRaises((OSError, PermissionError)),
                    ):
                        apply_changeset(
                            source_snapshot=source,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=changes,
                            policy=policy,
                            state_dir=Path(tmp) / "state",
                            transaction_id=uuid4().hex,
                        )
                self.assertEqual(
                    (root / "tracked.txt").read_text(encoding="utf-8"),
                    "source\n",
                )

    def test_recovery_rejects_swapped_ancestor_without_external_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "tracked.txt").write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex
            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "nested" / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
                changes = build_changeset(source.files, candidate)
                with self.assertRaises(RuntimeError):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _fault_after_mutation=0,
                    )

            with self.assertRaises(RuntimeError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    _fault_after_rollback_step="detach_action",
                )
            parked = Path(tmp) / "parked"
            nested.rename(parked)
            attacker = Path(tmp) / "attacker"
            attacker.mkdir()
            sentinel = attacker / "tracked.txt"
            sentinel.write_text("external\n", encoding="utf-8")
            try:
                nested.symlink_to(attacker, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            with self.assertRaises(WorkspaceSecurityError):
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "external\n")
            nested.unlink()
            parked.rename(nested)
            recover_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
            )
            self.assertEqual(
                (nested / "tracked.txt").read_text(encoding="utf-8"),
                "source\n",
            )

    def test_journal_capacity_exact_boundary_recovers_and_plus_one_rejects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "tracked.txt").write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
                changes = build_changeset(source.files, candidate)
                sizes: list[int] = []

                def capture(payload: dict[str, object]) -> None:
                    sizes.append(workspace_module._journal_worst_case_size(payload))
                    raise WorkspaceSecurityError("capacity captured")

                with (
                    mock.patch.object(
                        workspace_module,
                        "_assert_journal_capacity",
                        side_effect=capture,
                    ),
                    self.assertRaisesRegex(WorkspaceSecurityError, "captured"),
                ):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=state,
                        transaction_id=uuid4().hex,
                    )
                self.assertEqual(len(sizes), 1)
                exact = sizes[0]
                accepted_id = uuid4().hex
                with (
                    mock.patch.object(
                        workspace_module,
                        "_MAX_JOURNAL_BYTES",
                        exact,
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=state,
                        transaction_id=accepted_id,
                        _fault_after_mutation=0,
                    )
                with mock.patch.object(
                    workspace_module,
                    "_MAX_JOURNAL_BYTES",
                    exact,
                ):
                    recover_workspace_transaction(
                        state_dir=state,
                        transaction_id=accepted_id,
                        source_root=root,
                    )
                rejected_id = uuid4().hex
                with (
                    mock.patch.object(
                        workspace_module,
                        "_MAX_JOURNAL_BYTES",
                        exact - 1,
                    ),
                    self.assertRaisesRegex(WorkspaceSecurityError, "exceeds"),
                ):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=changes,
                        policy=policy,
                        state_dir=state,
                        transaction_id=rejected_id,
                    )
            self.assertEqual(
                (root / "tracked.txt").read_text(encoding="utf-8"),
                "source\n",
            )
            self.assertFalse((state / f"transaction-{rejected_id}").exists())
            self.assertFalse(any(root.rglob(".mymoe-*")))

    def test_committed_cleanup_marker_retries_every_boundary(self) -> None:
        for step in (
            "during_marker_write",
            "after_marker",
            "after_rename",
            "before_rmtree",
            "after_rmtree",
        ):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                (root / "tracked.txt").write_text("source\n", encoding="utf-8")
                policy = WorkspaceScopePolicy()
                source = snapshot_workspace(root, policy)
                state = Path(tmp) / "state"
                transaction_id = uuid4().hex
                with materialize_workspace(source, policy) as materialized:
                    (materialized.root / "tracked.txt").write_text(
                        "candidate\n",
                        encoding="utf-8",
                    )
                    candidate = materialized.snapshot()
                    result = apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        retain_committed_journal=True,
                        on_committed=lambda _: None,
                    )
                with self.assertRaises(RuntimeError):
                    finalize_committed_workspace_transaction(
                        state_dir=state,
                        transaction_id=transaction_id,
                        source_root=root,
                        expected_source_fingerprint=source.fingerprint,
                        _fault_after_cleanup_step=step,
                    )
                finalize_committed_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                )
                self.assertEqual(
                    (root / "tracked.txt").read_text(encoding="utf-8"),
                    "candidate\n",
                )
                self.assertEqual(
                    result.fingerprint, snapshot_workspace(root, policy).fingerprint
                )
                self.assertFalse((state / f"transaction-{transaction_id}").exists())
                self.assertFalse((state / f"cleanup-{transaction_id}").exists())
                self.assertFalse((state / f"cleanup-{transaction_id}.json").exists())

    def test_cleanup_rejects_unbound_directory_and_recovered_marker_only_retries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "tracked.txt").write_text("source\n", encoding="utf-8")
            source = snapshot_workspace(root, WorkspaceScopePolicy())
            state = Path(tmp) / "state"
            state.mkdir()
            transaction_id = uuid4().hex
            cleanup = state / f"cleanup-{transaction_id}"
            cleanup.mkdir()
            sentinel = cleanup / "sentinel.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            with self.assertRaises(WorkspaceSecurityError):
                finalize_committed_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "tracked.txt").write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex
            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
                with self.assertRaises(RuntimeError):
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        _fault_after_mutation=0,
                    )
            recover_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
                expected_source_fingerprint=source.fingerprint,
                retain_recovered_journal=True,
            )
            with self.assertRaises(RuntimeError):
                finalize_recovered_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                    _fault_after_cleanup_step="after_rmtree",
                )
            self.assertTrue((state / f"cleanup-{transaction_id}.json").exists())
            finalize_recovered_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
                expected_source_fingerprint=source.fingerprint,
            )
            self.assertFalse((state / f"cleanup-{transaction_id}.json").exists())

    def test_recovered_cleanup_marker_retries_every_boundary(self) -> None:
        for step in (
            "during_marker_write",
            "after_marker",
            "after_rename",
            "before_rmtree",
            "after_rmtree",
        ):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                (root / "tracked.txt").write_text("source\n", encoding="utf-8")
                policy = WorkspaceScopePolicy()
                source = snapshot_workspace(root, policy)
                state = Path(tmp) / "state"
                transaction_id = uuid4().hex
                with materialize_workspace(source, policy) as materialized:
                    (materialized.root / "tracked.txt").write_text(
                        "candidate\n",
                        encoding="utf-8",
                    )
                    candidate = materialized.snapshot()
                    with self.assertRaises(RuntimeError):
                        apply_changeset(
                            source_snapshot=source,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=build_changeset(source.files, candidate),
                            policy=policy,
                            state_dir=state,
                            transaction_id=transaction_id,
                            _fault_after_mutation=0,
                        )
                recover_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                    retain_recovered_journal=True,
                )
                with self.assertRaises(RuntimeError):
                    finalize_recovered_workspace_transaction(
                        state_dir=state,
                        transaction_id=transaction_id,
                        source_root=root,
                        expected_source_fingerprint=source.fingerprint,
                        _fault_after_cleanup_step=step,
                    )
                finalize_recovered_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                )
                self.assertEqual(
                    (root / "tracked.txt").read_text(encoding="utf-8"),
                    "source\n",
                )
                self.assertFalse((state / f"cleanup-{transaction_id}.json").exists())

    def test_cleanup_marker_tampering_and_ambiguous_state_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "tracked.txt").write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex
            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
                apply_changeset(
                    source_snapshot=source,
                    candidate_root=materialized.root,
                    candidate_files=candidate,
                    changes=build_changeset(source.files, candidate),
                    policy=policy,
                    state_dir=state,
                    transaction_id=transaction_id,
                    retain_committed_journal=True,
                    on_committed=lambda _: None,
                )
            with self.assertRaises(RuntimeError):
                finalize_committed_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                    _fault_after_cleanup_step="after_rename",
                )
            cleanup = state / f"cleanup-{transaction_id}"
            marker = state / f"cleanup-{transaction_id}.json"
            sentinel = cleanup / "sentinel.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            transaction = state / f"transaction-{transaction_id}"
            transaction.mkdir()
            with self.assertRaises(WorkspaceSecurityError):
                finalize_committed_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                )
            transaction.rmdir()
            payload = json.loads(marker.read_text(encoding="utf-8"))
            payload["journal_sha256"] = "0" * 64
            marker.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            marker.chmod(0o600)
            with self.assertRaises(WorkspaceSecurityError):
                finalize_committed_workspace_transaction(
                    state_dir=state,
                    transaction_id=transaction_id,
                    source_root=root,
                    expected_source_fingerprint=source.fingerprint,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_cleanup_marker_mode_and_canonical_encoding_fail_closed(self) -> None:
        for tamper in ("mode", "canonical"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                (root / "tracked.txt").write_text("source\n", encoding="utf-8")
                policy = WorkspaceScopePolicy()
                source = snapshot_workspace(root, policy)
                state = Path(tmp) / "state"
                transaction_id = uuid4().hex
                with materialize_workspace(source, policy) as materialized:
                    (materialized.root / "tracked.txt").write_text(
                        "candidate\n",
                        encoding="utf-8",
                    )
                    candidate = materialized.snapshot()
                    apply_changeset(
                        source_snapshot=source,
                        candidate_root=materialized.root,
                        candidate_files=candidate,
                        changes=build_changeset(source.files, candidate),
                        policy=policy,
                        state_dir=state,
                        transaction_id=transaction_id,
                        retain_committed_journal=True,
                        on_committed=lambda _: None,
                    )
                with self.assertRaises(RuntimeError):
                    finalize_committed_workspace_transaction(
                        state_dir=state,
                        transaction_id=transaction_id,
                        source_root=root,
                        expected_source_fingerprint=source.fingerprint,
                        _fault_after_cleanup_step="after_marker",
                    )
                transaction = state / f"transaction-{transaction_id}"
                sentinel = transaction / "sentinel.txt"
                sentinel.write_text("keep\n", encoding="utf-8")
                marker = state / f"cleanup-{transaction_id}.json"
                if tamper == "mode":
                    marker.chmod(0o444 if os.name == "nt" else 0o644)
                else:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    marker.write_text(
                        json.dumps(payload, indent=2) + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaises(WorkspaceSecurityError):
                    finalize_committed_workspace_transaction(
                        state_dir=state,
                        transaction_id=transaction_id,
                        source_root=root,
                        expected_source_fingerprint=source.fingerprint,
                    )
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_partial_cleanup_marker_temp_from_dead_writer_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "tracked.txt").write_text("source\n", encoding="utf-8")
            policy = WorkspaceScopePolicy()
            source = snapshot_workspace(root, policy)
            state = Path(tmp) / "state"
            transaction_id = uuid4().hex
            with materialize_workspace(source, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
                apply_changeset(
                    source_snapshot=source,
                    candidate_root=materialized.root,
                    candidate_files=candidate,
                    changes=build_changeset(source.files, candidate),
                    policy=policy,
                    state_dir=state,
                    transaction_id=transaction_id,
                    retain_committed_journal=True,
                    on_committed=lambda _: None,
                )

            abandoned = state / (
                f".cleanup-{transaction_id}.json.{uuid4().hex}.tmp"
            )
            abandoned.write_bytes(b"{")
            abandoned.chmod(0o600)
            finalize_committed_workspace_transaction(
                state_dir=state,
                transaction_id=transaction_id,
                source_root=root,
                expected_source_fingerprint=source.fingerprint,
            )

            self.assertTrue(abandoned.exists())
            self.assertFalse((state / f"transaction-{transaction_id}").exists())
            self.assertEqual(
                (root / "tracked.txt").read_text(encoding="utf-8"),
                "candidate\n",
            )

    def test_git_environment_is_minimal_and_runtime_compatible(self) -> None:
        poisoned = {
            "HOME": "/untrusted/home",
            "XDG_CONFIG_HOME": "/untrusted/config",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "/untrusted/helper",
        }
        with mock.patch.dict(os.environ, poisoned, clear=False):
            environment = workspace_module._sanitized_git_environment(
                commit_identity=None,
                executable_path=Path(sys.executable),
            )

        for name in poisoned:
            self.assertNotIn(name, environment)
        for name in (
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
        ):
            self.assertNotIn(name, environment)
        self.assertTrue(fingerprint_environment(environment).sha256)

    def test_synthetic_git_identity_is_validated(self) -> None:
        self.assertEqual(
            WorkspaceScopePolicy().synthetic_git_identity,
            GitIdentity(),
        )
        for kwargs in (
            {"name": ""},
            {"name": " invalid"},
            {"name": "invalid\nname"},
            {"email": "missing-at"},
            {"email": "two@@localhost"},
            {"email": "space @localhost"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(WorkspaceSecurityError):
                GitIdentity(**kwargs)

    def test_configured_identity_is_limited_to_synthetic_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            source_head = _git(root, "rev-parse", "HEAD")
            source_identity = _git(
                root,
                "log",
                "-1",
                "--format=%an%x00%ae%x00%cn%x00%ce",
            )
            configured = GitIdentity(
                name="Configured Materializer",
                email="configured@localhost",
            )
            policy = WorkspaceScopePolicy(synthetic_git_identity=configured)
            snapshot = snapshot_workspace(root, policy)

            with materialize_workspace(snapshot, policy) as materialized:
                synthetic_identity = _git(
                    materialized.root,
                    "log",
                    "-1",
                    "--format=%an%x00%ae%x00%cn%x00%ce",
                )

            self.assertEqual(
                synthetic_identity.split("\x00"),
                [
                    configured.name,
                    configured.email,
                    configured.name,
                    configured.email,
                ],
            )
            self.assertEqual(_git(root, "rev-parse", "HEAD"), source_head)
            self.assertEqual(
                _git(
                    root,
                    "log",
                    "-1",
                    "--format=%an%x00%ae%x00%cn%x00%ce",
                ),
                source_identity,
            )

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

    def test_non_git_snapshot_never_resolves_or_launches_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "value.txt").write_text("value", encoding="utf-8")
            with mock.patch.object(
                workspace_module,
                "_resolve_trusted_git_identity",
                side_effect=AssertionError("Git resolution must be skipped"),
            ):
                snapshot = snapshot_workspace(root, WorkspaceScopePolicy())

        self.assertFalse(snapshot.git_repository)

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

    def test_tracked_deletion_commits_without_mutating_the_git_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)
            index_before = _git_bytes(root, "ls-files", "--stage", "-z")

            with materialize_workspace(snapshot, policy) as materialized:
                (materialized.root / "tracked.txt").unlink()
                candidate = materialized.snapshot()
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

            self.assertFalse((root / "tracked.txt").exists())
            self.assertEqual(
                _git_bytes(root, "ls-files", "--stage", "-z"),
                index_before,
            )
            deleted = {item.path: item for item in result.files}["tracked.txt"]
            self.assertEqual(deleted.kind, "missing")

    def test_initially_missing_tracked_file_can_be_restored_transactionally(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            (root / "tracked.txt").unlink()
            policy = WorkspaceScopePolicy()
            snapshot = snapshot_workspace(root, policy)

            with materialize_workspace(snapshot, policy) as materialized:
                self.assertEqual(
                    {item.path: item for item in materialized.baseline_files}[
                        "tracked.txt"
                    ].kind,
                    "missing",
                )
                (materialized.root / "tracked.txt").write_text(
                    "restored\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
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

            self.assertEqual((root / "tracked.txt").read_text(), "restored\n")
            restored = {item.path: item for item in result.files}["tracked.txt"]
            self.assertEqual(restored.kind, "file")

    def test_missing_declared_ignored_path_allows_unrelated_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            policy = WorkspaceScopePolicy(
                ignored_paths=(IgnoredPathRule("optional.secret", "input_only"),)
            )
            snapshot = snapshot_workspace(root, policy)

            with materialize_workspace(snapshot, policy) as materialized:
                (materialized.root / "tracked.txt").write_text(
                    "candidate\n",
                    encoding="utf-8",
                )
                candidate = materialized.snapshot()
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
            optional = {item.path: item for item in result.files}["optional.secret"]
            self.assertEqual(optional.kind, "missing")

    def test_git_head_and_index_drift_roll_back_but_preserve_concurrent_git_state(
        self,
    ) -> None:
        for drift in ("head", "index"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                root.mkdir()
                _initialize_repo(root)
                policy = WorkspaceScopePolicy()
                snapshot = snapshot_workspace(root, policy)
                tracked_blob = _git(root, "rev-parse", "HEAD:tracked.txt")
                state = Path(tmp) / "state"
                with materialize_workspace(snapshot, policy) as materialized:
                    (materialized.root / "tracked.txt").write_text(
                        "candidate\n",
                        encoding="utf-8",
                    )
                    candidate = materialized.snapshot()
                    changes = build_changeset(materialized.baseline_files, candidate)

                    def mutate_git_authority(_target: Path) -> None:
                        if drift == "index":
                            _git(
                                root,
                                "update-index",
                                "--cacheinfo",
                                "100755",
                                tracked_blob,
                                "tracked.txt",
                            )
                        else:
                            _git(
                                root,
                                "-c",
                                "user.name=Antonio Antenore",
                                "-c",
                                "user.email=ant_ant95@hotmail.it",
                                "commit",
                                "-q",
                                "--allow-empty",
                                "-m",
                                "concurrent",
                            )

                    with self.assertRaisesRegex(
                        WorkspaceSecurityError,
                        "HEAD or index changed",
                    ):
                        apply_changeset(
                            source_snapshot=snapshot,
                            candidate_root=materialized.root,
                            candidate_files=candidate,
                            changes=changes,
                            policy=policy,
                            state_dir=state,
                            transaction_id=uuid4().hex,
                            _test_hook_after_detach=mutate_git_authority,
                        )

                self.assertEqual(
                    (root / "tracked.txt").read_text(encoding="utf-8"),
                    "initial\n",
                )
                after = snapshot_workspace(root, policy)
                if drift == "head":
                    self.assertNotEqual(after.head_sha, snapshot.head_sha)
                    self.assertEqual(after.index_sha256, snapshot.index_sha256)
                else:
                    self.assertEqual(after.head_sha, snapshot.head_sha)
                    self.assertNotEqual(after.index_sha256, snapshot.index_sha256)

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

    def test_git_attestation_ignores_ambient_state_and_fsmonitor_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            attacker = Path(tmp) / "attacker"
            attacker.mkdir()
            _initialize_repo(attacker)
            (attacker / "attacker.txt").write_text("attacker\n", encoding="utf-8")
            _git(attacker, "add", "attacker.txt")
            helper = Path(tmp) / "fsmonitor-helper"
            marker = Path(tmp) / "helper-ran"
            helper.write_text(
                f"#!/bin/sh\ntouch '{marker}'\nexit 0\n",
                encoding="utf-8",
            )
            helper.chmod(0o700)
            _git(root, "config", "core.fsmonitor", str(helper))
            poisoned = {
                "GIT_DIR": str(attacker / ".git"),
                "GIT_WORK_TREE": str(attacker),
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "core.fsmonitor",
                "GIT_CONFIG_VALUE_0": str(helper),
                "GIT_CONFIG_KEY_1": "credential.helper",
                "GIT_CONFIG_VALUE_1": f"!{helper}",
            }

            with mock.patch.dict(os.environ, poisoned, clear=False):
                snapshot = snapshot_workspace(root, WorkspaceScopePolicy())

            self.assertFalse(marker.exists())
            self.assertIn("tracked.txt", {item.path for item in snapshot.files})
            self.assertNotIn("attacker.txt", {item.path for item in snapshot.files})

    def test_git_attestation_never_runs_repository_clean_filters(self) -> None:
        if os.name == "nt":
            self.skipTest("executable marker helper uses a POSIX script")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            marker = Path(tmp) / "clean-filter-ran"
            helper = Path(tmp) / "clean-filter"
            helper.write_text(
                f"#!/bin/sh\ncat\n: > '{marker}'\n",
                encoding="utf-8",
            )
            helper.chmod(0o700)
            (root / ".gitattributes").write_text(
                "tracked.txt filter=untrusted\n",
                encoding="utf-8",
            )
            _git(root, "config", "filter.untrusted.clean", str(helper))
            _git(root, "config", "filter.untrusted.required", "true")
            _git(root, "add", ".gitattributes", "tracked.txt")
            _git(
                root,
                "-c",
                "user.name=Antonio Antenore",
                "-c",
                "user.email=ant_ant95@hotmail.it",
                "commit",
                "-q",
                "-m",
                "filter fixture",
            )
            marker.unlink(missing_ok=True)
            (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")

            snapshot = snapshot_workspace(root, WorkspaceScopePolicy())

            self.assertFalse(marker.exists())
            self.assertTrue(snapshot.git_repository)

    def test_ambient_path_cannot_select_the_git_executable(self) -> None:
        if os.name == "nt":
            self.skipTest("executable marker helper uses a POSIX script")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)
            attacker = Path(tmp) / "attacker-bin"
            attacker.mkdir()
            marker = Path(tmp) / "ambient-git-ran"
            fake_git = attacker / "git"
            fake_git.write_text(
                f"#!/bin/sh\n: > '{marker}'\nexit 99\n",
                encoding="utf-8",
            )
            fake_git.chmod(0o700)

            with mock.patch.dict(os.environ, {"PATH": str(attacker)}, clear=False):
                snapshot = snapshot_workspace(root, WorkspaceScopePolicy())

            self.assertFalse(marker.exists())
            self.assertTrue(snapshot.git_repository)

    def test_detected_repository_fails_closed_when_git_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)

            with (
                mock.patch.object(
                    workspace_module,
                    "_resolve_trusted_git_identity",
                    side_effect=WorkspaceSecurityError("fixed unavailable"),
                ),
                self.assertRaisesRegex(WorkspaceSecurityError, "fixed unavailable"),
            ):
                snapshot_workspace(root, WorkspaceScopePolicy())

    def test_detected_repository_never_downgrades_after_git_probe_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _initialize_repo(root)

            with (
                mock.patch.object(
                    workspace_module,
                    "_execute_git",
                    side_effect=WorkspaceSecurityError("fixed probe failure"),
                ),
                self.assertRaisesRegex(WorkspaceSecurityError, "fixed probe failure"),
            ):
                snapshot_workspace(root, WorkspaceScopePolicy())

    def test_repository_marker_changes_fail_closed_during_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "tracked.txt").write_text("initial\n", encoding="utf-8")

            with (
                mock.patch.object(
                    workspace_module,
                    "_has_repository_marker",
                    side_effect=(False, True),
                ),
                self.assertRaisesRegex(WorkspaceSecurityError, "marker changed"),
            ):
                snapshot_workspace(root, WorkspaceScopePolicy())

    def test_git_process_output_is_bounded_and_fails_closed(self) -> None:
        if os.name == "nt":
            self.skipTest("bounded fake executable fixture uses a shebang")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            fake_git = Path(tmp) / "bounded-git"
            fake_git.write_text(
                (
                    "#!/bin/sh\n"
                    # Use only shell builtins so the output-bound assertion is
                    # independent of PATH and of spaces in ``sys.executable``.
                    # The runtime must terminate this unbounded producer once
                    # the fixed Git stdout ceiling is crossed.
                    "while :; do\n"
                    "  printf '%8192s' ''\n"
                    "done\n"
                ),
                encoding="utf-8",
            )
            fake_git.chmod(0o700)
            identity = workspace_module.resolve_executable(
                str(fake_git),
                env={"PATH": str(fake_git.parent)},
            )

            with self.assertRaisesRegex(WorkspaceSecurityError, "output bound"):
                workspace_module._run_git(
                    ("rev-parse", "--is-inside-work-tree"),
                    root,
                    capture_bytes=True,
                    git_identity=identity,
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
        self.assertTrue(capability.backend)
        if capability.supported:
            self.assertIn(
                capability.backend,
                {
                    "darwin-renamex-excl",
                    "linux-renameat2-noreplace",
                    "windows-handle-write-through",
                },
            )

    def test_native_no_replace_preserves_a_concurrent_destination(self) -> None:
        capability = workspace_write_capability()
        if not capability.supported:
            self.skipTest(capability.reason)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.write_text("transaction", encoding="utf-8")
            target.write_text("concurrent", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                workspace_module._rename_no_replace(source, target)

            self.assertEqual(source.read_text(encoding="utf-8"), "transaction")
            self.assertEqual(target.read_text(encoding="utf-8"), "concurrent")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO fixture")
    def test_fifo_snapshot_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fifo = root / "blocked.txt"
            os.mkfifo(fifo, 0o600)
            source = (
                "from pathlib import Path\n"
                "from local_moe.assistant_bridge_workspace import "
                "WorkspaceScopePolicy, WorkspaceSecurityError, snapshot_workspace\n"
                f"root = Path({str(root)!r})\n"
                "try:\n"
                "    snapshot_workspace(root, WorkspaceScopePolicy())\n"
                "except WorkspaceSecurityError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('FIFO workspace entry was accepted')\n"
            )
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", source],
                    cwd=Path(__file__).resolve().parents[1],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3.0,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                self.fail("FIFO snapshot regression process exceeded its timeout")
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

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
                target = root / "tracked.txt"
                target.chmod(0o444 if os.name == "nt" else 0o700)
                try:
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
                finally:
                    if os.name == "nt":
                        target.chmod(0o600)

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
            before_mode = stat.S_IMODE((backups / "00000000.bin").stat().st_mode)
            after_mode = stat.S_IMODE(target.stat().st_mode)
            before = WorkspaceFile(
                "tracked.txt",
                "file",
                hashlib.sha256(before_data).hexdigest(),
                len(before_data),
                before_mode,
            )
            after = WorkspaceFile(
                "tracked.txt",
                "file",
                hashlib.sha256(after_data).hexdigest(),
                len(after_data),
                after_mode,
            )
            installed_identity = {
                "device_id": int(target.stat().st_dev),
                "inode": int(target.stat().st_ino),
            }
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
                                "installed_identity": installed_identity,
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

    def test_same_value_concurrent_file_is_never_claimed_by_digest(self) -> None:
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
            owner = transaction / "owned-install.bin"
            owner.write_bytes(after_data)
            before_mode = stat.S_IMODE((backups / "00000000.bin").stat().st_mode)
            after_mode = stat.S_IMODE(target.stat().st_mode)
            before = WorkspaceFile(
                "tracked.txt",
                "file",
                hashlib.sha256(before_data).hexdigest(),
                len(before_data),
                before_mode,
            )
            after = WorkspaceFile(
                "tracked.txt",
                "file",
                hashlib.sha256(after_data).hexdigest(),
                len(after_data),
                after_mode,
            )
            journal = transaction / "journal.json"
            journal.write_text(
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
                                "installed_identity": {
                                    "device_id": int(owner.stat().st_dev),
                                    "inode": int(owner.stat().st_ino),
                                },
                                "created_directories": [],
                                "status": "applied",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(WorkspaceSecurityError, "prove ownership"):
                recover_workspace_transaction(
                    state_dir=Path(tmp) / "state",
                    transaction_id=transaction_id,
                    source_root=root,
                )

            self.assertEqual(target.read_bytes(), after_data)
            self.assertTrue(journal.exists())

    def test_concurrent_empty_directory_is_preserved_during_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            concurrent = root / "new"
            concurrent.mkdir()
            transaction_id = uuid4().hex
            transaction = Path(tmp) / "state" / f"transaction-{transaction_id}"
            (transaction / "backups").mkdir(parents=True)
            after_data = b"candidate\n"
            after = WorkspaceFile(
                "new/value.txt",
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
                                "path": "new/value.txt",
                                "before": None,
                                "after": after.payload(),
                                "backup": None,
                                "backup_sha256": None,
                                "installed_identity": None,
                                "created_directories": ["new"],
                                "status": "mutating",
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

            self.assertTrue(concurrent.is_dir())

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
    _git(root, "config", "user.name", "Antonio Antenore")
    _git(root, "config", "user.email", "ant_ant95@hotmail.it")
    (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(
        root,
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
