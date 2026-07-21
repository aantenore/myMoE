from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import time
import tracemalloc
import unittest
from unittest import mock

from local_moe import artifact_tree
from local_moe.artifact_tree import (
    ArtifactTreeEntry,
    ArtifactTreeIdentity,
    ArtifactTreeLimits,
    hash_artifact_tree,
)
from local_moe.cell_contracts import CellContractError
from local_moe.secure_files import SecureFileLimitError


def _limits(**changes: int) -> ArtifactTreeLimits:
    values = {
        "max_files": 32,
        "max_total_bytes": 1024 * 1024,
        "max_depth": 8,
        "max_file_bytes": 512 * 1024,
    }
    values.update(changes)
    return ArtifactTreeLimits(**values)


class ArtifactTreeTests(unittest.TestCase):
    def test_hashes_single_file_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"GGUF-fixture")

            first = hash_artifact_tree(model, root=root, limits=_limits())
            second = hash_artifact_tree(model, root=root, limits=_limits())

        self.assertEqual(first, second)
        self.assertEqual(first.kind, "file")
        self.assertEqual(first.file_count, 1)
        self.assertEqual(first.entries[0].path, "model.gguf")
        self.assertEqual(
            first.entries[0].sha256,
            hashlib.sha256(b"GGUF-fixture").hexdigest(),
        )
        self.assertEqual(first.payload()["digest"], first.digest)

    @unittest.skipIf(
        os.name == "nt",
        "v1 directory identity requires secure descriptor-relative POSIX traversal",
    )
    def test_hashes_directory_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "mlx"
            (tree / "nested").mkdir(parents=True)
            (tree / "z.safetensors").write_bytes(b"z-weight")
            (tree / "nested" / "a.json").write_bytes(b"{}")

            first = hash_artifact_tree(tree, root=root, limits=_limits())
            second = hash_artifact_tree(tree, root=root, limits=_limits())

        self.assertEqual(first, second)
        self.assertEqual(
            [entry.path for entry in first.entries],
            ["nested/a.json", "z.safetensors"],
        )
        self.assertEqual(first.total_bytes, len(b"{}") + len(b"z-weight"))
        self.assertEqual(first.payload()["digest"], first.digest)
        with self.assertRaisesRegex(CellContractError, "does not match"):
            replace(first, digest="a" * 64)

    def test_enforces_single_file_byte_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"1234")

            for limits in (
                _limits(max_total_bytes=3),
                _limits(max_file_bytes=3),
            ):
                with self.subTest(limits=limits):
                    with self.assertRaises(CellContractError):
                        hash_artifact_tree(model, root=root, limits=limits)

    @unittest.skipIf(os.name == "nt", "POSIX descriptor-relative regression")
    def test_posix_file_hashing_uses_the_tighter_total_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"1234")
            limits = _limits(max_total_bytes=7, max_file_bytes=11)
            with mock.patch.object(
                artifact_tree,
                "hash_bounded_regular_descriptor",
                wraps=artifact_tree.hash_bounded_regular_descriptor,
            ) as secure_hash:
                identity = hash_artifact_tree(model, root=root, limits=limits)

        self.assertEqual(identity.total_bytes, 4)
        self.assertEqual(secure_hash.call_args.kwargs["maximum_bytes"], 7)

    @unittest.skipIf(
        os.name == "nt",
        "v1 directory bounds require secure descriptor-relative POSIX traversal",
    )
    def test_enforces_directory_file_count_byte_depth_and_file_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            (tree / "deep").mkdir(parents=True)
            (tree / "first.bin").write_bytes(b"1234")
            (tree / "deep" / "second.bin").write_bytes(b"5678")

            cases = (
                _limits(max_files=1),
                _limits(max_total_bytes=7),
                _limits(max_depth=1),
                _limits(max_file_bytes=3),
            )
            for limits in cases:
                with self.subTest(limits=limits):
                    with self.assertRaises(CellContractError):
                        hash_artifact_tree(tree, root=root, limits=limits)

    @unittest.skipIf(
        os.name == "nt",
        "v1 directory bounds require secure descriptor-relative POSIX traversal",
    )
    def test_directory_hashing_uses_the_remaining_total_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            tree.mkdir()
            (tree / "a.bin").write_bytes(b"123456")
            (tree / "b.bin").write_bytes(b"7890")
            limits = _limits(max_total_bytes=10, max_file_bytes=10)
            with mock.patch.object(
                artifact_tree,
                "hash_bounded_regular_descriptor",
                wraps=artifact_tree.hash_bounded_regular_descriptor,
            ) as secure_hash:
                identity = hash_artifact_tree(tree, root=root, limits=limits)

        self.assertEqual(identity.total_bytes, 10)
        self.assertEqual(
            [call.kwargs["maximum_bytes"] for call in secure_hash.call_args_list],
            [10, 4],
        )

    @unittest.skipIf(
        os.name == "nt",
        "v1 directory bounds require secure descriptor-relative POSIX traversal",
    )
    def test_entry_budget_bounds_enumeration_before_walking_a_wide_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "wide"
            tree.mkdir()
            for index in range(40):
                (tree / f"empty-{index:02d}").mkdir()
            original_scandir = os.scandir
            yielded = 0

            class CountingScandir:
                def __init__(self, descriptor: int) -> None:
                    self._inner = original_scandir(descriptor)

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def __exit__(self, *args):
                    return self._inner.__exit__(*args)

                def __iter__(self):
                    return self

                def __next__(self):
                    nonlocal yielded
                    item = next(self._inner)
                    yielded += 1
                    return item

            with (
                mock.patch.object(
                    artifact_tree.os,
                    "scandir",
                    side_effect=CountingScandir,
                ),
                mock.patch.object(
                    artifact_tree,
                    "_supports_secure_directory_walk",
                    return_value=True,
                ),
            ):
                with self.assertRaisesRegex(CellContractError, "entry bound"):
                    hash_artifact_tree(
                        tree,
                        root=root,
                        limits=_limits(max_files=3),
                    )

        self.assertEqual(yielded, 4)

    def test_rejects_invalid_limits_special_file_and_escape(self) -> None:
        for changes in (
            {"max_files": 0},
            {"max_files": True},
            {"max_depth": 65},
            {"max_total_bytes": 2 * 1024**4 + 1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(CellContractError):
                    _limits(**changes)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / f"{root.name}-outside.bin"
            outside.write_bytes(b"outside")
            try:
                with self.assertRaises(CellContractError):
                    hash_artifact_tree(outside, root=root, limits=_limits())
                with self.assertRaises(CellContractError):
                    hash_artifact_tree(
                        Path(os.devnull), root=Path(os.devnull).parent, limits=_limits()
                    )
            finally:
                outside.unlink(missing_ok=True)

    @unittest.skipIf(
        os.name == "nt",
        "v1 empty-directory inspection requires secure POSIX traversal",
    )
    def test_rejects_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()

            with self.assertRaises(CellContractError):
                hash_artifact_tree(empty, root=root, limits=_limits())

    @unittest.skipUnless(os.name == "nt", "Windows-specific fail-closed boundary")
    def test_directory_hashing_fails_closed_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "mlx"
            tree.mkdir()
            (tree / "config.json").write_bytes(b"{}")

            with self.assertRaises(CellContractError):
                hash_artifact_tree(tree, root=root, limits=_limits())

    def test_windows_file_hashing_does_not_probe_paths_before_secure_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    artifact_tree,
                    "hash_bounded_regular_file",
                    return_value=("a" * 64, 7),
                ),
                mock.patch.object(
                    Path,
                    "lstat",
                    side_effect=AssertionError("path probe must not run"),
                ),
            ):
                identity = artifact_tree._hash_windows_target(
                    root,
                    ("nested", "model.gguf"),
                    limits=_limits(),
                )

        self.assertEqual(identity.kind, "file")
        self.assertEqual(identity.total_bytes, 7)
        self.assertEqual(identity.entries[0].path, "model.gguf")

    def test_windows_file_hashing_uses_the_tighter_total_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            limits = _limits(max_total_bytes=7, max_file_bytes=11)
            with mock.patch.object(
                artifact_tree,
                "hash_bounded_regular_file",
                side_effect=SecureFileLimitError("bounded"),
            ) as secure_hash:
                with self.assertRaises(artifact_tree.ArtifactTreeLimitError):
                    artifact_tree._hash_windows_target(
                        root,
                        ("model.gguf",),
                        limits=limits,
                    )

        self.assertEqual(secure_hash.call_args.kwargs["maximum_bytes"], 7)

    @unittest.skipIf(
        os.name == "nt",
        "hardlink accounting requires directory traversal unavailable on Windows v1",
    )
    def test_rejects_symlinks_and_counts_hardlinks_once_when_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            tree.mkdir()
            source = tree / "source.bin"
            source.write_bytes(b"same-inode")
            linked = tree / "hard.bin"
            os.link(source, linked)

            identity = hash_artifact_tree(tree, root=root, limits=_limits())
            self.assertEqual(identity.file_count, 2)
            self.assertEqual(identity.total_bytes, 2 * len(b"same-inode"))
            self.assertEqual(identity.hashed_bytes, len(b"same-inode"))

            symlink = tree / "linked.bin"
            symlink.symlink_to(source)
            with self.assertRaises(CellContractError):
                hash_artifact_tree(tree, root=root, limits=_limits())

    @unittest.skipIf(
        os.name == "nt",
        "hardlink accounting requires directory traversal unavailable on Windows v1",
    )
    def test_physical_deduplication_does_not_change_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied = root / "copied"
            linked = root / "linked"
            copied.mkdir()
            linked.mkdir()
            content = b"same-logical-content"
            (copied / "a.bin").write_bytes(content)
            (copied / "b.bin").write_bytes(content)
            (linked / "a.bin").write_bytes(content)
            os.link(linked / "a.bin", linked / "b.bin")

            copied_identity = hash_artifact_tree(
                copied,
                root=root,
                limits=_limits(),
            )
            linked_identity = hash_artifact_tree(
                linked,
                root=root,
                limits=_limits(),
            )

        self.assertEqual(copied_identity.entries, linked_identity.entries)
        self.assertEqual(copied_identity.digest, linked_identity.digest)
        self.assertEqual(copied_identity.hashed_bytes, 2 * len(content))
        self.assertEqual(linked_identity.hashed_bytes, len(content))
        self.assertNotIn("hashed_bytes", copied_identity.payload())

    @unittest.skipIf(
        os.name == "nt",
        "Windows secure hashing owns and closes the complete pinned observation",
    )
    def test_revalidation_detects_change_after_streaming_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.gguf"
            target.write_bytes(b"stable")
            helper_name = (
                "hash_bounded_regular_file"
                if os.name == "nt"
                else "hash_bounded_regular_descriptor"
            )
            original = getattr(artifact_tree, helper_name)
            mutated = False

            def hash_then_mutate(*args, **kwargs):
                nonlocal mutated
                result = original(*args, **kwargs)
                if not mutated:
                    target.write_bytes(b"changed-after-hash")
                    mutated = True
                return result

            with mock.patch.object(
                artifact_tree,
                helper_name,
                side_effect=hash_then_mutate,
            ):
                with self.assertRaisesRegex(CellContractError, "changed"):
                    hash_artifact_tree(target, root=root, limits=_limits())

    @unittest.skipIf(
        os.name == "nt",
        "Windows secure hashing owns and closes the complete pinned observation",
    )
    def test_revalidation_detects_same_size_metadata_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "model.gguf"
            target.write_bytes(b"stable")
            helper_name = (
                "hash_bounded_regular_file"
                if os.name == "nt"
                else "hash_bounded_regular_descriptor"
            )
            original = getattr(artifact_tree, helper_name)

            def hash_then_touch(*args, **kwargs):
                result = original(*args, **kwargs)
                before = target.stat()
                os.utime(
                    target,
                    ns=(before.st_atime_ns, before.st_mtime_ns + 10_000_000_000),
                )
                return result

            with mock.patch.object(
                artifact_tree,
                helper_name,
                side_effect=hash_then_touch,
            ):
                with self.assertRaisesRegex(CellContractError, "changed"):
                    hash_artifact_tree(target, root=root, limits=_limits())

    @unittest.skipIf(os.name == "nt", "POSIX descriptor-relative regression")
    def test_ancestor_swap_cannot_redirect_hashing_to_a_replacement_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            tree = root / "tree"
            tree.mkdir(parents=True)
            (tree / "model.bin").write_bytes(b"GOOD!!")
            replacement = base / "replacement"
            (replacement / "tree").mkdir(parents=True)
            (replacement / "tree" / "model.bin").write_bytes(b"EVIL!!")
            original_open = artifact_tree._open_child_regular_file
            swapped = False

            def swap_ancestor_then_open(*args, **kwargs):
                nonlocal swapped
                if not swapped:
                    root.rename(base / "pinned-root")
                    replacement.rename(root)
                    swapped = True
                return original_open(*args, **kwargs)

            with mock.patch.object(
                artifact_tree,
                "_open_child_regular_file",
                side_effect=swap_ancestor_then_open,
            ):
                identity = hash_artifact_tree(tree, root=root, limits=_limits())

        self.assertTrue(swapped)
        self.assertEqual(
            identity.entries[0].sha256,
            hashlib.sha256(b"GOOD!!").hexdigest(),
        )
        self.assertNotEqual(
            identity.entries[0].sha256,
            hashlib.sha256(b"EVIL!!").hexdigest(),
        )

    @unittest.skipIf(os.name == "nt", "POSIX special-file fixture")
    def test_rejects_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo = root / "model.pipe"
            os.mkfifo(fifo)

            started = time.monotonic()
            with self.assertRaises(CellContractError):
                hash_artifact_tree(fifo, root=root, limits=_limits())
            self.assertLess(time.monotonic() - started, 1.0)

    def test_file_hashing_has_bounded_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "large.gguf"
            target.write_bytes(b"0123456789abcdef" * (32 * 1024))

            tracemalloc.start()
            try:
                identity = hash_artifact_tree(target, root=root, limits=_limits())
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

            self.assertEqual(identity.total_bytes, 512 * 1024)
            self.assertLess(peak, 1024 * 1024)

    @unittest.skipIf(
        os.name == "nt",
        "Windows has its own explicit directory fail-closed test",
    )
    def test_directory_walk_fails_closed_without_nofollow_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            tree.mkdir()
            (tree / "model.bin").write_bytes(b"fixture")

            with mock.patch.object(
                artifact_tree,
                "_supports_secure_directory_walk",
                return_value=False,
            ):
                with self.assertRaisesRegex(CellContractError, "unavailable"):
                    hash_artifact_tree(tree, root=root, limits=_limits())

    def test_identity_rejects_inconsistent_counts_and_unsorted_entries(self) -> None:
        entries = (
            ArtifactTreeEntry(
                path="a",
                size_bytes=1,
                sha256=hashlib.sha256(b"a").hexdigest(),
            ),
            ArtifactTreeEntry(
                path="b",
                size_bytes=1,
                sha256=hashlib.sha256(b"b").hexdigest(),
            ),
        )
        identity = ArtifactTreeIdentity(
            kind="directory",
            entries=entries,
            file_count=2,
            total_bytes=2,
            hashed_bytes=2,
        )

        with self.assertRaises(CellContractError):
            replace(identity, entries=tuple(reversed(identity.entries)), digest="")
        with self.assertRaises(CellContractError):
            ArtifactTreeIdentity(
                kind="directory",
                entries=identity.entries,
                file_count=3,
                total_bytes=identity.total_bytes,
                hashed_bytes=identity.hashed_bytes,
            )


if __name__ == "__main__":
    unittest.main()
