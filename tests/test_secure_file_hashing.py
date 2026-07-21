import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from local_moe import secure_files
from local_moe.cell_contracts import CellContractError
from local_moe.secure_files import (
    hash_bounded_regular_descriptor,
    hash_bounded_regular_file,
)


class SecureFileHashingTests(unittest.TestCase):
    def test_hashes_an_open_descriptor_without_closing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact.bin"
            target.write_bytes(b"descriptor-bound")
            descriptor = os.open(target, os.O_RDONLY)
            try:
                digest, size = hash_bounded_regular_descriptor(
                    descriptor,
                    maximum_bytes=1024,
                    label="descriptor artifact",
                )
                os.fstat(descriptor)
            finally:
                os.close(descriptor)

        self.assertEqual(digest, hashlib.sha256(b"descriptor-bound").hexdigest())
        self.assertEqual(size, len(b"descriptor-bound"))

    def test_hashes_nested_regular_and_empty_files_within_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "evidence"
            nested.mkdir()
            payload_bytes = b"\x00binary fixture\xff"
            payload = nested / "payload.bin"
            payload.write_bytes(payload_bytes)
            empty = root / "empty.bin"
            empty.write_bytes(b"")

            self.assertEqual(
                hash_bounded_regular_file(
                    payload,
                    root=root,
                    maximum_bytes=len(payload_bytes),
                    label="payload",
                ),
                (hashlib.sha256(payload_bytes).hexdigest(), len(payload_bytes)),
            )
            self.assertEqual(
                hash_bounded_regular_file(
                    empty,
                    maximum_bytes=1,
                    label="empty payload",
                ),
                (hashlib.sha256(b"").hexdigest(), 0),
            )

    def test_streams_without_calling_the_bounded_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload_bytes = (b"streaming-fixture" * 9_000) + b"tail"
            target = Path(directory) / "payload.bin"
            target.write_bytes(payload_bytes)
            original_read = os.read
            requested_sizes: list[int] = []

            def observed_read(descriptor: int, requested: int) -> bytes:
                requested_sizes.append(requested)
                return original_read(descriptor, requested)

            with (
                mock.patch.object(
                    secure_files,
                    "read_bounded_regular_file",
                    side_effect=AssertionError("bounded reader must not be called"),
                ),
                mock.patch.object(
                    secure_files.os,
                    "read",
                    side_effect=observed_read,
                ),
            ):
                observed = hash_bounded_regular_file(
                    target,
                    maximum_bytes=len(payload_bytes),
                    label="streamed payload",
                )

            self.assertEqual(
                observed,
                (hashlib.sha256(payload_bytes).hexdigest(), len(payload_bytes)),
            )
            self.assertGreater(len(requested_sizes), 2)
            self.assertTrue(
                all(0 < requested <= 64 * 1024 for requested in requested_sizes)
            )
            self.assertGreater(len(payload_bytes), max(requested_sizes))

    def test_rejects_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "payload.bin"
            target.write_bytes(b"12345")

            with self.assertRaisesRegex(CellContractError, "bounded"):
                hash_bounded_regular_file(
                    target,
                    maximum_bytes=4,
                    label="payload",
                )

    def test_rejects_escape_directory_and_invalid_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "outside.bin"
            outside.write_bytes(b"outside")
            invalid_actions = (
                lambda: hash_bounded_regular_file(
                    outside,
                    root=root,
                    maximum_bytes=16,
                    label="escape",
                ),
                lambda: hash_bounded_regular_file(
                    root,
                    root=root,
                    maximum_bytes=16,
                    label="directory",
                ),
                lambda: hash_bounded_regular_file(
                    outside,
                    maximum_bytes=True,
                    label="boolean bound",
                ),
                lambda: hash_bounded_regular_file(
                    outside,
                    maximum_bytes=0,
                    label="zero bound",
                ),
                lambda: hash_bounded_regular_file(
                    outside,
                    maximum_bytes=1.5,
                    label="non-integer bound",
                ),
                lambda: hash_bounded_regular_file(
                    outside,
                    maximum_bytes=16,
                    label=" ",
                ),
            )

            for action in invalid_actions:
                with self.subTest(action=action):
                    with self.assertRaises(CellContractError):
                        action()

    @unittest.skipIf(os.name == "nt", "symlink creation requires privileges on Windows")
    def test_rejects_symlinked_prefix_and_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            payload = real / "payload.bin"
            payload.write_bytes(b"fixture")
            linked_directory = root / "linked"
            linked_directory.symlink_to(real, target_is_directory=True)
            linked_file = root / "payload-link.bin"
            linked_file.symlink_to(payload)

            for target in (linked_directory / "payload.bin", linked_file):
                with self.subTest(target=target):
                    with self.assertRaises(CellContractError):
                        hash_bounded_regular_file(
                            target,
                            root=root,
                            maximum_bytes=64,
                            label="linked payload",
                        )

    @unittest.skipIf(os.name == "nt", "POSIX special-file fixtures only")
    def test_rejects_fifo_and_device_without_waiting_for_a_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "input.pipe"
            os.mkfifo(fifo)

            started = time.monotonic()
            with self.assertRaises(CellContractError):
                hash_bounded_regular_file(
                    fifo,
                    maximum_bytes=1024,
                    label="FIFO fixture",
                )
            self.assertLess(time.monotonic() - started, 1.0)

        with self.assertRaises(CellContractError):
            hash_bounded_regular_file(
                Path(os.devnull),
                maximum_bytes=1,
                label="device fixture",
            )

    @unittest.skipIf(os.name == "nt", "deterministic POSIX timestamp mutation")
    def test_rejects_file_mutated_during_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "payload.bin"
            target.write_bytes(b"stable before hashing")
            original_read = os.read
            mutated = False

            def read_and_mutate(descriptor: int, requested: int) -> bytes:
                nonlocal mutated
                chunk = original_read(descriptor, requested)
                if not mutated:
                    before = target.stat()
                    os.utime(
                        target,
                        ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000),
                    )
                    mutated = True
                return chunk

            with mock.patch.object(
                secure_files.os,
                "read",
                side_effect=read_and_mutate,
            ):
                with self.assertRaisesRegex(CellContractError, "changed"):
                    hash_bounded_regular_file(
                        target,
                        maximum_bytes=64,
                        label="mutating payload",
                    )


if __name__ == "__main__":
    unittest.main()
