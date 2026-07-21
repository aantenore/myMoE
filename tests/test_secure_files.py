import os
from pathlib import Path
import tempfile
import time
import unittest

from local_moe.cell_contracts import CellContractError
from local_moe.secure_files import read_bounded_regular_file
from local_moe.secure_files import _read_posix


class SecureFileTests(unittest.TestCase):
    def test_reads_nested_regular_and_empty_files_within_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "evidence"
            nested.mkdir()
            payload = nested / "payload.bin"
            payload.write_bytes(b"fixture")
            empty = root / "empty.bin"
            empty.write_bytes(b"")
            self.assertEqual(
                read_bounded_regular_file(
                    payload,
                    root=root,
                    maximum_bytes=7,
                    label="payload",
                ),
                b"fixture",
            )
            self.assertEqual(
                read_bounded_regular_file(
                    empty,
                    maximum_bytes=1,
                    label="empty payload",
                ),
                b"",
            )

    def test_rejects_escape_directory_and_invalid_bounds_uniformly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "outside.bin"
            outside.write_bytes(b"outside")
            for action in (
                lambda: read_bounded_regular_file(
                    outside, root=root, maximum_bytes=16, label="escape"
                ),
                lambda: read_bounded_regular_file(
                    root, root=root, maximum_bytes=16, label="directory"
                ),
                lambda: read_bounded_regular_file(
                    outside, maximum_bytes=bool(1), label="bad bound"
                ),
            ):
                with self.assertRaises(CellContractError):
                    action()

    def test_rejects_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "payload.bin"
            target.write_bytes(b"12345")
            with self.assertRaisesRegex(CellContractError, "bounded"):
                read_bounded_regular_file(
                    target,
                    maximum_bytes=4,
                    label="payload",
                )

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
                with self.assertRaises(CellContractError):
                    read_bounded_regular_file(
                        target,
                        root=root,
                        maximum_bytes=64,
                        label="linked payload",
                    )

    @unittest.skipIf(os.name == "nt", "POSIX FIFO fixture only")
    def test_fifo_is_rejected_without_waiting_for_a_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "input.pipe"
            os.mkfifo(fifo)

            started = time.monotonic()
            with self.assertRaises(CellContractError):
                read_bounded_regular_file(
                    fifo,
                    maximum_bytes=1024,
                    label="FIFO fixture",
                )

            self.assertLess(time.monotonic() - started, 1.0)

    @unittest.skipIf(os.name == "nt", "POSIX descriptor walk only")
    def test_descriptor_walk_rejects_symlink_above_trusted_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            real = base / "real"
            trusted = real / "trusted"
            trusted.mkdir(parents=True)
            payload = trusted / "payload.bin"
            payload.write_bytes(b"fixture")
            alias = base / "alias"
            alias.symlink_to(real, target_is_directory=True)

            with self.assertRaises(CellContractError):
                _read_posix(
                    alias / "trusted",
                    Path("payload.bin"),
                    maximum_bytes=64,
                    label="linked root",
                )

            self.assertEqual(
                read_bounded_regular_file(
                    alias / "trusted" / "payload.bin",
                    root=alias / "trusted",
                    maximum_bytes=64,
                    label="canonical trusted root",
                ),
                b"fixture",
            )


if __name__ == "__main__":
    unittest.main()
