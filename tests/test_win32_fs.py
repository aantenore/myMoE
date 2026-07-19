from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import local_moe._win32_fs as win32_fs


_FILE_ID = bytes.fromhex("00112233445566778899aabbccddeeff")


class _Function:
    def __init__(self, *, return_value: object = 1, implementation: object = None):
        self.return_value = return_value
        self.implementation = implementation
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: list[object] | None = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        if self.implementation is not None:
            return self.implementation(*args)
        return self.return_value


def _kernel32_with_identity(
    *,
    handle: int = 41,
    volume_serial: int = 0xFEDCBA9876543210,
    file_id: bytes = _FILE_ID,
    attributes: int = 0x20,
    reparse_tag: int = 0,
    failing_info_class: int | None = None,
) -> SimpleNamespace:
    def query(
        queried_handle: object,
        info_class: object,
        destination: object,
        size: object,
    ) -> int:
        del queried_handle, size
        if info_class == failing_info_class:
            return 0
        if info_class == win32_fs._FILE_ATTRIBUTE_TAG_INFO_CLASS:
            info = ctypes.cast(
                destination,
                ctypes.POINTER(win32_fs._FILE_ATTRIBUTE_TAG_INFO),
            ).contents
            info.FileAttributes = attributes
            info.ReparseTag = reparse_tag
            return 1
        if info_class == win32_fs._FILE_ID_INFO_CLASS:
            info = ctypes.cast(
                destination,
                ctypes.POINTER(win32_fs._FILE_ID_INFO),
            ).contents
            info.VolumeSerialNumber = volume_serial
            for index, value in enumerate(file_id):
                info.FileId.Identifier[index] = value
            return 1
        raise AssertionError(f"Unexpected information class: {info_class}")

    return SimpleNamespace(
        CreateFileW=_Function(return_value=handle),
        GetFileInformationByHandleEx=_Function(implementation=query),
        CloseHandle=_Function(return_value=1),
        MoveFileExW=_Function(return_value=1),
    )


class Win32FileIdentityTests(unittest.TestCase):
    def test_requires_a_complete_128_bit_file_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 16 bytes"):
            win32_fs.Win32FileIdentity(
                volume_serial=1,
                file_id=b"short",
                attributes=0,
                reparse_tag=0,
            )

    def test_normalizes_mutable_ids_and_compares_all_128_bits(self) -> None:
        low_bits = bytes.fromhex("0011223344556677")
        first = win32_fs.Win32FileIdentity(
            volume_serial=7,
            file_id=bytearray(low_bits + bytes.fromhex("8899aabbccddeeff")),
            attributes=0,
            reparse_tag=0,
        )
        second = win32_fs.Win32FileIdentity(
            volume_serial=7,
            file_id=low_bits + bytes.fromhex("8899aabbccddee00"),
            attributes=0,
            reparse_tag=0,
        )

        self.assertIs(type(first.file_id), bytes)
        self.assertNotEqual(first, second)
        self.assertFalse(first.same_file_as(second))

    def test_same_file_ignores_mutable_attributes_without_changing_equality(
        self,
    ) -> None:
        original = win32_fs.Win32FileIdentity(
            volume_serial=7,
            file_id=_FILE_ID,
            attributes=0,
            reparse_tag=0,
        )
        archived = win32_fs.Win32FileIdentity(
            volume_serial=7,
            file_id=_FILE_ID,
            attributes=0x20,
            reparse_tag=0,
        )

        self.assertNotEqual(original, archived)
        self.assertTrue(original.same_file_as(archived))
        self.assertTrue(archived.same_file_as(original))

    def test_same_file_rejects_a_different_volume_or_file_id(self) -> None:
        original = win32_fs.Win32FileIdentity(
            volume_serial=7,
            file_id=_FILE_ID,
            attributes=0,
            reparse_tag=0,
        )
        different_volume = win32_fs.Win32FileIdentity(
            volume_serial=8,
            file_id=_FILE_ID,
            attributes=0,
            reparse_tag=0,
        )
        different_file = win32_fs.Win32FileIdentity(
            volume_serial=7,
            file_id=_FILE_ID[:-1] + bytes([_FILE_ID[-1] ^ 0x01]),
            attributes=0,
            reparse_tag=0,
        )

        self.assertFalse(original.same_file_as(different_volume))
        self.assertFalse(original.same_file_as(different_file))
        self.assertFalse(original.same_file_as(object()))

    def test_reparse_state_is_fail_closed_for_attribute_or_tag(self) -> None:
        attribute_identity = win32_fs.Win32FileIdentity(
            volume_serial=1,
            file_id=_FILE_ID,
            attributes=win32_fs._FILE_ATTRIBUTE_REPARSE_POINT,
            reparse_tag=0,
        )
        tag_identity = win32_fs.Win32FileIdentity(
            volume_serial=1,
            file_id=_FILE_ID,
            attributes=0,
            reparse_tag=0xA000000C,
        )

        self.assertTrue(attribute_identity.is_reparse_point)
        self.assertTrue(tag_identity.is_reparse_point)


class OpenNoFollowTests(unittest.TestCase):
    def test_regular_file_transfers_handle_ownership_to_readonly_fd(self) -> None:
        kernel32 = _kernel32_with_identity()
        msvcrt = SimpleNamespace(open_osfhandle=_Function(return_value=73))

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(win32_fs, "_load_msvcrt", return_value=msvcrt),
        ):
            descriptor, identity = win32_fs.open_nofollow_fd(Path("safe.txt"))

        self.assertEqual(descriptor, 73)
        self.assertEqual(identity.volume_serial, 0xFEDCBA9876543210)
        self.assertEqual(identity.file_id, _FILE_ID)
        create_call = kernel32.CreateFileW.calls[0]
        self.assertEqual(create_call[0], "safe.txt")
        self.assertEqual(create_call[1], win32_fs._GENERIC_READ)
        self.assertEqual(
            create_call[2],
            win32_fs._FILE_SHARE_READ | win32_fs._FILE_SHARE_WRITE,
        )
        self.assertEqual(create_call[4], win32_fs._OPEN_EXISTING)
        self.assertEqual(create_call[5], win32_fs._FILE_FLAG_OPEN_REPARSE_POINT)
        self.assertEqual(
            msvcrt.open_osfhandle.calls,
            [(41, os.O_RDONLY | getattr(os, "O_BINARY", 0))],
        )
        self.assertEqual(kernel32.CloseHandle.calls, [])

    def test_writable_directory_sets_backup_semantics_and_optional_delete_share(
        self,
    ) -> None:
        kernel32 = _kernel32_with_identity(attributes=0x10)
        msvcrt = SimpleNamespace(open_osfhandle=_Function(return_value=74))

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(win32_fs, "_load_msvcrt", return_value=msvcrt),
        ):
            win32_fs.open_nofollow_fd(
                "directory",
                directory=True,
                writable=True,
                share_delete=True,
            )

        create_call = kernel32.CreateFileW.calls[0]
        self.assertEqual(
            create_call[1],
            win32_fs._GENERIC_READ | win32_fs._GENERIC_WRITE,
        )
        self.assertEqual(
            create_call[2],
            win32_fs._FILE_SHARE_READ
            | win32_fs._FILE_SHARE_WRITE
            | win32_fs._FILE_SHARE_DELETE,
        )
        self.assertEqual(
            create_call[5],
            win32_fs._FILE_FLAG_OPEN_REPARSE_POINT
            | win32_fs._FILE_FLAG_BACKUP_SEMANTICS,
        )
        self.assertEqual(
            msvcrt.open_osfhandle.calls,
            [(41, os.O_RDWR | getattr(os, "O_BINARY", 0))],
        )

    def test_reparse_point_is_rejected_before_crt_ownership_transfer(self) -> None:
        kernel32 = _kernel32_with_identity(
            attributes=win32_fs._FILE_ATTRIBUTE_REPARSE_POINT,
            reparse_tag=0xA000000C,
        )

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(win32_fs, "_load_msvcrt") as load_msvcrt,
            self.assertRaises(OSError) as raised,
        ):
            win32_fs.open_nofollow_fd("link")

        self.assertEqual(raised.exception.errno, errno.ELOOP)
        self.assertEqual(kernel32.CloseHandle.calls, [(41,)])
        load_msvcrt.assert_not_called()

    def test_each_identity_query_failure_closes_the_raw_handle(self) -> None:
        for failing_class in (
            win32_fs._FILE_ATTRIBUTE_TAG_INFO_CLASS,
            win32_fs._FILE_ID_INFO_CLASS,
        ):
            with self.subTest(failing_class=failing_class):
                kernel32 = _kernel32_with_identity(
                    failing_info_class=failing_class,
                )
                with (
                    mock.patch.object(
                        win32_fs,
                        "_load_kernel32",
                        return_value=kernel32,
                    ),
                    mock.patch.object(win32_fs, "_last_error", return_value=87),
                    self.assertRaises(OSError) as raised,
                ):
                    win32_fs.open_nofollow_fd("candidate")

                self.assertEqual(raised.exception.winerror, 87)
                self.assertEqual(kernel32.CloseHandle.calls, [(41,)])

    def test_open_osfhandle_failure_keeps_raw_handle_ownership_and_closes_it(
        self,
    ) -> None:
        kernel32 = _kernel32_with_identity()
        msvcrt = SimpleNamespace(
            open_osfhandle=_Function(
                implementation=lambda *_: (_ for _ in ()).throw(
                    OSError(errno.EMFILE, "descriptor table full")
                )
            )
        )

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(win32_fs, "_load_msvcrt", return_value=msvcrt),
            self.assertRaises(OSError) as raised,
        ):
            win32_fs.open_nofollow_fd("candidate")

        self.assertEqual(raised.exception.errno, errno.EMFILE)
        self.assertEqual(kernel32.CloseHandle.calls, [(41,)])

    def test_invalid_create_handle_maps_missing_path_without_closing_it(self) -> None:
        kernel32 = _kernel32_with_identity(handle=ctypes.c_void_p(-1).value)

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(
                win32_fs,
                "_last_error",
                return_value=win32_fs._ERROR_FILE_NOT_FOUND,
            ),
            self.assertRaises(FileNotFoundError) as raised,
        ):
            win32_fs.open_nofollow_fd("missing")

        self.assertEqual(raised.exception.winerror, win32_fs._ERROR_FILE_NOT_FOUND)
        self.assertEqual(kernel32.CloseHandle.calls, [])


class IdentityFromDescriptorTests(unittest.TestCase):
    def test_queries_file_id_and_attribute_tag_from_the_same_handle(self) -> None:
        kernel32 = _kernel32_with_identity(handle=59)
        msvcrt = SimpleNamespace(get_osfhandle=_Function(return_value=59))

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(win32_fs, "_load_msvcrt", return_value=msvcrt),
        ):
            identity = win32_fs.identity_from_fd(9)

        self.assertEqual(msvcrt.get_osfhandle.calls, [(9,)])
        self.assertEqual(
            [call[1] for call in kernel32.GetFileInformationByHandleEx.calls],
            [
                win32_fs._FILE_ATTRIBUTE_TAG_INFO_CLASS,
                win32_fs._FILE_ID_INFO_CLASS,
            ],
        )
        self.assertEqual(identity.file_id, _FILE_ID)
        self.assertEqual(kernel32.CloseHandle.calls, [])

    def test_invalid_crt_descriptor_fails_before_loading_kernel32(self) -> None:
        msvcrt = SimpleNamespace(get_osfhandle=_Function(return_value=-1))

        with (
            mock.patch.object(win32_fs, "_load_msvcrt", return_value=msvcrt),
            mock.patch.object(win32_fs, "_load_kernel32") as load_kernel32,
            self.assertRaises(OSError) as raised,
        ):
            win32_fs.identity_from_fd(9)

        self.assertEqual(raised.exception.errno, errno.EBADF)
        load_kernel32.assert_not_called()


class MoveNoReplaceTests(unittest.TestCase):
    def test_uses_write_through_without_replace_or_copy_flags(self) -> None:
        kernel32 = _kernel32_with_identity()

        with mock.patch.object(
            win32_fs,
            "_load_kernel32",
            return_value=kernel32,
        ):
            win32_fs.move_no_replace(Path("source.tmp"), Path("target.json"))

        self.assertEqual(
            kernel32.MoveFileExW.calls,
            [("source.tmp", "target.json", win32_fs._MOVEFILE_WRITE_THROUGH)],
        )

    def test_write_through_can_be_disabled_without_enabling_replace(self) -> None:
        kernel32 = _kernel32_with_identity()

        with mock.patch.object(
            win32_fs,
            "_load_kernel32",
            return_value=kernel32,
        ):
            win32_fs.move_no_replace("source", "target", write_through=False)

        self.assertEqual(kernel32.MoveFileExW.calls, [("source", "target", 0)])

    def test_collision_errors_are_normalized_to_file_exists(self) -> None:
        for error in (
            win32_fs._ERROR_FILE_EXISTS,
            win32_fs._ERROR_ALREADY_EXISTS,
        ):
            with self.subTest(error=error):
                kernel32 = _kernel32_with_identity()
                kernel32.MoveFileExW.return_value = 0
                with (
                    mock.patch.object(
                        win32_fs,
                        "_load_kernel32",
                        return_value=kernel32,
                    ),
                    mock.patch.object(win32_fs, "_last_error", return_value=error),
                    self.assertRaises(FileExistsError) as raised,
                ):
                    win32_fs.move_no_replace("source", "target")

                self.assertEqual(raised.exception.winerror, error)
                self.assertEqual(raised.exception.filename, "target")

    def test_other_move_failures_preserve_the_win32_error(self) -> None:
        kernel32 = _kernel32_with_identity()
        kernel32.MoveFileExW.return_value = 0

        with (
            mock.patch.object(win32_fs, "_load_kernel32", return_value=kernel32),
            mock.patch.object(win32_fs, "_last_error", return_value=5),
            self.assertRaises(OSError) as raised,
        ):
            win32_fs.move_no_replace("source", "target")

        self.assertEqual(raised.exception.winerror, 5)
        self.assertEqual(raised.exception.filename, "target")


class PosixImportabilityTests(unittest.TestCase):
    def test_native_loader_fails_closed_when_windll_is_unavailable(self) -> None:
        if hasattr(ctypes, "WinDLL"):
            self.skipTest("ctypes.WinDLL is available on this interpreter")

        with self.assertRaises(OSError) as raised:
            win32_fs._load_kernel32()

        self.assertEqual(raised.exception.errno, errno.ENOSYS)


if __name__ == "__main__":
    unittest.main()
