from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from local_moe import assistant_bridge_process as process_module


class AssistantBridgeProcessTests(unittest.TestCase):
    def test_windows_wait_proves_exit_and_fails_closed_otherwise(self) -> None:
        for wait_result, expected in (
            (0, False),
            (0x00000102, True),
            (0xFFFFFFFF, True),
        ):
            with self.subTest(wait_result=wait_result):
                kernel32 = SimpleNamespace(
                    OpenProcess=mock.Mock(return_value=123),
                    WaitForSingleObject=mock.Mock(return_value=wait_result),
                    CloseHandle=mock.Mock(return_value=1),
                )
                ctypes_api = SimpleNamespace(
                    c_uint32=int,
                    c_int=int,
                    c_void_p=lambda value: value,
                    get_last_error=mock.Mock(return_value=0),
                )

                self.assertEqual(
                    process_module._windows_process_is_alive(
                        42,
                        _ctypes=ctypes_api,
                        _kernel32=kernel32,
                    ),
                    expected,
                )
                kernel32.OpenProcess.assert_called_once_with(0x00100000, 0, 42)
                kernel32.WaitForSingleObject.assert_called_once_with(123, 0)
                kernel32.CloseHandle.assert_called_once_with(123)

    def test_windows_open_failure_proves_only_missing_pid(self) -> None:
        for error, expected in ((87, False), (5, True), (6, True)):
            with self.subTest(error=error):
                kernel32 = SimpleNamespace(
                    OpenProcess=mock.Mock(return_value=0),
                )
                ctypes_api = SimpleNamespace(
                    c_uint32=int,
                    c_int=int,
                    c_void_p=lambda value: value,
                    get_last_error=mock.Mock(return_value=error),
                )

                self.assertEqual(
                    process_module._windows_process_is_alive(
                        42,
                        _ctypes=ctypes_api,
                        _kernel32=kernel32,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
