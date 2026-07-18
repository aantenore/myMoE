from __future__ import annotations

import os
from typing import Any


def process_is_alive(pid: int) -> bool:
    """Return False only when process absence can be proven safely."""

    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_is_alive(
    pid: int,
    *,
    _ctypes: Any = None,
    _kernel32: Any = None,
) -> bool:
    if _ctypes is None:
        import ctypes

        _ctypes = ctypes
    if _kernel32 is None:
        _kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)

    open_process = _kernel32.OpenProcess
    open_process.argtypes = [_ctypes.c_uint32, _ctypes.c_int, _ctypes.c_uint32]
    open_process.restype = _ctypes.c_void_p
    handle = open_process(0x00100000, 0, pid)
    if not handle:
        # ERROR_INVALID_PARAMETER proves that no process owns this PID. Access
        # denial and unknown failures preserve the lock conservatively.
        return _ctypes.get_last_error() != 87
    try:
        wait = _kernel32.WaitForSingleObject
        wait.argtypes = [_ctypes.c_void_p, _ctypes.c_uint32]
        wait.restype = _ctypes.c_uint32
        result = wait(_ctypes.c_void_p(handle), 0)
        if result == 0:
            return False
        # WAIT_TIMEOUT proves liveness; unexpected results fail closed as live.
        return True
    finally:
        close = _kernel32.CloseHandle
        close.argtypes = [_ctypes.c_void_p]
        close.restype = _ctypes.c_int
        close(_ctypes.c_void_p(handle))
