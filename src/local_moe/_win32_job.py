from __future__ import annotations

import os


class WindowsJobError(RuntimeError):
    """Raised when a kill-on-close Windows Job Object cannot be configured."""


if os.name == "nt":  # pragma: no cover - exercised by the Windows CI runner.
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9

    class _IoCounters(ctypes.Structure):
        _fields_ = (
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        )

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL


class WindowsKillJob:
    """Own a Windows Job Object that terminates every descendant on close."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise WindowsJobError("Windows Job Objects are unavailable on this platform.")
        handle = _kernel32.CreateJobObjectW(None, None)  # type: ignore[name-defined]
        if not handle:
            raise WindowsJobError(_last_error("CreateJobObjectW"))
        self._handle = handle
        information = _ExtendedLimitInformation()  # type: ignore[name-defined]
        information.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE  # type: ignore[name-defined]
        )
        configured = _kernel32.SetInformationJobObject(  # type: ignore[name-defined]
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,  # type: ignore[name-defined]
            ctypes.byref(information),  # type: ignore[name-defined]
            ctypes.sizeof(information),  # type: ignore[name-defined]
        )
        if not configured:
            error_code = ctypes.get_last_error()  # type: ignore[name-defined]
            self.close()
            raise WindowsJobError(
                _last_error("SetInformationJobObject", code=error_code)
            )

    def assign(self, process_handle: int) -> None:
        if self._handle is None:
            raise WindowsJobError("Windows Job Object is closed.")
        assigned = _kernel32.AssignProcessToJobObject(  # type: ignore[name-defined]
            self._handle,
            wintypes.HANDLE(process_handle),  # type: ignore[name-defined]
        )
        if not assigned:
            raise WindowsJobError(_last_error("AssignProcessToJobObject"))

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        self._handle = None
        if handle is not None:
            _kernel32.CloseHandle(handle)  # type: ignore[name-defined]


def _last_error(operation: str, *, code: int | None = None) -> str:
    if os.name != "nt":
        return f"{operation} failed"
    resolved = ctypes.get_last_error() if code is None else code  # type: ignore[name-defined]
    return f"{operation} failed with Windows error {resolved}"
