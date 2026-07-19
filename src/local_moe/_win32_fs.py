from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import os


_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_MOVEFILE_WRITE_THROUGH = 0x00000008
_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
_FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
_FILE_ID_INFO_CLASS = 18

_DWORD = ctypes.c_uint32
_BOOL = ctypes.c_int
_HANDLE = ctypes.c_void_p


class _FILE_ID_128(ctypes.Structure):
    _fields_ = [("Identifier", ctypes.c_ubyte * 16)]


class _FILE_ID_INFO(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_ulonglong),
        ("FileId", _FILE_ID_128),
    ]


class _FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
    _fields_ = [
        ("FileAttributes", _DWORD),
        ("ReparseTag", _DWORD),
    ]


@dataclass(frozen=True)
class Win32FileIdentity:
    """Stable Win32 identity and final-component reparse metadata."""

    volume_serial: int
    file_id: bytes
    attributes: int
    reparse_tag: int

    def __post_init__(self) -> None:
        file_id = bytes(self.file_id)
        if len(file_id) != 16:
            raise ValueError("Win32 file IDs must contain exactly 16 bytes.")
        object.__setattr__(self, "file_id", file_id)

    def same_file_as(self, other: object) -> bool:
        """Compare only the stable Win32 file-identity components.

        File attributes and the reparse tag are observations that can change
        while a handle remains bound to the same file. Callers must validate
        those observations separately before relying on this continuity check.
        """

        return (
            isinstance(other, Win32FileIdentity)
            and self.volume_serial == other.volume_serial
            and self.file_id == other.file_id
        )

    @property
    def is_reparse_point(self) -> bool:
        return bool(
            self.attributes & _FILE_ATTRIBUTE_REPARSE_POINT or self.reparse_tag
        )


def open_nofollow_fd(
    path: str | os.PathLike[str],
    *,
    directory: bool = False,
    writable: bool = False,
    share_delete: bool = False,
) -> tuple[int, Win32FileIdentity]:
    """Open an existing Win32 path without following its final reparse point.

    The returned CRT descriptor owns the underlying Win32 handle and must be
    closed with :func:`os.close`. Before that ownership transfer, all failures
    close the raw handle. Omitting delete sharing pins the directory entry while
    the descriptor remains open.
    """

    kernel32 = _load_kernel32()
    create_file = kernel32.CreateFileW
    _set_signature(
        create_file,
        [
            ctypes.c_wchar_p,
            _DWORD,
            _DWORD,
            ctypes.c_void_p,
            _DWORD,
            _DWORD,
            _HANDLE,
        ],
        _HANDLE,
    )

    desired_access = _GENERIC_READ
    if writable:
        desired_access |= _GENERIC_WRITE
    share_mode = _FILE_SHARE_READ | _FILE_SHARE_WRITE
    if share_delete:
        share_mode |= _FILE_SHARE_DELETE
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS

    raw_handle = create_file(
        os.fspath(path),
        desired_access,
        share_mode,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    handle = _handle_value(raw_handle)
    if _is_invalid_handle(handle):
        _raise_win32_error(
            _last_error(),
            "CreateFileW could not open the path",
            os.fspath(path),
        )

    owns_raw_handle = True
    try:
        identity = _identity_from_handle(handle, kernel32=kernel32)
        if identity.is_reparse_point:
            raise OSError(
                errno.ELOOP,
                "Refusing to follow a Win32 symbolic link or reparse point",
                os.fspath(path),
            )

        msvcrt = _load_msvcrt()
        descriptor_flags = os.O_RDWR if writable else os.O_RDONLY
        descriptor_flags |= getattr(os, "O_BINARY", 0)
        descriptor = int(msvcrt.open_osfhandle(handle, descriptor_flags))
        if descriptor < 0:
            raise OSError(errno.EBADF, "open_osfhandle returned an invalid descriptor")
        owns_raw_handle = False
        return descriptor, identity
    finally:
        if owns_raw_handle:
            _close_handle(kernel32, handle)


def identity_from_fd(fd: int) -> Win32FileIdentity:
    """Return the 128-bit Win32 file identity and reparse metadata for a CRT fd."""

    msvcrt = _load_msvcrt()
    handle = _handle_value(msvcrt.get_osfhandle(fd))
    if _is_invalid_handle(handle):
        raise OSError(errno.EBADF, "The CRT descriptor has no valid Win32 handle")
    return _identity_from_handle(handle, kernel32=_load_kernel32())


def move_no_replace(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    write_through: bool = True,
) -> None:
    """Atomically move a Win32 path while refusing to replace the target."""

    kernel32 = _load_kernel32()
    move = kernel32.MoveFileExW
    _set_signature(
        move,
        [ctypes.c_wchar_p, ctypes.c_wchar_p, _DWORD],
        _BOOL,
    )
    flags = _MOVEFILE_WRITE_THROUGH if write_through else 0
    if move(os.fspath(source), os.fspath(target), flags):
        return

    error = _last_error()
    if error in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS}:
        exception = FileExistsError(
            error,
            "Win32 move target already exists",
            os.fspath(target),
        )
        exception.winerror = error
        raise exception
    _raise_win32_error(
        error,
        "MoveFileExW could not move the path without replacement",
        os.fspath(target),
    )


def _identity_from_handle(handle: int, *, kernel32: object) -> Win32FileIdentity:
    query = kernel32.GetFileInformationByHandleEx
    _set_signature(
        query,
        [_HANDLE, ctypes.c_int, ctypes.c_void_p, _DWORD],
        _BOOL,
    )

    attribute_info = _FILE_ATTRIBUTE_TAG_INFO()
    if not query(
        handle,
        _FILE_ATTRIBUTE_TAG_INFO_CLASS,
        ctypes.byref(attribute_info),
        ctypes.sizeof(attribute_info),
    ):
        _raise_win32_error(
            _last_error(),
            "GetFileInformationByHandleEx(FileAttributeTagInfo) failed",
        )

    file_id_info = _FILE_ID_INFO()
    if not query(
        handle,
        _FILE_ID_INFO_CLASS,
        ctypes.byref(file_id_info),
        ctypes.sizeof(file_id_info),
    ):
        _raise_win32_error(
            _last_error(),
            "GetFileInformationByHandleEx(FileIdInfo) failed",
        )

    return Win32FileIdentity(
        volume_serial=int(file_id_info.VolumeSerialNumber),
        file_id=bytes(file_id_info.FileId.Identifier),
        attributes=int(attribute_info.FileAttributes),
        reparse_tag=int(attribute_info.ReparseTag),
    )


def _load_kernel32() -> object:
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise OSError(errno.ENOSYS, "Win32 filesystem APIs are unavailable")
    return win_dll("kernel32", use_last_error=True)


def _load_msvcrt() -> object:
    try:
        import msvcrt
    except ImportError as exc:
        raise OSError(errno.ENOSYS, "The Windows CRT is unavailable") from exc
    return msvcrt


def _last_error() -> int:
    get_last_error = getattr(ctypes, "get_last_error", None)
    if get_last_error is None:
        return 0
    return int(get_last_error())


def _handle_value(handle: object) -> int | None:
    if handle is None:
        return None
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    if value is None:
        return None
    return int(value)


def _is_invalid_handle(handle: int | None) -> bool:
    return handle is None or handle in {-1, ctypes.c_void_p(-1).value}


def _close_handle(kernel32: object, handle: int) -> None:
    close_handle = kernel32.CloseHandle
    _set_signature(close_handle, [_HANDLE], _BOOL)
    close_handle(handle)


def _set_signature(function: object, argtypes: list[object], restype: object) -> None:
    function.argtypes = argtypes
    function.restype = restype


def _raise_win32_error(error: int, message: str, path: str | None = None) -> None:
    effective_error = int(error) or errno.EIO
    exception_type: type[OSError]
    if error in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
        exception_type = FileNotFoundError
    else:
        exception_type = OSError
    exception = exception_type(effective_error, message, path)
    exception.winerror = int(error)
    raise exception


__all__ = [
    "Win32FileIdentity",
    "identity_from_fd",
    "move_no_replace",
    "open_nofollow_fd",
]
