from __future__ import annotations

import os
from pathlib import Path
import stat

from .cell_contracts import CellContractError


_READ_CHUNK_BYTES = 64 * 1024


def read_bounded_regular_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one stable regular file without following path components.

    When ``root`` is provided, ``path`` must remain below that directory.  The
    root and every descendant prefix are held open for the duration of the
    read.  An empty regular file is valid; callers own content-level policy.
    """

    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
    ):
        raise CellContractError("maximum_bytes must be a positive integer.")
    if not isinstance(label, str) or not label.strip():
        raise CellContractError("Secure file label must be non-empty.")
    root_path, relative = _confined_paths(path, root=root, label=label)
    if os.name == "nt":
        return _read_windows(
            root_path,
            relative,
            maximum_bytes=maximum_bytes,
            label=label,
        )
    return _read_posix(
        root_path,
        relative,
        maximum_bytes=maximum_bytes,
        label=label,
    )


def _confined_paths(
    path: str | Path,
    *,
    root: str | Path | None,
    label: str,
) -> tuple[Path, Path]:
    try:
        if root is None:
            target = Path(os.path.abspath(os.fspath(path)))
            lexical_root = target.parent
            relative = Path(target.name)
        else:
            lexical_root = Path(os.path.abspath(os.fspath(root)))
            supplied = Path(os.fspath(path))
            target = Path(
                os.path.abspath(
                    os.fspath(
                        supplied
                        if supplied.is_absolute()
                        else lexical_root / supplied
                    )
                )
            )
            relative = target.relative_to(lexical_root)
        # POSIX opens the canonical trusted root one component at a time below.
        # Resolving it first preserves legitimate platform aliases (for example
        # macOS /var -> /private/var) while preventing an alias from being
        # followed later during the descriptor walk. Windows instead pins and
        # validates every original prefix as a non-reparse directory.
        root_path = (
            lexical_root
            if os.name == "nt"
            else lexical_root.resolve(strict=True)
        )
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise CellContractError(f"{label} path is invalid or leaves its root.") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise CellContractError(f"{label} must name a file below its root.")
    if os.name == "nt" and any(
        ":" in part or part.endswith((" ", ".")) for part in relative.parts
    ):
        raise CellContractError(f"{label} uses an ambiguous Windows path component.")
    return root_path, relative


def _read_posix(
    root: Path,
    relative: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise CellContractError(
            "Secure no-follow file reads are unavailable on this platform."
        )
    no_follow = getattr(os, "O_NOFOLLOW")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = (
        os.O_RDONLY | close_on_exec | no_follow | getattr(os, "O_DIRECTORY")
    )
    # A FIFO opened read-only can block before a descriptor exists to inspect.
    # Non-blocking mode is harmless for regular files and lets the type check
    # reject special files without waiting for a peer process.
    file_flags = (
        os.O_RDONLY
        | close_on_exec
        | no_follow
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptors: list[int] = []
    try:
        if not root.is_absolute() or not root.anchor:
            raise CellContractError(f"{label} root must be absolute.")
        current = os.open(root.anchor, directory_flags)
        descriptors.append(current)
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise CellContractError(f"{label} root must be a real directory.")
        for component in root.parts[1:]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise CellContractError(f"{label} root must be a real directory.")
        for component in relative.parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise CellContractError(f"{label} parent must be a real directory.")
        descriptor = os.open(relative.parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        return _read_stable_descriptor(
            descriptor,
            maximum_bytes=maximum_bytes,
            label=label,
        )
    except CellContractError:
        raise
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise CellContractError(f"Unable to securely read {label} file.") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_windows(
    root: Path,
    relative: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    from . import _win32_fs

    descriptors: list[tuple[int, object, os.stat_result, bool]] = []
    try:
        prefixes = _windows_directory_prefixes(root, label=label)
        current = prefixes[-1]
        for prefix in prefixes:
            descriptor, identity = _win32_fs.open_nofollow_fd(
                prefix,
                directory=True,
                writable=False,
                share_delete=False,
            )
            before = _validate_windows_descriptor(
                descriptor,
                identity,
                directory=True,
                label=f"{label} root prefix",
            )
            descriptors.append((descriptor, identity, before, True))
        for component in relative.parts[:-1]:
            current = current / component
            descriptor, identity = _win32_fs.open_nofollow_fd(
                current,
                directory=True,
                writable=False,
                share_delete=False,
            )
            before = _validate_windows_descriptor(
                descriptor,
                identity,
                directory=True,
                label=f"{label} parent",
            )
            descriptors.append((descriptor, identity, before, True))
        target = current / relative.parts[-1]
        descriptor, identity = _win32_fs.open_nofollow_fd(
            target,
            directory=False,
            writable=False,
            share_delete=False,
        )
        before = _validate_windows_descriptor(
            descriptor,
            identity,
            directory=False,
            label=label,
        )
        descriptors.append((descriptor, identity, before, False))
        content = _read_stable_descriptor(
            descriptor,
            maximum_bytes=maximum_bytes,
            label=label,
            expected=before,
        )
        for pinned, expected_identity, expected_stat, directory in descriptors:
            observed_identity = _win32_fs.identity_from_fd(pinned)
            observed_stat = _validate_windows_descriptor(
                pinned,
                observed_identity,
                directory=directory,
                label=label,
            )
            if (
                not expected_identity.same_file_as(observed_identity)
                or expected_identity.is_reparse_point
                or observed_identity.is_reparse_point
                or _type_bits(expected_stat.st_mode)
                != _type_bits(observed_stat.st_mode)
            ):
                raise CellContractError(
                    f"{label} identity changed while it was being read."
                )
        return content
    except CellContractError:
        raise
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise CellContractError(f"Unable to securely read {label} file.") from exc
    finally:
        for descriptor, _, _, _ in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _windows_directory_prefixes(root: Path, *, label: str) -> tuple[Path, ...]:
    anchor = root.anchor
    if not anchor:
        raise CellContractError(f"{label} root must be absolute.")
    current = Path(anchor)
    prefixes = [current]
    for component in root.parts[1:]:
        current = current / component
        prefixes.append(current)
    return tuple(prefixes)


def _validate_windows_descriptor(
    descriptor: int,
    identity: object,
    *,
    directory: bool,
    label: str,
) -> os.stat_result:
    from ._win32_fs import Win32FileIdentity, identity_from_fd

    if not isinstance(identity, Win32FileIdentity) or identity.is_reparse_point:
        raise CellContractError(f"{label} must not be a reparse point.")
    observed_identity = identity_from_fd(descriptor)
    if observed_identity.is_reparse_point or not identity.same_file_as(
        observed_identity
    ):
        raise CellContractError(f"{label} identity changed while it was being opened.")
    observed = os.fstat(descriptor)
    valid_type = (
        stat.S_ISDIR(observed.st_mode) if directory else stat.S_ISREG(observed.st_mode)
    )
    if not valid_type:
        expected = "directory" if directory else "regular file"
        raise CellContractError(f"{label} must be a real {expected}.")
    return observed


def _read_stable_descriptor(
    descriptor: int,
    *,
    maximum_bytes: int,
    label: str,
    expected: os.stat_result | None = None,
) -> bytes:
    before = os.fstat(descriptor) if expected is None else expected
    if not stat.S_ISREG(before.st_mode):
        raise CellContractError(f"{label} must be a regular non-link file.")
    if before.st_size < 0 or before.st_size > maximum_bytes:
        raise CellContractError(f"{label} exceeds the bounded size limit.")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > maximum_bytes:
            raise CellContractError(f"{label} exceeds the bounded size limit.")
        chunks.append(chunk)
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise CellContractError(f"{label} changed while it was being read.")
    if total != after.st_size:
        raise CellContractError(f"{label} changed while it was being read.")
    return b"".join(chunks)


def _type_bits(mode: int) -> int:
    return stat.S_IFMT(mode)


__all__ = ["read_bounded_regular_file"]
