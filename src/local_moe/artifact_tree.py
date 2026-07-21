from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat

from .cell_contracts import CellContractError
from .secure_files import (
    SecureFileLimitError,
    hash_bounded_regular_descriptor,
    hash_bounded_regular_file,
)
from .verified_routing_contracts import (
    require_non_negative_int,
    require_sha256,
    sha256_json,
)


HARD_MAX_FILES = 100_000
HARD_MAX_TOTAL_BYTES = 2 * 1024**4
HARD_MAX_DEPTH = 64
HARD_MAX_FILE_BYTES = 2 * 1024**4

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class ArtifactTreeLimitError(CellContractError):
    """Raised when an artifact exceeds an explicit traversal or hashing budget."""


def _bounded_int(value: object, label: str, maximum: int) -> int:
    try:
        rendered = require_non_negative_int(value, label)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CellContractError(str(exc)) from exc
    if rendered > maximum:
        raise CellContractError(f"{label} exceeds the supported bound.")
    return rendered


def _positive_int(value: object, label: str, maximum: int) -> int:
    rendered = _bounded_int(value, label, maximum)
    if rendered < 1:
        raise CellContractError(f"{label} must be positive.")
    return rendered


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CellContractError(f"{label} must be a relative POSIX path.")
    if not value or "\\" in value or "\x00" in value:
        raise CellContractError(f"{label} must be a relative POSIX path.")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or path.as_posix() != value
    ):
        raise CellContractError(f"{label} must stay below its artifact root.")
    return value


@dataclass(frozen=True)
class ArtifactTreeLimits:
    max_files: int
    max_total_bytes: int
    max_depth: int
    max_file_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_files",
            _positive_int(self.max_files, "max_files", HARD_MAX_FILES),
        )
        object.__setattr__(
            self,
            "max_total_bytes",
            _positive_int(
                self.max_total_bytes,
                "max_total_bytes",
                HARD_MAX_TOTAL_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_depth",
            _positive_int(self.max_depth, "max_depth", HARD_MAX_DEPTH),
        )
        object.__setattr__(
            self,
            "max_file_bytes",
            _positive_int(
                self.max_file_bytes,
                "max_file_bytes",
                HARD_MAX_FILE_BYTES,
            ),
        )


@dataclass(frozen=True)
class ArtifactTreeEntry:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "artifact path"))
        object.__setattr__(
            self,
            "size_bytes",
            _bounded_int(
                self.size_bytes,
                "artifact size_bytes",
                HARD_MAX_FILE_BYTES,
            ),
        )
        try:
            digest = require_sha256(self.sha256, "artifact sha256")
        except (TypeError, ValueError) as exc:
            raise CellContractError(str(exc)) from exc
        object.__setattr__(self, "sha256", digest)

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactTreeIdentity:
    kind: str
    entries: tuple[ArtifactTreeEntry, ...]
    file_count: int
    total_bytes: int
    hashed_bytes: int
    digest: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"directory", "file"}:
            raise CellContractError("Artifact identity kind is unsupported.")
        entries = tuple(self.entries) if isinstance(self.entries, (tuple, list)) else ()
        if any(not isinstance(item, ArtifactTreeEntry) for item in entries):
            raise CellContractError("Artifact identity entries are invalid.")
        if entries != tuple(sorted(entries, key=lambda item: item.path)):
            raise CellContractError("Artifact entries must be sorted by path.")
        paths = [item.path for item in entries]
        if len(paths) != len(set(paths)):
            raise CellContractError("Artifact entry paths must be unique.")
        count = _bounded_int(self.file_count, "file_count", HARD_MAX_FILES)
        total = _bounded_int(
            self.total_bytes,
            "total_bytes",
            HARD_MAX_TOTAL_BYTES,
        )
        hashed = _bounded_int(
            self.hashed_bytes,
            "hashed_bytes",
            HARD_MAX_TOTAL_BYTES,
        )
        if count != len(entries) or total != sum(item.size_bytes for item in entries):
            raise CellContractError(
                "Artifact identity counts do not match its entries."
            )
        if hashed > total:
            raise CellContractError("hashed_bytes cannot exceed total_bytes.")
        if self.kind == "file" and count != 1:
            raise CellContractError("A file artifact identity must contain one entry.")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "file_count", count)
        object.__setattr__(self, "total_bytes", total)
        object.__setattr__(self, "hashed_bytes", hashed)
        expected = sha256_json(self.content_payload())
        if self.digest:
            try:
                supplied = require_sha256(self.digest, "artifact tree digest")
            except (TypeError, ValueError) as exc:
                raise CellContractError(str(exc)) from exc
            if supplied != expected:
                raise CellContractError(
                    "Artifact tree digest does not match its content."
                )
        object.__setattr__(self, "digest", expected)

    def content_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "entries": [item.payload() for item in self.entries],
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }

    def payload(self) -> dict[str, object]:
        return {**self.content_payload(), "digest": self.digest}


@dataclass(frozen=True)
class _FileObservation:
    relative_parts: tuple[str, ...]
    fields: tuple[int, ...]


def hash_artifact_tree(
    path: str | Path,
    *,
    root: str | Path,
    limits: ArtifactTreeLimits,
) -> ArtifactTreeIdentity:
    """Hash a stable artifact file or tree under a trusted root.

    POSIX directory traversal is descriptor-relative and never follows links.
    Directory traversal fails closed on platforms without the required
    primitives. Files are streamed by :func:`hash_bounded_regular_file`.
    """

    if not isinstance(limits, ArtifactTreeLimits):
        raise CellContractError("Artifact tree limits are invalid.")
    root_path, relative = _confined_target(path, root=root)
    if os.name == "nt":
        return _hash_windows_target(root_path, relative, limits=limits)
    if not _supports_secure_directory_walk():
        raise CellContractError(
            "Secure no-follow artifact traversal is unavailable on this platform."
        )
    return _hash_posix_target(root_path, relative, limits=limits)


def _confined_target(
    path: str | Path,
    *,
    root: str | Path,
) -> tuple[Path, tuple[str, ...]]:
    try:
        lexical_root = Path(os.path.abspath(os.fspath(root)))
        supplied = Path(os.fspath(path))
        target = Path(
            os.path.abspath(
                os.fspath(
                    supplied if supplied.is_absolute() else lexical_root / supplied
                )
            )
        )
        relative = target.relative_to(lexical_root)
        root_path = (
            lexical_root if os.name == "nt" else lexical_root.resolve(strict=True)
        )
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise CellContractError("Artifact path is invalid or leaves its root.") from exc
    if os.name == "nt" and any(
        ":" in part or part.endswith((" ", ".")) for part in relative.parts
    ):
        raise CellContractError("Artifact path uses an ambiguous component.")
    return root_path, tuple(relative.parts)


def _supports_secure_directory_walk() -> bool:
    return bool(
        os.name != "nt"
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.scandir in os.supports_fd
    )


def _hash_windows_target(
    root: Path,
    relative: tuple[str, ...],
    *,
    limits: ArtifactTreeLimits,
) -> ArtifactTreeIdentity:
    if not relative:
        raise CellContractError(
            "Secure no-follow directory traversal is unavailable on Windows."
        )
    target = root.joinpath(*relative)
    try:
        digest, size = hash_bounded_regular_file(
            target,
            root=root,
            maximum_bytes=min(limits.max_file_bytes, limits.max_total_bytes),
            label="artifact file",
        )
    except SecureFileLimitError as exc:
        raise ArtifactTreeLimitError(
            "Artifact file exceeds the configured byte bound."
        ) from exc
    if size > limits.max_total_bytes:
        raise ArtifactTreeLimitError("Artifact file exceeds the configured byte bound.")
    return ArtifactTreeIdentity(
        kind="file",
        entries=(ArtifactTreeEntry(target.name, size, digest),),
        file_count=1,
        total_bytes=size,
        hashed_bytes=size,
    )


def _hash_posix_target(
    root: Path,
    relative: tuple[str, ...],
    *,
    limits: ArtifactTreeLimits,
) -> ArtifactTreeIdentity:
    pinned = _open_absolute_directory(root)
    try:
        current = pinned[-1]
        for component in relative[:-1]:
            current = _open_child_directory(current, component)
            pinned.append(current)
        if not relative:
            target_before = os.fstat(current)
            return _hash_directory_descriptor(
                current,
                target_before=target_before,
                limits=limits,
            )

        name = relative[-1]
        target_before = _stat_child(current, name)
        if _is_link_or_reparse(target_before):
            raise CellContractError(
                "Artifact links and reparse points are not allowed."
            )
        if stat.S_ISREG(target_before.st_mode):
            return _hash_single_posix_file(
                current,
                name,
                before=target_before,
                limits=limits,
            )
        if not stat.S_ISDIR(target_before.st_mode):
            raise CellContractError(
                "Artifact target must be a regular file or directory."
            )
        target_descriptor = _open_child_directory(current, name, expected=target_before)
        pinned.append(target_descriptor)
        return _hash_directory_descriptor(
            target_descriptor,
            target_before=target_before,
            limits=limits,
        )
    except CellContractError:
        raise
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise CellContractError("Artifact could not be inspected safely.") from exc
    finally:
        for descriptor in reversed(pinned):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _hash_single_posix_file(
    parent_descriptor: int,
    name: str,
    *,
    before: os.stat_result,
    limits: ArtifactTreeLimits,
) -> ArtifactTreeIdentity:
    size = int(before.st_size)
    if size < 0 or size > limits.max_file_bytes or size > limits.max_total_bytes:
        raise ArtifactTreeLimitError("Artifact file exceeds the configured byte bound.")
    descriptor = _open_child_regular_file(
        parent_descriptor,
        name,
        expected=before,
    )
    try:
        digest, observed_size = hash_bounded_regular_descriptor(
            descriptor,
            maximum_bytes=min(limits.max_file_bytes, limits.max_total_bytes),
            label="artifact file",
        )
        after = _stat_child(parent_descriptor, name)
        if (
            observed_size != size
            or not stat.S_ISREG(after.st_mode)
            or _stable_fields(after) != _stable_fields(before)
        ):
            raise CellContractError("Artifact file changed while it was being hashed.")
    finally:
        os.close(descriptor)
    return ArtifactTreeIdentity(
        kind="file",
        entries=(ArtifactTreeEntry(name, size, digest),),
        file_count=1,
        total_bytes=size,
        hashed_bytes=size,
    )


def _hash_directory_descriptor(
    descriptor: int,
    *,
    target_before: os.stat_result,
    limits: ArtifactTreeLimits,
) -> ArtifactTreeIdentity:
    entries: list[ArtifactTreeEntry] = []
    observations: list[_FileObservation] = []
    hardlinks: dict[tuple[int, int], tuple[str, int, tuple[int, ...]]] = {}
    totals = [0, 0]
    visited_entries = [0]
    _walk_directory(
        descriptor,
        entry_relative=(),
        depth=0,
        limits=limits,
        entries=entries,
        observations=observations,
        hardlinks=hardlinks,
        totals=totals,
        visited_entries=visited_entries,
    )
    after = os.fstat(descriptor)
    if _stable_fields(after) != _stable_fields(target_before):
        raise CellContractError("Artifact tree changed while it was being hashed.")
    for observation in observations:
        observed = _stat_relative_nofollow(descriptor, observation.relative_parts)
        if (
            not stat.S_ISREG(observed.st_mode)
            or _stable_fields(observed) != observation.fields
        ):
            raise CellContractError("Artifact tree changed while it was being hashed.")
    ordered = tuple(sorted(entries, key=lambda item: item.path))
    if not ordered:
        raise CellContractError("Artifact directory must contain at least one file.")
    return ArtifactTreeIdentity(
        kind="directory",
        entries=ordered,
        file_count=len(ordered),
        total_bytes=totals[0],
        hashed_bytes=totals[1],
    )


def _walk_directory(
    descriptor: int,
    *,
    entry_relative: tuple[str, ...],
    depth: int,
    limits: ArtifactTreeLimits,
    entries: list[ArtifactTreeEntry],
    observations: list[_FileObservation],
    hardlinks: dict[tuple[int, int], tuple[str, int, tuple[int, ...]]],
    totals: list[int],
    visited_entries: list[int],
) -> None:
    before_directory = os.fstat(descriptor)
    if not stat.S_ISDIR(before_directory.st_mode) or _is_link_or_reparse(
        before_directory
    ):
        raise CellContractError("Artifact directory must be a real directory.")
    names: list[str] = []
    try:
        with os.scandir(descriptor) as directory_entries:
            for directory_entry in directory_entries:
                if visited_entries[0] >= limits.max_files:
                    raise ArtifactTreeLimitError(
                        "Artifact tree exceeds the configured entry bound."
                    )
                visited_entries[0] += 1
                names.append(directory_entry.name)
        names.sort()
    except ArtifactTreeLimitError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise CellContractError(
            "Artifact directory could not be enumerated safely."
        ) from exc
    for name in names:
        if not isinstance(name, str) or name in {"", ".", ".."}:
            raise CellContractError("Artifact directory contains an invalid name.")
        relative_parts = (*entry_relative, name)
        child_depth = depth + 1
        if child_depth > limits.max_depth:
            raise ArtifactTreeLimitError(
                "Artifact tree exceeds the configured depth bound."
            )
        observed = _stat_child(descriptor, name)
        if _is_link_or_reparse(observed):
            raise CellContractError(
                "Artifact links and reparse points are not allowed."
            )
        if stat.S_ISDIR(observed.st_mode):
            child = _open_child_directory(descriptor, name, expected=observed)
            try:
                _walk_directory(
                    child,
                    entry_relative=relative_parts,
                    depth=child_depth,
                    limits=limits,
                    entries=entries,
                    observations=observations,
                    hardlinks=hardlinks,
                    totals=totals,
                    visited_entries=visited_entries,
                )
            finally:
                os.close(child)
            after_child = _stat_child(descriptor, name)
            if not stat.S_ISDIR(after_child.st_mode) or _stable_fields(
                after_child
            ) != _stable_fields(observed):
                raise CellContractError(
                    "Artifact tree changed while it was being hashed."
                )
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise CellContractError("Artifact tree contains a special file.")
        size = int(observed.st_size)
        if size < 0 or size > limits.max_file_bytes:
            raise ArtifactTreeLimitError(
                "Artifact file exceeds the configured byte bound."
            )
        new_total = totals[0] + size
        if new_total > limits.max_total_bytes:
            raise ArtifactTreeLimitError(
                "Artifact tree exceeds the configured byte bound."
            )
        inode_key = _hardlink_key(observed)
        cached = hardlinks.get(inode_key) if inode_key is not None else None
        if cached is None:
            remaining_total_bytes = limits.max_total_bytes - totals[0]
            if remaining_total_bytes < 1:
                raise ArtifactTreeLimitError(
                    "Artifact tree exceeds the configured byte bound."
                )
            file_descriptor = _open_child_regular_file(
                descriptor,
                name,
                expected=observed,
            )
            try:
                digest, observed_size = hash_bounded_regular_descriptor(
                    file_descriptor,
                    maximum_bytes=min(
                        limits.max_file_bytes,
                        remaining_total_bytes,
                    ),
                    label="artifact file",
                )
                if observed_size != size:
                    raise CellContractError(
                        "Artifact file changed while it was being hashed."
                    )
            finally:
                os.close(file_descriptor)
            totals[1] += observed_size
            if inode_key is not None:
                hardlinks[inode_key] = (digest, observed_size, _stable_fields(observed))
        else:
            digest, observed_size, cached_fields = cached
            if observed_size != size or cached_fields != _stable_fields(observed):
                raise CellContractError(
                    "Hard-linked artifact identity is inconsistent."
                )
        after_file = _stat_child(descriptor, name)
        if not stat.S_ISREG(after_file.st_mode) or _stable_fields(
            after_file
        ) != _stable_fields(observed):
            raise CellContractError("Artifact file changed while it was being hashed.")
        entry_path = _relative_path(
            PurePosixPath(*relative_parts).as_posix(),
            "artifact path",
        )
        entries.append(ArtifactTreeEntry(entry_path, size, digest))
        observations.append(_FileObservation(relative_parts, _stable_fields(observed)))
        totals[0] = new_total
    after_directory = os.fstat(descriptor)
    if _stable_fields(after_directory) != _stable_fields(before_directory):
        raise CellContractError("Artifact tree changed while it was being hashed.")


def _open_absolute_directory(path: Path) -> list[int]:
    if not path.is_absolute() or not path.anchor:
        raise CellContractError("Artifact root must be absolute.")
    descriptors: list[int] = []
    try:
        current = os.open(path.anchor, _DIRECTORY_FLAGS)
        descriptors.append(current)
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            raise CellContractError("Artifact root must be a real directory.")
        for component in path.parts[1:]:
            current = _open_child_directory(current, component)
            descriptors.append(current)
        return descriptors
    except CellContractError:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except (OSError, TypeError, ValueError) as exc:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise CellContractError("Artifact root could not be opened safely.") from exc


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    *,
    expected: os.stat_result | None = None,
) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode) or _is_link_or_reparse(observed):
            raise CellContractError("Artifact directory must be a real directory.")
        if expected is not None and not _same_identity(expected, observed):
            raise CellContractError("Artifact directory changed while it was opened.")
        return descriptor
    except CellContractError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise CellContractError(
            "Artifact directory could not be opened safely."
        ) from exc


def _open_child_regular_file(
    parent_descriptor: int,
    name: str,
    *,
    expected: os.stat_result,
) -> int:
    try:
        descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_descriptor)
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or _is_link_or_reparse(observed):
            raise CellContractError("Artifact file must be a real regular file.")
        if _stable_fields(observed) != _stable_fields(expected):
            raise CellContractError("Artifact file changed while it was opened.")
        return descriptor
    except CellContractError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise CellContractError("Artifact file could not be opened safely.") from exc


def _stat_child(parent_descriptor: int, name: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except (OSError, TypeError, ValueError) as exc:
        raise CellContractError(
            "Artifact entry could not be inspected safely."
        ) from exc


def _stat_relative_nofollow(
    root_descriptor: int,
    relative: tuple[str, ...],
) -> os.stat_result:
    if not relative:
        return os.fstat(root_descriptor)
    descriptors: list[int] = []
    current = root_descriptor
    try:
        for component in relative[:-1]:
            current = _open_child_directory(current, component)
            descriptors.append(current)
        return _stat_child(current, relative[-1])
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _is_link_or_reparse(observed: os.stat_result) -> bool:
    attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(observed.st_mode) or bool(
        int(getattr(observed, "st_file_attributes", 0)) & attribute
    )


def _stable_fields(observed: os.stat_result) -> tuple[int, ...]:
    return tuple(
        int(getattr(observed, name, 0))
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    )


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        int(first.st_dev),
        int(first.st_ino),
        stat.S_IFMT(first.st_mode),
    ) == (
        int(second.st_dev),
        int(second.st_ino),
        stat.S_IFMT(second.st_mode),
    )


def _hardlink_key(observed: os.stat_result) -> tuple[int, int] | None:
    device = int(observed.st_dev)
    inode = int(observed.st_ino)
    links = int(getattr(observed, "st_nlink", 1))
    return (device, inode) if device and inode and links > 1 else None


__all__ = [
    "ArtifactTreeLimitError",
    "ArtifactTreeEntry",
    "ArtifactTreeIdentity",
    "ArtifactTreeLimits",
    "HARD_MAX_DEPTH",
    "HARD_MAX_FILE_BYTES",
    "HARD_MAX_FILES",
    "HARD_MAX_TOTAL_BYTES",
    "hash_artifact_tree",
]
