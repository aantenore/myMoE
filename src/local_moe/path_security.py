from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import stat


class PathBoundaryError(ValueError):
    """Raised when a requested file escapes its configured directory boundary."""


def resolve_existing_file(
    candidate: str | Path,
    *,
    allowed_roots: Iterable[str | Path],
    label: str = "file",
) -> Path:
    """Resolve an existing regular file below one of the configured roots.

    Relative API paths may either include the configured directory (for backwards
    compatibility) or be relative to it. Symlinks that escape a root are rejected
    because both the roots and the candidate are resolved before comparison.
    """

    roots = tuple(_resolve_root(root, label=label) for root in allowed_roots)
    if not roots:
        raise PathBoundaryError(f"No allowed {label} directories are configured.")

    requested = Path(candidate)
    attempts = [requested]
    if not requested.is_absolute():
        attempts.extend(root / requested for root in roots)

    last_error: OSError | None = None
    for attempt in attempts:
        try:
            resolved = attempt.resolve(strict=True)
        except OSError as exc:
            last_error = exc
            continue
        if not resolved.is_file():
            continue
        if any(_is_within(resolved, root) for root in roots):
            return resolved

    detail = f": {last_error}" if last_error is not None else ""
    raise PathBoundaryError(
        f"Requested {label} must be an existing regular file inside a configured directory{detail}"
    )


def read_text_file(
    candidate: str | Path,
    *,
    allowed_roots: Iterable[str | Path],
    label: str = "file",
    max_bytes: int,
) -> tuple[Path, str]:
    """Read a bounded, non-symlink regular file inside configured roots."""

    resolved = resolve_existing_file(candidate, allowed_roots=allowed_roots, label=label)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PathBoundaryError(f"Requested {label} is not a regular file.")
        if metadata.st_size > max_bytes:
            raise PathBoundaryError(
                f"Requested {label} exceeds the configured {max_bytes}-byte limit."
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            return resolved, stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _resolve_root(root: str | Path, *, label: str) -> Path:
    try:
        resolved = Path(root).resolve(strict=True)
    except OSError as exc:
        raise PathBoundaryError(f"Configured {label} directory is unavailable: {root}") from exc
    if not resolved.is_dir():
        raise PathBoundaryError(f"Configured {label} root is not a directory: {root}")
    return resolved


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True
