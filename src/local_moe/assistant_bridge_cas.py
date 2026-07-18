from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import tempfile
import unicodedata
from typing import Any, Iterator, Mapping, Sequence

from .assistant_bridge_integrity import canonical_json_bytes, sha256_bytes
from .assistant_bridge_two_phase_contracts import (
    ArtifactDescriptor,
    MAX_CANDIDATE_FILES,
    MAX_CANDIDATE_FILE_BYTES,
    MAX_CANDIDATE_METADATA_BYTES,
    MAX_CANDIDATE_TOTAL_BYTES,
    TwoPhaseContractError,
    require_sha256,
)


CAS_SCHEMA_VERSION = "1.0"
_EMPTY_SHA256 = sha256_bytes(b"")
_FILE_FIELDS = {"path", "kind", "sha256", "size", "mode", "direction"}
_DIRECTIONS = {"input_only", "round_trip"}
_SOURCE_FIELDS = {
    "rootSha256",
    "fingerprint",
    "gitRepository",
    "headSha",
    "indexSha256",
}
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class ContentAddressedStoreError(ValueError):
    """Raised when immutable candidate content cannot be trusted."""


class ContentAddressedStoreUninitializedError(ContentAddressedStoreError):
    """Raised when a read-only CAS has not been initialized yet."""


def _require_cas_sha256(value: str, label: str) -> str:
    try:
        return require_sha256(value, label)
    except TwoPhaseContractError as exc:
        raise ContentAddressedStoreError(str(exc)) from exc


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or not path.parts
        or path == PurePosixPath(".")
        or any(part in {"", "."} for part in path.parts)
        or _portable_path_component(path.parts[0]) == ".git"
    ):
        raise ContentAddressedStoreError("Candidate path is unsafe.")
    return value


def _portable_path_component(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _portable_path_key(value: str) -> str:
    return "/".join(
        _portable_path_component(part) for part in PurePosixPath(value).parts
    )


def _require_portable_unique_paths(
    paths: Sequence[str], *, label: str
) -> None:
    portable: dict[str, str] = {}
    for path in paths:
        key = _portable_path_key(path)
        existing = portable.get(key)
        if existing is not None and existing != path:
            raise ContentAddressedStoreError(
                f"{label} contains non-portable path collisions."
            )
        portable[key] = path


def _regular_file_state(path: Path) -> os.stat_result:
    try:
        state = path.lstat()
    except OSError as exc:
        raise ContentAddressedStoreError("CAS artifact is unavailable.") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise ContentAddressedStoreError("CAS artifact must be a regular file.")
    return state


class ContentAddressedStore:
    """Immutable SHA-256 CAS with RFC 8785 structured artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        create_if_missing: bool = True,
    ) -> None:
        if not isinstance(create_if_missing, bool):
            raise ContentAddressedStoreError("CAS creation policy is invalid.")
        raw = Path(root).expanduser()
        if raw.is_symlink():
            raise ContentAddressedStoreError("CAS root cannot be a symbolic link.")
        try:
            if create_if_missing:
                raw.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root = raw.resolve(strict=True)
        except FileNotFoundError as exc:
            if not create_if_missing:
                raise ContentAddressedStoreUninitializedError(
                    "CAS root is unavailable."
                ) from exc
            raise ContentAddressedStoreError("CAS root is unavailable.") from exc
        except OSError as exc:
            raise ContentAddressedStoreError("CAS root is unavailable.") from exc
        if not self.root.is_dir():
            raise ContentAddressedStoreError("CAS root must be a directory.")
        objects = self.root / "objects"
        if objects.exists():
            self._validate_internal_directory(objects)
        elif create_if_missing:
            objects.mkdir(mode=0o700)
        else:
            raise ContentAddressedStoreUninitializedError(
                "CAS object store is unavailable."
            )
        self._objects = objects / "sha256"
        if self._objects.exists():
            self._validate_internal_directory(self._objects)
        elif create_if_missing:
            self._objects.mkdir(mode=0o700)
        else:
            raise ContentAddressedStoreUninitializedError(
                "CAS object store is unavailable."
            )
        self._validate_internal_directory(self._objects)

    def put_bytes(self, value: bytes, *, media_type: str) -> ArtifactDescriptor:
        if not isinstance(value, bytes):
            raise ContentAddressedStoreError("CAS accepts bytes only.")
        descriptor = ArtifactDescriptor(
            media_type=media_type,
            sha256=sha256_bytes(value),
            size_bytes=len(value),
        )
        target = self._object_path(descriptor.sha256, create_parent=True)
        if target.exists():
            if self.get_bytes(descriptor) != value:
                raise ContentAddressedStoreError(
                    "Existing CAS object does not match its digest."
                )
            return descriptor
        temporary = target.parent / f".{descriptor.sha256}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor_fd = -1
        try:
            descriptor_fd = os.open(temporary, flags, 0o600)
            _write_all(descriptor_fd, value)
            os.fsync(descriptor_fd)
            os.close(descriptor_fd)
            descriptor_fd = -1
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                pass
            _fsync_directory(target.parent)
        except OSError as exc:
            raise ContentAddressedStoreError("CAS object could not be persisted.") from exc
        finally:
            if descriptor_fd >= 0:
                os.close(descriptor_fd)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if self.get_bytes(descriptor) != value:
            raise ContentAddressedStoreError("CAS object failed post-write verification.")
        return descriptor

    def put_json(self, value: Mapping[str, Any], *, media_type: str) -> ArtifactDescriptor:
        return self.put_bytes(canonical_json_bytes(dict(value)), media_type=media_type)

    def get_bytes(self, descriptor: ArtifactDescriptor) -> bytes:
        value = self._read_verified_artifact(
            descriptor,
            collect=True,
            max_bytes=None,
        )
        assert value is not None
        return value

    def _read_verified_artifact(
        self,
        descriptor: ArtifactDescriptor,
        *,
        collect: bool,
        max_bytes: int | None,
        output_fd: int | None = None,
    ) -> bytes | None:
        if max_bytes is not None and descriptor.size_bytes > max_bytes:
            raise ContentAddressedStoreError("CAS artifact exceeds its read bound.")
        target = self._object_path(descriptor.sha256, create_parent=False)
        before = _regular_file_state(target)
        if before.st_size != descriptor.size_bytes:
            raise ContentAddressedStoreError("CAS artifact size binding failed.")
        descriptor_fd = -1
        try:
            descriptor_fd = os.open(target, _READ_FLAGS)
            opened = os.fstat(descriptor_fd)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise ContentAddressedStoreError("CAS artifact identity changed.")
            chunks: list[bytes] | None = [] if collect else None
            digest = hashlib.sha256()
            remaining = descriptor.size_bytes
            while remaining:
                chunk = os.read(descriptor_fd, min(1024 * 1024, remaining))
                if not chunk:
                    raise ContentAddressedStoreError("CAS artifact is truncated.")
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
                if output_fd is not None:
                    _write_all(output_fd, chunk)
                remaining -= len(chunk)
            if os.read(descriptor_fd, 1):
                raise ContentAddressedStoreError(
                    "CAS artifact exceeds its size binding."
                )
            after = os.fstat(descriptor_fd)
        except OSError as exc:
            raise ContentAddressedStoreError(
                "CAS artifact could not be read safely."
            ) from exc
        finally:
            if descriptor_fd >= 0:
                os.close(descriptor_fd)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ContentAddressedStoreError("CAS artifact changed while read.")
        if digest.hexdigest() != descriptor.sha256:
            raise ContentAddressedStoreError("CAS artifact digest binding failed.")
        return None if chunks is None else b"".join(chunks)

    def get_json(
        self,
        descriptor: ArtifactDescriptor,
        *,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        raw = self._read_verified_artifact(
            descriptor,
            collect=True,
            max_bytes=max_bytes,
        )
        assert raw is not None
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContentAddressedStoreError("CAS JSON artifact is invalid.") from exc
        if not isinstance(value, dict):
            raise ContentAddressedStoreError("CAS JSON artifact must be an object.")
        if canonical_json_bytes(value) != raw:
            raise ContentAddressedStoreError("CAS JSON artifact is not RFC 8785 canonical.")
        return value

    def store_candidate(
        self,
        candidate_root: str | Path,
        candidate_files: Sequence[Mapping[str, Any]],
        changes: Sequence[Mapping[str, Any]],
        *,
        source_fingerprint: str,
        source_identity: Mapping[str, Any],
    ) -> tuple[ArtifactDescriptor, ArtifactDescriptor]:
        _require_cas_sha256(source_fingerprint, "source_fingerprint")
        if (
            len(candidate_files) > MAX_CANDIDATE_FILES
            or len(changes) > MAX_CANDIDATE_FILES
        ):
            raise ContentAddressedStoreError(
                "Candidate file-count bound was exceeded."
            )
        normalized_source = _normalize_source_identity(
            source_identity, source_fingerprint=source_fingerprint
        )
        raw_root = Path(candidate_root)
        if raw_root.is_symlink():
            raise ContentAddressedStoreError("Candidate root cannot be a symbolic link.")
        root = raw_root.resolve(strict=True)
        file_records: list[dict[str, object]] = []
        seen: set[str] = set()
        total_bytes = 0
        for raw in sorted(candidate_files, key=lambda item: str(item.get("path", ""))):
            record = _normalize_file_record(raw)
            path = str(record["path"])
            if path in seen:
                raise ContentAddressedStoreError("Candidate manifest contains duplicates.")
            seen.add(path)
            kind = str(record["kind"])
            digest = str(record["sha256"])
            size = int(record["size"])
            content: ArtifactDescriptor | None = None
            if kind == "file":
                if size > MAX_CANDIDATE_FILE_BYTES:
                    raise ContentAddressedStoreError(
                        "Candidate file byte bound was exceeded."
                    )
                total_bytes += size
                if total_bytes > MAX_CANDIDATE_TOTAL_BYTES:
                    raise ContentAddressedStoreError(
                        "Candidate total byte bound was exceeded."
                    )
                value = _read_candidate_file(root, path, expected_size=size)
                if sha256_bytes(value) != digest:
                    raise ContentAddressedStoreError(
                        "Candidate file no longer matches its manifest."
                    )
                content = self.put_bytes(
                    value,
                    media_type="application/octet-stream",
                )
            file_records.append(
                {
                    "path": path,
                    "kind": kind,
                    "sha256": digest,
                    "size": size,
                    "mode": record["mode"],
                    "direction": record["direction"],
                    "content": None if content is None else content.payload(),
                }
            )
        _require_portable_unique_paths(
            [str(item["path"]) for item in file_records],
            label="Candidate manifest",
        )
        normalized_changes = sorted(
            (_normalize_change(item) for item in changes),
            key=lambda item: str(item["path"]),
        )
        if len({str(item["path"]) for item in normalized_changes}) != len(
            normalized_changes
        ):
            raise ContentAddressedStoreError("Changeset contains duplicate paths.")
        _require_portable_unique_paths(
            [str(item["path"]) for item in normalized_changes],
            label="Changeset",
        )
        _validate_changes_against_manifest(normalized_changes, file_records)
        changeset_payload = {
            "schemaVersion": CAS_SCHEMA_VERSION,
            "sourceFingerprint": source_fingerprint,
            "changes": normalized_changes,
        }
        changeset_bytes = canonical_json_bytes(changeset_payload)
        if len(changeset_bytes) > MAX_CANDIDATE_METADATA_BYTES:
            raise ContentAddressedStoreError(
                "Candidate changeset byte bound was exceeded."
            )
        changeset = self.put_bytes(
            changeset_bytes,
            media_type="application/vnd.mymoe.changeset+json",
        )
        manifest_payload = {
            "schemaVersion": CAS_SCHEMA_VERSION,
            "sourceFingerprint": source_fingerprint,
            "source": normalized_source,
            "files": file_records,
            "changeset": changeset.payload(),
        }
        manifest_bytes = canonical_json_bytes(manifest_payload)
        if len(manifest_bytes) > MAX_CANDIDATE_METADATA_BYTES:
            raise ContentAddressedStoreError(
                "Candidate manifest byte bound was exceeded."
            )
        manifest = self.put_bytes(
            manifest_bytes,
            media_type="application/vnd.mymoe.workspace-manifest+json",
        )
        return manifest, changeset

    @contextmanager
    def materialize_candidate(
        self,
        manifest_descriptor: ArtifactDescriptor,
    ) -> Iterator[Path]:
        _, _, validated_files = self._validated_candidate_descriptors(
            manifest_descriptor,
            expected_changeset=None,
        )
        with tempfile.TemporaryDirectory(
            prefix="mymoe-cas-candidate-"
        ) as temporary:
            root = Path(temporary).resolve(strict=True)
            for raw, descriptor in validated_files:
                path = str(raw["path"])
                if descriptor is None:
                    continue
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if not target.parent.resolve(strict=True).is_relative_to(root):
                    raise ContentAddressedStoreError(
                        "Candidate materialization escaped its root."
                    )
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
                    os, "O_NOFOLLOW", 0
                )
                descriptor_fd = os.open(target, flags, int(raw["mode"]))
                try:
                    self._read_verified_artifact(
                        descriptor,
                        collect=False,
                        max_bytes=MAX_CANDIDATE_FILE_BYTES,
                        output_fd=descriptor_fd,
                    )
                    os.fsync(descriptor_fd)
                finally:
                    os.close(descriptor_fd)
            yield root

    def validate_candidate_closure(
        self,
        manifest_descriptor: ArtifactDescriptor,
        changeset_descriptor: ArtifactDescriptor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read and hash every artifact in a candidate without materializing it."""

        manifest, changeset, validated_files = (
            self._validated_candidate_descriptors(
                manifest_descriptor,
                expected_changeset=changeset_descriptor,
            )
        )
        for _, descriptor in validated_files:
            if descriptor is not None:
                self._read_verified_artifact(
                    descriptor,
                    collect=False,
                    max_bytes=MAX_CANDIDATE_FILE_BYTES,
                )
        return manifest, changeset

    def _validated_candidate_descriptors(
        self,
        manifest_descriptor: ArtifactDescriptor,
        *,
        expected_changeset: ArtifactDescriptor | None,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[tuple[dict[str, object], ArtifactDescriptor | None]],
    ]:
        if (
            manifest_descriptor.media_type
            != "application/vnd.mymoe.workspace-manifest+json"
        ):
            raise ContentAddressedStoreError(
                "Candidate manifest media type is invalid."
            )
        manifest = self.get_json(
            manifest_descriptor,
            max_bytes=MAX_CANDIDATE_METADATA_BYTES,
        )
        if (
            set(manifest)
            != {
                "schemaVersion",
                "sourceFingerprint",
                "source",
                "files",
                "changeset",
            }
            or manifest.get("schemaVersion") != CAS_SCHEMA_VERSION
        ):
            raise ContentAddressedStoreError(
                "Candidate manifest schema is unsupported."
            )
        _require_cas_sha256(
            str(manifest.get("sourceFingerprint", "")),
            "source_fingerprint",
        )
        raw_source = manifest.get("source")
        if not isinstance(raw_source, Mapping):
            raise ContentAddressedStoreError("Candidate source identity is invalid.")
        _normalize_source_identity(
            raw_source,
            source_fingerprint=str(manifest["sourceFingerprint"]),
        )
        raw_changeset = manifest.get("changeset")
        if not isinstance(raw_changeset, Mapping):
            raise ContentAddressedStoreError(
                "Candidate changeset descriptor is invalid."
            )
        try:
            changeset_descriptor = ArtifactDescriptor.from_payload(raw_changeset)
        except TwoPhaseContractError as exc:
            raise ContentAddressedStoreError(str(exc)) from exc
        if (
            changeset_descriptor.media_type
            != "application/vnd.mymoe.changeset+json"
        ):
            raise ContentAddressedStoreError("Candidate changeset media type is invalid.")
        if (
            expected_changeset is not None
            and changeset_descriptor != expected_changeset
        ):
            raise ContentAddressedStoreError(
                "Candidate manifest does not bind the requested changeset."
            )
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ContentAddressedStoreError("Candidate manifest files are invalid.")
        normalized = _validated_manifest_records(files)
        changeset = self.get_json(
            changeset_descriptor,
            max_bytes=MAX_CANDIDATE_METADATA_BYTES,
        )
        _validate_stored_changeset(
            changeset,
            source_fingerprint=str(manifest["sourceFingerprint"]),
            files=normalized,
        )
        validated_files: list[
            tuple[dict[str, object], ArtifactDescriptor | None]
        ] = []
        if len(normalized) > MAX_CANDIDATE_FILES:
            raise ContentAddressedStoreError(
                "Candidate manifest file-count bound was exceeded."
            )
        total_bytes = 0
        for raw in normalized:
            if raw["kind"] == "missing":
                validated_files.append((raw, None))
                continue
            descriptor_raw = raw["content"]
            if not isinstance(descriptor_raw, Mapping):
                raise ContentAddressedStoreError(
                    "Candidate content descriptor is missing."
                )
            try:
                descriptor = ArtifactDescriptor.from_payload(descriptor_raw)
            except TwoPhaseContractError as exc:
                raise ContentAddressedStoreError(str(exc)) from exc
            if (
                descriptor.media_type != "application/octet-stream"
                or descriptor.sha256 != raw["sha256"]
                or descriptor.size_bytes != raw["size"]
            ):
                raise ContentAddressedStoreError(
                    "Candidate content descriptor is incoherent."
                )
            if descriptor.size_bytes > MAX_CANDIDATE_FILE_BYTES:
                raise ContentAddressedStoreError(
                    "Candidate file byte bound was exceeded."
                )
            total_bytes += descriptor.size_bytes
            if total_bytes > MAX_CANDIDATE_TOTAL_BYTES:
                raise ContentAddressedStoreError(
                    "Candidate total byte bound was exceeded."
                )
            validated_files.append((raw, descriptor))
        return manifest, changeset, validated_files

    def load_candidate(
        self,
        manifest_descriptor: ArtifactDescriptor,
        changeset_descriptor: ArtifactDescriptor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return fully validated immutable-artifact payload copies."""

        return self.validate_candidate_closure(
            manifest_descriptor,
            changeset_descriptor,
        )

    def _object_path(self, digest: str, *, create_parent: bool) -> Path:
        _require_cas_sha256(digest, "CAS digest")
        parent = self._objects / digest[:2]
        if create_parent:
            parent.mkdir(mode=0o700, exist_ok=True)
        self._validate_internal_directory(parent)
        return parent / digest[2:]

    def _validate_internal_directory(self, path: Path) -> None:
        try:
            state = path.lstat()
        except OSError as exc:
            raise ContentAddressedStoreError("CAS directory is unavailable.") from exc
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise ContentAddressedStoreError("CAS directory must not be a link.")
        try:
            path.resolve(strict=True).relative_to(self.root)
        except ValueError as exc:
            raise ContentAddressedStoreError("CAS path escaped its root.") from exc


def _normalize_change(value: Mapping[str, Any]) -> dict[str, object]:
    if set(value) != {"path", "before", "after"}:
        raise ContentAddressedStoreError("Changeset record shape is invalid.")
    path = _safe_relative_path(str(value.get("path", "")))
    before = value.get("before")
    after = value.get("after")
    for label, item in (("before", before), ("after", after)):
        if item is not None and not isinstance(item, Mapping):
            raise ContentAddressedStoreError(f"Changeset {label} record is invalid.")
    normalized_before = None if before is None else _normalize_file_record(before)
    normalized_after = None if after is None else _normalize_file_record(after)
    if any(
        item is not None and item["path"] != path
        for item in (normalized_before, normalized_after)
    ):
        raise ContentAddressedStoreError("Changeset record path binding is invalid.")
    if normalized_before is None and normalized_after is None:
        raise ContentAddressedStoreError("Changeset record contains no change.")
    if normalized_before == normalized_after:
        raise ContentAddressedStoreError("Changeset before and after are identical.")
    return {
        "path": path,
        "before": normalized_before,
        "after": normalized_after,
    }


def _normalize_file_record(value: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FILE_FIELDS:
        raise ContentAddressedStoreError("Candidate file record shape is invalid.")
    path = value.get("path")
    kind = value.get("kind")
    digest = value.get("sha256")
    size = value.get("size")
    mode = value.get("mode")
    direction = value.get("direction")
    if not isinstance(path, str):
        raise ContentAddressedStoreError("Candidate path type is invalid.")
    path = _safe_relative_path(path)
    if kind not in {"file", "missing"} or not isinstance(kind, str):
        raise ContentAddressedStoreError("Candidate file kind is unsupported.")
    if not isinstance(digest, str):
        raise ContentAddressedStoreError("Candidate file digest type is invalid.")
    _require_cas_sha256(digest, "candidate file sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 <= size <= 2**63 - 1
    ):
        raise ContentAddressedStoreError("Candidate file size is outside safe bounds.")
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
        raise ContentAddressedStoreError("Candidate file mode is outside safe bounds.")
    if direction not in _DIRECTIONS or not isinstance(direction, str):
        raise ContentAddressedStoreError("Candidate file direction is invalid.")
    if kind == "missing":
        if size != 0 or mode != 0 or digest != _EMPTY_SHA256:
            raise ContentAddressedStoreError("Missing candidate record is incoherent.")
    elif mode == 0:
        raise ContentAddressedStoreError("Regular candidate file mode is invalid.")
    return {
        "path": path,
        "kind": kind,
        "sha256": digest,
        "size": size,
        "mode": mode,
        "direction": direction,
    }


def _validated_manifest_records(values: Sequence[Any]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != _FILE_FIELDS | {"content"}:
            raise ContentAddressedStoreError("Candidate manifest file record is invalid.")
        normalized = _normalize_file_record(
            {name: value[name] for name in _FILE_FIELDS}
        )
        content = value.get("content")
        if normalized["kind"] == "missing" and content is not None:
            raise ContentAddressedStoreError("Missing candidate has content.")
        if normalized["kind"] == "file" and not isinstance(content, Mapping):
            raise ContentAddressedStoreError("Candidate content descriptor is missing.")
        records.append({**normalized, "content": content})
    paths = [str(item["path"]) for item in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContentAddressedStoreError(
            "Candidate manifest paths must be ordered and unique."
        )
    _require_portable_unique_paths(paths, label="Candidate manifest")
    return records


def _validate_changes_against_manifest(
    changes: Sequence[Mapping[str, Any]],
    files: Sequence[Mapping[str, Any]],
) -> None:
    manifest = {
        str(item["path"]): {name: item[name] for name in _FILE_FIELDS}
        for item in files
    }
    for change in changes:
        path = str(change["path"])
        after = change["after"]
        if after is not None and after != manifest.get(path):
            raise ContentAddressedStoreError(
                "Changeset after record does not match the candidate manifest."
            )
        if after is None and path in manifest:
            raise ContentAddressedStoreError(
                "Changeset deletion is not represented by a missing manifest record."
            )


def _validate_stored_changeset(
    value: Mapping[str, Any],
    *,
    source_fingerprint: str,
    files: Sequence[Mapping[str, Any]],
) -> None:
    if set(value) != {"schemaVersion", "sourceFingerprint", "changes"}:
        raise ContentAddressedStoreError("Candidate changeset shape is invalid.")
    if (
        value.get("schemaVersion") != CAS_SCHEMA_VERSION
        or value.get("sourceFingerprint") != source_fingerprint
    ):
        raise ContentAddressedStoreError("Candidate changeset binding is invalid.")
    raw_changes = value.get("changes")
    if not isinstance(raw_changes, list) or not all(
        isinstance(item, Mapping) for item in raw_changes
    ):
        raise ContentAddressedStoreError("Candidate changeset records are invalid.")
    normalized = [_normalize_change(item) for item in raw_changes]
    paths = [str(item["path"]) for item in normalized]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ContentAddressedStoreError(
            "Candidate changeset paths must be ordered and unique."
        )
    _require_portable_unique_paths(paths, label="Candidate changeset")
    _validate_changes_against_manifest(normalized, files)


def _normalize_source_identity(
    value: Mapping[str, Any],
    *,
    source_fingerprint: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
        raise ContentAddressedStoreError("Candidate source identity shape is invalid.")
    root_sha256 = value.get("rootSha256")
    fingerprint = value.get("fingerprint")
    git_repository = value.get("gitRepository")
    head_sha = value.get("headSha")
    index_sha256 = value.get("indexSha256")
    if not isinstance(root_sha256, str) or not isinstance(fingerprint, str):
        raise ContentAddressedStoreError("Candidate source digest types are invalid.")
    _require_cas_sha256(root_sha256, "source root sha256")
    _require_cas_sha256(fingerprint, "source fingerprint")
    if fingerprint != source_fingerprint:
        raise ContentAddressedStoreError("Candidate source fingerprint is incoherent.")
    if not isinstance(git_repository, bool):
        raise ContentAddressedStoreError("Candidate source Git flag is invalid.")
    if head_sha is not None and not isinstance(head_sha, str):
        raise ContentAddressedStoreError("Candidate source HEAD is invalid.")
    if not isinstance(index_sha256, str):
        raise ContentAddressedStoreError("Candidate source index digest is invalid.")
    _require_cas_sha256(index_sha256, "source index sha256")
    if git_repository:
        if not head_sha or len(head_sha) > 128:
            raise ContentAddressedStoreError("Candidate source HEAD is invalid.")
    elif head_sha is not None:
        raise ContentAddressedStoreError("Non-Git source cannot bind a HEAD.")
    return {
        "rootSha256": root_sha256,
        "fingerprint": fingerprint,
        "gitRepository": git_repository,
        "headSha": head_sha,
        "indexSha256": index_sha256,
    }


def _read_candidate_file(root: Path, relative: str, *, expected_size: int) -> bytes:
    target = root / relative
    try:
        resolved_parent = target.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContentAddressedStoreError("Candidate parent path is unsafe.") from exc
    before = _regular_file_state(target)
    if before.st_size != expected_size:
        raise ContentAddressedStoreError("Candidate file size binding failed.")
    descriptor = os.open(target, _READ_FLAGS)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ContentAddressedStoreError("Candidate file identity changed.")
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContentAddressedStoreError("Candidate file is truncated.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContentAddressedStoreError("Candidate file exceeds its size binding.")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ContentAddressedStoreError("Candidate file changed while read.")
    return b"".join(chunks)


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("CAS write made no progress")
        offset += written


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
