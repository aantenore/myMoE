"""Filesystem exchange adapter for an out-of-process signed verifier."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import stat
import time
from typing import Any

from .assistant_bridge_integrity import canonical_json_bytes, sha256_bytes
from .assistant_bridge_two_phase_contracts import CandidateBinding


DIRECTORY_PAIRED_ATTESTATION_CONTRACT = "mymoe-directory-paired-attestation/v1"
DIRECTORY_PAIRED_ATTESTATION_REQUEST_CONTRACT = (
    "DirectoryPairedAttestationRequest"
)
DIRECTORY_PAIRED_ATTESTATION_RESPONSE_CONTRACT = (
    "DirectoryPairedAttestationResponse"
)
_SCHEMA_VERSION = "1.0"
_REQUEST_ID_BYTES = 32
_MAX_REQUEST_BYTES_LIMIT = 8 * 1024 * 1024
_MAX_RESPONSE_BYTES_LIMIT = 64 * 1024 * 1024
_MAX_ENVELOPES_LIMIT = 64
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class DirectoryPairedAttestationError(ValueError):
    """Raised when the sidecar exchange cannot be trusted or completed."""


class DirectoryPairedAttestationTimeout(DirectoryPairedAttestationError):
    """Raised when no valid sidecar response arrives before the deadline."""


@dataclass(frozen=True)
class _DirectoryIdentity:
    device: int
    inode: int
    mode: int


class DirectoryPairedAttestationProducer:
    """Exchange claim-bound requests for signed DSSE envelopes via directories.

    The producer never imports verifier code and never receives signing material.
    An independently launched sidecar reads canonical request files and writes
    canonical response files containing only base64-encoded DSSE envelopes.
    """

    def __init__(
        self,
        exchange_dir: str | Path,
        *,
        poll_interval_seconds: float = 0.05,
        maximum_wait_seconds: float = 60.0,
        maximum_request_bytes: int = 1024 * 1024,
        maximum_response_bytes: int = 32 * 1024 * 1024,
        maximum_envelopes: int = 64,
    ) -> None:
        self.poll_interval_seconds = _bounded_number(
            poll_interval_seconds,
            "poll interval",
            minimum=0.001,
            maximum=1.0,
        )
        self.maximum_wait_seconds = _bounded_number(
            maximum_wait_seconds,
            "maximum wait",
            minimum=0.01,
            maximum=3600.0,
        )
        self.maximum_request_bytes = _bounded_integer(
            maximum_request_bytes,
            "maximum request bytes",
            minimum=1024,
            maximum=_MAX_REQUEST_BYTES_LIMIT,
        )
        self.maximum_response_bytes = _bounded_integer(
            maximum_response_bytes,
            "maximum response bytes",
            minimum=1024,
            maximum=_MAX_RESPONSE_BYTES_LIMIT,
        )
        self.maximum_envelopes = _bounded_integer(
            maximum_envelopes,
            "maximum envelopes",
            minimum=1,
            maximum=_MAX_ENVELOPES_LIMIT,
        )
        self.root = _pin_existing_private_directory(
            exchange_dir,
            "attestation exchange directory",
        )
        self.requests_dir = self.root / "requests"
        self.responses_dir = self.root / "responses"
        self._root_identity = _private_directory_identity(
            self.root,
            "attestation exchange directory",
        )
        self._requests_identity = _private_directory_identity(
            self.requests_dir,
            "attestation request directory",
        )
        self._responses_identity = _private_directory_identity(
            self.responses_dir,
            "attestation response directory",
        )

    @property
    def configuration_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "schemaVersion": _SCHEMA_VERSION,
                    "contract": DIRECTORY_PAIRED_ATTESTATION_CONTRACT,
                    "exchangeRootSha256": sha256_bytes(
                        str(self.root).encode("utf-8")
                    ),
                    "requestDirectory": "requests",
                    "responseDirectory": "responses",
                    "pollIntervalSeconds": self.poll_interval_seconds,
                    "maximumWaitSeconds": self.maximum_wait_seconds,
                    "maximumRequestBytes": self.maximum_request_bytes,
                    "maximumResponseBytes": self.maximum_response_bytes,
                    "maximumEnvelopes": self.maximum_envelopes,
                    "writePolicy": "atomic-no-clobber",
                    "responsePolicy": "canonical-bounded-exact-binding",
                }
            )
        )

    @property
    def semantic_configuration_sha256(self) -> str:
        """Identify the exchange protocol without freezing its state directory."""

        return sha256_bytes(
            canonical_json_bytes(
                {
                    "schemaVersion": _SCHEMA_VERSION,
                    "contract": DIRECTORY_PAIRED_ATTESTATION_CONTRACT,
                    "requestDirectory": "requests",
                    "responseDirectory": "responses",
                    "pollIntervalSeconds": self.poll_interval_seconds,
                    "maximumWaitSeconds": self.maximum_wait_seconds,
                    "maximumRequestBytes": self.maximum_request_bytes,
                    "maximumResponseBytes": self.maximum_response_bytes,
                    "maximumEnvelopes": self.maximum_envelopes,
                    "writePolicy": "atomic-no-clobber",
                    "responsePolicy": "canonical-bounded-exact-binding",
                }
            )
        )

    @property
    def state_paths(self) -> tuple[Path, ...]:
        return (self.root,)

    def attest(
        self,
        binding: CandidateBinding,
        workspace: Path,
        deadline: float,
    ) -> tuple[bytes, ...]:
        if not isinstance(binding, CandidateBinding):
            raise TypeError("binding must be a CandidateBinding.")
        workspace_root = _pin_verifier_workspace(workspace)
        deadline_value = _finite_number(deadline, "attestation deadline")
        wait_seconds = deadline_value - time.time()
        if wait_seconds <= 0 or wait_seconds > self.maximum_wait_seconds:
            raise DirectoryPairedAttestationError(
                "Attestation deadline is outside the configured wait bound."
            )
        self._validate_directories()

        request_id = secrets.token_hex(_REQUEST_ID_BYTES)
        request_path = self.requests_dir / f"request-{request_id}.json"
        response_path = self.responses_dir / f"response-{request_id}.json"
        if _lstat_optional(request_path) is not None:
            raise DirectoryPairedAttestationError(
                "Attestation request id collided with an existing artifact."
            )
        if _lstat_optional(response_path) is not None:
            raise DirectoryPairedAttestationError(
                "Attestation response exists before its request."
            )
        request = {
            "schemaVersion": _SCHEMA_VERSION,
            "contract": DIRECTORY_PAIRED_ATTESTATION_REQUEST_CONTRACT,
            "requestId": request_id,
            "bindingSha256": binding.binding_sha256,
            "binding": binding.payload(),
            "verifierWorkspacePath": str(workspace_root),
            "deadline": deadline_value,
        }
        encoded_request = canonical_json_bytes(request)
        if len(encoded_request) > self.maximum_request_bytes:
            raise DirectoryPairedAttestationError(
                "Attestation request exceeds its configured byte bound."
            )
        _atomic_no_clobber(request_path, encoded_request)
        self._validate_directories()
        if _read_secure_file(
            request_path,
            maximum_bytes=self.maximum_request_bytes,
            label="attestation request",
        ) != encoded_request:
            raise DirectoryPairedAttestationError(
                "Attestation request changed after publication."
            )

        monotonic_deadline = time.monotonic() + wait_seconds
        while True:
            remaining = monotonic_deadline - time.monotonic()
            if remaining <= 0:
                raise DirectoryPairedAttestationTimeout(
                    "Timed out waiting for the signed attestation sidecar."
                )
            if _lstat_optional(response_path) is not None:
                response_bytes = _read_secure_file(
                    response_path,
                    maximum_bytes=self.maximum_response_bytes,
                    label="attestation response",
                )
                envelopes = self._decode_response(
                    response_bytes,
                    request_id=request_id,
                    binding_sha256=binding.binding_sha256,
                )
                self._validate_directories()
                return envelopes
            time.sleep(min(self.poll_interval_seconds, remaining))

    def _validate_directories(self) -> None:
        _assert_directory_identity(
            self.root,
            self._root_identity,
            "attestation exchange directory",
        )
        _assert_directory_identity(
            self.requests_dir,
            self._requests_identity,
            "attestation request directory",
        )
        _assert_directory_identity(
            self.responses_dir,
            self._responses_identity,
            "attestation response directory",
        )

    def _decode_response(
        self,
        value: bytes,
        *,
        request_id: str,
        binding_sha256: str,
    ) -> tuple[bytes, ...]:
        try:
            raw = json.loads(value, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectoryPairedAttestationError(
                "Attestation response is not valid JSON."
            ) from exc
        if not isinstance(raw, dict) or set(raw) != {
            "schemaVersion",
            "contract",
            "requestId",
            "bindingSha256",
            "envelopes",
        }:
            raise DirectoryPairedAttestationError(
                "Attestation response shape is invalid."
            )
        try:
            canonical = canonical_json_bytes(raw)
        except ValueError as exc:
            raise DirectoryPairedAttestationError(
                "Attestation response is outside the canonical JSON profile."
            ) from exc
        if canonical != value:
            raise DirectoryPairedAttestationError(
                "Attestation response must use canonical JSON."
            )
        if (
            raw["schemaVersion"] != _SCHEMA_VERSION
            or raw["contract"] != DIRECTORY_PAIRED_ATTESTATION_RESPONSE_CONTRACT
            or raw["requestId"] != request_id
            or raw["bindingSha256"] != binding_sha256
        ):
            raise DirectoryPairedAttestationError(
                "Attestation response does not match its exact request binding."
            )
        encoded_envelopes = raw["envelopes"]
        if (
            not isinstance(encoded_envelopes, list)
            or not encoded_envelopes
            or len(encoded_envelopes) > self.maximum_envelopes
        ):
            raise DirectoryPairedAttestationError(
                "Attestation response envelope count is outside safe bounds."
            )
        envelopes: list[bytes] = []
        total_bytes = 0
        for encoded in encoded_envelopes:
            if not isinstance(encoded, str) or not encoded:
                raise DirectoryPairedAttestationError(
                    "Attestation response envelope encoding is invalid."
                )
            try:
                envelope = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise DirectoryPairedAttestationError(
                    "Attestation response envelope is not canonical base64."
                ) from exc
            if base64.b64encode(envelope).decode("ascii") != encoded:
                raise DirectoryPairedAttestationError(
                    "Attestation response envelope is not canonical base64."
                )
            total_bytes += len(envelope)
            if not envelope or total_bytes > self.maximum_response_bytes:
                raise DirectoryPairedAttestationError(
                    "Attestation response envelopes exceed safe bounds."
                )
            envelopes.append(envelope)
        if len(envelopes) != len(set(envelopes)):
            raise DirectoryPairedAttestationError(
                "Attestation response repeats an envelope."
            )
        return tuple(envelopes)


def _pin_existing_private_directory(value: str | Path, label: str) -> Path:
    declared = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    try:
        parent = declared.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DirectoryPairedAttestationError(
            f"{label} parent is unavailable."
        ) from exc
    target = parent / declared.name
    _private_directory_identity(target, label)
    return target


def _pin_verifier_workspace(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("workspace must be a Path.")
    declared = Path(os.path.abspath(os.fspath(value.expanduser())))
    try:
        details = declared.lstat()
        resolved = declared.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DirectoryPairedAttestationError(
            "Verifier workspace is unavailable."
        ) from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise DirectoryPairedAttestationError(
            "Verifier workspace must be a non-link directory."
        )
    return resolved


def _private_directory_identity(path: Path, label: str) -> _DirectoryIdentity:
    try:
        details = path.lstat()
    except OSError as exc:
        raise DirectoryPairedAttestationError(f"{label} is unavailable.") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise DirectoryPairedAttestationError(
            f"{label} must be a non-link directory."
        )
    if os.name == "posix" and (
        stat.S_IMODE(details.st_mode) != 0o700 or details.st_uid != os.getuid()
    ):
        raise DirectoryPairedAttestationError(
            f"{label} permissions must be owner-only."
        )
    return _DirectoryIdentity(
        device=details.st_dev,
        inode=details.st_ino,
        mode=details.st_mode,
    )


def _assert_directory_identity(
    path: Path,
    expected: _DirectoryIdentity,
    label: str,
) -> None:
    current = _private_directory_identity(path, label)
    if current != expected:
        raise DirectoryPairedAttestationError(f"{label} identity changed.")


def _read_secure_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise DirectoryPairedAttestationError(f"{label} is unavailable.") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise DirectoryPairedAttestationError(
            f"{label} is not a bounded single-link regular file."
        )
    if os.name == "posix" and (
        stat.S_IMODE(before.st_mode) != 0o600 or before.st_uid != os.getuid()
    ):
        raise DirectoryPairedAttestationError(
            f"{label} permissions must be owner-only."
        )
    descriptor = -1
    try:
        descriptor = os.open(path, _READ_FLAGS)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
        ):
            raise DirectoryPairedAttestationError(
                f"{label} identity changed before reading."
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise DirectoryPairedAttestationError(f"{label} is truncated.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DirectoryPairedAttestationError(
                f"{label} exceeds its size binding."
            )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise DirectoryPairedAttestationError(f"{label} could not be read.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DirectoryPairedAttestationError(f"{label} changed while reading.")
    return b"".join(chunks)


def _atomic_no_clobber(path: Path, value: bytes) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(temporary, _WRITE_FLAGS, 0o600)
        _write_all(descriptor, value)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise DirectoryPairedAttestationError(
                "Attestation request temporary file is unsafe."
            )
        if os.name == "posix" and stat.S_IMODE(opened.st_mode) != 0o600:
            raise DirectoryPairedAttestationError(
                "Attestation request permissions must be owner-only."
            )
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError as exc:
            raise DirectoryPairedAttestationError(
                "Attestation request already exists; overwrite is forbidden."
            ) from exc
        os.unlink(temporary)
        _fsync_directory(path.parent)
        _read_secure_file(
            path,
            maximum_bytes=max(len(value), 1),
            label="attestation request",
        )
    except OSError as exc:
        raise DirectoryPairedAttestationError(
            "Attestation request could not be published atomically."
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        if linked and _lstat_optional(path) is None:
            raise DirectoryPairedAttestationError(
                "Attestation request disappeared after publication."
            )


def _write_all(descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("write made no progress")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectoryPairedAttestationError(
                "Attestation response contains duplicate JSON keys."
            )
        result[key] = value
    return result


def _lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DirectoryPairedAttestationError(
            "Attestation exchange artifact cannot be inspected safely."
        ) from exc


def _finite_number(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise DirectoryPairedAttestationError(f"{label} must be finite.")
    return float(value)


def _bounded_number(
    value: float,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    normalized = _finite_number(value, label)
    if not minimum <= normalized <= maximum:
        raise DirectoryPairedAttestationError(f"{label} is outside safe bounds.")
    return normalized


def _bounded_integer(
    value: int,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise DirectoryPairedAttestationError(f"{label} is outside safe bounds.")
    return value
