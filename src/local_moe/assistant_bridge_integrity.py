from __future__ import annotations

import hashlib
from typing import Any

import rfc8785


class IntegrityContractError(ValueError):
    """Raised when a value cannot be bound by the bridge integrity profile."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return RFC 8785 JSON Canonicalization Scheme bytes."""

    try:
        encoded = rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise IntegrityContractError(
            "Value is not representable by the RFC 8785 integrity profile."
        ) from exc
    if not isinstance(encoded, bytes):  # Defensive against dependency drift.
        raise IntegrityContractError("RFC 8785 encoder returned a non-byte value.")
    return encoded


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))
