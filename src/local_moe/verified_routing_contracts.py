from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any


CONTRACT_VERSION = "1.0"
DIFFICULTIES = ("simple", "medium", "complex", "very_complex")
EVIDENCE_STRENGTHS = (
    "implicit",
    "judge",
    "user",
    "independent",
    "deterministic",
)
OUTCOME_STATUSES = ("failed", "inconclusive", "passed")
ROUTE_PLANS = ("local", "local_then_verify", "premium")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VerifiedRoutingError(ValueError):
    """Raised when shadow-routing evidence violates its typed contract."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_sha256(value: object, label: str) -> str:
    rendered = str(value or "")
    if _SHA256.fullmatch(rendered) is None:
        raise VerifiedRoutingError(f"{label} must be a lowercase SHA-256 digest.")
    return rendered


def require_safe_id(value: object, label: str) -> str:
    rendered = str(value or "").strip()
    if _SAFE_ID.fullmatch(rendered) is None:
        raise VerifiedRoutingError(f"{label} must be a safe identifier.")
    return rendered


def require_identifier_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise VerifiedRoutingError(f"{label} must be a list.")
    items = tuple(require_safe_id(item, label) for item in value)
    if len(set(items)) != len(items):
        raise VerifiedRoutingError(f"{label} must not contain duplicates.")
    return items


def require_finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise VerifiedRoutingError(f"{label} must be numeric.")
    try:
        rendered = float(value)
    except (TypeError, ValueError) as exc:
        raise VerifiedRoutingError(f"{label} must be numeric.") from exc
    if not math.isfinite(rendered):
        raise VerifiedRoutingError(f"{label} must be finite.")
    if minimum is not None and rendered < minimum:
        raise VerifiedRoutingError(f"{label} must be >= {minimum}.")
    if maximum is not None and rendered > maximum:
        raise VerifiedRoutingError(f"{label} must be <= {maximum}.")
    return rendered


def require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise VerifiedRoutingError(f"{label} must be an integer.")
    try:
        rendered = int(value)
    except (TypeError, ValueError) as exc:
        raise VerifiedRoutingError(f"{label} must be an integer.") from exc
    if rendered < 0 or rendered != float(value):
        raise VerifiedRoutingError(f"{label} must be a non-negative integer.")
    return rendered


def require_utc_timestamp(value: object, label: str) -> str:
    rendered = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VerifiedRoutingError(f"{label} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise VerifiedRoutingError(f"{label} must use UTC.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def reject_unknown(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise VerifiedRoutingError(
            f"Unknown {label} fields: {', '.join(unknown)}."
        )
