from __future__ import annotations

from typing import Any

REDACTED_VALUE = "[redacted]"

SECRET_KEY_MARKERS = (
    "api",
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
)


def is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SECRET_KEY_MARKERS)


def sanitize_diagnostic_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): (REDACTED_VALUE if is_secret_key(str(key)) else sanitize_diagnostic_value(nested))
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_diagnostic_value(item) for item in value]
    return str(type(value).__name__)


def public_env_summary(env: dict[str, str]) -> dict[str, Any]:
    return {
        "env": {},
        "env_configured": bool(env),
        "env_count": len(env),
    }
