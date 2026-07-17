from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from importlib import import_module
import logging
import re
import threading
from typing import Any, Protocol


REDACTED_VALUE = "[redacted]"
DETECT_SECRETS_REQUIREMENT = "detect-secrets>=1.5,<1.6"


class SecretRedactionError(ValueError):
    """Raised when user-controlled content cannot be redacted safely."""


class ResidualAssuranceUnavailableError(SecretRedactionError):
    """Raised when the configured residual detector cannot provide assurance."""


@dataclass(frozen=True)
class TextSecretPattern:
    """A named pattern whose ``secret`` group is replaced, never reported."""

    name: str
    regex: re.Pattern[str] = field(repr=False)

    def __post_init__(self) -> None:
        if not self.name or "secret" not in self.regex.groupindex:
            raise SecretRedactionError(
                "Text secret patterns require a name and a named 'secret' group."
            )


def _pattern(name: str, expression: str, flags: int = 0) -> TextSecretPattern:
    return TextSecretPattern(name=name, regex=re.compile(expression, flags))


_NON_SECRET_TOKEN_KEY = (
    r"(?!(?:max|min|input|output|total|prompt|completion)[_-]?"
    r"tokens?(?:[_-](?:budget|count))?\b)"
)
_KEY_PREFIX = r"(?:[A-Za-z0-9]+[_-])*"
_SENSITIVE_KEY = (
    rf"{_NON_SECRET_TOKEN_KEY}{_KEY_PREFIX}"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"authorization|client[_-]?secret|private[_-]?key|secret[_-]?access[_-]?key|"
    r"password|passwd|pwd|credential(?:s)?|secret|token)"
)
_ASSIGNMENT_SENSITIVE_KEY = (
    rf"{_NON_SECRET_TOKEN_KEY}{_KEY_PREFIX}"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|private[_-]?key|secret[_-]?access[_-]?key|"
    r"password|passwd|pwd|credential(?:s)?|secret|token)"
)

DEFAULT_TEXT_PATTERNS = (
    _pattern(
        "json-double-quoted-secret",
        rf'(?P<prefix>"{_SENSITIVE_KEY}"\s*:\s*")'
        r'(?P<secret>(?:\\.|[^"\\])*)(?P<suffix>")',
        re.IGNORECASE,
    ),
    _pattern(
        "json-single-quoted-secret",
        rf"(?P<prefix>'{_SENSITIVE_KEY}'\s*:\s*')"
        r"(?P<secret>(?:\\.|[^'\\])*)(?P<suffix>')",
        re.IGNORECASE,
    ),
    _pattern(
        "quoted-secret-assignment",
        rf"(?P<prefix>(?<!['\"])\b{_ASSIGNMENT_SENSITIVE_KEY}\b\s*[:=]\s*)"
        r"(?P<quote>['\"])(?P<secret>(?:\\.|(?!\2).)*)(?P<suffix>(?P=quote))",
        re.IGNORECASE,
    ),
    _pattern(
        "unquoted-secret-assignment",
        rf"(?P<prefix>(?<!['\"])\b{_ASSIGNMENT_SENSITIVE_KEY}\b\s*[:=]\s*)"
        r"(?!['\"])(?P<secret>[^\r\n,;}\]]*?\S)(?P<suffix>\s*(?=$|[,;}\]]))",
        re.IGNORECASE | re.MULTILINE,
    ),
    _pattern(
        "authorization-header",
        r"(?P<prefix>\bauthorization\s*:\s*(?:bearer|basic|token|apikey)?\s*)"
        r"(?P<secret>[^\s,;]+)",
        re.IGNORECASE,
    ),
    _pattern(
        "slack-webhook",
        r"(?P<secret>https://hooks\.slack\.com/services/"
        r"[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+)",
        re.IGNORECASE,
    ),
    _pattern(
        "slack-token",
        r"(?P<secret>\bxox(?:a|b|p|o|s|r)-(?:\d+-)+[A-Za-z0-9]+\b)",
        re.IGNORECASE,
    ),
    _pattern(
        "github-token",
        r"(?P<secret>\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36}\b|"
        r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b)",
    ),
    _pattern(
        "gitlab-token",
        r"(?P<secret>\b(?:glpat|gldt|glft|glsoat|glrt)-[A-Za-z0-9_-]{20,50}\b|"
        r"\bGR1348941[A-Za-z0-9_-]{20,50}\b|"
        r"\bglcbt-(?:[0-9A-Fa-f]{2}_)?[A-Za-z0-9_-]{20,50}\b|"
        r"\bglimt-[A-Za-z0-9_-]{25}\b|"
        r"\bglptt-[A-Za-z0-9_-]{40}\b|"
        r"\bglagent-[A-Za-z0-9_-]{50,1024}\b|"
        r"\bgloas-[A-Za-z0-9_-]{64}\b)",
    ),
    _pattern(
        "aws-access-key",
        r"(?P<secret>\b(?:A3T[A-Z0-9]|ABIA|ACCA|AKIA|ASIA)[A-Z0-9]{16}\b)",
    ),
    _pattern(
        "basic-auth-url",
        r"(?P<prefix>https?://[^:/\s]+:)(?P<secret>[^@/\s]+)(?P<suffix>@)",
        re.IGNORECASE,
    ),
    _pattern(
        "secret-query-parameter",
        rf"(?P<prefix>[?&]{_SENSITIVE_KEY}=)(?P<secret>[^&#\s]+)",
        re.IGNORECASE,
    ),
    _pattern(
        "private-key",
        r"(?P<secret>-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?"
        r"-----END [^-\r\n]*PRIVATE KEY-----)",
        re.DOTALL,
    ),
)

DEFAULT_USER_CONTROLLED_FIELDS = frozenset(
    {
        "constraints",
        "content",
        "diff",
        "diff_text",
        "error",
        "message",
        "objective",
        "output",
        "prompt",
        "stderr",
        "stdout",
    }
)
DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "authorization",
        "client_secret",
        "private_key",
        "secret_access_key",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "secret",
        "token",
    }
)
DEFAULT_GENERATED_METADATA_FIELDS = frozenset(
    {
        "artifact_sha256",
        "checksum",
        "digest",
        "fingerprint",
        "hash",
        "objective_sha256",
        "sha1",
        "sha256",
        "sha512",
        "task_fingerprint",
        "workspace_fingerprint",
    }
)
DEFAULT_RESIDUAL_PLUGINS = (
    "AWSKeyDetector",
    "AzureStorageKeyDetector",
    "Base64HighEntropyString",
    "BasicAuthDetector",
    "DiscordBotTokenDetector",
    "GitHubTokenDetector",
    "GitLabTokenDetector",
    "HexHighEntropyString",
    "JwtTokenDetector",
    "KeywordDetector",
    "NpmDetector",
    "OpenAIDetector",
    "PrivateKeyDetector",
    "PypiTokenDetector",
    "SendGridDetector",
    "SlackDetector",
    "SquareOAuthDetector",
    "StripeDetector",
    "TelegramBotTokenDetector",
    "TwilioKeyDetector",
)

_GENERATED_KEY_MARKER = re.compile(
    r"(?:^|_)(?:sha(?:1|224|256|384|512)|checksum|digest|fingerprint|hash)(?:_|$)",
    re.IGNORECASE,
)
_GENERATED_TEXT_VALUE = re.compile(
    r"(?P<prefix>\b(?:[A-Za-z0-9_.-]*_)?"
    r"(?:sha(?:1|224|256|384|512)|checksum|digest|fingerprint|hash)"
    r"\s*[:=]\s*['\"]?)(?P<value>[0-9A-Fa-f]{32,128})(?P<suffix>['\"]?)",
    re.IGNORECASE,
)
_DETECT_SECRETS_LOCK = threading.RLock()


@dataclass(frozen=True)
class SecretRedactionPolicy:
    """Configures which fields and patterns may cross a remote boundary.

    ``generated_metadata_fields`` is retained for API compatibility, but its
    entries are interpreted as exact normalized paths.  A bare name therefore
    trusts only that root field; an identically named nested input is scanned.
    """

    user_controlled_fields: frozenset[str] = DEFAULT_USER_CONTROLLED_FIELDS
    sensitive_keys: frozenset[str] = DEFAULT_SENSITIVE_KEYS
    generated_metadata_fields: frozenset[str] = DEFAULT_GENERATED_METADATA_FIELDS
    patterns: tuple[TextSecretPattern, ...] = DEFAULT_TEXT_PATTERNS
    replacement: str = REDACTED_VALUE
    require_residual_assurance: bool = True
    residual_plugins: tuple[str, ...] = DEFAULT_RESIDUAL_PLUGINS

    def __post_init__(self) -> None:
        if not self.replacement or "\r" in self.replacement or "\n" in self.replacement:
            raise SecretRedactionError("The redaction replacement must be one line.")
        if not self.residual_plugins and self.require_residual_assurance:
            raise SecretRedactionError(
                "Residual assurance requires at least one configured detector plugin."
            )
        for value in (
            *self.user_controlled_fields,
            *self.sensitive_keys,
            *self.generated_metadata_fields,
        ):
            if not value or value != value.lower():
                raise SecretRedactionError(
                    "Field and key selectors must be non-empty lowercase strings."
                )
        for name in self.residual_plugins:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) is None:
                raise SecretRedactionError(
                    "Residual plugin names must be safe identifiers."
                )


@dataclass(frozen=True)
class SecretRedactionResult:
    """A safe result whose representation never includes source content."""

    value: object = field(repr=False)
    redaction_count: int
    structured_redaction_count: int
    pattern_redaction_count: int
    residual_redaction_count: int
    skipped_generated_field_count: int
    residual_assured: bool
    residual_detector: str | None


class ResidualSecretDetector(Protocol):
    name: str

    def redact(self, value: str, replacement: str) -> tuple[str, int]: ...


@dataclass(frozen=True)
class _DetectSecretsApi:
    version: str
    scan_line: Callable[[str], Any] = field(repr=False)
    transient_settings: Callable[[dict[str, Any]], AbstractContextManager[Any]] = field(
        repr=False
    )


class DetectSecretsResidualDetector:
    """Offline, no-allowlist adapter for the detect-secrets 1.5 scanner."""

    name = "detect-secrets/1.5"

    def __init__(self, plugin_names: Sequence[str] = DEFAULT_RESIDUAL_PLUGINS) -> None:
        self._plugin_names = tuple(plugin_names)
        self._api = _load_detect_secrets_api()

    def redact(self, value: str, replacement: str) -> tuple[str, int]:
        shielded, generated_values = _shield_generated_text_values(value)
        config = {
            "plugins_used": [{"name": name} for name in self._plugin_names],
            # Empty is intentional: it disables inline allowlists and the
            # verification-policy filter that invokes provider network checks.
            "filters_used": [],
        }
        logger = logging.getLogger("detect-secrets")
        try:
            with _DETECT_SECRETS_LOCK:
                previous_disabled = logger.disabled
                try:
                    logger.disabled = True
                    with self._api.transient_settings(config):
                        redacted, count = _redact_residual_lines(
                            shielded,
                            replacement,
                            self._api.scan_line,
                        )
                finally:
                    logger.disabled = previous_disabled
        except ResidualAssuranceUnavailableError:
            raise
        except Exception:
            raise ResidualAssuranceUnavailableError(
                "detect-secrets residual assurance failed closed."
            ) from None
        return _restore_generated_text_values(redacted, generated_values), count


def redact_text(
    value: str,
    policy: SecretRedactionPolicy | None = None,
    *,
    residual_detector: ResidualSecretDetector | None = None,
) -> SecretRedactionResult:
    """Redact one user-controlled string according to an explicit policy."""

    active = policy or SecretRedactionPolicy()
    detector = _resolve_residual_detector(active, residual_detector)
    redacted, pattern_count = _apply_text_patterns(value, active)
    residual_count = 0
    if detector is not None:
        redacted, residual_count = _redact_with_detector(
            redacted,
            active,
            detector,
        )
    return SecretRedactionResult(
        value=redacted,
        redaction_count=pattern_count + residual_count,
        structured_redaction_count=0,
        pattern_redaction_count=pattern_count,
        residual_redaction_count=residual_count,
        skipped_generated_field_count=0,
        residual_assured=detector is not None,
        residual_detector=detector.name if detector is not None else None,
    )


def redact_user_controlled_fields(
    payload: Mapping[str, object],
    policy: SecretRedactionPolicy | None = None,
    *,
    residual_detector: ResidualSecretDetector | None = None,
) -> SecretRedactionResult:
    """Return a redacted copy while leaving generated metadata unscanned."""

    return _redact_object(
        payload,
        policy or SecretRedactionPolicy(),
        residual_detector=residual_detector,
        scan_all_strings=False,
        require_json_compatible=False,
    )


def redact_outbound_object(
    payload: object,
    policy: SecretRedactionPolicy | None = None,
    *,
    residual_detector: ResidualSecretDetector | None = None,
) -> SecretRedactionResult:
    """Redact and finally scan an entire outbound JSON-compatible object.

    Unlike the compatibility wrapper :func:`redact_user_controlled_fields`,
    every string is pattern-scanned even when its path is not selected.  When
    residual assurance is enabled, every string value is also passed through
    the residual detector.  Mapping keys are checked with deterministic secret
    patterns and fail closed if redaction would be necessary because renaming a
    key could overwrite another entry.
    """

    return _redact_object(
        payload,
        policy or SecretRedactionPolicy(),
        residual_detector=residual_detector,
        scan_all_strings=True,
        require_json_compatible=True,
    )


def _redact_object(
    payload: object,
    policy: SecretRedactionPolicy,
    *,
    residual_detector: ResidualSecretDetector | None,
    scan_all_strings: bool,
    require_json_compatible: bool,
) -> SecretRedactionResult:
    active = policy
    detector = _resolve_residual_detector(active, residual_detector)
    counters = _Counters()
    redacted = _redact_structure(
        payload,
        active,
        detector,
        counters,
        path=(),
        scan_all_strings=scan_all_strings,
        require_json_compatible=require_json_compatible,
    )
    if detector is not None:
        redacted = _scan_residual_outbound(
            redacted,
            active,
            detector,
            counters,
            path=(),
        )
    return SecretRedactionResult(
        value=redacted,
        redaction_count=(counters.structured + counters.pattern + counters.residual),
        structured_redaction_count=counters.structured,
        pattern_redaction_count=counters.pattern,
        residual_redaction_count=counters.residual,
        skipped_generated_field_count=counters.skipped_generated,
        residual_assured=detector is not None,
        residual_detector=detector.name if detector is not None else None,
    )


@dataclass
class _Counters:
    structured: int = 0
    pattern: int = 0
    residual: int = 0
    skipped_generated: int = 0


def _resolve_residual_detector(
    policy: SecretRedactionPolicy,
    residual_detector: ResidualSecretDetector | None,
) -> ResidualSecretDetector | None:
    if residual_detector is not None:
        return residual_detector
    if not policy.require_residual_assurance:
        return None
    return DetectSecretsResidualDetector(policy.residual_plugins)


def _redact_structure(
    value: object,
    policy: SecretRedactionPolicy,
    detector: ResidualSecretDetector | None,
    counters: _Counters,
    *,
    path: tuple[str, ...],
    scan_all_strings: bool,
    require_json_compatible: bool,
) -> object:
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise SecretRedactionError(
                    "User-controlled mappings require string field names."
                )
            _validate_mapping_key_patterns(raw_key, policy)
            normalized = _normalize_key(raw_key)
            nested_path = (*path, normalized)
            if _is_valid_generated_field(nested_path, nested, policy):
                output[raw_key] = nested
                counters.skipped_generated += 1
            elif _is_sensitive_key(normalized, policy):
                output[raw_key] = policy.replacement
                counters.structured += 1
            else:
                selected = (
                    scan_all_strings
                    or _is_selected_field(
                        nested_path,
                        policy.user_controlled_fields,
                    )
                    or _is_generated_field(nested_path, policy)
                )
                output[raw_key] = _redact_structure(
                    nested,
                    policy,
                    detector,
                    counters,
                    path=nested_path,
                    scan_all_strings=selected,
                    require_json_compatible=require_json_compatible,
                )
        return output
    if isinstance(value, list):
        return [
            _redact_structure(
                nested,
                policy,
                detector,
                counters,
                path=path,
                scan_all_strings=scan_all_strings,
                require_json_compatible=require_json_compatible,
            )
            for nested in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_structure(
                nested,
                policy,
                detector,
                counters,
                path=path,
                scan_all_strings=scan_all_strings,
                require_json_compatible=require_json_compatible,
            )
            for nested in value
        )
    if isinstance(value, str) and scan_all_strings:
        redacted, pattern_count = _apply_text_patterns(value, policy)
        counters.pattern += pattern_count
        if detector is not None:
            try:
                redacted, residual_count = detector.redact(
                    redacted,
                    policy.replacement,
                )
            except ResidualAssuranceUnavailableError:
                raise
            except Exception:
                raise ResidualAssuranceUnavailableError(
                    "Residual secret assurance failed closed."
                ) from None
            counters.residual += residual_count
        return redacted
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if scan_all_strings or require_json_compatible:
        raise SecretRedactionError(
            "Outbound content must contain JSON-compatible values."
        )
    return value


def _scan_residual_outbound(
    value: object,
    policy: SecretRedactionPolicy,
    detector: ResidualSecretDetector,
    counters: _Counters,
    *,
    path: tuple[str, ...],
) -> object:
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise SecretRedactionError(
                    "Outbound mappings require string field names."
                )
            _validate_mapping_key_patterns(raw_key, policy)
            normalized = _normalize_key(raw_key)
            nested_path = (*path, normalized)
            if _is_valid_generated_field(nested_path, nested, policy):
                output[raw_key] = nested
            else:
                output[raw_key] = _scan_residual_outbound(
                    nested,
                    policy,
                    detector,
                    counters,
                    path=nested_path,
                )
        return output
    if isinstance(value, list):
        return [
            _scan_residual_outbound(
                nested,
                policy,
                detector,
                counters,
                path=path,
            )
            for nested in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _scan_residual_outbound(
                nested,
                policy,
                detector,
                counters,
                path=path,
            )
            for nested in value
        )
    if isinstance(value, str):
        redacted, residual_count = _redact_with_detector(value, policy, detector)
        counters.residual += residual_count
        return redacted
    return value


def _apply_text_patterns(
    value: str,
    policy: SecretRedactionPolicy,
) -> tuple[str, int]:
    redacted = value
    count = 0
    for rule in policy.patterns:
        redacted, replacements = rule.regex.subn(
            lambda match: _replace_secret_group(match, policy.replacement),
            redacted,
        )
        count += replacements
    return redacted, count


def _replace_secret_group(match: re.Match[str], replacement: str) -> str:
    start, end = match.span("secret")
    relative_start = start - match.start()
    relative_end = end - match.start()
    return match.group(0)[:relative_start] + replacement + match.group(0)[relative_end:]


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _is_generated_field(
    path: tuple[str, ...],
    policy: SecretRedactionPolicy,
) -> bool:
    return ".".join(path) in policy.generated_metadata_fields


def _is_valid_generated_field(
    path: tuple[str, ...],
    value: object,
    policy: SecretRedactionPolicy,
) -> bool:
    if not _is_generated_field(path, policy):
        return False
    key = path[-1]
    expected_length = 40 if key == "sha1" else 128 if key == "sha512" else 64

    def is_digest(candidate: object) -> bool:
        return (
            isinstance(candidate, str)
            and re.fullmatch(
                rf"[0-9a-f]{{{expected_length}}}",
                candidate,
            )
            is not None
        )

    if isinstance(value, (list, tuple)):
        return all(is_digest(item) for item in value)
    return is_digest(value)


def _validate_mapping_key_patterns(
    key: str,
    policy: SecretRedactionPolicy,
) -> None:
    redacted, count = _apply_text_patterns(key, policy)
    if redacted != key or count:
        raise SecretRedactionError(
            "Secrets in outbound mapping keys cannot be redacted safely."
        )


def _redact_with_detector(
    value: str,
    policy: SecretRedactionPolicy,
    detector: ResidualSecretDetector,
) -> tuple[str, int]:
    try:
        redacted, count = detector.redact(value, policy.replacement)
    except ResidualAssuranceUnavailableError:
        raise
    except Exception:
        raise ResidualAssuranceUnavailableError(
            "Residual secret assurance failed closed."
        ) from None
    if (
        not isinstance(redacted, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or (count == 0 and redacted != value)
        or (count > 0 and redacted == value)
    ):
        raise ResidualAssuranceUnavailableError(
            "Residual secret assurance returned an unverifiable redaction."
        )
    return redacted, count


def _is_sensitive_key(key: str, policy: SecretRedactionPolicy) -> bool:
    if (
        key in {"max_token", "min_token", "token_budget"}
        or key.endswith("_token_budget")
        or key.endswith("_token_count")
        or key.endswith("_tokens")
    ):
        return False
    if key in policy.sensitive_keys:
        return True
    return any(
        key.endswith(f"_{suffix}")
        for suffix in (
            "api_key",
            "access_token",
            "auth_token",
            "client_secret",
            "credential",
            "credentials",
            "password",
            "private_key",
            "secret",
            "secret_access_key",
            "token",
        )
    )


def _is_selected_field(path: tuple[str, ...], selectors: frozenset[str]) -> bool:
    if "*" in selectors:
        return True
    dotted = ".".join(path)
    return bool(path) and (path[-1] in selectors or dotted in selectors)


def _load_detect_secrets_api() -> _DetectSecretsApi:
    try:
        version_module = import_module("detect_secrets.__version__")
        scan_module = import_module("detect_secrets.core.scan")
        settings_module = import_module("detect_secrets.settings")
    except ModuleNotFoundError as exc:
        raise ResidualAssuranceUnavailableError(
            f"Residual assurance requires optional dependency {DETECT_SECRETS_REQUIREMENT}."
        ) from exc
    except Exception as exc:
        raise ResidualAssuranceUnavailableError(
            "The detect-secrets residual detector could not be initialized."
        ) from exc
    version = getattr(version_module, "VERSION", "")
    if re.fullmatch(r"1\.5(?:\.\d+)?", str(version)) is None:
        raise ResidualAssuranceUnavailableError(
            "Residual assurance requires a compatible detect-secrets 1.5.x runtime."
        )
    scan_line = getattr(scan_module, "scan_line", None)
    transient_settings = getattr(settings_module, "transient_settings", None)
    if not callable(scan_line) or not callable(transient_settings):
        raise ResidualAssuranceUnavailableError(
            "The detect-secrets 1.5.x adapter API is unavailable."
        )
    return _DetectSecretsApi(
        version=str(version),
        scan_line=scan_line,
        transient_settings=transient_settings,
    )


def _redact_residual_lines(
    value: str,
    replacement: str,
    scan_line: Callable[[str], Any],
) -> tuple[str, int]:
    output: list[str] = []
    count = 0
    for line in value.splitlines(keepends=True) or [value]:
        secrets: set[str] = set()
        for finding in scan_line(line):
            secret = getattr(finding, "secret_value", None)
            if not isinstance(secret, str) or not secret:
                raise ResidualAssuranceUnavailableError(
                    "detect-secrets returned an unredactable residual finding."
                )
            secrets.add(secret)
        ordered = sorted(secrets, key=len, reverse=True)
        for secret in ordered:
            if secret not in line:
                raise ResidualAssuranceUnavailableError(
                    "detect-secrets returned an unredactable residual finding."
                )
        if ordered:
            matcher = re.compile("|".join(re.escape(secret) for secret in ordered))
            redacted_line, replacements = matcher.subn(lambda _: replacement, line)
            count += replacements
        else:
            redacted_line = line
        output.append(redacted_line)
    return "".join(output), count


def _shield_generated_text_values(
    value: str,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    protected: list[tuple[str, str]] = []
    sentinels: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        sentinel_index = len(protected)
        sentinel = f"__GENERATED_METADATA_VALUE_{sentinel_index}__"
        while sentinel in value or sentinel in sentinels:
            sentinel_index += 1
            sentinel = f"__GENERATED_METADATA_VALUE_{sentinel_index}__"
        sentinels.add(sentinel)
        protected.append((sentinel, match.group("value")))
        return f"{match.group('prefix')}{sentinel}{match.group('suffix')}"

    return _GENERATED_TEXT_VALUE.sub(replace, value), tuple(protected)


def _restore_generated_text_values(
    value: str,
    protected: tuple[tuple[str, str], ...],
) -> str:
    restored = value
    for sentinel, original in protected:
        restored = restored.replace(sentinel, original)
    return restored


__all__ = [
    "DETECT_SECRETS_REQUIREMENT",
    "DEFAULT_GENERATED_METADATA_FIELDS",
    "DEFAULT_RESIDUAL_PLUGINS",
    "DEFAULT_SENSITIVE_KEYS",
    "DEFAULT_TEXT_PATTERNS",
    "DEFAULT_USER_CONTROLLED_FIELDS",
    "DetectSecretsResidualDetector",
    "REDACTED_VALUE",
    "ResidualAssuranceUnavailableError",
    "ResidualSecretDetector",
    "SecretRedactionError",
    "SecretRedactionPolicy",
    "SecretRedactionResult",
    "TextSecretPattern",
    "redact_outbound_object",
    "redact_text",
    "redact_user_controlled_fields",
]
