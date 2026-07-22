from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


LOCAL_CASCADE_SCHEMA_VERSION = "1.1"
TASK_KINDS = ("classification", "extraction", "summarization")
OUTPUT_FORMATS = ("json_object", "text")
TOKEN_SOURCES = ("actual", "estimated", "unknown")
ATTEMPT_STATUSES = ("abstained", "completed", "error")
VERIFICATION_STATUSES = ("escalate", "passed")
RUN_STATUSES = ("all_abstained", "exhausted", "passed")
REQUESTED_EXECUTION_SCOPES = ("offline_local",)
EXECUTION_SCOPE_ATTESTATIONS = ("adapter_declared_unverified",)
ATTEMPT_ERROR_REASON_CODES = (
    "attempt_port_error",
    "attempt_result_contract_error",
)
CONTENT_VERIFIER_REASON_CODES = (
    "content_too_long",
    "content_too_short",
    "empty_content",
    "forbidden_term_present",
    "invalid_json",
    "json_field_type_mismatch",
    "json_not_object",
    "json_string_value_not_allowed",
    "missing_json_field",
    "missing_required_term",
    "unexpected_json_field",
)
TOKEN_LIMIT_REASON_CODES = (
    "input_token_limit_exceeded",
    "output_token_limit_exceeded",
)
JSON_VALUE_KINDS = (
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
)

MAX_TASK_CHARACTERS = 65_536
MAX_RESULT_CHARACTERS = 262_144
MAX_TIERS = 16
MAX_REASON_CODES = 32
MAX_JSON_FIELDS = 128

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")
_JSON_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^cascade-run-[0-9a-f]{32}$")


class LocalCascadeContractError(ValueError):
    """Raised when a LocalCascade value violates its versioned contract."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalCascadeContractError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise LocalCascadeContractError(f"{label} field names must be strings.")
    return value


def _exact_fields(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    present = set(raw)
    missing = sorted(expected - present)
    unknown = sorted(present - expected)
    if missing:
        raise LocalCascadeContractError(
            f"Missing {label} fields: {', '.join(missing)}."
        )
    if unknown:
        raise LocalCascadeContractError(
            f"Unknown {label} fields: {', '.join(unknown)}."
        )


def _string(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise LocalCascadeContractError(f"{label} must be a string.")
    rendered = value.strip()
    if not rendered:
        raise LocalCascadeContractError(f"{label} must not be empty.")
    if len(rendered) > maximum:
        raise LocalCascadeContractError(
            f"{label} must contain at most {maximum} characters."
        )
    return rendered


def _safe_id(value: object, label: str) -> str:
    rendered = _string(value, label, maximum=256)
    if _SAFE_ID.fullmatch(rendered) is None:
        raise LocalCascadeContractError(f"{label} must be a safe identifier.")
    return rendered


def _field_name(value: object, label: str) -> str:
    rendered = _string(value, label, maximum=128)
    if _JSON_FIELD_NAME.fullmatch(rendered) is None:
        raise LocalCascadeContractError(f"{label} must be a top-level JSON field name.")
    return rendered


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LocalCascadeContractError(f"{label} must be an integer.")
    if value < minimum:
        raise LocalCascadeContractError(f"{label} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise LocalCascadeContractError(f"{label} must be <= {maximum}.")
    return value


def _finite_float(
    value: object,
    label: str,
    *,
    minimum: float = 0.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalCascadeContractError(f"{label} must be numeric.")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < minimum:
        raise LocalCascadeContractError(f"{label} must be finite and >= {minimum}.")
    return rendered


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise LocalCascadeContractError(f"{label} must be a boolean.")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LocalCascadeContractError(
            f"{label} must be null or a lowercase SHA-256 digest."
        )
    return value


def _schema_contract(
    schema_version: object,
    contract: object,
    *,
    expected_contract: str,
) -> tuple[str, str]:
    if schema_version != LOCAL_CASCADE_SCHEMA_VERSION:
        raise LocalCascadeContractError(
            f"Unsupported {expected_contract} schema_version."
        )
    if contract != expected_contract:
        raise LocalCascadeContractError(
            f"Unsupported contract; expected {expected_contract}."
        )
    return str(schema_version), str(contract)


def _string_tuple(
    value: object,
    label: str,
    *,
    maximum_items: int,
    maximum_characters: int,
    identifiers: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise LocalCascadeContractError(f"{label} must be a list.")
    if len(value) > maximum_items:
        raise LocalCascadeContractError(
            f"{label} must contain at most {maximum_items} items."
        )
    items = tuple(
        _safe_id(item, label)
        if identifiers
        else _string(item, label, maximum=maximum_characters)
        for item in value
    )
    if len(set(items)) != len(items):
        raise LocalCascadeContractError(f"{label} must not contain duplicates.")
    return items


@dataclass(frozen=True)
class LocalCascadeTaskV1:
    task_id: str
    kind: str
    instruction: str
    output_format: str
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeTaskV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeTaskV1",
        )
        object.__setattr__(self, "task_id", _safe_id(self.task_id, "task_id"))
        if self.kind not in TASK_KINDS:
            raise LocalCascadeContractError(
                f"task kind must be one of: {', '.join(TASK_KINDS)}."
            )
        object.__setattr__(
            self,
            "instruction",
            _string(
                self.instruction,
                "task instruction",
                maximum=MAX_TASK_CHARACTERS,
            ),
        )
        if self.output_format not in OUTPUT_FORMATS:
            raise LocalCascadeContractError(
                f"task output_format must be one of: {', '.join(OUTPUT_FORMATS)}."
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "task_id": self.task_id,
            "kind": self.kind,
            "instruction": self.instruction,
            "output_format": self.output_format,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeTaskV1:
        raw = _mapping(value, "LocalCascadeTaskV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "task_id",
                "kind",
                "instruction",
                "output_format",
            },
            "LocalCascadeTaskV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            task_id=raw["task_id"],  # type: ignore[arg-type]
            kind=raw["kind"],  # type: ignore[arg-type]
            instruction=raw["instruction"],  # type: ignore[arg-type]
            output_format=raw["output_format"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LocalCascadeTierV1:
    tier_id: str
    cost_rank: int
    model_ref: str
    max_input_tokens: int
    max_output_tokens: int
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeTierV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeTierV1",
        )
        object.__setattr__(self, "tier_id", _safe_id(self.tier_id, "tier_id"))
        object.__setattr__(
            self, "cost_rank", _integer(self.cost_rank, "cost_rank", maximum=1_000_000)
        )
        object.__setattr__(self, "model_ref", _safe_id(self.model_ref, "model_ref"))
        object.__setattr__(
            self,
            "max_input_tokens",
            _integer(
                self.max_input_tokens,
                "max_input_tokens",
                minimum=1,
                maximum=10_000_000,
            ),
        )
        object.__setattr__(
            self,
            "max_output_tokens",
            _integer(
                self.max_output_tokens,
                "max_output_tokens",
                minimum=1,
                maximum=1_000_000,
            ),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "tier_id": self.tier_id,
            "cost_rank": self.cost_rank,
            "model_ref": self.model_ref,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeTierV1:
        raw = _mapping(value, "LocalCascadeTierV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "tier_id",
                "cost_rank",
                "model_ref",
                "max_input_tokens",
                "max_output_tokens",
            },
            "LocalCascadeTierV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            tier_id=raw["tier_id"],  # type: ignore[arg-type]
            cost_rank=raw["cost_rank"],  # type: ignore[arg-type]
            model_ref=raw["model_ref"],  # type: ignore[arg-type]
            max_input_tokens=raw["max_input_tokens"],  # type: ignore[arg-type]
            max_output_tokens=raw["max_output_tokens"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LocalCascadeJsonFieldV1:
    name: str
    value_kind: str
    required: bool
    allowed_string_values: tuple[str, ...] = ()
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeJsonFieldV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeJsonFieldV1",
        )
        object.__setattr__(self, "name", _field_name(self.name, "JSON field name"))
        if self.value_kind not in JSON_VALUE_KINDS:
            raise LocalCascadeContractError(
                f"JSON value_kind must be one of: {', '.join(JSON_VALUE_KINDS)}."
            )
        object.__setattr__(self, "required", _boolean(self.required, "required"))
        values = _string_tuple(
            self.allowed_string_values,
            "allowed_string_values",
            maximum_items=128,
            maximum_characters=512,
        )
        if values and self.value_kind != "string":
            raise LocalCascadeContractError(
                "allowed_string_values can only constrain a string field."
            )
        object.__setattr__(self, "allowed_string_values", values)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "name": self.name,
            "value_kind": self.value_kind,
            "required": self.required,
            "allowed_string_values": list(self.allowed_string_values),
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeJsonFieldV1:
        raw = _mapping(value, "LocalCascadeJsonFieldV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "name",
                "value_kind",
                "required",
                "allowed_string_values",
            },
            "LocalCascadeJsonFieldV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            name=raw["name"],  # type: ignore[arg-type]
            value_kind=raw["value_kind"],  # type: ignore[arg-type]
            required=raw["required"],  # type: ignore[arg-type]
            allowed_string_values=raw["allowed_string_values"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LocalCascadeVerifierV1:
    output_format: str
    min_characters: int
    max_characters: int
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    case_sensitive_terms: bool = False
    json_fields: tuple[LocalCascadeJsonFieldV1, ...] = ()
    allow_extra_json_fields: bool = False
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeVerifierV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeVerifierV1",
        )
        if self.output_format not in OUTPUT_FORMATS:
            raise LocalCascadeContractError(
                f"verifier output_format must be one of: {', '.join(OUTPUT_FORMATS)}."
            )
        minimum = _integer(
            self.min_characters,
            "min_characters",
            maximum=MAX_RESULT_CHARACTERS,
        )
        maximum = _integer(
            self.max_characters,
            "max_characters",
            minimum=1,
            maximum=MAX_RESULT_CHARACTERS,
        )
        if minimum > maximum:
            raise LocalCascadeContractError(
                "min_characters must not exceed max_characters."
            )
        object.__setattr__(self, "min_characters", minimum)
        object.__setattr__(self, "max_characters", maximum)
        required = _string_tuple(
            self.required_terms,
            "required_terms",
            maximum_items=128,
            maximum_characters=512,
        )
        forbidden = _string_tuple(
            self.forbidden_terms,
            "forbidden_terms",
            maximum_items=128,
            maximum_characters=512,
        )
        object.__setattr__(self, "required_terms", required)
        object.__setattr__(self, "forbidden_terms", forbidden)
        object.__setattr__(
            self,
            "case_sensitive_terms",
            _boolean(self.case_sensitive_terms, "case_sensitive_terms"),
        )
        if not isinstance(self.json_fields, (list, tuple)):
            raise LocalCascadeContractError("json_fields must be a list.")
        if len(self.json_fields) > MAX_JSON_FIELDS:
            raise LocalCascadeContractError(
                f"json_fields must contain at most {MAX_JSON_FIELDS} items."
            )
        fields = tuple(
            item
            if isinstance(item, LocalCascadeJsonFieldV1)
            else LocalCascadeJsonFieldV1.from_payload(item)
            for item in self.json_fields
        )
        names = [field.name for field in fields]
        if len(set(names)) != len(names):
            raise LocalCascadeContractError(
                "json_fields must not contain duplicate names."
            )
        object.__setattr__(self, "json_fields", fields)
        object.__setattr__(
            self,
            "allow_extra_json_fields",
            _boolean(self.allow_extra_json_fields, "allow_extra_json_fields"),
        )
        if self.output_format == "text":
            if fields or self.allow_extra_json_fields:
                raise LocalCascadeContractError(
                    "text verification cannot declare JSON field policy."
                )
        elif not fields:
            raise LocalCascadeContractError(
                "json_object verification requires at least one JSON field."
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "output_format": self.output_format,
            "min_characters": self.min_characters,
            "max_characters": self.max_characters,
            "required_terms": list(self.required_terms),
            "forbidden_terms": list(self.forbidden_terms),
            "case_sensitive_terms": self.case_sensitive_terms,
            "json_fields": [field.payload() for field in self.json_fields],
            "allow_extra_json_fields": self.allow_extra_json_fields,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeVerifierV1:
        raw = _mapping(value, "LocalCascadeVerifierV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "output_format",
                "min_characters",
                "max_characters",
                "required_terms",
                "forbidden_terms",
                "case_sensitive_terms",
                "json_fields",
                "allow_extra_json_fields",
            },
            "LocalCascadeVerifierV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            output_format=raw["output_format"],  # type: ignore[arg-type]
            min_characters=raw["min_characters"],  # type: ignore[arg-type]
            max_characters=raw["max_characters"],  # type: ignore[arg-type]
            required_terms=raw["required_terms"],  # type: ignore[arg-type]
            forbidden_terms=raw["forbidden_terms"],  # type: ignore[arg-type]
            case_sensitive_terms=raw["case_sensitive_terms"],  # type: ignore[arg-type]
            json_fields=raw["json_fields"],  # type: ignore[arg-type]
            allow_extra_json_fields=raw["allow_extra_json_fields"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LocalCascadeConfigV1:
    cascade_id: str
    tiers: tuple[LocalCascadeTierV1, ...]
    verifier: LocalCascadeVerifierV1
    max_attempts: int
    requested_execution_scope: str = "offline_local"
    allow_network: bool = False
    allow_tools: bool = False
    allow_writes: bool = False
    parallel_attempts: int = 1
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeConfigV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeConfigV1",
        )
        object.__setattr__(self, "cascade_id", _safe_id(self.cascade_id, "cascade_id"))
        if not isinstance(self.tiers, (list, tuple)):
            raise LocalCascadeContractError("tiers must be a list.")
        if not 1 <= len(self.tiers) <= MAX_TIERS:
            raise LocalCascadeContractError(
                f"tiers must contain between 1 and {MAX_TIERS} items."
            )
        tiers = tuple(
            item
            if isinstance(item, LocalCascadeTierV1)
            else LocalCascadeTierV1.from_payload(item)
            for item in self.tiers
        )
        ids = [tier.tier_id for tier in tiers]
        ranks = [tier.cost_rank for tier in tiers]
        if len(set(ids)) != len(ids):
            raise LocalCascadeContractError("tier_id values must be unique.")
        if len(set(ranks)) != len(ranks):
            raise LocalCascadeContractError("cost_rank values must be unique.")
        object.__setattr__(self, "tiers", tiers)
        verifier = (
            self.verifier
            if isinstance(self.verifier, LocalCascadeVerifierV1)
            else LocalCascadeVerifierV1.from_payload(self.verifier)
        )
        object.__setattr__(self, "verifier", verifier)
        object.__setattr__(
            self,
            "max_attempts",
            _integer(
                self.max_attempts,
                "max_attempts",
                minimum=1,
                maximum=len(tiers),
            ),
        )
        if self.requested_execution_scope not in REQUESTED_EXECUTION_SCOPES:
            raise LocalCascadeContractError(
                "requested_execution_scope must be offline_local."
            )
        if _boolean(self.allow_network, "allow_network"):
            raise LocalCascadeContractError(
                "LocalCascade cannot allow external-network authority."
            )
        if _boolean(self.allow_tools, "allow_tools"):
            raise LocalCascadeContractError("LocalCascade cannot allow tools.")
        if _boolean(self.allow_writes, "allow_writes"):
            raise LocalCascadeContractError("LocalCascade cannot allow writes.")
        if (
            _integer(
                self.parallel_attempts,
                "parallel_attempts",
                minimum=1,
                maximum=1,
            )
            != 1
        ):
            raise LocalCascadeContractError(
                "LocalCascade requires exactly one sequential attempt."
            )

    @property
    def ordered_tiers(self) -> tuple[LocalCascadeTierV1, ...]:
        return tuple(sorted(self.tiers, key=lambda tier: tier.cost_rank))[
            : self.max_attempts
        ]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "cascade_id": self.cascade_id,
            "tiers": [tier.payload() for tier in self.tiers],
            "verifier": self.verifier.payload(),
            "max_attempts": self.max_attempts,
            "requested_execution_scope": self.requested_execution_scope,
            "allow_network": self.allow_network,
            "allow_tools": self.allow_tools,
            "allow_writes": self.allow_writes,
            "parallel_attempts": self.parallel_attempts,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeConfigV1:
        raw = _mapping(value, "LocalCascadeConfigV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "cascade_id",
                "tiers",
                "verifier",
                "max_attempts",
                "requested_execution_scope",
                "allow_network",
                "allow_tools",
                "allow_writes",
                "parallel_attempts",
            },
            "LocalCascadeConfigV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            cascade_id=raw["cascade_id"],  # type: ignore[arg-type]
            tiers=raw["tiers"],  # type: ignore[arg-type]
            verifier=raw["verifier"],  # type: ignore[arg-type]
            max_attempts=raw["max_attempts"],  # type: ignore[arg-type]
            requested_execution_scope=raw["requested_execution_scope"],  # type: ignore[arg-type]
            allow_network=raw["allow_network"],  # type: ignore[arg-type]
            allow_tools=raw["allow_tools"],  # type: ignore[arg-type]
            allow_writes=raw["allow_writes"],  # type: ignore[arg-type]
            parallel_attempts=raw["parallel_attempts"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LocalCascadeTokenCountV1:
    source: str
    count: int | None
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeTokenCountV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeTokenCountV1",
        )
        if self.source not in TOKEN_SOURCES:
            raise LocalCascadeContractError(
                f"token source must be one of: {', '.join(TOKEN_SOURCES)}."
            )
        if self.source == "unknown":
            if self.count is not None:
                raise LocalCascadeContractError(
                    "unknown token counts must use a null count."
                )
        else:
            object.__setattr__(
                self,
                "count",
                _integer(self.count, "token count", maximum=1_000_000_000),
            )

    @classmethod
    def unknown(cls) -> LocalCascadeTokenCountV1:
        return cls(source="unknown", count=None)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "source": self.source,
            "count": self.count,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeTokenCountV1:
        raw = _mapping(value, "LocalCascadeTokenCountV1")
        _exact_fields(
            raw,
            {"schema_version", "contract", "source", "count"},
            "LocalCascadeTokenCountV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            source=raw["source"],  # type: ignore[arg-type]
            count=raw["count"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class LocalCascadeAttemptRequestV1:
    task: LocalCascadeTaskV1
    tier: LocalCascadeTierV1
    attempt_number: int
    verifier_reason_codes: tuple[str, ...]
    requested_execution_scope: str = "offline_local"
    allow_network: bool = False
    allow_tools: bool = False
    allow_writes: bool = False
    parallel_attempts: int = 1
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeAttemptRequestV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeAttemptRequestV1",
        )
        if not isinstance(self.task, LocalCascadeTaskV1):
            raise LocalCascadeContractError(
                "attempt request task must be LocalCascadeTaskV1."
            )
        if not isinstance(self.tier, LocalCascadeTierV1):
            raise LocalCascadeContractError(
                "attempt request tier must be LocalCascadeTierV1."
            )
        object.__setattr__(
            self,
            "attempt_number",
            _integer(
                self.attempt_number,
                "attempt_number",
                minimum=1,
                maximum=MAX_TIERS,
            ),
        )
        codes = _string_tuple(
            self.verifier_reason_codes,
            "verifier_reason_codes",
            maximum_items=MAX_REASON_CODES,
            maximum_characters=128,
            identifiers=True,
        )
        object.__setattr__(self, "verifier_reason_codes", codes)
        if self.requested_execution_scope not in REQUESTED_EXECUTION_SCOPES:
            raise LocalCascadeContractError(
                "attempt requested_execution_scope must be offline_local."
            )
        if any(
            (
                _boolean(self.allow_network, "allow_network"),
                _boolean(self.allow_tools, "allow_tools"),
                _boolean(self.allow_writes, "allow_writes"),
            )
        ):
            raise LocalCascadeContractError(
                "attempt requests cannot enable external-network authority, tools, or writes."
            )
        if (
            _integer(
                self.parallel_attempts,
                "parallel_attempts",
                minimum=1,
                maximum=1,
            )
            != 1
        ):
            raise LocalCascadeContractError("attempt requests must be sequential.")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "task": self.task.payload(),
            "tier": self.tier.payload(),
            "attempt_number": self.attempt_number,
            "verifier_reason_codes": list(self.verifier_reason_codes),
            "requested_execution_scope": self.requested_execution_scope,
            "allow_network": self.allow_network,
            "allow_tools": self.allow_tools,
            "allow_writes": self.allow_writes,
            "parallel_attempts": self.parallel_attempts,
        }


@dataclass(frozen=True)
class LocalCascadeAttemptResultV1:
    status: str
    content: str | None
    input_tokens: LocalCascadeTokenCountV1
    output_tokens: LocalCascadeTokenCountV1
    network_calls: int = 0
    tool_calls: int = 0
    write_operations: int = 0
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeAttemptResultV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeAttemptResultV1",
        )
        if self.status not in {"abstained", "completed"}:
            raise LocalCascadeContractError(
                "attempt result status must be abstained or completed."
            )
        if self.status == "abstained":
            if self.content is not None:
                raise LocalCascadeContractError(
                    "an abstained attempt cannot contain content."
                )
        elif not isinstance(self.content, str):
            raise LocalCascadeContractError(
                "a completed attempt must contain text content."
            )
        elif len(self.content) > MAX_RESULT_CHARACTERS:
            raise LocalCascadeContractError(
                f"attempt content exceeds {MAX_RESULT_CHARACTERS} characters."
            )
        for field in ("input_tokens", "output_tokens"):
            value = getattr(self, field)
            if not isinstance(value, LocalCascadeTokenCountV1):
                raise LocalCascadeContractError(
                    f"{field} must be LocalCascadeTokenCountV1."
                )
        for field in ("network_calls", "tool_calls", "write_operations"):
            count = _integer(getattr(self, field), field, maximum=1_000_000)
            if count != 0:
                raise LocalCascadeContractError(
                    "LocalCascade rejects reported external-network, tool, and write activity."
                )


@dataclass(frozen=True)
class LocalCascadeAttemptReceiptV1:
    attempt_number: int
    tier_id: str
    cost_rank: int
    request_sha256: str
    output_sha256: str | None
    attempt_status: str
    verification_status: str
    verifier_reason_codes: tuple[str, ...]
    duration_ms: float
    input_tokens: LocalCascadeTokenCountV1
    output_tokens: LocalCascadeTokenCountV1
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeAttemptReceiptV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeAttemptReceiptV1",
        )
        object.__setattr__(
            self,
            "attempt_number",
            _integer(
                self.attempt_number,
                "attempt_number",
                minimum=1,
                maximum=MAX_TIERS,
            ),
        )
        object.__setattr__(self, "tier_id", _safe_id(self.tier_id, "tier_id"))
        object.__setattr__(
            self, "cost_rank", _integer(self.cost_rank, "cost_rank", maximum=1_000_000)
        )
        if (
            not isinstance(self.request_sha256, str)
            or _SHA256.fullmatch(self.request_sha256) is None
        ):
            raise LocalCascadeContractError(
                "request_sha256 must be a lowercase SHA-256 digest."
            )
        object.__setattr__(
            self,
            "output_sha256",
            _optional_sha256(self.output_sha256, "output_sha256"),
        )
        if self.attempt_status not in ATTEMPT_STATUSES:
            raise LocalCascadeContractError("attempt_status is unsupported.")
        if self.verification_status not in VERIFICATION_STATUSES:
            raise LocalCascadeContractError("verification_status is unsupported.")
        codes = _string_tuple(
            self.verifier_reason_codes,
            "verifier_reason_codes",
            maximum_items=MAX_REASON_CODES,
            maximum_characters=128,
            identifiers=True,
        )
        if self.verification_status == "passed" and codes:
            raise LocalCascadeContractError(
                "a passed verification cannot contain reason codes."
            )
        if self.verification_status == "escalate" and not codes:
            raise LocalCascadeContractError(
                "an escalation requires verifier reason codes."
            )
        if self.attempt_status == "completed":
            if self.output_sha256 is None:
                raise LocalCascadeContractError(
                    "a completed attempt requires an output_sha256."
                )
            if self.verification_status == "escalate" and not set(codes).issubset(
                set(CONTENT_VERIFIER_REASON_CODES) | set(TOKEN_LIMIT_REASON_CODES)
            ):
                raise LocalCascadeContractError(
                    "a completed escalation contains unsupported verifier reasons."
                )
        else:
            if self.output_sha256 is not None:
                raise LocalCascadeContractError(
                    "an abstained or errored attempt cannot contain output_sha256."
                )
            if self.verification_status != "escalate":
                raise LocalCascadeContractError(
                    "only a completed attempt can pass verification."
                )
            if self.attempt_status == "abstained":
                allowed = {"attempt_abstained", *TOKEN_LIMIT_REASON_CODES}
                if "attempt_abstained" not in codes or not set(codes).issubset(allowed):
                    raise LocalCascadeContractError(
                        "an abstained attempt requires consistent reason codes."
                    )
            elif len(codes) != 1 or codes[0] not in ATTEMPT_ERROR_REASON_CODES:
                raise LocalCascadeContractError(
                    "an errored attempt requires one supported error reason."
                )
        object.__setattr__(self, "verifier_reason_codes", codes)
        object.__setattr__(
            self, "duration_ms", _finite_float(self.duration_ms, "duration_ms")
        )
        for field in ("input_tokens", "output_tokens"):
            if not isinstance(getattr(self, field), LocalCascadeTokenCountV1):
                raise LocalCascadeContractError(
                    f"{field} must be LocalCascadeTokenCountV1."
                )
        if self.attempt_status == "error" and any(
            getattr(self, field).source != "unknown"
            for field in ("input_tokens", "output_tokens")
        ):
            raise LocalCascadeContractError(
                "errored attempts must keep token counts unknown."
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "attempt_number": self.attempt_number,
            "tier_id": self.tier_id,
            "cost_rank": self.cost_rank,
            "request_sha256": self.request_sha256,
            "output_sha256": self.output_sha256,
            "attempt_status": self.attempt_status,
            "verification_status": self.verification_status,
            "verifier_reason_codes": list(self.verifier_reason_codes),
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens.payload(),
            "output_tokens": self.output_tokens.payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeAttemptReceiptV1:
        raw = _mapping(value, "LocalCascadeAttemptReceiptV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "attempt_number",
                "tier_id",
                "cost_rank",
                "request_sha256",
                "output_sha256",
                "attempt_status",
                "verification_status",
                "verifier_reason_codes",
                "duration_ms",
                "input_tokens",
                "output_tokens",
            },
            "LocalCascadeAttemptReceiptV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            attempt_number=raw["attempt_number"],  # type: ignore[arg-type]
            tier_id=raw["tier_id"],  # type: ignore[arg-type]
            cost_rank=raw["cost_rank"],  # type: ignore[arg-type]
            request_sha256=raw["request_sha256"],  # type: ignore[arg-type]
            output_sha256=raw["output_sha256"],  # type: ignore[arg-type]
            attempt_status=raw["attempt_status"],  # type: ignore[arg-type]
            verification_status=raw["verification_status"],  # type: ignore[arg-type]
            verifier_reason_codes=raw["verifier_reason_codes"],  # type: ignore[arg-type]
            duration_ms=raw["duration_ms"],  # type: ignore[arg-type]
            input_tokens=LocalCascadeTokenCountV1.from_payload(raw["input_tokens"]),
            output_tokens=LocalCascadeTokenCountV1.from_payload(raw["output_tokens"]),
        )


@dataclass(frozen=True)
class LocalCascadeTokenTotalsV1:
    actual_input_tokens: int
    actual_output_tokens: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    unknown_input_attempts: int
    unknown_output_attempts: int
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeTokenTotalsV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeTokenTotalsV1",
        )
        for field in (
            "actual_input_tokens",
            "actual_output_tokens",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "unknown_input_attempts",
            "unknown_output_attempts",
        ):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field, maximum=1_000_000_000),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "unknown_input_attempts": self.unknown_input_attempts,
            "unknown_output_attempts": self.unknown_output_attempts,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeTokenTotalsV1:
        raw = _mapping(value, "LocalCascadeTokenTotalsV1")
        fields = {
            "schema_version",
            "contract",
            "actual_input_tokens",
            "actual_output_tokens",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "unknown_input_attempts",
            "unknown_output_attempts",
        }
        _exact_fields(raw, fields, "LocalCascadeTokenTotalsV1")
        return cls(**{field: raw[field] for field in fields})  # type: ignore[arg-type]


def build_local_cascade_evidence_payload(
    *,
    task_sha256: str,
    config_sha256: str,
    status: str,
    selected_tier_id: str | None,
    attempts: tuple[LocalCascadeAttemptReceiptV1, ...],
    token_totals: LocalCascadeTokenTotalsV1,
    requested_execution_scope: str,
    execution_scope_attestation: str,
) -> dict[str, object]:
    """Build stable evidence without run identity or volatile timings."""

    stable_attempts: list[dict[str, object]] = []
    for attempt in attempts:
        payload = attempt.payload()
        payload.pop("duration_ms")
        stable_attempts.append(payload)
    return {
        "schema_version": LOCAL_CASCADE_SCHEMA_VERSION,
        "contract": "LocalCascadeEvidenceV1",
        "task_sha256": task_sha256,
        "config_sha256": config_sha256,
        "status": status,
        "selected_tier_id": selected_tier_id,
        "attempt_count": len(attempts),
        "attempts": stable_attempts,
        "token_totals": token_totals.payload(),
        "requested_execution_scope": requested_execution_scope,
        "execution_scope_attestation": execution_scope_attestation,
    }


def build_local_cascade_evidence_sha256(
    *,
    task_sha256: str,
    config_sha256: str,
    status: str,
    selected_tier_id: str | None,
    attempts: tuple[LocalCascadeAttemptReceiptV1, ...],
    token_totals: LocalCascadeTokenTotalsV1,
    requested_execution_scope: str = "offline_local",
    execution_scope_attestation: str = "adapter_declared_unverified",
) -> str:
    return sha256_json(
        build_local_cascade_evidence_payload(
            task_sha256=task_sha256,
            config_sha256=config_sha256,
            status=status,
            selected_tier_id=selected_tier_id,
            attempts=attempts,
            token_totals=token_totals,
            requested_execution_scope=requested_execution_scope,
            execution_scope_attestation=execution_scope_attestation,
        )
    )


@dataclass(frozen=True)
class LocalCascadeReceiptV1:
    run_id: str
    task_sha256: str
    config_sha256: str
    status: str
    selected_tier_id: str | None
    attempt_count: int
    total_duration_ms: float
    attempts: tuple[LocalCascadeAttemptReceiptV1, ...]
    token_totals: LocalCascadeTokenTotalsV1
    evidence_sha256: str
    requested_execution_scope: str = "offline_local"
    execution_scope_attestation: str = "adapter_declared_unverified"
    parallel_attempts: int = 1
    schema_version: str = LOCAL_CASCADE_SCHEMA_VERSION
    contract: str = "LocalCascadeReceiptV1"

    def __post_init__(self) -> None:
        _schema_contract(
            self.schema_version,
            self.contract,
            expected_contract="LocalCascadeReceiptV1",
        )
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise LocalCascadeContractError(
                "run_id must be cascade-run- followed by 32 lowercase hex characters."
            )
        for field in ("task_sha256", "config_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise LocalCascadeContractError(
                    f"{field} must be a lowercase SHA-256 digest."
                )
        if self.status not in RUN_STATUSES:
            raise LocalCascadeContractError("LocalCascade status is unsupported.")
        if self.selected_tier_id is not None:
            object.__setattr__(
                self,
                "selected_tier_id",
                _safe_id(self.selected_tier_id, "selected_tier_id"),
            )
        if (self.status == "passed") != (self.selected_tier_id is not None):
            raise LocalCascadeContractError(
                "selected_tier_id must be present exactly when status is passed."
            )
        if not isinstance(self.attempts, (list, tuple)):
            raise LocalCascadeContractError("attempts must be a list.")
        attempts = tuple(
            item
            if isinstance(item, LocalCascadeAttemptReceiptV1)
            else LocalCascadeAttemptReceiptV1.from_payload(item)
            for item in self.attempts
        )
        if not attempts or len(attempts) > MAX_TIERS:
            raise LocalCascadeContractError(
                f"attempts must contain between 1 and {MAX_TIERS} items."
            )
        object.__setattr__(self, "attempts", attempts)
        count = _integer(
            self.attempt_count,
            "attempt_count",
            minimum=1,
            maximum=MAX_TIERS,
        )
        if count != len(attempts):
            raise LocalCascadeContractError(
                "attempt_count must equal the number of attempt receipts."
            )
        object.__setattr__(self, "attempt_count", count)
        if tuple(item.attempt_number for item in attempts) != tuple(
            range(1, count + 1)
        ):
            raise LocalCascadeContractError(
                "attempt receipt numbers must be contiguous and ordered."
            )
        tier_ids = tuple(item.tier_id for item in attempts)
        cost_ranks = tuple(item.cost_rank for item in attempts)
        if len(set(tier_ids)) != len(tier_ids):
            raise LocalCascadeContractError(
                "attempt receipt tier_id values must be unique."
            )
        if len(set(cost_ranks)) != len(cost_ranks) or cost_ranks != tuple(
            sorted(cost_ranks)
        ):
            raise LocalCascadeContractError(
                "attempt receipt cost ranks must be unique and increasing."
            )
        object.__setattr__(
            self,
            "total_duration_ms",
            _finite_float(self.total_duration_ms, "total_duration_ms"),
        )
        if not isinstance(self.token_totals, LocalCascadeTokenTotalsV1):
            raise LocalCascadeContractError(
                "token_totals must be LocalCascadeTokenTotalsV1."
            )
        expected_totals = build_token_totals(attempts)
        if self.token_totals != expected_totals:
            raise LocalCascadeContractError(
                "token_totals must exactly match the attempt receipts."
            )
        passed_attempts = tuple(
            item for item in attempts if item.verification_status == "passed"
        )
        if self.status == "passed":
            if (
                len(passed_attempts) != 1
                or passed_attempts[0] is not attempts[-1]
                or self.selected_tier_id != passed_attempts[0].tier_id
            ):
                raise LocalCascadeContractError(
                    "a passed run requires one final matching passed attempt."
                )
        elif passed_attempts:
            raise LocalCascadeContractError(
                "a non-passed run cannot contain a passed attempt."
            )
        if self.status == "all_abstained" and any(
            item.attempt_status != "abstained" for item in attempts
        ):
            raise LocalCascadeContractError(
                "all_abstained requires every attempt to abstain."
            )
        if self.status == "exhausted" and all(
            item.attempt_status == "abstained" for item in attempts
        ):
            raise LocalCascadeContractError(
                "an all-abstained run cannot use exhausted status."
            )
        if self.requested_execution_scope not in REQUESTED_EXECUTION_SCOPES:
            raise LocalCascadeContractError(
                "receipt requested_execution_scope must be offline_local."
            )
        if self.execution_scope_attestation not in EXECUTION_SCOPE_ATTESTATIONS:
            raise LocalCascadeContractError(
                "receipt execution_scope_attestation must remain explicitly unverified."
            )
        if (
            _integer(
                self.parallel_attempts,
                "parallel_attempts",
                minimum=1,
                maximum=1,
            )
            != 1
        ):
            raise LocalCascadeContractError("receipt parallel_attempts must be one.")
        if (
            not isinstance(self.evidence_sha256, str)
            or _SHA256.fullmatch(self.evidence_sha256) is None
        ):
            raise LocalCascadeContractError(
                "evidence_sha256 must be a lowercase SHA-256 digest."
            )
        expected_evidence = build_local_cascade_evidence_sha256(
            task_sha256=self.task_sha256,
            config_sha256=self.config_sha256,
            status=self.status,
            selected_tier_id=self.selected_tier_id,
            attempts=attempts,
            token_totals=self.token_totals,
            requested_execution_scope=self.requested_execution_scope,
            execution_scope_attestation=self.execution_scope_attestation,
        )
        if self.evidence_sha256 != expected_evidence:
            raise LocalCascadeContractError(
                "evidence_sha256 does not match the semantic receipt evidence."
            )

    def evidence_payload(self) -> dict[str, object]:
        return build_local_cascade_evidence_payload(
            task_sha256=self.task_sha256,
            config_sha256=self.config_sha256,
            status=self.status,
            selected_tier_id=self.selected_tier_id,
            attempts=self.attempts,
            token_totals=self.token_totals,
            requested_execution_scope=self.requested_execution_scope,
            execution_scope_attestation=self.execution_scope_attestation,
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "run_id": self.run_id,
            "task_sha256": self.task_sha256,
            "config_sha256": self.config_sha256,
            "status": self.status,
            "selected_tier_id": self.selected_tier_id,
            "attempt_count": self.attempt_count,
            "total_duration_ms": self.total_duration_ms,
            "attempts": [attempt.payload() for attempt in self.attempts],
            "token_totals": self.token_totals.payload(),
            "evidence_sha256": self.evidence_sha256,
            "requested_execution_scope": self.requested_execution_scope,
            "execution_scope_attestation": self.execution_scope_attestation,
            "parallel_attempts": self.parallel_attempts,
        }

    @classmethod
    def from_payload(cls, value: object) -> LocalCascadeReceiptV1:
        raw = _mapping(value, "LocalCascadeReceiptV1")
        _exact_fields(
            raw,
            {
                "schema_version",
                "contract",
                "run_id",
                "task_sha256",
                "config_sha256",
                "status",
                "selected_tier_id",
                "attempt_count",
                "total_duration_ms",
                "attempts",
                "token_totals",
                "evidence_sha256",
                "requested_execution_scope",
                "execution_scope_attestation",
                "parallel_attempts",
            },
            "LocalCascadeReceiptV1",
        )
        return cls(
            schema_version=raw["schema_version"],  # type: ignore[arg-type]
            contract=raw["contract"],  # type: ignore[arg-type]
            run_id=raw["run_id"],  # type: ignore[arg-type]
            task_sha256=raw["task_sha256"],  # type: ignore[arg-type]
            config_sha256=raw["config_sha256"],  # type: ignore[arg-type]
            status=raw["status"],  # type: ignore[arg-type]
            selected_tier_id=raw["selected_tier_id"],  # type: ignore[arg-type]
            attempt_count=raw["attempt_count"],  # type: ignore[arg-type]
            total_duration_ms=raw["total_duration_ms"],  # type: ignore[arg-type]
            attempts=raw["attempts"],  # type: ignore[arg-type]
            token_totals=LocalCascadeTokenTotalsV1.from_payload(raw["token_totals"]),
            evidence_sha256=raw["evidence_sha256"],  # type: ignore[arg-type]
            requested_execution_scope=raw["requested_execution_scope"],  # type: ignore[arg-type]
            execution_scope_attestation=raw["execution_scope_attestation"],  # type: ignore[arg-type]
            parallel_attempts=raw["parallel_attempts"],  # type: ignore[arg-type]
        )


def build_token_totals(
    attempts: tuple[LocalCascadeAttemptReceiptV1, ...],
) -> LocalCascadeTokenTotalsV1:
    buckets = {
        "actual_input_tokens": 0,
        "actual_output_tokens": 0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "unknown_input_attempts": 0,
        "unknown_output_attempts": 0,
    }
    for attempt in attempts:
        for direction in ("input", "output"):
            usage = getattr(attempt, f"{direction}_tokens")
            if usage.source == "unknown":
                buckets[f"unknown_{direction}_attempts"] += 1
            else:
                buckets[f"{usage.source}_{direction}_tokens"] += usage.count or 0
    return LocalCascadeTokenTotalsV1(**buckets)
