from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
from typing import Any, Iterable, Mapping, Sequence

from .verified_routing_contracts import (
    CONTRACT_VERSION,
    VerifiedRoutingError,
    reject_unknown,
    require_safe_id,
    require_sha256,
    sha256_json,
)


PRICING_CONTRACT = "VerifiedPairedPricingContract"

_MAX_PRICING_ITEMS = 4_096
_MAX_COMMANDS = 1_024
_MAX_TOKEN_COUNT = 1_000_000_000_000
_MAX_DECIMAL_INPUT_CHARS = 128
_MAX_DECIMAL_SIGNIFICANT_DIGITS = 64
_MAX_DECIMAL_ADJUSTED_EXPONENT = 72

_PRICING_ITEM_FIELDS = {
    "provider_id",
    "model",
    "prompt_usd_per_million",
    "completion_usd_per_million",
}
_PRICING_FIELDS = {
    "schema_version",
    "contract",
    "items",
    "pricing_sha256",
}
_COMMAND_METADATA_FIELDS = {
    "provider_id",
    "model",
    "provider_runtime_sha256",
    "prompt_tokens",
    "completion_tokens",
}
_COMMAND_EVIDENCE_FIELDS = _COMMAND_METADATA_FIELDS | {
    "cost_usd",
    "command_sha256",
}
_PAIRED_EVIDENCE_FIELDS = {
    "pricing_sha256",
    "commands",
    "total_cost_usd",
    "cost_sha256",
}


class IncompleteCostEvidenceError(VerifiedRoutingError):
    """Raised when command metadata cannot produce complete cost evidence."""


@dataclass(frozen=True)
class PricingItem:
    """One provider/model price, expressed as exact decimal USD strings."""

    provider_id: str
    model: str
    prompt_usd_per_million: str
    completion_usd_per_million: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            require_safe_id(self.provider_id, "provider_id"),
        )
        object.__setattr__(self, "model", require_safe_id(self.model, "model"))
        object.__setattr__(
            self,
            "prompt_usd_per_million",
            _canonical_non_negative_decimal(
                self.prompt_usd_per_million,
                "prompt_usd_per_million",
            ),
        )
        object.__setattr__(
            self,
            "completion_usd_per_million",
            _canonical_non_negative_decimal(
                self.completion_usd_per_million,
                "completion_usd_per_million",
            ),
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider_id, self.model)

    def payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "prompt_usd_per_million": self.prompt_usd_per_million,
            "completion_usd_per_million": self.completion_usd_per_million,
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "PricingItem":
        payload = _mapping(raw, "pricing item")
        _require_exact_fields(payload, _PRICING_ITEM_FIELDS, "pricing item")
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PricingContract:
    """Canonical, content-addressed prices for verified paired execution."""

    items: tuple[PricingItem, ...]
    schema_version: str = CONTRACT_VERSION
    contract: str = PRICING_CONTRACT
    pricing_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise VerifiedRoutingError("Unsupported pricing schema_version.")
        if self.contract != PRICING_CONTRACT:
            raise VerifiedRoutingError("Pricing contract is unsupported.")
        items = tuple(self.items)
        if not items or any(not isinstance(item, PricingItem) for item in items):
            raise VerifiedRoutingError(
                "Pricing contract items must be a non-empty list of PricingItem."
            )
        if len(items) > _MAX_PRICING_ITEMS:
            raise VerifiedRoutingError(
                "Pricing contract contains too many provider/model items."
            )
        keys = tuple(item.key for item in items)
        if len(keys) != len(set(keys)):
            raise VerifiedRoutingError(
                "Pricing contract provider/model pairs must be unique."
            )
        if keys != tuple(sorted(keys)):
            raise VerifiedRoutingError(
                "Pricing contract items must use canonical provider/model order."
            )
        object.__setattr__(self, "items", items)
        object.__setattr__(
            self,
            "pricing_sha256",
            sha256_json(self.content_payload()),
        )

    @classmethod
    def build(
        cls,
        items: Iterable[PricingItem | Mapping[str, object]],
    ) -> "PricingContract":
        normalized = tuple(
            sorted(
                (
                    item
                    if isinstance(item, PricingItem)
                    else PricingItem.from_payload(item)
                    for item in items
                ),
                key=lambda item: item.key,
            )
        )
        return cls(items=normalized)

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "PricingContract":
        payload = _mapping(raw, "pricing contract")
        _require_exact_fields(payload, _PRICING_FIELDS, "pricing contract")
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise VerifiedRoutingError("Pricing contract items must be a list.")
        items = tuple(PricingItem.from_payload(item) for item in raw_items)
        contract = cls(
            items=items,
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            contract=payload["contract"],  # type: ignore[arg-type]
        )
        supplied_digest = require_sha256(
            payload["pricing_sha256"], "pricing_sha256"
        )
        if supplied_digest != contract.pricing_sha256:
            raise VerifiedRoutingError("Pricing contract digest is invalid.")
        return contract

    @classmethod
    def from_json(cls, raw: str | bytes) -> "PricingContract":
        return cls.from_payload(_decode_json_object(raw, "pricing contract"))

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "items": [item.payload() for item in self.items],
        }

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["pricing_sha256"] = self.pricing_sha256
        return payload

    def item_for(self, provider_id: str, model: str) -> PricingItem:
        key = (provider_id, model)
        for item in self.items:
            if item.key == key:
                return item
        raise IncompleteCostEvidenceError(
            "Cost evidence is incomplete: no pricing rate exists for "
            f"provider/model {provider_id}/{model}."
        )


@dataclass(frozen=True)
class CommandCostEvidence:
    """Content-free token and cost metadata for one provider command."""

    provider_id: str
    model: str
    provider_runtime_sha256: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: str
    command_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            require_safe_id(self.provider_id, "provider_id"),
        )
        object.__setattr__(self, "model", require_safe_id(self.model, "model"))
        object.__setattr__(
            self,
            "provider_runtime_sha256",
            require_sha256(
                self.provider_runtime_sha256,
                "provider_runtime_sha256",
            ),
        )
        object.__setattr__(
            self,
            "prompt_tokens",
            _require_token_count(self.prompt_tokens, "prompt_tokens"),
        )
        object.__setattr__(
            self,
            "completion_tokens",
            _require_token_count(self.completion_tokens, "completion_tokens"),
        )
        object.__setattr__(
            self,
            "cost_usd",
            _canonical_non_negative_decimal(self.cost_usd, "cost_usd"),
        )
        object.__setattr__(
            self,
            "command_sha256",
            sha256_json(self.content_payload()),
        )

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "CommandCostEvidence":
        payload = _mapping(raw, "command cost evidence")
        _require_exact_fields(
            payload,
            _COMMAND_EVIDENCE_FIELDS,
            "command cost evidence",
        )
        evidence = cls(
            provider_id=payload["provider_id"],  # type: ignore[arg-type]
            model=payload["model"],  # type: ignore[arg-type]
            provider_runtime_sha256=payload[  # type: ignore[arg-type]
                "provider_runtime_sha256"
            ],
            prompt_tokens=payload["prompt_tokens"],  # type: ignore[arg-type]
            completion_tokens=payload[  # type: ignore[arg-type]
                "completion_tokens"
            ],
            cost_usd=payload["cost_usd"],  # type: ignore[arg-type]
        )
        supplied_digest = require_sha256(
            payload["command_sha256"], "command_sha256"
        )
        if supplied_digest != evidence.command_sha256:
            raise VerifiedRoutingError("Command cost evidence digest is invalid.")
        return evidence

    @classmethod
    def from_json(cls, raw: str | bytes) -> "CommandCostEvidence":
        return cls.from_payload(_decode_json_object(raw, "command cost evidence"))

    def metadata_payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "provider_runtime_sha256": self.provider_runtime_sha256,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def content_payload(self) -> dict[str, object]:
        payload = self.metadata_payload()
        payload["cost_usd"] = self.cost_usd
        return payload

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["command_sha256"] = self.command_sha256
        return payload


@dataclass(frozen=True)
class PairedCostEvidence:
    """Content-addressed exact cost evidence tied to one pricing contract."""

    pricing_sha256: str
    commands: tuple[CommandCostEvidence, ...]
    total_cost_usd: str
    cost_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pricing_sha256",
            require_sha256(self.pricing_sha256, "pricing_sha256"),
        )
        commands = tuple(self.commands)
        if not commands or any(
            not isinstance(command, CommandCostEvidence) for command in commands
        ):
            raise VerifiedRoutingError(
                "Paired cost evidence commands must be a non-empty list."
            )
        if len(commands) > _MAX_COMMANDS:
            raise VerifiedRoutingError(
                "Paired cost evidence contains too many commands."
            )
        object.__setattr__(self, "commands", commands)
        total = _canonical_non_negative_decimal(
            self.total_cost_usd,
            "total_cost_usd",
        )
        expected_total = _canonical_decimal(
            _sum_exact(Decimal(command.cost_usd) for command in commands)
        )
        if total != expected_total:
            raise VerifiedRoutingError(
                "Paired cost evidence total does not equal its command costs."
            )
        object.__setattr__(self, "total_cost_usd", total)
        object.__setattr__(
            self,
            "cost_sha256",
            sha256_json(self.content_payload()),
        )

    @classmethod
    def from_payload(
        cls,
        raw: Mapping[str, object],
        *,
        pricing: PricingContract | Mapping[str, object] | None = None,
    ) -> "PairedCostEvidence":
        payload = _mapping(raw, "paired cost evidence")
        _require_exact_fields(
            payload,
            _PAIRED_EVIDENCE_FIELDS,
            "paired cost evidence",
        )
        raw_commands = payload["commands"]
        if not isinstance(raw_commands, list):
            raise VerifiedRoutingError(
                "Paired cost evidence commands must be a list."
            )
        evidence = cls(
            pricing_sha256=payload["pricing_sha256"],  # type: ignore[arg-type]
            commands=tuple(
                CommandCostEvidence.from_payload(command)
                for command in raw_commands
            ),
            total_cost_usd=payload["total_cost_usd"],  # type: ignore[arg-type]
        )
        supplied_digest = require_sha256(payload["cost_sha256"], "cost_sha256")
        if supplied_digest != evidence.cost_sha256:
            raise VerifiedRoutingError("Paired cost evidence digest is invalid.")
        if pricing is not None:
            evidence.validate_against(pricing)
        return evidence

    @classmethod
    def from_json(
        cls,
        raw: str | bytes,
        *,
        pricing: PricingContract | Mapping[str, object] | None = None,
    ) -> "PairedCostEvidence":
        return cls.from_payload(
            _decode_json_object(raw, "paired cost evidence"),
            pricing=pricing,
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "pricing_sha256": self.pricing_sha256,
            "commands": [command.payload() for command in self.commands],
            "total_cost_usd": self.total_cost_usd,
        }

    def payload(self) -> dict[str, object]:
        payload = self.content_payload()
        payload["cost_sha256"] = self.cost_sha256
        return payload

    def validate_against(
        self,
        pricing: PricingContract | Mapping[str, object],
    ) -> None:
        normalized_pricing = _pricing_contract(pricing)
        if self.pricing_sha256 != normalized_pricing.pricing_sha256:
            raise VerifiedRoutingError(
                "Paired cost evidence references a different pricing contract."
            )
        rebuilt = build_cost_evidence(
            normalized_pricing,
            (command.metadata_payload() for command in self.commands),
        )
        if rebuilt.payload() != self.payload():
            raise VerifiedRoutingError(
                "Paired cost evidence does not match the referenced prices."
            )


def build_cost_evidence(
    pricing: PricingContract | Mapping[str, object],
    commands: Iterable[Mapping[str, object]],
) -> PairedCostEvidence:
    """Calculate exact costs without retaining prompts, outputs, or credentials."""

    normalized_pricing = _pricing_contract(pricing)
    evidence: list[CommandCostEvidence] = []
    for raw_command in commands:
        if len(evidence) >= _MAX_COMMANDS:
            raise VerifiedRoutingError(
                "Paired cost evidence contains too many commands."
            )
        metadata = _command_metadata(raw_command)
        item = normalized_pricing.item_for(
            metadata["provider_id"],
            metadata["model"],
        )
        prompt_cost = _token_cost(
            Decimal(item.prompt_usd_per_million),
            metadata["prompt_tokens"],
        )
        completion_cost = _token_cost(
            Decimal(item.completion_usd_per_million),
            metadata["completion_tokens"],
        )
        cost_usd = _canonical_decimal(
            _sum_exact((prompt_cost, completion_cost))
        )
        evidence.append(
            CommandCostEvidence(
                provider_id=metadata["provider_id"],
                model=metadata["model"],
                provider_runtime_sha256=metadata["provider_runtime_sha256"],
                prompt_tokens=metadata["prompt_tokens"],
                completion_tokens=metadata["completion_tokens"],
                cost_usd=cost_usd,
            )
        )
    if not evidence:
        raise IncompleteCostEvidenceError(
            "Cost evidence is incomplete: at least one command is required."
        )
    total = _canonical_decimal(
        _sum_exact(Decimal(command.cost_usd) for command in evidence)
    )
    return PairedCostEvidence(
        pricing_sha256=normalized_pricing.pricing_sha256,
        commands=tuple(evidence),
        total_cost_usd=total,
    )


def _pricing_contract(
    pricing: PricingContract | Mapping[str, object],
) -> PricingContract:
    if isinstance(pricing, PricingContract):
        return pricing
    return PricingContract.from_payload(pricing)


def _command_metadata(raw: Mapping[str, object]) -> dict[str, Any]:
    payload = _mapping(raw, "command cost metadata")
    reject_unknown(payload, _COMMAND_METADATA_FIELDS, "command cost metadata")
    missing = sorted(_COMMAND_METADATA_FIELDS.difference(payload))
    if missing:
        raise IncompleteCostEvidenceError(
            "Cost evidence is incomplete; missing command metadata fields: "
            f"{', '.join(missing)}."
        )
    for name in ("provider_id", "model", "provider_runtime_sha256"):
        value = payload[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            raise IncompleteCostEvidenceError(
                f"Cost evidence is incomplete: {name} is required."
            )
    for name in ("prompt_tokens", "completion_tokens"):
        if payload[name] is None:
            raise IncompleteCostEvidenceError(
                f"Cost evidence is incomplete: {name} is required."
            )
    return {
        "provider_id": require_safe_id(payload["provider_id"], "provider_id"),
        "model": require_safe_id(payload["model"], "model"),
        "provider_runtime_sha256": require_sha256(
            payload["provider_runtime_sha256"],
            "provider_runtime_sha256",
        ),
        "prompt_tokens": _require_token_count(
            payload["prompt_tokens"], "prompt_tokens"
        ),
        "completion_tokens": _require_token_count(
            payload["completion_tokens"], "completion_tokens"
        ),
    }


def _require_token_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerifiedRoutingError(f"{label} must be a non-negative integer.")
    if value < 0:
        raise VerifiedRoutingError(f"{label} must be a non-negative integer.")
    if value > _MAX_TOKEN_COUNT:
        raise VerifiedRoutingError(f"{label} exceeds the supported bound.")
    return value


def _canonical_non_negative_decimal(value: object, label: str) -> str:
    if isinstance(value, bool):
        raise VerifiedRoutingError(f"{label} must be numeric, not boolean.")
    if isinstance(value, str) and len(value) > _MAX_DECIMAL_INPUT_CHARS:
        raise VerifiedRoutingError(f"{label} exceeds the supported precision.")
    if isinstance(value, int) and value.bit_length() > 384:
        raise VerifiedRoutingError(f"{label} exceeds the supported precision.")
    try:
        if isinstance(value, Decimal):
            number = value
        elif isinstance(value, float):
            number = Decimal(str(value))
        elif isinstance(value, (str, int)):
            number = Decimal(value)
        else:
            raise TypeError
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise VerifiedRoutingError(f"{label} must be a decimal number.") from exc
    if not number.is_finite():
        raise VerifiedRoutingError(f"{label} must be finite.")
    if number < 0:
        raise VerifiedRoutingError(f"{label} must be non-negative.")
    if number != 0:
        _, digits, _ = number.as_tuple()
        if (
            len(digits) > _MAX_DECIMAL_SIGNIFICANT_DIGITS
            or abs(number.adjusted()) > _MAX_DECIMAL_ADJUSTED_EXPONENT
        ):
            raise VerifiedRoutingError(
                f"{label} exceeds the supported magnitude or precision."
            )
    return _canonical_decimal(number)


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _token_cost(rate: Decimal, tokens: int) -> Decimal:
    coefficient, exponent = _decimal_components(rate)
    return _decimal_from_components(coefficient * tokens, exponent - 6)


def _sum_exact(values: Iterable[Decimal]) -> Decimal:
    components = [_decimal_components(value) for value in values]
    if not components:
        return Decimal(0)
    common_exponent = min(exponent for _, exponent in components)
    total = sum(
        coefficient * (10 ** (exponent - common_exponent))
        for coefficient, exponent in components
    )
    return _decimal_from_components(total, common_exponent)


def _decimal_components(value: Decimal) -> tuple[int, int]:
    if not value.is_finite() or value < 0:
        raise VerifiedRoutingError("Internal cost values must be finite and non-negative.")
    sign, digits, exponent = value.as_tuple()
    if sign:
        raise VerifiedRoutingError("Internal cost values must be non-negative.")
    coefficient = int("".join(str(digit) for digit in digits) or "0")
    return coefficient, int(exponent)


def _decimal_from_components(coefficient: int, exponent: int) -> Decimal:
    if coefficient == 0:
        return Decimal(0)
    digits = tuple(int(digit) for digit in str(coefficient))
    return Decimal((0, digits, exponent))


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise VerifiedRoutingError(f"{label} must be an object with string keys.")
    return dict(value)


def _require_exact_fields(
    raw: dict[str, Any],
    fields: set[str],
    label: str,
) -> None:
    reject_unknown(raw, fields, label)
    missing = sorted(fields.difference(raw))
    if missing:
        raise VerifiedRoutingError(
            f"Missing {label} fields: {', '.join(missing)}."
        )


def _decode_json_object(raw: str | bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except VerifiedRoutingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedRoutingError(f"{label} is not valid JSON.") from exc
    return _mapping(value, label)


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise VerifiedRoutingError(f"Duplicate JSON field: {key}.")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise VerifiedRoutingError(f"JSON numeric constant {value} is not finite.")
