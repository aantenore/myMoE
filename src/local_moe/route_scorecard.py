from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .route_outcomes import VerifiedOutcomeRecord
from .verified_routing_contracts import (
    CONTRACT_VERSION,
    DIFFICULTIES,
    EVIDENCE_STRENGTHS,
    ROUTE_PLANS,
    VerifiedRoutingError,
    now_utc,
    reject_unknown,
    require_finite_number,
    require_non_negative_int,
    require_safe_id,
    require_sha256,
    require_utc_timestamp,
    sha256_json,
)


_ROOT_FIELDS = {
    "schema_version",
    "generated_at",
    "expires_at",
    "minimum_evidence_strength",
    "source_digest",
    "entries",
    "digest",
}
_ENTRY_FIELDS = {
    "config_sha256",
    "route",
    "capability",
    "difficulty",
    "verified_samples",
    "success_rate",
    "p95_latency_ms",
    "mean_tokens",
    "cost_sample_count",
    "mean_cost_usd",
    "mean_premium_calls",
    "mean_egress_chars",
}


class RouteScorecardFreshnessError(VerifiedRoutingError):
    """Raised when a structurally valid scorecard is not fresh enough to use."""


@dataclass(frozen=True)
class RouteScorecardEntry:
    config_sha256: str
    route: str
    capability: str
    difficulty: str
    verified_samples: int
    success_rate: float
    p95_latency_ms: float
    mean_tokens: float
    cost_sample_count: int
    mean_cost_usd: float | None
    mean_premium_calls: float
    mean_egress_chars: float

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.config_sha256,
            self.route,
            self.capability,
            self.difficulty,
        )

    def payload(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "route": self.route,
            "capability": self.capability,
            "difficulty": self.difficulty,
            "verified_samples": self.verified_samples,
            "success_rate": self.success_rate,
            "p95_latency_ms": self.p95_latency_ms,
            "mean_tokens": self.mean_tokens,
            "cost_sample_count": self.cost_sample_count,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_premium_calls": self.mean_premium_calls,
            "mean_egress_chars": self.mean_egress_chars,
        }


@dataclass(frozen=True)
class RouteScorecard:
    generated_at: str
    expires_at: str
    minimum_evidence_strength: str
    source_digest: str
    entries: tuple[RouteScorecardEntry, ...]
    digest: str
    schema_version: str = CONTRACT_VERSION

    def payload(self) -> dict[str, object]:
        body = self.content_payload()
        body["digest"] = self.digest
        return body

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "expires_at": self.expires_at,
            "minimum_evidence_strength": self.minimum_evidence_strength,
            "source_digest": self.source_digest,
            "entries": [entry.payload() for entry in self.entries],
        }

    def entries_for(
        self,
        *,
        config_sha256: str,
        route: str,
        capabilities: Iterable[str],
        difficulty: str,
    ) -> tuple[RouteScorecardEntry, ...]:
        requested = tuple(capabilities) or ("*",)
        index = {entry.key: entry for entry in self.entries}
        matches: list[RouteScorecardEntry] = []
        for capability in requested:
            key = (config_sha256, route, capability, difficulty)
            entry = index.get(key)
            if entry is None:
                return ()
            matches.append(entry)
        return tuple(matches)

    def conservative_entry(
        self,
        *,
        config_sha256: str,
        route: str,
        capabilities: Iterable[str],
        difficulty: str,
    ) -> RouteScorecardEntry | None:
        matches = self.entries_for(
            config_sha256=config_sha256,
            route=route,
            capabilities=capabilities,
            difficulty=difficulty,
        )
        if not matches:
            return None
        capability = "*" if len(matches) == 1 and matches[0].capability == "*" else "+".join(
            sorted(entry.capability for entry in matches)
        )
        return RouteScorecardEntry(
            config_sha256=config_sha256,
            route=route,
            capability=capability,
            difficulty=difficulty,
            verified_samples=min(entry.verified_samples for entry in matches),
            success_rate=min(entry.success_rate for entry in matches),
            p95_latency_ms=max(entry.p95_latency_ms for entry in matches),
            mean_tokens=max(entry.mean_tokens for entry in matches),
            cost_sample_count=min(entry.cost_sample_count for entry in matches),
            mean_cost_usd=(
                None
                if any(entry.mean_cost_usd is None for entry in matches)
                else max(float(entry.mean_cost_usd) for entry in matches)
            ),
            mean_premium_calls=max(entry.mean_premium_calls for entry in matches),
            mean_egress_chars=max(entry.mean_egress_chars for entry in matches),
        )


def build_route_scorecard(
    records: Iterable[object],
    *,
    minimum_evidence_strength: str = "independent",
    generated_at: str | None = None,
    ttl_seconds: int = 86_400,
) -> RouteScorecard:
    minimum_evidence_strength = _require_evidence_strength(
        minimum_evidence_strength,
        "minimum_evidence_strength",
    )
    ttl_seconds = require_non_negative_int(ttl_seconds, "ttl_seconds")
    if ttl_seconds == 0:
        raise VerifiedRoutingError("ttl_seconds must be positive.")
    generated_at = require_utc_timestamp(generated_at or now_utc(), "generated_at")
    generated = _parse_timestamp(generated_at)
    expires_at = (generated + timedelta(seconds=ttl_seconds)).replace(
        microsecond=0
    ).isoformat()

    normalized = [_verified_outcome_payload(record) for record in records]
    if not normalized:
        raise VerifiedRoutingError("At least one outcome record is required.")
    ids = [str(record["record_id"]) for record in normalized]
    if len(ids) != len(set(ids)):
        raise VerifiedRoutingError("Outcome record_id values must be unique.")
    normalized.sort(key=lambda record: str(record["record_id"]))
    source_digest = sha256_json({"records": normalized})

    minimum_rank = EVIDENCE_STRENGTHS.index(minimum_evidence_strength)
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for record in normalized:
        if EVIDENCE_STRENGTHS.index(str(record["evidence_strength"])) < minimum_rank:
            continue
        if record["outcome"] == "inconclusive":
            continue
        capabilities = tuple(record["capabilities"]) or ("*",)
        for capability in capabilities:
            key = (
                str(record["config_sha256"]),
                str(record["planned_route"]),
                str(capability),
                str(record["difficulty"]),
            )
            grouped.setdefault(key, []).append(record)
    if not grouped:
        raise VerifiedRoutingError(
            "No binary outcomes satisfy the minimum evidence strength."
        )

    entries = tuple(
        _aggregate_entry(key, grouped[key]) for key in sorted(grouped)
    )
    content = {
        "schema_version": CONTRACT_VERSION,
        "generated_at": generated_at,
        "expires_at": expires_at,
        "minimum_evidence_strength": minimum_evidence_strength,
        "source_digest": source_digest,
        "entries": [entry.payload() for entry in entries],
    }
    return RouteScorecard(
        generated_at=generated_at,
        expires_at=expires_at,
        minimum_evidence_strength=minimum_evidence_strength,
        source_digest=source_digest,
        entries=entries,
        digest=sha256_json(content),
    )


def load_route_scorecard(
    path: str | Path,
    *,
    now: str | datetime | None = None,
    max_age_seconds: int | None = None,
    require_fresh: bool = True,
) -> RouteScorecard:
    raw = _load_json(Path(path))
    if not isinstance(raw, dict):
        raise VerifiedRoutingError("Route scorecard must be a JSON object.")
    return route_scorecard_from_payload(
        raw,
        now=now,
        max_age_seconds=max_age_seconds,
        require_fresh=require_fresh,
    )


def route_scorecard_from_payload(
    raw: Mapping[str, object],
    *,
    now: str | datetime | None = None,
    max_age_seconds: int | None = None,
    require_fresh: bool = True,
) -> RouteScorecard:
    data = dict(raw)
    reject_unknown(data, _ROOT_FIELDS, "route scorecard")
    if data.get("schema_version") != CONTRACT_VERSION:
        raise VerifiedRoutingError("Unsupported route scorecard schema_version.")
    generated_at = require_utc_timestamp(data.get("generated_at"), "generated_at")
    expires_at = require_utc_timestamp(data.get("expires_at"), "expires_at")
    generated = _parse_timestamp(generated_at)
    expires = _parse_timestamp(expires_at)
    if expires <= generated:
        raise VerifiedRoutingError("expires_at must be after generated_at.")
    minimum_evidence_strength = _require_evidence_strength(
        data.get("minimum_evidence_strength"),
        "minimum_evidence_strength",
    )
    source_digest = require_sha256(data.get("source_digest"), "source_digest")
    digest = require_sha256(data.get("digest"), "digest")

    entries_raw = data.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise VerifiedRoutingError("entries must be a non-empty list.")
    entries = tuple(_entry_from_payload(item) for item in entries_raw)
    keys = [entry.key for entry in entries]
    if keys != sorted(keys):
        raise VerifiedRoutingError("entries must be sorted by their compound key.")
    if len(keys) != len(set(keys)):
        raise VerifiedRoutingError("entries must have unique compound keys.")

    scorecard = RouteScorecard(
        generated_at=generated_at,
        expires_at=expires_at,
        minimum_evidence_strength=minimum_evidence_strength,
        source_digest=source_digest,
        entries=entries,
        digest=digest,
    )
    if sha256_json(scorecard.content_payload()) != digest:
        raise VerifiedRoutingError("Route scorecard digest does not match its content.")
    _validate_freshness(
        scorecard,
        now=now,
        max_age_seconds=max_age_seconds,
        require_fresh=require_fresh,
    )
    return scorecard


def write_route_scorecard(path: str | Path, scorecard: RouteScorecard) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(scorecard.payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_outcome_payloads(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records: list[dict[str, object]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            parsed = _loads_json(line, f"outcome JSONL line {line_number}")
            if not isinstance(parsed, dict):
                raise VerifiedRoutingError(
                    f"Outcome JSONL line {line_number} must be an object."
                )
            records.append(parsed)
        return records
    parsed = _load_json(source)
    if isinstance(parsed, list):
        records_raw = parsed
    elif isinstance(parsed, dict) and set(parsed) == {"records"}:
        records_raw = parsed["records"]
    else:
        raise VerifiedRoutingError(
            "Outcome JSON must be a list or an object containing only records."
        )
    if not isinstance(records_raw, list):
        raise VerifiedRoutingError("records must be a list.")
    if not all(isinstance(item, dict) for item in records_raw):
        raise VerifiedRoutingError("Each outcome record must be an object.")
    return [dict(item) for item in records_raw]


def _aggregate_entry(
    key: tuple[str, str, str, str],
    records: list[dict[str, object]],
) -> RouteScorecardEntry:
    samples = len(records)
    latencies = [float(record["latency_ms"]) for record in records]
    token_totals = [
        float(record["prompt_tokens"]) + float(record["completion_tokens"])
        for record in records
    ]
    return RouteScorecardEntry(
        config_sha256=key[0],
        route=key[1],
        capability=key[2],
        difficulty=key[3],
        verified_samples=samples,
        success_rate=_mean(
            [1.0 if record["outcome"] == "passed" else 0.0 for record in records]
        ),
        p95_latency_ms=_nearest_rank_percentile(latencies, 0.95),
        mean_tokens=_mean(token_totals),
        cost_sample_count=len(
            [record for record in records if record["estimated_cost_usd"] is not None]
        ),
        mean_cost_usd=_optional_mean(
            [
                float(record["estimated_cost_usd"])
                for record in records
                if record["estimated_cost_usd"] is not None
            ]
        ),
        mean_premium_calls=_mean(
            [float(record["premium_calls"]) for record in records]
        ),
        mean_egress_chars=_mean(
            [float(record["remote_payload_chars"]) for record in records]
        ),
    )


def _verified_outcome_payload(record: object) -> dict[str, object]:
    if isinstance(record, VerifiedOutcomeRecord):
        verified = record
    elif isinstance(record, Mapping):
        verified = VerifiedOutcomeRecord.from_payload(record)
    elif hasattr(record, "payload") and callable(record.payload):
        payload = record.payload()
        if not isinstance(payload, Mapping):
            raise VerifiedRoutingError("Outcome payload() must return a mapping.")
        verified = VerifiedOutcomeRecord.from_payload(payload)
    else:
        raise VerifiedRoutingError("Outcome records must expose VerifiedOutcomeRecord payloads.")
    return verified.payload()


def _entry_from_payload(raw: object) -> RouteScorecardEntry:
    if not isinstance(raw, dict):
        raise VerifiedRoutingError("Each scorecard entry must be an object.")
    reject_unknown(raw, _ENTRY_FIELDS, "route scorecard entry")
    missing = sorted(_ENTRY_FIELDS.difference(raw))
    if missing:
        raise VerifiedRoutingError(
            f"Missing route scorecard entry fields: {', '.join(missing)}."
        )
    route = str(raw["route"])
    if route not in ROUTE_PLANS:
        raise VerifiedRoutingError("Scorecard entry route is not supported.")
    capability = str(raw["capability"])
    if capability != "*":
        capability = require_safe_id(capability, "capability")
    difficulty = str(raw["difficulty"])
    if difficulty not in DIFFICULTIES:
        raise VerifiedRoutingError("Scorecard entry difficulty is not supported.")
    verified_samples = require_non_negative_int(
        raw["verified_samples"], "verified_samples"
    )
    if verified_samples == 0:
        raise VerifiedRoutingError("verified_samples must be positive.")
    cost_sample_count = require_non_negative_int(
        raw["cost_sample_count"], "cost_sample_count"
    )
    if cost_sample_count > verified_samples:
        raise VerifiedRoutingError(
            "cost_sample_count cannot exceed verified_samples."
        )
    mean_cost_usd = _optional_non_negative_number(
        raw["mean_cost_usd"], "mean_cost_usd"
    )
    if (cost_sample_count == 0) != (mean_cost_usd is None):
        raise VerifiedRoutingError(
            "mean_cost_usd must be present exactly when cost samples exist."
        )
    return RouteScorecardEntry(
        config_sha256=require_sha256(raw["config_sha256"], "config_sha256"),
        route=route,
        capability=capability,
        difficulty=difficulty,
        verified_samples=verified_samples,
        success_rate=require_finite_number(
            raw["success_rate"], "success_rate", minimum=0.0, maximum=1.0
        ),
        p95_latency_ms=require_finite_number(
            raw["p95_latency_ms"], "p95_latency_ms", minimum=0.0
        ),
        mean_tokens=require_finite_number(
            raw["mean_tokens"], "mean_tokens", minimum=0.0
        ),
        cost_sample_count=cost_sample_count,
        mean_cost_usd=mean_cost_usd,
        mean_premium_calls=require_finite_number(
            raw["mean_premium_calls"], "mean_premium_calls", minimum=0.0
        ),
        mean_egress_chars=require_finite_number(
            raw["mean_egress_chars"], "mean_egress_chars", minimum=0.0
        ),
    )


def _validate_freshness(
    scorecard: RouteScorecard,
    *,
    now: str | datetime | None,
    max_age_seconds: int | None,
    require_fresh: bool,
) -> None:
    if not require_fresh:
        return
    current = _coerce_now(now)
    generated = _parse_timestamp(scorecard.generated_at)
    expires = _parse_timestamp(scorecard.expires_at)
    if generated > current:
        raise RouteScorecardFreshnessError("Route scorecard was generated in the future.")
    if current >= expires:
        raise RouteScorecardFreshnessError("Route scorecard has expired.")
    if max_age_seconds is not None:
        maximum = require_non_negative_int(max_age_seconds, "max_age_seconds")
        if (current - generated).total_seconds() > maximum:
            raise RouteScorecardFreshnessError(
                "Route scorecard exceeds the configured maximum age."
            )


def _require_evidence_strength(value: object, label: str) -> str:
    rendered = str(value or "")
    if rendered not in EVIDENCE_STRENGTHS:
        raise VerifiedRoutingError(f"{label} is not supported.")
    return rendered


def _mean(values: list[float]) -> float:
    return round(math.fsum(values) / len(values), 12)


def _optional_mean(values: list[float]) -> float | None:
    return None if not values else _mean(values)


def _optional_non_negative_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return require_finite_number(value, label, minimum=0.0)


def _nearest_rank_percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 12)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _coerce_now(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise VerifiedRoutingError("now must be timezone-aware.")
        return value.astimezone(timezone.utc)
    return _parse_timestamp(require_utc_timestamp(value, "now"))


def _load_json(path: Path) -> object:
    return _loads_json(path.read_text(encoding="utf-8"), str(path))


def _loads_json(text: str, label: str) -> object:
    try:
        return json.loads(
            text,
            parse_constant=lambda value: (_raise_non_finite(value, label)),
        )
    except json.JSONDecodeError as exc:
        raise VerifiedRoutingError(f"Invalid JSON in {label}.") from exc


def _raise_non_finite(value: str, label: str) -> object:
    raise VerifiedRoutingError(f"Non-finite number {value} is not allowed in {label}.")


__all__ = [
    "RouteScorecard",
    "RouteScorecardEntry",
    "RouteScorecardFreshnessError",
    "build_route_scorecard",
    "load_outcome_payloads",
    "load_route_scorecard",
    "route_scorecard_from_payload",
    "write_route_scorecard",
]
