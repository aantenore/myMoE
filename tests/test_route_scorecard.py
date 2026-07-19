from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.route_scorecard import (
    RouteScorecardFreshnessError,
    build_route_scorecard,
    load_route_scorecard,
    route_scorecard_from_payload,
)
from local_moe.verified_routing_contracts import VerifiedRoutingError
from local_moe.verified_routing_contracts import sha256_json


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_SHA256 = "1" * 64


class RouteScorecardTests(unittest.TestCase):
    def test_builds_content_addressed_verified_aggregates(self) -> None:
        records = [
            _record("r-1", outcome="passed", latency_ms=100, prompt_tokens=20),
            _record("r-2", outcome="failed", latency_ms=300, prompt_tokens=40),
            _record("r-3", outcome="passed", latency_ms=200, prompt_tokens=60),
            _record(
                "r-weak",
                outcome="failed",
                evidence_strength="user",
                latency_ms=900,
            ),
            _record(
                "r-open",
                outcome="inconclusive",
                evidence_strength="deterministic",
                latency_ms=900,
            ),
            _record("r-abstained", abstained=True, confidence=0.99),
            _record("r-low-confidence", confidence=0.69),
        ]

        scorecard = build_route_scorecard(
            reversed(records),
            generated_at="2026-07-19T00:00:00+00:00",
            ttl_seconds=3600,
        )

        self.assertEqual(len(scorecard.entries), 1)
        entry = scorecard.entries[0]
        self.assertEqual(entry.verified_samples, 3)
        self.assertAlmostEqual(entry.success_rate, 2 / 3)
        self.assertEqual(entry.p95_latency_ms, 300)
        self.assertEqual(entry.mean_tokens, 50)
        self.assertEqual(scorecard.minimum_confidence, 0.7)
        self.assertEqual(len(scorecard.digest), 64)
        self.assertEqual(
            scorecard.digest,
            build_route_scorecard(
                records,
                generated_at="2026-07-19T00:00:00+00:00",
                ttl_seconds=3600,
            ).digest,
        )

    def test_empty_and_multiple_capabilities_use_exact_sets(self) -> None:
        records = [
            _record("wild-1", capabilities=[]),
            _record("multi-1", capabilities=["analysis", "code"]),
            _record("multi-2", capabilities=["analysis"], outcome="failed"),
        ]
        scorecard = build_route_scorecard(
            records,
            generated_at="2026-07-19T00:00:00+00:00",
        )

        empty = scorecard.conservative_entry(
            config_sha256=CONFIG_SHA256,
            route="local",
            capabilities=(),
            difficulty="medium",
        )
        combined = scorecard.conservative_entry(
            config_sha256=CONFIG_SHA256,
            route="local",
            capabilities=("analysis", "code"),
            difficulty="medium",
        )

        self.assertIsNotNone(empty)
        self.assertEqual(empty.capabilities, ())
        self.assertIsNotNone(combined)
        self.assertEqual(combined.capabilities, ("analysis", "code"))
        self.assertEqual(combined.success_rate, 1.0)
        self.assertEqual(combined.verified_samples, 1)

    def test_disjoint_single_capability_cells_do_not_support_a_combined_request(self) -> None:
        scorecard = build_route_scorecard(
            [
                _record("analysis-only", capabilities=["analysis"]),
                _record("code-only", capabilities=["code"]),
                _record("literal-plus", capabilities=["analysis+code"]),
            ],
            generated_at="2026-07-19T00:00:00+00:00",
        )

        combined = scorecard.conservative_entry(
            config_sha256=CONFIG_SHA256,
            route="local",
            capabilities=("analysis", "code"),
            difficulty="medium",
        )
        literal_plus = scorecard.conservative_entry(
            config_sha256=CONFIG_SHA256,
            route="local",
            capabilities=("analysis+code",),
            difficulty="medium",
        )

        self.assertIsNone(combined)
        self.assertIsNotNone(literal_plus)
        self.assertEqual(literal_plus.capabilities, ("analysis+code",))

    def test_abstained_and_low_confidence_records_are_excluded(self) -> None:
        accepted = _record("accepted", confidence=0.7)
        abstained = _record("abstained", abstained=True, confidence=0.99)
        low_confidence = _record("low-confidence", confidence=0.699)

        scorecard = build_route_scorecard(
            [accepted, abstained, low_confidence],
            minimum_confidence=0.7,
            generated_at="2026-07-19T00:00:00+00:00",
        )

        self.assertEqual(scorecard.entries[0].verified_samples, 1)
        with self.assertRaisesRegex(VerifiedRoutingError, "confidence floors"):
            build_route_scorecard(
                [abstained, low_confidence],
                minimum_confidence=0.7,
                generated_at="2026-07-19T00:00:00+00:00",
            )

        with self.assertRaisesRegex(VerifiedRoutingError, "minimum_confidence"):
            build_route_scorecard(
                [accepted],
                minimum_confidence=1.01,
                generated_at="2026-07-19T00:00:00+00:00",
            )

    def test_fixture_loads_strictly_and_detects_content_tampering(self) -> None:
        scorecard = load_route_scorecard(
            FIXTURES / "verified-routing-scorecard.json",
            now="2026-07-20T00:00:00+00:00",
        )
        self.assertEqual(scorecard.schema_version, "1.0")
        self.assertEqual(scorecard.minimum_confidence, 0.7)
        self.assertEqual(len(scorecard.entries), 3)
        self.assertEqual(scorecard.entries[0].capabilities, ("analysis",))

        tampered = scorecard.payload()
        tampered["entries"][0]["success_rate"] = 1.0
        with self.assertRaisesRegex(VerifiedRoutingError, "digest"):
            route_scorecard_from_payload(
                tampered,
                now="2026-07-20T00:00:00+00:00",
            )

    def test_loader_rejects_expired_future_and_over_age_scorecards(self) -> None:
        path = FIXTURES / "verified-routing-scorecard.json"
        with self.assertRaises(RouteScorecardFreshnessError):
            load_route_scorecard(path, now="2031-01-01T00:00:00+00:00")
        with self.assertRaises(RouteScorecardFreshnessError):
            load_route_scorecard(path, now="2026-01-01T00:00:00+00:00")
        with self.assertRaises(RouteScorecardFreshnessError):
            load_route_scorecard(
                path,
                now="2026-07-21T00:00:00+00:00",
                max_age_seconds=3600,
            )

    def test_loader_rejects_non_finite_numbers_before_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scorecard.json"
            path.write_text('{"success_rate": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(VerifiedRoutingError, "Non-finite"):
                load_route_scorecard(path)

    def test_builder_rejects_duplicate_records_and_unknown_fields(self) -> None:
        record = _record("same")
        with self.assertRaisesRegex(VerifiedRoutingError, "unique"):
            build_route_scorecard([record, record])
        unknown = dict(record)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
            build_route_scorecard([unknown])


def _record(
    record_id: str,
    *,
    capabilities: list[str] | None = None,
    outcome: str = "passed",
    evidence_strength: str = "independent",
    latency_ms: int = 100,
    prompt_tokens: int = 20,
    estimated_cost_usd: float | None = 0.0,
    confidence: float = 0.9,
    abstained: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "config_sha256": CONFIG_SHA256,
        "route_receipt_id": f"receipt-{record_id}",
        "route_receipt_sha256": "2" * 64,
        "task_fingerprint": "3" * 64,
        "profile": "balanced",
        "planned_route": "local",
        "final_provider": "local-provider",
        "capabilities": ["analysis"] if capabilities is None else capabilities,
        "difficulty": "medium",
        "confidence": confidence,
        "source": "test-fixture",
        "abstained": abstained,
        "outcome": outcome,
        "evidence_strength": evidence_strength,
        "evidence_sha256": "4" * 64,
        "failure_class": "verification-failed" if outcome == "failed" else "none",
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 10,
        "premium_calls": 0,
        "remote_payload_chars": 0,
        "estimated_cost_usd": estimated_cost_usd,
        "created_at": "2026-07-19T00:00:00+00:00",
        "provider_runtime_sha256": "5" * 64,
        "model": "local-model",
    }
    payload["record_id"] = f"outcome-{sha256_json(payload)}"
    return payload


if __name__ == "__main__":
    unittest.main()
