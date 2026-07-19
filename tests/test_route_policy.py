from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from local_moe.route_policy import (
    load_route_policy,
    recommend_shadow_route,
    route_policy_from_payload,
)
from local_moe.route_scorecard import (
    load_route_scorecard,
    route_scorecard_from_payload,
)
from local_moe.route_signals import TaskSignals
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    sha256_json,
)


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_SHA256 = "1" * 64
TASK_FINGERPRINT = "3" * 64
NOW = "2026-07-20T00:00:00+00:00"


class RoutePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_route_policy(
            FIXTURES / "verified-routing-policy.json"
        )
        self.scorecard = load_route_scorecard(
            FIXTURES / "verified-routing-scorecard.json",
            now=NOW,
        )

    def test_policy_is_strict_shadow_only_and_content_digested(self) -> None:
        self.assertEqual(self.policy.mode, "shadow")
        self.assertEqual(len(self.policy.digest), 64)

        raw = self.policy.payload()
        raw["mode"] = "active"
        with self.assertRaisesRegex(VerifiedRoutingError, "shadow"):
            route_policy_from_payload(raw)
        raw = self.policy.payload()
        raw["unexpected"] = True
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
            route_policy_from_payload(raw)

    def test_policy_profiles_are_deep_immutable_and_digest_stable(self) -> None:
        digest = self.policy.digest
        with self.assertRaises(TypeError):
            self.policy.profiles["balanced"] = self.policy.profiles["economy"]
        with self.assertRaises(AttributeError):
            self.policy.profiles["balanced"].weights.cost = 0.0

        detached = self.policy.payload()
        detached["profiles"]["balanced"]["weights"]["cost"] = 0.0

        self.assertGreater(
            self.policy.profiles["balanced"].weights.cost,
            0.0,
        )
        self.assertEqual(self.policy.digest, digest)

    def test_profiles_choose_deterministically_without_applying(self) -> None:
        signals = _signals()
        economy = recommend_shadow_route(
            _receipt(profile="economy"),
            signals,
            self.scorecard,
            self.policy,
            profile="economy",
            now=NOW,
        )
        quality = recommend_shadow_route(
            _receipt(profile="quality"),
            signals,
            self.scorecard,
            self.policy,
            profile="quality",
            now=NOW,
        )

        self.assertEqual(economy.recommended_route, "local")
        self.assertEqual(quality.recommended_route, "local_then_verify")
        self.assertFalse(economy.applied)
        self.assertFalse(quality.applied)
        self.assertEqual(quality.policy_digest, self.policy.digest)
        self.assertEqual(quality.scorecard_digest, self.scorecard.digest)
        self.assertNotIn("objective", json.dumps(quality.payload()))

    def test_hard_eligibility_never_expands_remote_scope_or_budget(self) -> None:
        decision = recommend_shadow_route(
            _receipt(
                profile="privacy",
                remote_allowed=False,
                premium_call_budget=0,
            ),
            _signals(),
            self.scorecard,
            self.policy,
            profile="privacy",
            now=NOW,
        )

        self.assertEqual(decision.recommended_route, "local")
        candidates = {candidate.route: candidate for candidate in decision.candidates}
        self.assertTrue(candidates["local"].hard_eligible)
        self.assertFalse(candidates["premium"].hard_eligible)
        self.assertIn("remote_not_allowed", candidates["premium"].rejection_codes)
        self.assertIn(
            "premium_budget_unavailable",
            candidates["local_then_verify"].rejection_codes,
        )

    def test_offline_profile_explicitly_excludes_every_remote_route(self) -> None:
        decision = recommend_shadow_route(
            _receipt(
                profile="offline",
                remote_allowed=True,
                premium_call_budget=1,
            ),
            _signals(),
            self.scorecard,
            self.policy,
            profile="offline",
            now=NOW,
        )

        self.assertEqual(decision.recommended_route, "local")
        candidates = {candidate.route: candidate for candidate in decision.candidates}
        for route in ("local_then_verify", "premium"):
            self.assertFalse(candidates[route].hard_eligible)
            self.assertIn(
                "offline_remote_forbidden",
                candidates[route].rejection_codes,
            )

        with self.assertRaisesRegex(VerifiedRoutingError, "Offline"):
            recommend_shadow_route(
                _receipt(route="premium", profile="offline"),
                _signals(),
                self.scorecard,
                self.policy,
                profile="offline",
                now=NOW,
            )

    def test_blocked_receipt_is_terminal_without_scorecard_lookup(self) -> None:
        decision = recommend_shadow_route(
            _receipt(route="blocked", profile="balanced"),
            _signals(capabilities=("unknown-capability",)),
            self.scorecard,
            self.policy,
            profile="balanced",
            now="2035-01-01T00:00:00+00:00",
        )

        self.assertEqual(decision.baseline_route, "blocked")
        self.assertEqual(decision.recommended_route, "blocked")
        self.assertTrue(decision.abstained)
        self.assertEqual(decision.candidates, ())
        self.assertIn("receipt_blocked", decision.reason_codes)

    def test_low_confidence_stale_and_out_of_distribution_abstain(self) -> None:
        low = recommend_shadow_route(
            _receipt(),
            _signals(confidence=0.2),
            self.scorecard,
            self.policy,
            profile="balanced",
            now=NOW,
        )
        stale = recommend_shadow_route(
            _receipt(),
            _signals(),
            self.scorecard,
            self.policy,
            profile="balanced",
            now="2035-01-01T00:00:00+00:00",
        )
        ood = recommend_shadow_route(
            _receipt(),
            _signals(capabilities=("code",)),
            self.scorecard,
            self.policy,
            profile="balanced",
            now=NOW,
        )

        self.assertTrue(low.abstained)
        self.assertTrue(stale.abstained)
        self.assertTrue(ood.abstained)
        self.assertEqual(low.recommended_route, low.baseline_route)
        self.assertIn("scorecard_stale", stale.reason_codes)
        self.assertIn("scorecard_out_of_distribution", ood.reason_codes)

    def test_incomplete_cost_evidence_abstains_for_cost_weighted_profile(self) -> None:
        raw = self.scorecard.payload()
        for entry in raw["entries"]:
            entry["cost_sample_count"] = 0
            entry["mean_cost_usd"] = None
        content = dict(raw)
        content.pop("digest")
        raw["digest"] = sha256_json(content)
        incomplete = route_scorecard_from_payload(raw, now=NOW)

        decision = recommend_shadow_route(
            _receipt(),
            _signals(),
            incomplete,
            self.policy,
            profile="balanced",
            now=NOW,
        )

        self.assertTrue(decision.abstained)
        self.assertEqual(decision.recommended_route, decision.baseline_route)
        self.assertIn("incomplete_cost_evidence", decision.reason_codes)

    def test_zero_cost_weight_ignores_missing_cost_in_pareto_filter(self) -> None:
        policy_raw = self.policy.payload()
        policy_raw["profiles"]["balanced"]["weights"]["cost"] = 0.0
        policy = route_policy_from_payload(policy_raw)

        scorecard_raw = self.scorecard.payload()
        for entry in scorecard_raw["entries"]:
            entry["cost_sample_count"] = 0
            entry["mean_cost_usd"] = None
        content = dict(scorecard_raw)
        content.pop("digest")
        scorecard_raw["digest"] = sha256_json(content)
        scorecard = route_scorecard_from_payload(scorecard_raw, now=NOW)

        decision = recommend_shadow_route(
            _receipt(),
            _signals(),
            scorecard,
            policy,
            profile="balanced",
            now=NOW,
        )

        self.assertFalse(decision.abstained)
        self.assertFalse(decision.applied)
        self.assertNotIn("incomplete_cost_evidence", decision.reason_codes)
        self.assertTrue(
            all(
                candidate.mean_cost_usd is None
                for candidate in decision.candidates
            )
        )

    def test_profile_and_task_signal_binding_are_enforced(self) -> None:
        with self.assertRaisesRegex(VerifiedRoutingError, "profile"):
            recommend_shadow_route(
                _receipt(profile="economy"),
                _signals(),
                self.scorecard,
                self.policy,
                profile="balanced",
                now=NOW,
            )
        with self.assertRaisesRegex(VerifiedRoutingError, "signals"):
            recommend_shadow_route(
                _receipt(),
                _signals(request_fingerprint="9" * 64),
                self.scorecard,
                self.policy,
                profile="balanced",
                now=NOW,
            )


def _receipt(
    *,
    route: str = "local",
    profile: str = "balanced",
    local_gaps: tuple[str, ...] = (),
    premium_gaps: tuple[str, ...] = (),
    remote_allowed: bool = True,
    premium_call_budget: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        route=route,
        config_sha256=CONFIG_SHA256,
        local_gaps=local_gaps,
        premium_gaps=premium_gaps,
        remote_allowed=remote_allowed,
        premium_call_budget=premium_call_budget,
        task={
            "profile": profile,
            "task_fingerprint": TASK_FINGERPRINT,
        },
    )


def _signals(
    *,
    capabilities: tuple[str, ...] = ("analysis",),
    confidence: float = 0.9,
    request_fingerprint: str = TASK_FINGERPRINT,
) -> TaskSignals:
    return TaskSignals(
        request_fingerprint=request_fingerprint,
        capabilities=capabilities,
        difficulty="medium",
        confidence=confidence,
        abstained=False,
        source="test-fixture",
    )


if __name__ == "__main__":
    unittest.main()
