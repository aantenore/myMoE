from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from local_moe.route_policy import (
    CandidateRouteScore,
    RoutePolicyNormalization,
    RoutePolicyWeights,
    RouteProfilePolicy,
    ShadowRouteDecision,
    load_route_policy,
    recommend_shadow_route,
    route_policy_from_payload,
)
from local_moe.route_scorecard import (
    load_route_scorecard,
    route_scorecard_from_payload,
)
from local_moe.route_signals import MetadataTaskSignalProvider, TaskSignals
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    sha256_json,
)


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_SHA256 = "1" * 64
TASK_FINGERPRINT = "3" * 64
NOW = "2026-07-20T00:00:00+00:00"
OBJECTIVE_SHA256 = "2" * 64
WORKSPACE_SHA256 = "4" * 64


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

    def test_direct_policy_constructors_enforce_semantic_invariants(self) -> None:
        weights = self.policy.profiles["balanced"].weights

        with self.assertRaises(VerifiedRoutingError):
            RoutePolicyWeights(
                quality=0.5,
                cost=-0.1,
                latency=0.2,
                egress=0.2,
                premium=0.2,
            )
        with self.assertRaisesRegex(VerifiedRoutingError, "positive sum"):
            RoutePolicyWeights(
                quality=0.0,
                cost=0.0,
                latency=0.0,
                egress=0.0,
                premium=0.0,
            )
        with self.assertRaisesRegex(VerifiedRoutingError, "positive"):
            RoutePolicyNormalization(
                cost_usd=0.0,
                latency_ms=1.0,
                egress_chars=1.0,
                premium_calls=1.0,
            )
        with self.assertRaisesRegex(VerifiedRoutingError, "min_samples"):
            RouteProfilePolicy(
                weights=weights,
                min_success_rate=0.7,
                min_samples=0,
                min_confidence=0.7,
            )
        with self.assertRaisesRegex(VerifiedRoutingError, "digest"):
            replace(self.policy, digest="0" * 64)

    def test_candidate_contract_rejects_impossible_evidence(self) -> None:
        candidate = CandidateRouteScore(
            route="local",
            hard_eligible=True,
            pareto_eligible=True,
            utility=0.2,
            verified_samples=2,
            success_rate=0.8,
            p95_latency_ms=100.0,
            mean_tokens=50.0,
            cost_sample_count=2,
            mean_cost_usd=0.01,
            mean_premium_calls=0.0,
            mean_egress_chars=0.0,
            rejection_codes=(),
        )

        invalid_changes = (
            {"success_rate": -0.1},
            {"cost_sample_count": 3},
            {"cost_sample_count": 0, "mean_cost_usd": 0.01},
            {"verified_samples": None},
            {"hard_eligible": False},
            {"pareto_eligible": True, "rejection_codes": ("rejected",)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(VerifiedRoutingError):
                    replace(candidate, **changes)

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
        self.assertNotIn('"objective":', json.dumps(quality.payload()))

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
            _receipt(
                route="blocked",
                profile="balanced",
                capabilities=("unknown-capability",),
            ),
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
        low_provider = MetadataTaskSignalProvider(
            confidence_with_capabilities=0.2,
            minimum_confidence=0.6,
        )
        low = recommend_shadow_route(
            _receipt(),
            _signals(provider=low_provider),
            self.scorecard,
            self.policy,
            profile="balanced",
            now=NOW,
            signal_provider=low_provider,
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
            _receipt(capabilities=("code",)),
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

    def test_capability_set_must_exactly_match_the_route_receipt(self) -> None:
        mismatched = _signals(capabilities=("code",))

        with self.assertRaisesRegex(VerifiedRoutingError, "capabilities"):
            recommend_shadow_route(
                _receipt(capabilities=("analysis",)),
                mismatched,
                self.scorecard,
                self.policy,
                profile="balanced",
                now=NOW,
            )

    def test_route_receipt_requires_the_complete_exact_bridge_task(self) -> None:
        for missing_field in ("objective_sha256", "allow_remote"):
            receipt = _receipt()
            receipt.raw_payload["task"].pop(missing_field)
            with self.subTest(missing_field=missing_field):
                with self.assertRaisesRegex(VerifiedRoutingError, "Missing"):
                    recommend_shadow_route(
                        receipt,
                        _signals(),
                        self.scorecard,
                        self.policy,
                        profile="balanced",
                        now=NOW,
                    )

        receipt = _receipt()
        receipt.raw_payload["task"]["unattested"] = True
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
            recommend_shadow_route(
                receipt,
                _signals(),
                self.scorecard,
                self.policy,
                profile="balanced",
                now=NOW,
            )

    def test_recomputed_signal_digest_cannot_bypass_provider_binding(self) -> None:
        payload = _signals().payload()
        payload["difficulty"] = "simple"
        unsigned = dict(payload)
        unsigned.pop("signals_sha256")
        payload["signals_sha256"] = sha256_json(unsigned)
        tampered = TaskSignals.from_payload(payload)

        with self.assertRaisesRegex(VerifiedRoutingError, "signal provider"):
            recommend_shadow_route(
                _receipt(),
                tampered,
                self.scorecard,
                self.policy,
                profile="balanced",
                now=NOW,
            )

    def test_unattested_context_override_cannot_change_decision_signals(self) -> None:
        signals = MetadataTaskSignalProvider().signals_from_metadata(
            _task(),
            context_tokens=5000,
        )

        with self.assertRaisesRegex(VerifiedRoutingError, "signal provider"):
            recommend_shadow_route(
                _receipt(),
                signals,
                self.scorecard,
                self.policy,
                profile="balanced",
                now=NOW,
            )

    def test_decision_round_trip_and_content_tamper_are_strict(self) -> None:
        decision = recommend_shadow_route(
            _receipt(route="local_then_verify"),
            _signals(),
            self.scorecard,
            self.policy,
            profile="balanced",
            now=NOW,
        )
        payload = decision.payload()

        self.assertEqual(ShadowRouteDecision.from_payload(payload), decision)
        self.assertEqual(decision.route_receipt_id, "route-policy-test")
        self.assertEqual(decision.task_signals, _signals())
        tampered = json.loads(json.dumps(payload))
        tampered["recommended_route"] = "premium"
        with self.assertRaisesRegex(VerifiedRoutingError, "digest"):
            ShadowRouteDecision.from_payload(tampered)

        nested_tamper = json.loads(json.dumps(payload))
        nested_tamper["task_signals"]["confidence"] = 0.8
        with self.assertRaisesRegex(VerifiedRoutingError, "digest"):
            ShadowRouteDecision.from_payload(nested_tamper)

        impossible_payloads = []
        recommends_blocked = json.loads(json.dumps(payload))
        recommends_blocked["recommended_route"] = "blocked"
        impossible_payloads.append(recommends_blocked)
        abstained_reroute = json.loads(json.dumps(payload))
        abstained_reroute["abstained"] = True
        abstained_reroute["recommended_route"] = "premium"
        impossible_payloads.append(abstained_reroute)
        ineligible_winner = json.loads(json.dumps(payload))
        winner = next(
            item
            for item in ineligible_winner["candidates"]
            if item["route"] == ineligible_winner["recommended_route"]
        )
        winner["pareto_eligible"] = False
        impossible_payloads.append(ineligible_winner)
        missing_candidate = json.loads(json.dumps(payload))
        missing_candidate["candidates"].pop()
        impossible_payloads.append(missing_candidate)

        for impossible in impossible_payloads:
            unsigned = dict(impossible)
            unsigned.pop("decision_sha256")
            impossible["decision_sha256"] = sha256_json(unsigned)
            with self.subTest(impossible=impossible):
                with self.assertRaises(VerifiedRoutingError):
                    ShadowRouteDecision.from_payload(impossible)


def _receipt(
    *,
    route: str = "local",
    profile: str = "balanced",
    capabilities: tuple[str, ...] = ("analysis",),
    local_gaps: tuple[str, ...] = (),
    premium_gaps: tuple[str, ...] = (),
    remote_allowed: bool = True,
    premium_call_budget: int = 1,
) -> SimpleNamespace:
    task = _task(profile=profile, capabilities=capabilities)
    local_runtime = {
        "provider_id": "local-a",
        "model": "local/model-a",
        "execution_scope": "device_only",
    }
    local_runtime["runtime_sha256"] = sha256_json(local_runtime)
    premium_runtime = {
        "provider_id": "premium-a",
        "model": "premium/model-a",
        "execution_scope": "paid_remote",
    }
    premium_runtime["runtime_sha256"] = sha256_json(premium_runtime)
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "contract": "RouteDecisionReceipt",
        "receipt_id": "route-policy-test",
        "task": task,
        "route": route,
        "local_provider": "local-a",
        "premium_provider": "premium-a",
        "local_gaps": list(local_gaps),
        "premium_gaps": list(premium_gaps),
        "remote_allowed": remote_allowed,
        "premium_call_budget": premium_call_budget,
        "rationale_codes": ["profile_selected"],
        "expected_flow": ["local", "verify"],
        "config_sha256": CONFIG_SHA256,
        "workspace": {"fingerprint": WORKSPACE_SHA256},
        "local_runtime": local_runtime,
        "premium_runtime": premium_runtime,
    }
    return SimpleNamespace(
        raw_payload=payload,
        receipt_id=payload["receipt_id"],
        route=route,
        config_sha256=CONFIG_SHA256,
        local_gaps=local_gaps,
        premium_gaps=premium_gaps,
        remote_allowed=remote_allowed,
        premium_call_budget=premium_call_budget,
        task=task,
    )


def _task(
    *,
    profile: str = "balanced",
    capabilities: tuple[str, ...] = ("analysis",),
) -> dict[str, object]:
    return {
        "task_id": "task-policy-test",
        "objective_sha256": OBJECTIVE_SHA256,
        "task_fingerprint": TASK_FINGERPRINT,
        "objective_chars": 400,
        "profile": profile,
        "capability_demand": {
            "required": list(capabilities),
            "tools": [],
            "risk_class": "read_only",
        },
        "constraint_count": 2,
        "no_change_expected": False,
        "required_verifier_ids": ["tests"],
        "allow_remote": True,
        "allow_remote_workspace": False,
        "max_premium_calls": 1,
    }


def _signals(
    *,
    capabilities: tuple[str, ...] = ("analysis",),
    request_fingerprint: str = TASK_FINGERPRINT,
    provider: MetadataTaskSignalProvider | None = None,
) -> TaskSignals:
    task = _task(capabilities=capabilities)
    task["task_fingerprint"] = request_fingerprint
    return (provider or MetadataTaskSignalProvider()).signals_from_metadata(
        task,
    )


if __name__ == "__main__":
    unittest.main()
