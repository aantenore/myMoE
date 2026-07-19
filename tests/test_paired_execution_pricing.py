from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
import unittest

from local_moe.paired_execution_pricing import (
    CommandCostEvidence,
    IncompleteCostEvidenceError,
    PairedCostEvidence,
    PricingContract,
    PricingItem,
    build_cost_evidence,
)
from local_moe.verified_routing_contracts import (
    VerifiedRoutingError,
    sha256_json,
)


LOCAL_RUNTIME = "1" * 64
PREMIUM_RUNTIME = "2" * 64


class PairedExecutionPricingTests(unittest.TestCase):
    def test_zero_priced_local_command_has_exact_zero_cost(self) -> None:
        pricing = PricingContract.build(
            [
                PricingItem(
                    provider_id="local",
                    model="local-model",
                    prompt_usd_per_million="0.0000",
                    completion_usd_per_million=0,
                )
            ]
        )

        evidence = build_cost_evidence(
            pricing,
            [
                {
                    "provider_id": "local",
                    "model": "local-model",
                    "provider_runtime_sha256": LOCAL_RUNTIME,
                    "prompt_tokens": 80_000,
                    "completion_tokens": 5_000,
                }
            ],
        )

        self.assertEqual(evidence.commands[0].cost_usd, "0")
        self.assertEqual(evidence.total_cost_usd, "0")
        serialized = json.dumps(evidence.payload())
        self.assertNotIn('"prompt":', serialized)
        self.assertNotIn('"output":', serialized)
        self.assertNotIn('"credential":', serialized)

    def test_multiple_commands_sum_exact_decimal_costs_without_rounding(self) -> None:
        pricing = PricingContract.build(
            [
                PricingItem("provider-b", "model-b", "0.8", "1.2"),
                PricingItem("provider-a", "model-a", "2.500", "10"),
                PricingItem("provider-c", "model-c", "0.001", "0"),
            ]
        )
        commands = [
            {
                "provider_id": "provider-a",
                "model": "model-a",
                "provider_runtime_sha256": PREMIUM_RUNTIME,
                "prompt_tokens": 1_000,
                "completion_tokens": 500,
            },
            {
                "provider_id": "provider-b",
                "model": "model-b",
                "provider_runtime_sha256": PREMIUM_RUNTIME,
                "prompt_tokens": 125_000,
                "completion_tokens": 25_000,
            },
            {
                "provider_id": "provider-c",
                "model": "model-c",
                "provider_runtime_sha256": PREMIUM_RUNTIME,
                "prompt_tokens": 1,
                "completion_tokens": 0,
            },
        ]

        evidence = build_cost_evidence(pricing, commands)

        self.assertEqual(
            [command.cost_usd for command in evidence.commands],
            ["0.0075", "0.13", "0.000000001"],
        )
        self.assertEqual(evidence.total_cost_usd, "0.137500001")
        self.assertEqual(
            Decimal(evidence.total_cost_usd),
            sum(
                (Decimal(command.cost_usd) for command in evidence.commands),
                start=Decimal(0),
            ),
        )
        self.assertEqual(
            PairedCostEvidence.from_payload(
                evidence.payload(), pricing=pricing
            ).payload(),
            evidence.payload(),
        )

    def test_missing_usage_model_rate_or_command_is_explicitly_incomplete(
        self,
    ) -> None:
        pricing = PricingContract.build(
            [PricingItem("provider", "known-model", "1", "2")]
        )
        complete = {
            "provider_id": "provider",
            "model": "known-model",
            "provider_runtime_sha256": PREMIUM_RUNTIME,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

        missing_usage = dict(complete)
        missing_usage.pop("completion_tokens")
        with self.assertRaisesRegex(
            IncompleteCostEvidenceError,
            "missing command metadata fields: completion_tokens",
        ):
            build_cost_evidence(pricing, [missing_usage])

        missing_model = dict(complete, model=None)
        with self.assertRaisesRegex(
            IncompleteCostEvidenceError,
            "model is required",
        ):
            build_cost_evidence(pricing, [missing_model])

        unknown_rate = dict(complete, model="unpriced-model")
        with self.assertRaisesRegex(
            IncompleteCostEvidenceError,
            "no pricing rate exists",
        ):
            build_cost_evidence(pricing, [unknown_rate])

        with self.assertRaisesRegex(
            IncompleteCostEvidenceError,
            "at least one command is required",
        ):
            build_cost_evidence(pricing, [])

    def test_metadata_boundary_rejects_content_and_caller_costs(self) -> None:
        pricing = PricingContract.build(
            [PricingItem("provider", "model", "1", "2")]
        )
        command = {
            "provider_id": "provider",
            "model": "model",
            "provider_runtime_sha256": PREMIUM_RUNTIME,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

        for forbidden in ("prompt", "output", "api_key", "cost_usd"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
                    build_cost_evidence(
                        pricing,
                        [dict(command, **{forbidden: "not-persisted"})],
                    )

    def test_pricing_and_cost_tampering_is_rejected(self) -> None:
        pricing = PricingContract.build(
            [PricingItem("provider", "model", "1", "2")]
        )
        evidence = build_cost_evidence(
            pricing,
            [
                {
                    "provider_id": "provider",
                    "model": "model",
                    "provider_runtime_sha256": PREMIUM_RUNTIME,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            ],
        )

        tampered_pricing = deepcopy(pricing.payload())
        tampered_pricing["items"][0]["prompt_usd_per_million"] = "99"
        with self.assertRaisesRegex(VerifiedRoutingError, "digest is invalid"):
            PricingContract.from_payload(tampered_pricing)

        tampered_command = deepcopy(evidence.payload())
        tampered_command["commands"][0]["cost_usd"] = "99"
        with self.assertRaisesRegex(VerifiedRoutingError, "digest is invalid"):
            PairedCostEvidence.from_payload(tampered_command)

        tampered_total = deepcopy(evidence.payload())
        tampered_total["total_cost_usd"] = "99"
        content = dict(tampered_total)
        content.pop("cost_sha256")
        tampered_total["cost_sha256"] = sha256_json(content)
        with self.assertRaisesRegex(VerifiedRoutingError, "does not equal"):
            PairedCostEvidence.from_payload(tampered_total)

        forged_command = CommandCostEvidence(
            provider_id="provider",
            model="model",
            provider_runtime_sha256=PREMIUM_RUNTIME,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd="99",
        )
        forged = PairedCostEvidence(
            pricing_sha256=pricing.pricing_sha256,
            commands=(forged_command,),
            total_cost_usd="99",
        )
        with self.assertRaisesRegex(VerifiedRoutingError, "does not match"):
            forged.validate_against(pricing)

    def test_nonfinite_negative_boolean_and_noninteger_numbers_are_rejected(
        self,
    ) -> None:
        invalid_rates = (
            float("nan"),
            float("inf"),
            Decimal("-Infinity"),
            -0.01,
            True,
        )
        for rate in invalid_rates:
            with self.subTest(rate=rate):
                with self.assertRaises(VerifiedRoutingError):
                    PricingItem("provider", "model", rate, "0")

        pricing = PricingContract.build(
            [PricingItem("provider", "model", "1", "2")]
        )
        base = {
            "provider_id": "provider",
            "model": "model",
            "provider_runtime_sha256": PREMIUM_RUNTIME,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
        for token_count in (-1, True, 1.0, float("nan"), float("inf")):
            with self.subTest(token_count=token_count):
                with self.assertRaises(VerifiedRoutingError):
                    build_cost_evidence(
                        pricing,
                        [dict(base, prompt_tokens=token_count)],
                    )

    def test_extreme_numeric_inputs_fail_before_unbounded_decimal_rendering(self) -> None:
        for rate in ("1e1000000", "1e-1000000", "9" * 129):
            with self.subTest(rate=rate[:20]):
                with self.assertRaisesRegex(
                    VerifiedRoutingError,
                    "supported magnitude|supported precision",
                ):
                    PricingItem("provider", "model", rate, "0")

        pricing = PricingContract.build(
            [PricingItem("provider", "model", "1", "2")]
        )
        command = {
            "provider_id": "provider",
            "model": "model",
            "provider_runtime_sha256": PREMIUM_RUNTIME,
            "prompt_tokens": 1_000_000_000_001,
            "completion_tokens": 0,
        }
        with self.assertRaisesRegex(VerifiedRoutingError, "supported bound"):
            build_cost_evidence(pricing, [command])

    def test_pricing_items_are_unique_and_canonically_ordered(self) -> None:
        first = PricingItem("a-provider", "model", "1", "2")
        last = PricingItem("z-provider", "model", "3", "4")

        with self.assertRaisesRegex(VerifiedRoutingError, "canonical"):
            PricingContract(items=(last, first))
        with self.assertRaisesRegex(VerifiedRoutingError, "must be unique"):
            PricingContract(items=(first, first))

        built = PricingContract.build((last, first))
        self.assertEqual(
            [item.provider_id for item in built.items],
            ["a-provider", "z-provider"],
        )

        reversed_payload = deepcopy(built.payload())
        reversed_payload["items"].reverse()
        with self.assertRaisesRegex(VerifiedRoutingError, "canonical"):
            PricingContract.from_payload(reversed_payload)

    def test_strict_parsers_reject_unknown_missing_and_duplicate_json_fields(
        self,
    ) -> None:
        pricing = PricingContract.build(
            [PricingItem("provider", "model", "1.000", "2.000")]
        )
        self.assertEqual(
            PricingContract.from_json(json.dumps(pricing.payload())).payload(),
            pricing.payload(),
        )

        unknown = deepcopy(pricing.payload())
        unknown["currency"] = "USD"
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
            PricingContract.from_payload(unknown)

        missing = deepcopy(pricing.payload())
        missing.pop("contract")
        with self.assertRaisesRegex(VerifiedRoutingError, "Missing"):
            PricingContract.from_payload(missing)

        with self.assertRaisesRegex(VerifiedRoutingError, "Duplicate JSON field"):
            PricingContract.from_json(
                '{"schema_version":"1.0","schema_version":"1.0"}'
            )

        with self.assertRaisesRegex(VerifiedRoutingError, "not finite"):
            PricingContract.from_json(
                '{"schema_version":NaN,"contract":"x","items":[],"pricing_sha256":"x"}'
            )


if __name__ == "__main__":
    unittest.main()
