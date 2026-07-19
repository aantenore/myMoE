from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import math
import unittest

from local_moe.route_signals import (
    MetadataTaskSignalProvider,
    TaskSignals,
    signals_from_route_receipt,
)
from local_moe.verified_routing_contracts import VerifiedRoutingError


def _fingerprint(value: str = "task") -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata(
    *,
    required: tuple[str, ...] = ("analysis",),
    tools: tuple[str, ...] = (),
    risk_class: str = "read_only",
    objective_chars: object = 400,
    constraint_count: object = 1,
) -> dict[str, object]:
    return {
        "task_fingerprint": _fingerprint(),
        "objective_chars": objective_chars,
        "capability_demand": {
            "required": list(required),
            "tools": list(tools),
            "risk_class": risk_class,
        },
        "constraint_count": constraint_count,
    }


def _receipt(task: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "contract": "RouteDecisionReceipt",
        "task": task if task is not None else _metadata(),
    }


class _ReceiptObject:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def payload(self) -> dict[str, object]:
        return self._payload


class _MismatchedProvider:
    def signals_from_metadata(
        self,
        task_metadata: object,
        *,
        context_tokens: int | None = None,
    ) -> TaskSignals:
        return TaskSignals(
            request_fingerprint=_fingerprint("different"),
            capabilities=("analysis",),
            difficulty="simple",
            confidence=1.0,
            abstained=False,
            source="test-provider",
        )


class TaskSignalsContractTests(unittest.TestCase):
    def test_payload_round_trip_is_content_free_and_immutable(self) -> None:
        private_material = "opaque-private-material-7431"
        signals = TaskSignals(
            request_fingerprint=_fingerprint(private_material),
            capabilities=("tests", "analysis"),
            difficulty="medium",
            confidence=0.8,
            abstained=False,
            source="metadata-test",
            objective_chars=len(private_material),
            context_tokens=10,
            constraint_count=2,
            tool_count=1,
        )

        payload = signals.payload()
        rendered = json.dumps(payload, sort_keys=True)

        self.assertEqual(TaskSignals.from_payload(payload), signals)
        self.assertEqual(payload["capabilities"], ["analysis", "tests"])
        self.assertNotIn(private_material, rendered)
        with self.assertRaises(FrozenInstanceError):
            signals.confidence = 0.2  # type: ignore[misc]

    def test_from_payload_rejects_unknown_or_missing_fields(self) -> None:
        payload = TaskSignals(
            request_fingerprint=_fingerprint(),
            capabilities=(),
            difficulty="simple",
            confidence=0.2,
            abstained=True,
            source="metadata-test",
        ).payload()

        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown"):
            TaskSignals.from_payload({**payload, "unexpected": 1})
        incomplete = dict(payload)
        incomplete.pop("source")
        with self.assertRaisesRegex(VerifiedRoutingError, "Missing"):
            TaskSignals.from_payload(incomplete)

    def test_payload_digest_rejects_content_tamper(self) -> None:
        payload = TaskSignals(
            request_fingerprint=_fingerprint(),
            capabilities=("analysis",),
            difficulty="simple",
            confidence=0.9,
            abstained=False,
            source="metadata-test",
        ).payload()
        payload["confidence"] = 0.8

        with self.assertRaisesRegex(VerifiedRoutingError, "digest"):
            TaskSignals.from_payload(payload)

    def test_constructor_rejects_nan_boolean_numbers_and_invalid_enums(self) -> None:
        base = {
            "request_fingerprint": _fingerprint(),
            "capabilities": ("analysis",),
            "difficulty": "simple",
            "confidence": 0.5,
            "abstained": False,
            "source": "metadata-test",
        }
        invalid = (
            {"confidence": math.nan},
            {"confidence": True},
            {"objective_chars": True},
            {"abstained": 1},
            {"difficulty": "unknown"},
            {"source": True},
            {"request_fingerprint": True},
            {"capabilities": (True,)},
        )
        for replacement in invalid:
            with self.subTest(replacement=replacement):
                with self.assertRaises(VerifiedRoutingError):
                    TaskSignals(**{**base, **replacement})  # type: ignore[arg-type]


class MetadataTaskSignalProviderTests(unittest.TestCase):
    def test_explicit_capabilities_are_sorted_and_deterministic(self) -> None:
        provider = MetadataTaskSignalProvider()
        task = _metadata(
            required=("tests", "analysis"),
            tools=("shell",),
            constraint_count=2,
        )

        first = provider.signals_from_metadata(task)
        second = provider.signals_from_metadata(task)

        self.assertEqual(first, second)
        self.assertEqual(first.capabilities, ("analysis", "tests"))
        self.assertEqual(first.tool_count, 1)
        self.assertEqual(first.context_tokens, 100)
        self.assertEqual(first.difficulty, "medium")
        self.assertFalse(first.abstained)

    def test_provider_configuration_digest_changes_with_behavior(self) -> None:
        default = MetadataTaskSignalProvider()
        configured = MetadataTaskSignalProvider(constraint_maxima=(2, 4, 10))

        self.assertNotEqual(default.config_sha256, configured.config_sha256)
        self.assertEqual(
            default.signals_from_metadata(_metadata()).provider_config_sha256,
            default.config_sha256,
        )

    def test_structural_thresholds_select_the_highest_difficulty_band(self) -> None:
        provider = MetadataTaskSignalProvider(
            objective_char_maxima=(10, 20, 30),
            context_token_maxima=(100, 200, 300),
            constraint_maxima=(10, 20, 30),
            tool_maxima=(10, 20, 30),
            capability_maxima=(10, 20, 30),
        )

        signals = provider.signals_from_metadata(
            _metadata(objective_chars=21, constraint_count=0)
        )

        self.assertEqual(signals.difficulty, "complex")
        risk_signals = MetadataTaskSignalProvider().signals_from_metadata(
            _metadata(risk_class="write_external")
        )

        self.assertEqual(risk_signals.difficulty, "complex")

    def test_missing_capabilities_abstains_unless_threshold_is_configured(self) -> None:
        signals = MetadataTaskSignalProvider().signals_from_metadata(
            _metadata(required=())
        )

        self.assertEqual(signals.confidence, 0.35)
        self.assertTrue(signals.abstained)
        provider = MetadataTaskSignalProvider(
            confidence_without_capabilities=0.7,
            minimum_confidence=0.65,
        )
        configured = provider.signals_from_metadata(_metadata(required=()))

        self.assertEqual(configured.confidence, 0.7)
        self.assertFalse(configured.abstained)

    def test_unknown_risk_is_out_of_distribution_and_abstains(self) -> None:
        signals = MetadataTaskSignalProvider().signals_from_metadata(
            _metadata(risk_class="novel-risk")
        )

        self.assertEqual(signals.difficulty, "very_complex")
        self.assertEqual(signals.confidence, 0.0)
        self.assertTrue(signals.abstained)

    def test_supported_maximum_overflow_is_out_of_distribution(self) -> None:
        provider = MetadataTaskSignalProvider(max_capability_count=1)

        signals = provider.signals_from_metadata(
            _metadata(required=("analysis", "tests"))
        )

        self.assertEqual(signals.difficulty, "very_complex")
        self.assertTrue(signals.abstained)

    def test_explicit_context_cannot_understate_the_structural_estimate(self) -> None:
        provider = MetadataTaskSignalProvider(objective_chars_per_context_token=4)
        task = _metadata(objective_chars=401)

        with self.assertRaisesRegex(VerifiedRoutingError, "structural estimate"):
            provider.signals_from_metadata(task, context_tokens=100)
        signals = provider.signals_from_metadata(task, context_tokens=101)
        self.assertEqual(signals.context_tokens, 101)

    def test_metadata_and_demand_unknown_fields_fail_closed(self) -> None:
        task = _metadata()
        task["raw_text"] = "not-accepted"
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown task metadata"):
            MetadataTaskSignalProvider().signals_from_metadata(task)

        task = _metadata()
        demand = dict(task["capability_demand"])  # type: ignore[arg-type]
        demand["extra"] = 1
        task["capability_demand"] = demand
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown capability_demand"):
            MetadataTaskSignalProvider().signals_from_metadata(task)

    def test_metadata_boolean_and_non_string_identifiers_fail_closed(self) -> None:
        with self.assertRaises(VerifiedRoutingError):
            MetadataTaskSignalProvider().signals_from_metadata(
                _metadata(objective_chars=True)
            )
        task = _metadata()
        demand = dict(task["capability_demand"])  # type: ignore[arg-type]
        demand["required"] = [True]
        task["capability_demand"] = demand
        with self.assertRaises(VerifiedRoutingError):
            MetadataTaskSignalProvider().signals_from_metadata(task)

    def test_invalid_provider_configuration_fails_closed(self) -> None:
        invalid = (
            {"minimum_confidence": math.nan},
            {"objective_char_maxima": (1, True, 3)},
            {"objective_char_maxima": (3, 2, 1)},
            {"objective_chars_per_context_token": 0},
            {"risk_difficulties": (("new-risk", "invalid"),)},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(VerifiedRoutingError):
                    MetadataTaskSignalProvider(**values)  # type: ignore[arg-type]


class RouteReceiptSignalTests(unittest.TestCase):
    def test_mapping_and_payload_object_delegate_to_the_provider(self) -> None:
        receipt = _receipt()

        from_mapping = signals_from_route_receipt(receipt)
        from_object = signals_from_route_receipt(_ReceiptObject(receipt))

        self.assertEqual(from_mapping, from_object)
        self.assertEqual(from_mapping.request_fingerprint, _fingerprint())

    def test_receipt_contract_and_unknown_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(VerifiedRoutingError, "contract"):
            signals_from_route_receipt({"contract": "Other", "task": _metadata()})
        with self.assertRaisesRegex(VerifiedRoutingError, "Unknown route receipt"):
            signals_from_route_receipt({**_receipt(), "extra": 1})
        with self.assertRaisesRegex(VerifiedRoutingError, "missing task"):
            signals_from_route_receipt({"contract": "RouteDecisionReceipt"})

    def test_provider_cannot_change_the_request_binding(self) -> None:
        with self.assertRaisesRegex(VerifiedRoutingError, "different request"):
            signals_from_route_receipt(
                _receipt(),
                provider=_MismatchedProvider(),
            )


if __name__ == "__main__":
    unittest.main()
