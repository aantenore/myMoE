from __future__ import annotations

import json
import unittest

from local_moe.local_cascade import (
    LocalCascadeRun,
    run_local_cascade,
    verify_local_cascade_content,
)
from local_moe.local_cascade_contracts import (
    LocalCascadeAttemptRequestV1,
    LocalCascadeAttemptResultV1,
    LocalCascadeConfigV1,
    LocalCascadeContractError,
    LocalCascadeJsonFieldV1,
    LocalCascadeReceiptV1,
    LocalCascadeTaskV1,
    LocalCascadeTierV1,
    LocalCascadeTokenCountV1,
    LocalCascadeVerifierV1,
    canonical_json,
)


class RecordingAttemptPort:
    def __init__(self, results: list[LocalCascadeAttemptResultV1]) -> None:
        self.results = list(results)
        self.requests: list[LocalCascadeAttemptRequestV1] = []

    def attempt(
        self, request: LocalCascadeAttemptRequestV1
    ) -> LocalCascadeAttemptResultV1:
        self.requests.append(request)
        return self.results.pop(0)


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def token(source: str, count: int | None) -> LocalCascadeTokenCountV1:
    return LocalCascadeTokenCountV1(source=source, count=count)


def completed(
    content: str,
    *,
    input_source: str = "actual",
    input_count: int | None = 10,
    output_source: str = "actual",
    output_count: int | None = 5,
) -> LocalCascadeAttemptResultV1:
    return LocalCascadeAttemptResultV1(
        status="completed",
        content=content,
        input_tokens=token(input_source, input_count),
        output_tokens=token(output_source, output_count),
    )


def abstained() -> LocalCascadeAttemptResultV1:
    return LocalCascadeAttemptResultV1(
        status="abstained",
        content=None,
        input_tokens=token("unknown", None),
        output_tokens=token("unknown", None),
    )


def tier(tier_id: str, cost_rank: int) -> LocalCascadeTierV1:
    return LocalCascadeTierV1(
        tier_id=tier_id,
        cost_rank=cost_rank,
        model_ref=f"local/{tier_id}",
        max_input_tokens=4_096,
        max_output_tokens=512,
    )


def text_verifier() -> LocalCascadeVerifierV1:
    return LocalCascadeVerifierV1(
        output_format="text",
        min_characters=10,
        max_characters=80,
        required_terms=("local",),
        forbidden_terms=("unverified",),
    )


def text_config(*tiers: LocalCascadeTierV1) -> LocalCascadeConfigV1:
    return LocalCascadeConfigV1(
        cascade_id="summary-default",
        tiers=tiers,
        verifier=text_verifier(),
        max_attempts=len(tiers),
    )


def task(
    instruction: str = "Summarize the device note.",
) -> LocalCascadeTaskV1:
    return LocalCascadeTaskV1(
        task_id="summary-1",
        kind="summarization",
        instruction=instruction,
        output_format="text",
    )


class LocalCascadeExecutionTests(unittest.TestCase):
    def test_cheapest_tier_passes_and_stops_the_cascade(self) -> None:
        port = RecordingAttemptPort(
            [completed("A local summary that meets every deterministic rule.")]
        )
        config = text_config(tier("large", 20), tier("small", 1))

        result = run_local_cascade(
            task(),
            config,
            port,
            clock=IncrementingClock(),
        )

        self.assertIsInstance(result, LocalCascadeRun)
        self.assertEqual(
            result.content,
            "A local summary that meets every deterministic rule.",
        )
        self.assertEqual(result.receipt.status, "passed")
        self.assertEqual(result.receipt.selected_tier_id, "small")
        self.assertEqual(result.receipt.attempt_count, 1)
        self.assertEqual([item.tier.tier_id for item in port.requests], ["small"])
        self.assertEqual(port.requests[0].verifier_reason_codes, ())

    def test_failed_content_is_not_forwarded_during_escalation(self) -> None:
        private_failed_content = "RAW_FAILED_BODY_7a3d"
        original_instruction = "ORIGINAL_TASK_BODY_92b1"
        port = RecordingAttemptPort(
            [
                completed(private_failed_content),
                completed("A local result accepted by deterministic checks."),
            ]
        )

        result = run_local_cascade(
            task(original_instruction),
            text_config(tier("tiny", 0), tier("medium", 5)),
            port,
            clock=IncrementingClock(),
        )

        self.assertEqual(result.receipt.status, "passed")
        self.assertEqual(len(port.requests), 2)
        second = port.requests[1]
        self.assertEqual(second.task.instruction, original_instruction)
        self.assertEqual(
            second.verifier_reason_codes,
            ("missing_required_term",),
        )
        self.assertNotIn(private_failed_content, canonical_json(second.payload()))
        self.assertEqual(
            set(second.payload()),
            {
                "schema_version",
                "contract",
                "task",
                "tier",
                "attempt_number",
                "verifier_reason_codes",
                "execution_scope",
                "allow_network",
                "allow_tools",
                "allow_writes",
                "parallel_attempts",
            },
        )

    def test_all_abstain_has_no_final_content(self) -> None:
        port = RecordingAttemptPort([abstained(), abstained()])

        result = run_local_cascade(
            task(),
            text_config(tier("tiny", 0), tier("small", 1)),
            port,
            clock=IncrementingClock(),
        )

        self.assertIsNone(result.content)
        self.assertEqual(result.receipt.status, "all_abstained")
        self.assertIsNone(result.receipt.selected_tier_id)
        self.assertEqual(
            port.requests[1].verifier_reason_codes,
            ("attempt_abstained",),
        )

    def test_receipt_is_metadata_only(self) -> None:
        task_secret = "TASK_SECRET_4b8d"
        failed_secret = "FAILED_SECRET_d812"
        accepted_secret = "FINAL_SECRET_f913"
        port = RecordingAttemptPort(
            [
                completed(failed_secret),
                completed(f"A local summary {accepted_secret} is accepted."),
            ]
        )

        result = run_local_cascade(
            task(task_secret),
            text_config(tier("tiny", 0), tier("small", 1)),
            port,
            clock=IncrementingClock(),
        )
        receipt_json = canonical_json(result.receipt.payload())

        self.assertEqual(
            result.content,
            f"A local summary {accepted_secret} is accepted.",
        )
        for raw_body in (task_secret, failed_secret, accepted_secret):
            self.assertNotIn(raw_body, receipt_json)
        self.assertNotIn("instruction", receipt_json)
        self.assertNotIn("content", receipt_json)
        self.assertIn("output_sha256", receipt_json)

    def test_offline_constraints_are_fixed_for_every_attempt(self) -> None:
        port = RecordingAttemptPort([abstained(), abstained()])

        run_local_cascade(
            task(),
            text_config(tier("tiny", 0), tier("small", 1)),
            port,
            clock=IncrementingClock(),
        )

        for request in port.requests:
            self.assertEqual(request.execution_scope, "offline_local")
            self.assertFalse(request.allow_network)
            self.assertFalse(request.allow_tools)
            self.assertFalse(request.allow_writes)
            self.assertEqual(request.parallel_attempts, 1)

        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeConfigV1(
                cascade_id="unsafe",
                tiers=(tier("tiny", 0),),
                verifier=text_verifier(),
                max_attempts=1,
                allow_network=True,
            )
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeAttemptResultV1(
                status="completed",
                content="A local result.",
                input_tokens=token("actual", 1),
                output_tokens=token("actual", 1),
                tool_calls=1,
            )

    def test_token_sources_remain_separate_and_unknown_is_not_estimated(self) -> None:
        port = RecordingAttemptPort(
            [
                completed(
                    "too short",
                    input_source="actual",
                    input_count=11,
                    output_source="estimated",
                    output_count=3,
                ),
                completed(
                    "A local summary accepted by deterministic checks.",
                    input_source="unknown",
                    input_count=None,
                    output_source="actual",
                    output_count=7,
                ),
            ]
        )

        result = run_local_cascade(
            task(),
            text_config(tier("tiny", 0), tier("small", 1)),
            port,
            clock=IncrementingClock(),
        )
        totals = result.receipt.token_totals

        self.assertEqual(totals.actual_input_tokens, 11)
        self.assertEqual(totals.actual_output_tokens, 7)
        self.assertEqual(totals.estimated_input_tokens, 0)
        self.assertEqual(totals.estimated_output_tokens, 3)
        self.assertEqual(totals.unknown_input_attempts, 1)
        self.assertEqual(totals.unknown_output_attempts, 0)
        self.assertEqual(result.receipt.attempts[1].input_tokens.source, "unknown")
        with self.assertRaises(LocalCascadeContractError):
            token("unknown", 0)


class LocalCascadeVerifierTests(unittest.TestCase):
    def test_json_classification_uses_allowed_values(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=100,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="label",
                    value_kind="string",
                    required=True,
                    allowed_string_values=("bug", "feature"),
                ),
            ),
        )

        accepted = verify_local_cascade_content('{"label":"bug"}', verifier)
        rejected = verify_local_cascade_content('{"label":"maybe"}', verifier)

        self.assertTrue(accepted.passed)
        self.assertEqual(
            rejected.reason_codes,
            ("json_string_value_not_allowed",),
        )

    def test_json_extraction_checks_shape_types_and_duplicates(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=200,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="name",
                    value_kind="string",
                    required=True,
                ),
                LocalCascadeJsonFieldV1(
                    name="quantity",
                    value_kind="integer",
                    required=True,
                ),
            ),
        )

        accepted = verify_local_cascade_content(
            '{"name":"cable","quantity":2}', verifier
        )
        wrong_type = verify_local_cascade_content(
            '{"name":"cable","quantity":true}', verifier
        )
        duplicate = verify_local_cascade_content(
            '{"name":"cable","name":"wire","quantity":2}', verifier
        )

        self.assertTrue(accepted.passed)
        self.assertIn("json_field_type_mismatch", wrong_type.reason_codes)
        self.assertEqual(duplicate.reason_codes, ("invalid_json",))

    def test_text_bounds_are_deterministic(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="text",
            min_characters=5,
            max_characters=12,
        )

        self.assertTrue(verify_local_cascade_content("five!", verifier).passed)
        self.assertEqual(
            verify_local_cascade_content("x", verifier).reason_codes,
            ("content_too_short",),
        )
        self.assertEqual(
            verify_local_cascade_content("x" * 13, verifier).reason_codes,
            ("content_too_long",),
        )


class LocalCascadeStrictContractTests(unittest.TestCase):
    def test_task_tier_verifier_and_config_round_trip_strictly(self) -> None:
        original_task = task()
        original_config = text_config(tier("small", 1), tier("tiny", 0))

        self.assertEqual(
            LocalCascadeTaskV1.from_payload(original_task.payload()),
            original_task,
        )
        self.assertEqual(
            LocalCascadeConfigV1.from_payload(original_config.payload()),
            original_config,
        )
        self.assertEqual(
            LocalCascadeTierV1.from_payload(tier("tiny", 0).payload()),
            tier("tiny", 0),
        )
        self.assertEqual(
            LocalCascadeVerifierV1.from_payload(text_verifier().payload()),
            text_verifier(),
        )

    def test_unknown_fields_and_versions_are_rejected(self) -> None:
        task_payload = task().payload()
        task_payload["extra"] = True
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeTaskV1.from_payload(task_payload)

        verifier_payload = text_verifier().payload()
        verifier_payload["schema_version"] = "2.0"
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeVerifierV1.from_payload(verifier_payload)

        tier_payload = tier("tiny", 0).payload()
        tier_payload["cost_rank"] = True
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeTierV1.from_payload(tier_payload)

        config_payload = text_config(tier("tiny", 0)).payload()
        config_payload["parallel_attempts"] = 2
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeConfigV1.from_payload(config_payload)

    def test_receipt_round_trip_has_no_unversioned_nested_values(self) -> None:
        result = run_local_cascade(
            task(),
            text_config(tier("tiny", 0)),
            RecordingAttemptPort(
                [completed("A local summary accepted by deterministic checks.")]
            ),
            clock=IncrementingClock(),
        )
        rendered = json.loads(canonical_json(result.receipt.payload()))

        parsed = LocalCascadeReceiptV1.from_payload(rendered)

        self.assertEqual(parsed, result.receipt)
        self.assertEqual(
            rendered["attempts"][0]["input_tokens"]["schema_version"],
            "1.0",
        )


if __name__ == "__main__":
    unittest.main()
