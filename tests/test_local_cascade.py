from __future__ import annotations

import json
import unittest

from local_moe.local_cascade import (
    MAX_JSON_NESTING_DEPTH,
    MAX_JSON_NODES,
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
    def __init__(self, step: float = 0.01) -> None:
        self.value = 100.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
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
                "requested_execution_scope",
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
            self.assertEqual(request.requested_execution_scope, "offline_local")
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

    def test_reported_token_limits_force_escalation_but_unknown_stays_unknown(
        self,
    ) -> None:
        port = RecordingAttemptPort(
            [
                completed(
                    "A local summary accepted by deterministic checks.",
                    input_count=5_000,
                    output_source="estimated",
                    output_count=999,
                ),
                completed(
                    "A local summary accepted by deterministic checks.",
                    input_source="unknown",
                    input_count=None,
                    output_source="unknown",
                    output_count=None,
                ),
            ]
        )

        result = run_local_cascade(
            task(),
            text_config(tier("tiny", 0), tier("small", 1)),
            port,
            clock=IncrementingClock(),
        )

        self.assertEqual(result.receipt.status, "passed")
        self.assertEqual(result.receipt.selected_tier_id, "small")
        self.assertEqual(
            result.receipt.attempts[0].verifier_reason_codes,
            (
                "input_token_limit_exceeded",
                "output_token_limit_exceeded",
            ),
        )
        self.assertEqual(result.receipt.attempts[1].input_tokens.source, "unknown")
        self.assertEqual(result.receipt.attempts[1].output_tokens.source, "unknown")

    def test_nested_json_parser_failure_escalates_instead_of_aborting(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=262_144,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="label",
                    value_kind="string",
                    required=True,
                    allowed_string_values=("bug", "feature"),
                ),
            ),
        )
        config = LocalCascadeConfigV1(
            cascade_id="classification-default",
            tiers=(tier("tiny", 0), tier("small", 1)),
            verifier=verifier,
            max_attempts=2,
        )
        json_task = LocalCascadeTaskV1(
            task_id="classification-1",
            kind="classification",
            instruction="Classify the local issue.",
            output_format="json_object",
        )
        nested = '{"label":' + ('{"nested":' * 1_200) + "null" + ("}" * 1_200) + "}"
        port = RecordingAttemptPort([completed(nested), completed('{"label":"bug"}')])

        result = run_local_cascade(
            json_task,
            config,
            port,
            clock=IncrementingClock(),
        )

        self.assertEqual(result.receipt.status, "passed")
        self.assertEqual(result.receipt.selected_tier_id, "small")
        self.assertEqual(
            result.receipt.attempts[0].verifier_reason_codes,
            ("invalid_json",),
        )


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

    def test_json_number_overflow_is_rejected_as_invalid_json(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=100,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="score",
                    value_kind="number",
                    required=True,
                ),
            ),
        )

        result = verify_local_cascade_content('{"score":1e400}', verifier)

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_codes, ("invalid_json",))

    def test_json_object_depth_limit_is_explicit_and_deterministic(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=262_144,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="payload",
                    value_kind="object",
                    required=True,
                ),
            ),
        )

        at_limit = (
            '{"payload":'
            + ('{"nested":' * (MAX_JSON_NESTING_DEPTH - 1))
            + "null"
            + ("}" * (MAX_JSON_NESTING_DEPTH - 1))
            + "}"
        )
        above_limit = (
            '{"payload":'
            + ('{"nested":' * MAX_JSON_NESTING_DEPTH)
            + "null"
            + ("}" * MAX_JSON_NESTING_DEPTH)
            + "}"
        )

        self.assertTrue(verify_local_cascade_content(at_limit, verifier).passed)
        self.assertEqual(
            verify_local_cascade_content(above_limit, verifier).reason_codes,
            ("invalid_json",),
        )

    def test_json_array_depth_limit_is_explicit_and_deterministic(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=262_144,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="payload",
                    value_kind="array",
                    required=True,
                ),
            ),
        )

        at_limit = (
            '{"payload":'
            + ("[" * (MAX_JSON_NESTING_DEPTH - 1))
            + "0"
            + ("]" * (MAX_JSON_NESTING_DEPTH - 1))
            + "}"
        )
        above_limit = (
            '{"payload":'
            + ("[" * MAX_JSON_NESTING_DEPTH)
            + "0"
            + ("]" * MAX_JSON_NESTING_DEPTH)
            + "}"
        )

        self.assertTrue(verify_local_cascade_content(at_limit, verifier).passed)
        self.assertEqual(
            verify_local_cascade_content(above_limit, verifier).reason_codes,
            ("invalid_json",),
        )

    def test_json_node_limit_counts_values_without_recursive_walk(self) -> None:
        verifier = LocalCascadeVerifierV1(
            output_format="json_object",
            min_characters=2,
            max_characters=262_144,
            json_fields=(
                LocalCascadeJsonFieldV1(
                    name="items",
                    value_kind="array",
                    required=True,
                ),
            ),
        )
        at_limit = json.dumps({"items": [0] * (MAX_JSON_NODES - 2)})
        above_limit = json.dumps({"items": [0] * (MAX_JSON_NODES - 1)})

        self.assertTrue(verify_local_cascade_content(at_limit, verifier).passed)
        self.assertEqual(
            verify_local_cascade_content(above_limit, verifier).reason_codes,
            ("invalid_json",),
        )

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

    def test_scope_is_explicitly_requested_and_unverified(self) -> None:
        config = text_config(tier("tiny", 0))
        config_payload = config.payload()
        self.assertEqual(config.requested_execution_scope, "offline_local")
        self.assertIn("requested_execution_scope", config_payload)
        self.assertNotIn("execution_scope", config_payload)

        legacy_payload = config.payload()
        legacy_payload["execution_scope"] = legacy_payload.pop(
            "requested_execution_scope"
        )
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeConfigV1.from_payload(legacy_payload)

        port = RecordingAttemptPort(
            [completed("A local summary accepted by deterministic checks.")]
        )
        result = run_local_cascade(
            task(),
            config,
            port,
            clock=IncrementingClock(),
        )
        self.assertEqual(
            port.requests[0].requested_execution_scope,
            "offline_local",
        )
        request_payload = port.requests[0].payload()
        self.assertIn("requested_execution_scope", request_payload)
        self.assertNotIn("execution_scope", request_payload)
        self.assertEqual(
            result.receipt.requested_execution_scope,
            "offline_local",
        )
        self.assertEqual(
            result.receipt.execution_scope_attestation,
            "adapter_declared_unverified",
        )
        receipt_payload = result.receipt.payload()
        self.assertNotIn("execution_scope", receipt_payload)
        self.assertEqual(
            receipt_payload["execution_scope_attestation"],
            "adapter_declared_unverified",
        )

    def test_receipt_parser_rejects_cross_field_contradictions(self) -> None:
        result = run_local_cascade(
            task(),
            text_config(tier("tiny", 0), tier("small", 1)),
            RecordingAttemptPort(
                [
                    completed("too short"),
                    completed("A local summary accepted by deterministic checks."),
                ]
            ),
            clock=IncrementingClock(),
        )

        bad_status = result.receipt.payload()
        bad_status["status"] = "exhausted"
        bad_status["selected_tier_id"] = None
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_status)

        bad_selected_tier = result.receipt.payload()
        bad_selected_tier["selected_tier_id"] = "tiny"
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_selected_tier)

        bad_count = result.receipt.payload()
        bad_count["attempt_count"] = 1
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_count)

        bad_totals = result.receipt.payload()
        totals = bad_totals["token_totals"]
        assert isinstance(totals, dict)
        totals["actual_input_tokens"] += 1
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_totals)

        missing_output_digest = result.receipt.payload()
        attempts = missing_output_digest["attempts"]
        assert isinstance(attempts, list)
        attempts[0]["output_sha256"] = None
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(missing_output_digest)

        bad_sequence = result.receipt.payload()
        attempts = bad_sequence["attempts"]
        assert isinstance(attempts, list)
        attempts[1]["attempt_number"] = 3
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_sequence)

        bad_reason = result.receipt.payload()
        attempts = bad_reason["attempts"]
        assert isinstance(attempts, list)
        attempts[0]["verifier_reason_codes"] = ["attempt_port_error"]
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_reason)

        bad_evidence = result.receipt.payload()
        bad_evidence["evidence_sha256"] = "0" * 64
        with self.assertRaises(LocalCascadeContractError):
            LocalCascadeReceiptV1.from_payload(bad_evidence)

    def test_run_ids_are_unique_but_evidence_digests_are_correlatable(self) -> None:
        raw_content = "A local summary accepted by deterministic checks."

        def run(step: float) -> LocalCascadeRun:
            return run_local_cascade(
                task(),
                text_config(tier("tiny", 0)),
                RecordingAttemptPort([completed(raw_content)]),
                clock=IncrementingClock(step),
            )

        first = run(0.01)
        second = run(0.02)

        self.assertNotEqual(first.receipt.run_id, second.receipt.run_id)
        self.assertRegex(first.receipt.run_id, r"^cascade-run-[0-9a-f]{32}$")
        self.assertNotEqual(
            first.receipt.total_duration_ms,
            second.receipt.total_duration_ms,
        )
        self.assertEqual(
            first.receipt.evidence_sha256,
            second.receipt.evidence_sha256,
        )
        self.assertEqual(first.receipt.task_sha256, second.receipt.task_sha256)
        self.assertEqual(
            first.receipt.attempts[0].output_sha256,
            second.receipt.attempts[0].output_sha256,
        )
        self.assertNotIn(raw_content, canonical_json(first.receipt.payload()))

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
            "1.1",
        )
        self.assertEqual(rendered["schema_version"], "1.1")
        self.assertEqual(
            parsed.evidence_sha256,
            result.receipt.evidence_sha256,
        )


if __name__ == "__main__":
    unittest.main()
