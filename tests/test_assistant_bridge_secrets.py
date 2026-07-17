from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from local_moe.assistant_bridge_secrets import (
    DetectSecretsResidualDetector,
    REDACTED_VALUE,
    ResidualAssuranceUnavailableError,
    SecretRedactionError,
    SecretRedactionPolicy,
    TextSecretPattern,
    redact_text,
    redact_user_controlled_fields,
)


HAS_DETECT_SECRETS = importlib.util.find_spec("detect_secrets") is not None


def structural_policy(**overrides: object) -> SecretRedactionPolicy:
    values: dict[str, object] = {"require_residual_assurance": False}
    values.update(overrides)
    return SecretRedactionPolicy(**values)  # type: ignore[arg-type]


class _ResidualDetector:
    name = "test-residual"

    def __init__(self, needle: str = "residual-secret") -> None:
        self.needle = needle
        self.values: list[str] = []

    def redact(self, value: str, replacement: str) -> tuple[str, int]:
        self.values.append(value)
        count = value.count(self.needle)
        return value.replace(self.needle, replacement), count


class _BrokenResidualDetector:
    name = "broken-residual"

    def redact(self, value: str, replacement: str) -> tuple[str, int]:
        del value, replacement
        raise RuntimeError("sensitive diagnostic must not escape")


class AssistantBridgeSecretRedactionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_DETECT_SECRETS, "detect-secrets 1.5 optional dependency")
    def test_overlapping_findings_do_not_redact_the_replacement_recursively(self) -> None:
        result = redact_text("Bearer bearer-secret")

        self.assertEqual(result.value, "[redacted] [redacted]")
        self.assertNotIn("[r[", str(result.value))

    def test_redacts_named_provider_and_structured_secret_repros(self) -> None:
        slack = "xoxb" + "-123456789012-123456789012-abcdefghijklmnopqrstuvwx"
        slack_webhook = (
            "https://hooks.slack.com/services/TA2345678/BA2345678/"
            "abcdefghijklmnopqrstuvwx"
        )
        github = "ghp_" + "A" * 36
        github_fine_grained = "github_pat_" + "B" * 30
        gitlab = "glpat-" + "c" * 20
        aws = "AKIA" + "D" * 16
        basic = "dXNlcjpwYXNzd29yZA=="
        payload = {
            "objective": "\n".join(
                (
                    f"Slack {slack}",
                    f"Webhook {slack_webhook}",
                    f"GitHub {github} {github_fine_grained}",
                    f"GitLab {gitlab}",
                    f"AWS {aws}",
                    f"Authorization: Basic {basic}",
                    'JSON {"api_key":"json-value"}',
                    'JSON {"OPENAI_API_KEY":"prefixed-json-value"}',
                    "PASSWORD=two assignment words",
                    'client_secret="quoted secret words"',
                    "AWS_SECRET_ACCESS_KEY=aws/secret+material=",
                    "GITHUB_TOKEN=opaque-provider-value",
                )
            ),
            "config": {
                "client_secret": "nested-client-value",
                "credentials": {"username": "alice", "password": "nested-value"},
                "safe": "visible",
            },
        }

        result = redact_user_controlled_fields(payload, structural_policy())
        rendered = json.dumps(result.value, sort_keys=True)

        for secret in (
            slack,
            slack_webhook,
            github,
            github_fine_grained,
            gitlab,
            aws,
            basic,
            "json-value",
            "prefixed-json-value",
            "two assignment words",
            "quoted secret words",
            "aws/secret+material=",
            "opaque-provider-value",
            "nested-client-value",
            "nested-value",
        ):
            self.assertNotIn(secret, rendered)
        self.assertIn(REDACTED_VALUE, rendered)
        self.assertIn("Authorization: Basic [redacted]", rendered)
        self.assertEqual(result.value["config"]["safe"], "visible")  # type: ignore[index]
        self.assertGreaterEqual(result.redaction_count, 11)

    def test_does_not_mutate_input_and_skips_generated_metadata_fields(self) -> None:
        fingerprint = "0123456789abcdef" * 4
        artifact_digest = "fedcba9876543210" * 4
        payload = {
            "objective": "safe objective",
            "task_fingerprint": fingerprint,
            "artifact_sha256": artifact_digest,
            "nested": {"receipt_digest": fingerprint},
        }
        original = deepcopy(payload)
        detector = _ResidualDetector()

        result = redact_user_controlled_fields(
            payload,
            SecretRedactionPolicy(),
            residual_detector=detector,
        )

        self.assertEqual(payload, original)
        self.assertEqual(result.value["task_fingerprint"], fingerprint)  # type: ignore[index]
        self.assertEqual(result.value["artifact_sha256"], artifact_digest)  # type: ignore[index]
        self.assertEqual(result.value["nested"]["receipt_digest"], fingerprint)  # type: ignore[index]
        self.assertEqual(detector.values, ["safe objective"])
        self.assertEqual(result.skipped_generated_field_count, 3)

    def test_configurable_field_path_and_custom_pattern(self) -> None:
        custom = TextSecretPattern(
            name="custom-ticket-secret",
            regex=re.compile(r"CUSTOM\[(?P<secret>[^]]+)\]"),
        )
        policy = structural_policy(
            user_controlled_fields=frozenset({"request.body"}),
            patterns=(custom,),
        )
        payload = {
            "request": {"body": "CUSTOM[hide-me]", "title": "CUSTOM[keep-me]"},
            "body": "CUSTOM[not-selected]",
        }

        result = redact_user_controlled_fields(payload, policy)

        self.assertEqual(result.value["request"]["body"], "CUSTOM[[redacted]]")  # type: ignore[index]
        self.assertEqual(result.value["request"]["title"], "CUSTOM[keep-me]")  # type: ignore[index]
        self.assertEqual(result.value["body"], "CUSTOM[not-selected]")  # type: ignore[index]

    def test_reasonable_false_positives_remain_visible(self) -> None:
        objective = (
            "Tune the token budget and API design; see issue gh-123, UUID "
            "123e4567-e89b-12d3-a456-426614174000, region eu-west-1, and "
            "https://example.test/docs?topic=tokenization."
        )
        payload = {
            "objective": objective,
            "token_budget": 8192,
            "max_token": 4096,
            "min_tokens": 128,
            "output_token_count": 42,
            "monkey": "banana",
            "keynote": "architecture",
        }

        result = redact_user_controlled_fields(payload, structural_policy())

        self.assertEqual(result.value, payload)
        self.assertEqual(result.redaction_count, 0)

    def test_residual_detector_redacts_missed_content_and_result_repr_is_safe(
        self,
    ) -> None:
        detector = _ResidualDetector()
        result = redact_text(
            "prefix residual-secret suffix",
            SecretRedactionPolicy(patterns=()),
            residual_detector=detector,
        )

        self.assertEqual(result.value, "prefix [redacted] suffix")
        self.assertEqual(result.residual_redaction_count, 1)
        self.assertTrue(result.residual_assured)
        self.assertNotIn("residual-secret", repr(result))

    def test_residual_failure_is_sanitized_and_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ResidualAssuranceUnavailableError,
            r"failed closed",
        ) as raised:
            redact_text(
                "source-secret",
                SecretRedactionPolicy(patterns=()),
                residual_detector=_BrokenResidualDetector(),
            )
        self.assertNotIn("sensitive diagnostic", str(raised.exception))
        self.assertNotIn("source-secret", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)

    def test_missing_optional_dependency_is_explicit_when_assurance_required(
        self,
    ) -> None:
        missing = ModuleNotFoundError(
            "No module named detect_secrets",
            name="detect_secrets",
        )
        with patch(
            "local_moe.assistant_bridge_secrets.import_module",
            side_effect=missing,
        ):
            with self.assertRaisesRegex(
                ResidualAssuranceUnavailableError,
                r"detect-secrets>=1\.5,<1\.6",
            ):
                redact_text("safe input")

    def test_residual_assurance_can_only_be_disabled_explicitly(self) -> None:
        result = redact_text(
            "API_KEY=local-secret",
            structural_policy(),
        )
        self.assertEqual(result.value, "API_KEY=[redacted]")
        self.assertFalse(result.residual_assured)
        self.assertIsNone(result.residual_detector)

    def test_invalid_policy_and_non_json_user_content_fail_closed(self) -> None:
        with self.assertRaises(SecretRedactionError):
            SecretRedactionPolicy(user_controlled_fields=frozenset({"Objective"}))
        with self.assertRaises(SecretRedactionError):
            TextSecretPattern(name="missing-group", regex=re.compile("secret"))
        with self.assertRaisesRegex(SecretRedactionError, "JSON-compatible"):
            redact_user_controlled_fields(
                {"objective": object()},
                structural_policy(),
            )

    def test_adapter_rejects_an_incompatible_runtime_without_exposing_version(
        self,
    ) -> None:
        modules = {
            "detect_secrets.__version__": SimpleNamespace(VERSION="2.0-secret-build"),
            "detect_secrets.core.scan": SimpleNamespace(scan_line=lambda value: ()),
            "detect_secrets.settings": SimpleNamespace(
                transient_settings=lambda value: value
            ),
        }
        with patch(
            "local_moe.assistant_bridge_secrets.import_module",
            side_effect=lambda name: modules[name],
        ):
            with self.assertRaisesRegex(
                ResidualAssuranceUnavailableError,
                r"compatible detect-secrets 1\.5\.x",
            ) as raised:
                DetectSecretsResidualDetector()
        self.assertNotIn("2.0-secret-build", str(raised.exception))


@unittest.skipUnless(HAS_DETECT_SECRETS, "detect-secrets 1.5 optional dependency")
class DetectSecretsResidualIntegrationTests(unittest.TestCase):
    def test_inline_allowlist_is_ignored(self) -> None:
        secret = "not-a-placeholder-value-123456"
        result = redact_text(
            f'password = "{secret}"  # pragma: allowlist secret',
            SecretRedactionPolicy(
                patterns=(),
                residual_plugins=("KeywordDetector",),
            ),
        )

        self.assertNotIn(secret, str(result.value))
        self.assertEqual(result.residual_redaction_count, 1)

    def test_provider_verification_never_calls_the_network(self) -> None:
        slack = "xoxb" + "-123456789012-123456789012-abcdefghijklmnopqrstuvwx"
        with patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("network verification was attempted"),
        ) as request:
            result = redact_text(
                slack,
                SecretRedactionPolicy(
                    patterns=(),
                    residual_plugins=("SlackDetector",),
                ),
            )

        request.assert_not_called()
        self.assertEqual(result.value, REDACTED_VALUE)

    def test_generated_fingerprint_is_shielded_from_entropy_detector(self) -> None:
        fingerprint = "c4a11f7e9d356a208b4ef9072da9513f68cba501de94726f9350ac62d81e47bf"
        residual_hex = (
            "f037bc9a61ed54208d37a1cf5e9462b897a3cd501f286eb49d30c7fa81e6542b"
        )
        result = redact_text(
            f'task_fingerprint="{fingerprint}" opaque="{residual_hex}"',
            SecretRedactionPolicy(
                patterns=(),
                residual_plugins=("HexHighEntropyString",),
            ),
        )

        self.assertIn(fingerprint, str(result.value))
        self.assertNotIn(residual_hex, str(result.value))


if __name__ == "__main__":
    unittest.main()
