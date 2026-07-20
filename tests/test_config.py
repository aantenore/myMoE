from __future__ import annotations

from copy import deepcopy
import unittest

from local_moe.config import ConfigError, parse_config, runtime_config_sha256


def _base_config() -> dict[str, object]:
    return {
        "routing": {
            "top_k": 1,
            "fallback_order": ["general"],
            "aggregation": "best",
        },
        "experts": [
            {
                "id": "coder",
                "provider": "synthetic",
                "model": "synthetic-coder",
                "role": "coding",
            },
            {
                "id": "general",
                "provider": "synthetic",
                "model": "synthetic-general",
                "role": "general",
            },
        ],
        "rules": [
            {
                "expert_id": "coder",
                "keywords": ["python"],
                "weight": 3.0,
            }
        ],
    }


class ConfigTests(unittest.TestCase):
    def test_runtime_config_digest_covers_timeout_and_all_params(self) -> None:
        raw = _base_config()
        raw["experts"][0]["timeout_seconds"] = 45.0  # type: ignore[index]
        raw["experts"][0]["params"] = {  # type: ignore[index]
            "temperature": 0.2,
            "nested": {"alpha": 1, "beta": ["x", "y"]},
        }
        baseline = runtime_config_sha256(parse_config(raw))

        timeout_changed = deepcopy(raw)
        timeout_changed["experts"][0]["timeout_seconds"] = 46.0  # type: ignore[index]
        params_changed = deepcopy(raw)
        params = params_changed["experts"][0]["params"]  # type: ignore[index]
        params["nested"]["beta"] = ["x", "z"]  # type: ignore[index]

        self.assertRegex(baseline, r"\A[0-9a-f]{64}\Z")
        self.assertNotEqual(
            baseline,
            runtime_config_sha256(parse_config(timeout_changed)),
        )
        self.assertNotEqual(
            baseline,
            runtime_config_sha256(parse_config(params_changed)),
        )

    def test_runtime_config_digest_canonicalizes_parameter_key_order(self) -> None:
        first = _base_config()
        first["experts"][0]["params"] = {  # type: ignore[index]
            "z": 1,
            "a": {"y": 2, "b": 3},
        }
        second = deepcopy(first)
        second["experts"][0]["params"] = {  # type: ignore[index]
            "a": {"b": 3, "y": 2},
            "z": 1,
        }

        self.assertEqual(
            runtime_config_sha256(parse_config(first)),
            runtime_config_sha256(parse_config(second)),
        )

    def test_rejects_missing_experts(self) -> None:
        raw = _base_config()
        raw["experts"] = []

        with self.assertRaisesRegex(ConfigError, "At least one expert"):
            parse_config(raw)

    def test_rejects_duplicate_expert_ids(self) -> None:
        raw = _base_config()
        raw["experts"] = [
            {
                "id": "coder",
                "provider": "synthetic",
                "model": "a",
                "role": "coding",
            },
            {
                "id": "coder",
                "provider": "synthetic",
                "model": "b",
                "role": "coding",
            },
        ]

        with self.assertRaisesRegex(ConfigError, "unique"):
            parse_config(raw)

    def test_rejects_expert_ids_that_are_not_safe_header_tokens(self) -> None:
        for expert_id in (
            "coder\r\nX-Injected: true",
            "coder/other",
            " coder",
            "éxpert",
            "x" * 81,
        ):
            with self.subTest(expert_id=expert_id):
                raw = _base_config()
                raw["experts"][0]["id"] = expert_id  # type: ignore[index]
                with self.assertRaisesRegex(ConfigError, "Expert id must"):
                    parse_config(raw)

    def test_rejects_unknown_rule_expert(self) -> None:
        raw = _base_config()
        raw["rules"] = [{"expert_id": "missing", "keywords": ["x"]}]

        with self.assertRaisesRegex(ConfigError, "Rule references unknown expert"):
            parse_config(raw)

    def test_rejects_unknown_fallback_expert(self) -> None:
        raw = _base_config()
        raw["routing"] = {"fallback_order": ["missing"]}

        with self.assertRaisesRegex(ConfigError, "Fallback references unknown expert"):
            parse_config(raw)

    def test_rejects_invalid_top_k(self) -> None:
        raw = _base_config()
        raw["routing"] = {"top_k": 0}

        with self.assertRaisesRegex(ConfigError, "top_k"):
            parse_config(raw)

    def test_rejects_top_k_larger_than_expert_count(self) -> None:
        raw = _base_config()
        raw["routing"] = {"top_k": 3}

        with self.assertRaisesRegex(ConfigError, "number of experts"):
            parse_config(raw)

    def test_rejects_unknown_aggregation(self) -> None:
        raw = _base_config()
        raw["routing"] = {"aggregation": "vote"}

        with self.assertRaisesRegex(ConfigError, "Unsupported aggregation"):
            parse_config(raw)

    def test_rejects_unknown_routing_strategy(self) -> None:
        raw = _base_config()
        raw["routing"] = {"strategy": "llm_judge"}

        with self.assertRaisesRegex(ConfigError, "Unsupported routing strategy"):
            parse_config(raw)

    def test_rejects_unknown_semantic_route_expert(self) -> None:
        raw = _base_config()
        raw["routing"] = {
            "strategy": "hybrid",
            "semantic": {
                "enabled": True,
                "examples": [
                    {
                        "expert_id": "missing",
                        "utterances": ["analyze this"],
                    }
                ],
            },
        }

        with self.assertRaisesRegex(ConfigError, "Semantic route references unknown expert"):
            parse_config(raw)

    def test_rejects_invalid_semantic_examples_shape(self) -> None:
        raw = _base_config()
        raw["routing"] = {
            "strategy": "hybrid",
            "semantic": {
                "enabled": True,
                "examples": "general",
            },
        }

        with self.assertRaisesRegex(ConfigError, "semantic.examples"):
            parse_config(raw)

    def test_parses_hybrid_semantic_routing(self) -> None:
        raw = _base_config()
        raw["routing"] = {
            "strategy": "hybrid",
            "semantic": {
                "enabled": True,
                "examples": [
                    {
                        "expert_id": "general",
                        "utterances": ["riassumi questa nota"],
                        "weight": 1.2,
                    }
                ],
            },
        }

        config = parse_config(raw)

        self.assertEqual(config.routing.strategy, "hybrid")
        self.assertTrue(config.routing.semantic.enabled)
        self.assertEqual(config.routing.semantic.examples[0].expert_id, "general")

    def test_rejects_enabled_distilled_routing_without_artifact_path(self) -> None:
        raw = _base_config()
        raw["routing"] = {
            "strategy": "distilled",
            "distilled": {
                "enabled": True,
            },
        }

        with self.assertRaisesRegex(ConfigError, "artifact_path"):
            parse_config(raw)

    def test_parses_distilled_routing(self) -> None:
        raw = _base_config()
        raw["routing"] = {
            "strategy": "distilled",
            "distilled": {
                "enabled": True,
                "artifact_path": "outputs/router-distilled-extended.json",
                "min_confidence": 0.1,
                "weight": 1.5,
            },
        }

        config = parse_config(raw)

        self.assertEqual(config.routing.strategy, "distilled")
        self.assertTrue(config.routing.distilled.enabled)
        self.assertEqual(config.routing.distilled.artifact_path, "outputs/router-distilled-extended.json")


if __name__ == "__main__":
    unittest.main()
