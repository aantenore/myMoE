from __future__ import annotations

import unittest

from local_moe.config import ConfigError, parse_config


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


if __name__ == "__main__":
    unittest.main()
