from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import (
    MAX_APP_CONFIG_BYTES,
    app_config_payload,
    load_app_config,
)


class AppConfigTests(unittest.TestCase):
    def test_rejects_duplicate_keys_and_oversized_config_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate_root = root / "duplicate-root.json"
            duplicate_root.write_text(
                '{"name":"decoy","name":"myMoE"}', encoding="utf-8"
            )
            duplicate_nested = root / "duplicate-nested.json"
            duplicate_nested.write_text(
                '{"advisor":{"enabled":false,"enabled":true}}',
                encoding="utf-8",
            )
            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(MAX_APP_CONFIG_BYTES + 1)

            for path in (duplicate_root, duplicate_nested, oversized):
                with self.subTest(path=path.name), self.assertRaises(ValueError):
                    load_app_config(path)

    def test_loads_response_language_policy_and_serializes_same_contract(self) -> None:
        config = load_app_config("configs/app.json")

        self.assertTrue(config.language.respond_in_user_language)
        self.assertEqual(config.runtime.profile_dir, "configs")
        self.assertEqual(config.runtime.evaluation_dir, "experiments")
        self.assertTrue(
            app_config_payload(config)["language"]["respond_in_user_language"]
        )
        self.assertEqual(
            app_config_payload(config)["gateway"],
            {
                "enabled": True,
                "model_alias": "mymoe",
                "max_request_bytes": 8 * 1024 * 1024,
                "max_response_bytes": 32 * 1024 * 1024,
                "allow_non_loopback": False,
                "api_key_env": "",
            },
        )

    def test_gateway_defaults_preserve_older_app_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text("{}", encoding="utf-8")

            gateway = load_app_config(path).gateway

        self.assertTrue(gateway.enabled)
        self.assertEqual(gateway.model_alias, "mymoe")
        self.assertEqual(gateway.max_request_bytes, 8 * 1024 * 1024)
        self.assertEqual(gateway.max_response_bytes, 32 * 1024 * 1024)
        self.assertFalse(gateway.allow_non_loopback)
        self.assertEqual(gateway.api_key_env, "")

    def test_advisor_defaults_to_disabled_for_older_app_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text("{}", encoding="utf-8")

            config = load_app_config(path)

        self.assertFalse(config.advisor.enabled)
        self.assertEqual(config.advisor.catalog_path, "")
        self.assertEqual(config.advisor.evaluation_contract_path, "")
        self.assertEqual(config.advisor.allowed_profiles, ("balanced",))
        public = app_config_payload(config)["advisor"]
        self.assertNotIn("catalog_path", public)
        self.assertNotIn("evaluation_contract_path", public)

    def test_loads_enabled_advisor_policy_without_exposing_source_paths(self) -> None:
        config = load_app_config("configs/app.json")

        self.assertTrue(config.advisor.enabled)
        self.assertEqual(config.advisor.workload_id, "local-summary")
        self.assertEqual(config.advisor.capabilities, ("summarization",))
        self.assertEqual(config.advisor.tool_surfaces, ())
        self.assertEqual(config.advisor.risk_class, "compute_only")
        self.assertEqual(config.advisor.context_tokens, 4096)
        public = app_config_payload(config)["advisor"]
        self.assertEqual(public["default_profile"], "balanced")
        self.assertEqual(public["workload"]["id"], "local-summary")
        self.assertNotIn("catalog_path", public)
        self.assertNotIn("evaluation_contract_path", public)

    def test_enabled_advisor_requires_both_source_paths(self) -> None:
        for missing in ("catalog_path", "evaluation_contract_path"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                advisor = {
                    "enabled": True,
                    "catalog_path": "catalog.json",
                    "evaluation_contract_path": "evaluation.json",
                }
                del advisor[missing]
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps({"advisor": advisor}), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, f"advisor.{missing}"):
                    load_app_config(path)

    def test_advisor_rejects_invalid_profile_and_workload_shapes(self) -> None:
        valid = {
            "enabled": True,
            "catalog_path": "catalog.json",
            "evaluation_contract_path": "evaluation.json",
            "allowed_profiles": ["balanced"],
            "default_profile": "balanced",
            "workload_id": "local-summary",
            "capabilities": ["summarization"],
            "tool_surfaces": [],
            "risk_class": "compute_only",
            "context_tokens": 4096,
            "max_request_bytes": 65536,
            "max_task_chars": 16384,
        }
        cases = (
            ("allowed_profiles", [], "allowed_profiles"),
            ("allowed_profiles", ["balanced", "balanced"], "allowed_profiles"),
            ("allowed_profiles", "balanced", "allowed_profiles"),
            ("default_profile", "quality", "default_profile"),
            ("workload_id", 7, "workload_id"),
            ("capabilities", [], "capabilities"),
            ("capabilities", ["summarization", 7], "capabilities"),
            ("tool_surfaces", "none", "tool_surfaces"),
            ("risk_class", "", "risk_class"),
        )
        for field, value, message in cases:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmp,
            ):
                advisor = dict(valid)
                advisor[field] = value
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps({"advisor": advisor}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_app_config(path)

    def test_advisor_rejects_coerced_booleans_and_out_of_range_limits(self) -> None:
        valid = {
            "enabled": True,
            "catalog_path": "catalog.json",
            "evaluation_contract_path": "evaluation.json",
        }
        cases = (
            ("enabled", "true"),
            ("catalog_path", 7),
            ("catalog_path", " catalog.json"),
            ("evaluation_contract_path", "evaluation.json\n"),
            ("context_tokens", True),
            ("context_tokens", 0),
            ("context_tokens", 1_048_577),
            ("max_request_bytes", 0),
            ("max_request_bytes", 262_145),
            ("max_task_chars", 0),
            ("max_task_chars", 131_073),
        )
        for field, value in cases:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmp,
            ):
                advisor = dict(valid)
                advisor[field] = value
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps({"advisor": advisor}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"advisor.{field}"):
                    load_app_config(path)

    def test_gateway_accepts_explicit_loopback_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text(
                json.dumps(
                    {
                        "gateway": {
                            "enabled": False,
                            "model_alias": "editor-local",
                            "max_request_bytes": 1024,
                            "max_response_bytes": 2048,
                            "allow_non_loopback": False,
                            "api_key_env": "LOCAL_GATEWAY_KEY",
                        }
                    }
                ),
                encoding="utf-8",
            )

            gateway = load_app_config(path).gateway

        self.assertFalse(gateway.enabled)
        self.assertEqual(gateway.model_alias, "editor-local")
        self.assertEqual(gateway.max_request_bytes, 1024)
        self.assertEqual(gateway.max_response_bytes, 2048)
        self.assertFalse(gateway.allow_non_loopback)
        self.assertEqual(gateway.api_key_env, "LOCAL_GATEWAY_KEY")

    def test_preserves_disabled_response_language_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text(
                json.dumps(
                    {
                        "language": {
                            "mode": "auto",
                            "respond_in_user_language": False,
                            "supported": ["auto", "en"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_app_config(path)

        self.assertFalse(config.language.respond_in_user_language)

    def test_rejects_unknown_language_policy_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text(
                json.dumps({"language": {"unexpected_option": True}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unknown app config keys in 'language': unexpected_option",
            ):
                load_app_config(path)

    def test_rejects_non_object_language_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text(
                json.dumps({"language": ["auto"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "App config section 'language' must be an object",
            ):
                load_app_config(path)

    def test_rejects_unknown_keys_in_every_app_config_layer(self) -> None:
        base = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
        cases = (
            (base | {"unexpected": True}, "root"),
            (base | {"runtime": {**base["runtime"], "unexpected": True}}, "runtime"),
            (
                base | {"extensions": {**base["extensions"], "unexpected": True}},
                "extensions",
            ),
            (
                base | {"permissions": {**base["permissions"], "unexpected": True}},
                "permissions",
            ),
            (
                base | {"gateway": {**base["gateway"], "unexpected": True}},
                "gateway",
            ),
            (
                base | {"advisor": {**base["advisor"], "unexpected": True}},
                "advisor",
            ),
        )
        for raw, section in cases:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    load_app_config(path)

    def test_bridge_policy_distinguishes_local_only_from_hybrid(self) -> None:
        raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
        for policy in ("disabled", "local_only", "hybrid_receipt_confirmation"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                raw["permissions"]["assistant_bridge_execution_policy"] = policy
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                config = load_app_config(path)
                self.assertEqual(
                    config.permissions.assistant_bridge_execution_policy,
                    policy,
                )

    def test_boolean_policy_fields_are_not_coerced_from_strings(self) -> None:
        raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
        raw["permissions"]["allow_process_execution"] = "false"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow_process_execution"):
                load_app_config(path)

    def test_gateway_boolean_fields_are_not_coerced_from_strings(self) -> None:
        base = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
        for field in ("enabled", "allow_non_loopback"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                raw = json.loads(json.dumps(base))
                raw["gateway"][field] = "false"
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, f"gateway.{field}"):
                    load_app_config(path)

    def test_gateway_rejects_invalid_aliases_and_size_limits(self) -> None:
        base = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
        cases = (
            ("model_alias", "", "gateway.model_alias"),
            ("model_alias", "two words", "gateway.model_alias"),
            ("model_alias", "x" * 81, "gateway.model_alias"),
            ("max_request_bytes", 0, "gateway.max_request_bytes"),
            ("max_request_bytes", True, "gateway.max_request_bytes"),
            (
                "max_request_bytes",
                64 * 1024 * 1024 + 1,
                "gateway.max_request_bytes",
            ),
            ("max_response_bytes", 0, "gateway.max_response_bytes"),
            (
                "max_response_bytes",
                256 * 1024 * 1024 + 1,
                "gateway.max_response_bytes",
            ),
        )
        for field, value, message in cases:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmp,
            ):
                raw = json.loads(json.dumps(base))
                raw["gateway"][field] = value
                path = Path(tmp) / "app.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_app_config(path)

    def test_non_loopback_gateway_requires_api_key_environment_variable(
        self,
    ) -> None:
        raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
        raw["gateway"]["allow_non_loopback"] = True
        raw["gateway"]["api_key_env"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gateway.api_key_env"):
                load_app_config(path)


if __name__ == "__main__":
    unittest.main()
