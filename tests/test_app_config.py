from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import app_config_payload, load_app_config


class AppConfigTests(unittest.TestCase):
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
