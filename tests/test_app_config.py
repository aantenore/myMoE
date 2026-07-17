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


if __name__ == "__main__":
    unittest.main()
