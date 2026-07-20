from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.config import load_config
from local_moe.context_policy import load_context_policy
from local_moe.package_defaults import (
    packaged_default_path,
    resolve_app_config_path,
    resolve_app_config_reference,
)


class PackageDefaultsTests(unittest.TestCase):
    def test_packaged_defaults_form_a_standalone_valid_configuration(self) -> None:
        app_path = packaged_default_path("app.json")
        app_config = load_app_config(app_path)
        moe_path = resolve_app_config_reference(
            app_config.default_moe_config,
            app_path,
        )
        context_path = resolve_app_config_reference(
            app_config.runtime.context_policy_config,
            app_path,
        )

        config = load_config(moe_path)
        context = load_context_policy(
            context_path,
            app_config.runtime.context_policy_profile,
        )

        self.assertEqual(config.experts[0].id, "local")
        self.assertEqual(context.context_limit_tokens, 16384)

    def test_explicit_app_config_is_never_replaced_by_a_packaged_default(self) -> None:
        requested = Path("missing") / "custom-app.json"

        self.assertEqual(resolve_app_config_path(requested), requested)

    def test_empty_working_directory_uses_packaged_app_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_app_config_path(working_directory=tmp)

        self.assertEqual(resolved, packaged_default_path("app.json"))

    def test_packaged_references_cannot_escape_the_defaults_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "stay inside defaults"):
            resolve_app_config_reference(
                "../outside.json",
                packaged_default_path("app.json"),
            )
