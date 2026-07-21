from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_moe.app_config import load_app_config
from local_moe.browser_capability import BrowserCapabilityConfig
from local_moe.browser_setup import (
    _read_packaged_json,
    _write_exclusive_text,
    materialize_browser_workspace,
)
from local_moe.extensions import load_mcp_servers


ROOT = Path(__file__).resolve().parents[1]


class BrowserSetupTests(unittest.TestCase):
    def test_materialized_workspace_is_self_contained_and_fail_closed_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "Space & Amp" / "browser workspace"
            result = materialize_browser_workspace(destination)
            app = load_app_config(result["app_config"])
            servers = load_mcp_servers(result["mcp_config"])

            self.assertEqual(result["status"], "created")
            self.assertTrue(app.permissions.allow_process_execution)
            self.assertEqual(len(servers), 1)
            BrowserCapabilityConfig.from_server(servers[0])
            self.assertTrue(Path(app.default_moe_config).is_absolute())
            self.assertTrue(Path(app.extensions.mcp_config).is_absolute())
            self.assertEqual(
                result["next"]["offline_canary_argv"],
                [
                    "mymoe",
                    "--app-config",
                    str(destination.resolve() / "app.browser.json"),
                    "--browser-canary",
                    "browser-local",
                    "--browser-canary-confirm",
                ],
            )
            prefetch = result["next"]["prefetch_provider_while_online_argv"]
            self.assertEqual(prefetch[0:2], ["mymoe", "browser-prefetch"])
            self.assertEqual(result["provider"], {"package": "@playwright/mcp", "version": "0.0.78"})

            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in destination.iterdir()
                if path.is_file()
            }
            with self.assertRaises(FileExistsError):
                materialize_browser_workspace(destination)
            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in destination.iterdir()
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_packaged_mcp_template_matches_source_checkout_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = materialize_browser_workspace(Path(tmp) / "browser")
            materialized = json.loads(Path(result["mcp_config"]).read_text(encoding="utf-8"))
            checkout = json.loads(
                (ROOT / "configs" / "mcp.playwright-browser.example.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(materialized, checkout)

    def test_browser_init_cli_runs_without_an_existing_app_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "from-cli"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "browser-init",
                    "--out",
                    str(destination),
                ],
                cwd=tmp,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "created")
            self.assertTrue(Path(payload["app_config"]).is_file())

    def test_exclusive_writer_never_follows_an_existing_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside.json"
            target.write_text("unchanged\n", encoding="utf-8")
            link = root / "app.browser.json"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")

            with self.assertRaises(FileExistsError):
                _write_exclusive_text(link, "replacement\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged\n")

    def test_invalid_packaged_metadata_writes_nothing(self) -> None:
        original = _read_packaged_json

        def invalid_mcp(relative_path: str):
            if relative_path.endswith("mcp.playwright-browser.example.json"):
                return {"servers": []}
            return original(relative_path)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "not-created"
            with (
                patch(
                    "local_moe.browser_setup._read_packaged_json",
                    side_effect=invalid_mcp,
                ),
                self.assertRaisesRegex(ValueError, "exactly one server"),
            ):
                materialize_browser_workspace(destination)
            self.assertFalse(destination.exists())

    def test_follow_up_commands_derive_the_server_name_from_the_template(self) -> None:
        original = _read_packaged_json

        def renamed_mcp(relative_path: str):
            payload = original(relative_path)
            if relative_path.endswith("mcp.playwright-browser.example.json"):
                payload["servers"][0]["name"] = "browser-renamed"
            return payload

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "local_moe.browser_setup._read_packaged_json",
                side_effect=renamed_mcp,
            ):
                result = materialize_browser_workspace(Path(tmp) / "browser")
            self.assertEqual(
                result["next"]["offline_canary_argv"][-2], "browser-renamed"
            )
            self.assertEqual(
                result["next"]["prefetch_provider_while_online_argv"][-1],
                "browser-renamed",
            )


if __name__ == "__main__":
    unittest.main()
