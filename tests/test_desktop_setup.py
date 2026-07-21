from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from local_moe.app_config import load_app_config
from local_moe.desktop_capability import DesktopCapabilityConfig
from local_moe.desktop_setup import materialize_desktop_workspace
from local_moe.desktop_setup import _read_packaged_json
from local_moe.extensions import load_mcp_servers


class DesktopSetupTests(unittest.TestCase):
    def test_materializes_bound_installable_workspace_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "cua-driver"
            binary.write_bytes(b"pinned-provider")
            binary.chmod(0o700)
            destination = root / "Space & Amp" / "desktop workspace"
            with (
                patch(
                    "local_moe.desktop_setup._provider_binary",
                    return_value=binary,
                ),
                patch(
                    "local_moe.desktop_setup._provider_version",
                    return_value="0.10.0",
                ),
                patch("local_moe.desktop_setup._disable_provider_telemetry"),
                patch(
                    "local_moe.desktop_setup._resolve_process_identity",
                    return_value={
                        "pid": 4242,
                        "name": "Offline Editor",
                        "started_at": "1753084800.000000",
                        "executable_sha256": "b" * 64,
                    },
                ),
            ):
                result = materialize_desktop_workspace(
                    destination,
                    target_id="offline-editor",
                    target_pid=4242,
                    window_id=17,
                )

            app = load_app_config(result["app_config"])
            servers = load_mcp_servers(result["mcp_config"])
            config = DesktopCapabilityConfig.from_server(servers[0])
            expected_provider_sha256 = hashlib.sha256(b"pinned-provider").hexdigest()

            self.assertEqual(result["status"], "created")
            self.assertEqual(config.target_id, "offline-editor")
            self.assertEqual(config.pid, 4242)
            self.assertEqual(config.window_id, 17)
            self.assertEqual(
                config.provider_executable_sha256,
                expected_provider_sha256,
            )
            self.assertEqual(config.process_executable_sha256, "b" * 64)
            self.assertTrue(Path(app.default_moe_config).is_absolute())
            self.assertTrue(Path(app.extensions.mcp_config).is_absolute())
            self.assertEqual(
                result["next"]["offline_canary_argv"][-2:],
                ["desktop-local", "--desktop-canary-confirm"],
            )
            self.assertNotIn("pid", result["target"])
            self.assertNotIn("window_id", result["target"])
            if os.name == "posix":
                directory_modes = [
                    stat.S_IMODE(path.stat().st_mode)
                    for path in (destination, destination / "plugins", destination / "skills")
                ]
                self.assertTrue(all(mode == 0o700 for mode in directory_modes))
                modes = [
                    stat.S_IMODE(path.stat().st_mode)
                    for path in destination.iterdir()
                    if path.is_file()
                ]
                self.assertTrue(modes)
                self.assertTrue(all(mode == 0o600 for mode in modes))

            before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in destination.iterdir()
                if path.is_file()
            }
            with (
                patch(
                    "local_moe.desktop_setup._provider_binary",
                    return_value=binary,
                ),
                patch(
                    "local_moe.desktop_setup._provider_version",
                    return_value="0.10.0",
                ),
                patch(
                    "local_moe.desktop_setup._disable_provider_telemetry"
                ) as repeat_disable_telemetry,
                patch(
                    "local_moe.desktop_setup._resolve_process_identity",
                    return_value={
                        "pid": 4242,
                        "name": "Offline Editor",
                        "started_at": "1753084800.000000",
                        "executable_sha256": "b" * 64,
                    },
                ),
                self.assertRaises(FileExistsError),
            ):
                materialize_desktop_workspace(
                    destination,
                    target_id="offline-editor",
                    target_pid=4242,
                    window_id=17,
                )
            repeat_disable_telemetry.assert_not_called()
            after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in destination.iterdir()
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_source_examples_are_parseable_but_explicitly_unbound(self) -> None:
        app = load_app_config("configs/app.desktop.example.json")
        servers = load_mcp_servers(app.extensions.mcp_config)
        config = DesktopCapabilityConfig.from_server(servers[0])
        source = json.loads(
            Path("configs/mcp.cua-desktop.example.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config.provider, "cua_driver")
        self.assertEqual(config.target_id, "replace-target")
        self.assertEqual(
            source["servers"][0]["desktop_capability"][
                "provider_executable_sha256"
            ],
            "0" * 64,
        )
        self.assertEqual(
            _read_packaged_json(
                "templates/desktop/mcp.cua-desktop.example.json"
            ),
            source,
        )
        self.assertEqual(
            _read_packaged_json("templates/desktop/app.desktop.example.json"),
            json.loads(
                Path("configs/app.desktop.example.json").read_text(
                    encoding="utf-8"
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
