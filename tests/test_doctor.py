from __future__ import annotations

import unittest

from local_moe.app_config import load_app_config
from local_moe.config import load_config
from local_moe.doctor import build_doctor_report
from local_moe.extensions import load_extension_registry


class DoctorTests(unittest.TestCase):
    def test_builds_ready_report_for_synthetic_profile(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = load_config("tests/fixtures/moe.synthetic.json")
        registry = load_extension_registry(
            plugins_dir=app_config.extensions.plugins_dir,
            skills_dir=app_config.extensions.skills_dir,
            tools_config=app_config.extensions.tools_config,
            mcp_config=app_config.extensions.mcp_config,
            cron_config=app_config.extensions.cron_config,
        )

        report = build_doctor_report(
            config_path="tests/fixtures/moe.synthetic.json",
            config=config,
            app_config=app_config,
            registry=registry,
        )

        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertTrue(report["recommendations"])
        self.assertEqual(checks["setup"]["status"], "pass")
        self.assertEqual(checks["health"]["status"], "pass")
        self.assertEqual(checks["extensions"]["status"], "pass")
        self.assertIn("extension_audit", report)
        self.assertTrue(report["extensions"]["tools"])


if __name__ == "__main__":
    unittest.main()
