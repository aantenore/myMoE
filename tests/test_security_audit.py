from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from local_moe.app_config import load_app_config
from local_moe.config import load_config, parse_config
from local_moe.extensions import load_extension_registry
from local_moe.security_audit import build_security_audit_report, render_security_audit_markdown


class SecurityAuditTests(unittest.TestCase):
    def test_default_profile_reports_ready_security_posture(self) -> None:
        app_config = load_app_config("configs/app.json")
        config = load_config("tests/fixtures/moe.synthetic.json")

        report = build_security_audit_report(
            config_path="tests/fixtures/moe.synthetic.json",
            config=config,
            app_config=app_config,
        )

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["mcp"]["env_var_count"], 0)
        self.assertEqual(report["model_endpoints"]["remote_count"], 0)
        self.assertIn("environment variable names and values", " ".join(report["privacy"]["excludes"]))

    def test_warns_without_leaking_mcp_env_names_or_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = root / "app.json"
            mcp_path = root / "mcp.json"
            cron_path = root / "cron.json"
            tools_path = root / "tools.json"
            skills_dir = root / "skills"
            plugins_dir = root / "plugins"
            skills_dir.mkdir()
            plugins_dir.mkdir()
            raw_app = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
            raw_app["runtime"]["work_dir"] = str(root / "runtime")
            raw_app["runtime"]["cron_confirm_writes"] = True
            raw_app["extensions"]["mcp_config"] = str(mcp_path)
            raw_app["extensions"]["cron_config"] = str(cron_path)
            raw_app["extensions"]["tools_config"] = str(tools_path)
            raw_app["extensions"]["skills_dir"] = str(skills_dir)
            raw_app["extensions"]["plugins_dir"] = str(plugins_dir)
            raw_app["permissions"]["allow_process_execution"] = True
            app_config_path.write_text(json.dumps(raw_app), encoding="utf-8")
            mcp_path.write_text(
                json.dumps(
                    {
                        "servers": [
                            {
                                "name": "secure",
                                "description": "Secret-backed MCP server.",
                                "command": "python",
                                "enabled": True,
                                "risk_class": "process_execution",
                                "capabilities": ["tools"],
                                "env": {"MCP_SECRET_TOKEN": "super-secret"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cron_path.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "id": "write-job",
                                "description": "Write local state.",
                                "enabled": True,
                                "schedule": {"type": "interval", "seconds": 3600},
                                "command": ["memory.prune_expired"],
                                "risk_class": "write_local",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tools_path.write_text('{"tools":[]}', encoding="utf-8")
            app_config = load_app_config(app_config_path)
            registry = load_extension_registry(
                plugins_dir=plugins_dir,
                skills_dir=skills_dir,
                tools_config=tools_path,
                mcp_config=mcp_path,
                cron_config=cron_path,
            )
            config = parse_config(
                {
                    "routing": {"top_k": 1},
                    "experts": [
                        {
                            "id": "remote",
                            "provider": "openai_compatible",
                            "model": "remote-model",
                            "role": "general",
                            "base_url": "https://token@example.com/v1",
                        }
                    ],
                    "rules": [{"expert_id": "remote", "keywords": ["test"], "weight": 1.0}],
                }
            )

            report = build_security_audit_report(
                config_path="inline",
                config=config,
                app_config=app_config,
                app_config_path=str(app_config_path),
                registry=registry,
            )
            markdown = render_security_audit_markdown(report)

        serialized = json.dumps(report)
        self.assertEqual(report["status"], "attention")
        self.assertGreaterEqual(report["summary"]["warnings"], 1)
        self.assertEqual(report["mcp"]["env_var_count"], 1)
        self.assertEqual(report["mcp"]["servers"][0]["env_count"], 1)
        self.assertEqual(report["model_endpoints"]["remote_count"], 1)
        self.assertNotIn("MCP_SECRET_TOKEN", serialized)
        self.assertNotIn("super-secret", serialized)
        self.assertNotIn("token@example.com", serialized)
        self.assertIn("# myMoE Security Audit", markdown)
        self.assertNotIn("super-secret", markdown)


if __name__ == "__main__":
    unittest.main()
