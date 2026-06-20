from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.mcp_test_utils import write_fake_mcp_server, write_temp_mcp_app_config


ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


class CliTests(unittest.TestCase):
    def test_eval_mode_prints_router_metrics(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--eval",
                "experiments/eval_set.jsonl",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["accuracy"], 1.0)
        self.assertEqual(payload["total"], 8)

    def test_prompt_mode_runs_synthetic_generation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--prompt",
                "Write Python tests for a class",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("[coding:synthetic-coder]", completed.stdout)
        self.assertIn('"correlation_id"', completed.stdout)
        self.assertIn('"disagreement": null', completed.stdout)

    def test_doctor_prints_runtime_and_extensions(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--doctor",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertIn(payload["status"], {"ready", "attention", "blocked"})
        self.assertIn("checks", payload)
        self.assertIn("recommendations", payload)
        self.assertEqual(payload["app"]["mode"], "local_model_required")
        self.assertIn("runtime", payload)
        self.assertIn("setup", payload)
        self.assertIn("health", payload)
        self.assertIn("hardware_fit", payload)
        self.assertIn("hardware_fit", {item["id"] for item in payload["checks"]})
        self.assertIn("extension_audit", payload)
        self.assertIn("download_command_display", payload["setup"])
        self.assertTrue(payload["extensions"]["tools"])

    def test_setup_prints_model_asset_status(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--setup",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["models"], [])
        self.assertIn("scripts/bootstrap_runtime.py", payload["download_command_display"])

    def test_support_bundle_prints_privacy_safe_payload(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--support-bundle",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["doctor"]["status"], "ready")
        self.assertIn("chat transcripts", payload["privacy"]["excludes"])
        self.assertIn("quality_gate", payload)
        self.assertIn("performance", payload)

    def test_performance_report_prints_runtime_decision(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--performance-report",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertIn(payload["status"], {"ready", "ready_partial"})
        self.assertEqual(payload["decision"]["primary_general"]["candidate_id"], "qwen3-30b-a3b-2507-mlx-4bit")
        self.assertNotIn("content_excerpt", completed.stdout)

    def test_performance_report_can_render_markdown(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--performance-report",
                "--performance-report-format",
                "markdown",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("# myMoE Performance Report", completed.stdout)
        self.assertIn("Primary general expert", completed.stdout)

    def test_prepare_runtime_preview_prints_setup_run(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--prepare-runtime",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "planned")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["setup_before"]["status"], "ready")
        self.assertEqual(payload["steps"], [])

    def test_models_status_prints_managed_process_contract(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--models-status",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["servers"], [])

    def test_models_logs_prints_sanitized_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_temp_openai_config(root)
            app_config_path = _write_temp_app_config(root, config_path)
            log_path = root / "runtime" / "model-1.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "starting\napi_key=sk-abcdefghijklmnop\nready\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--config",
                    str(config_path),
                    "--app-config",
                    str(app_config_path),
                    "--models-logs",
                    "--models-log-lines",
                    "2",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["logs"][0]["line_count"], 2)
        self.assertIn("[REDACTED_SECRET]", "\n".join(payload["logs"][0]["lines"]))
        self.assertNotIn("sk-abcdefghijklmnop", completed.stdout)

    def test_cron_status_prints_jobs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--cron-status",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertIn("jobs", payload)
        self.assertIn("memory-maintenance", {item["id"] for item in payload["jobs"]})

    def test_run_cron_dry_run_prints_due_jobs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--run-cron",
                "--cron-dry-run",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertIn("results", payload)
        self.assertTrue(all(item["status"] == "dry_run" for item in payload["results"]))

    def test_run_tool_prints_tool_result(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--run-tool",
                "mcp.search_capabilities",
                "--tool-input",
                '{"query":"filesystem"}',
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["payload"]["servers"][0]["name"], "filesystem")

    def test_run_tool_lists_enabled_mcp_server_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_script = write_fake_mcp_server(root / "fake_mcp.py")
            app_config = write_temp_mcp_app_config(root, server_script)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--run-tool",
                    "mcp.list_tools",
                    "--tool-input",
                    '{"server":"fake","confirm_process_execution":true,"timeout_seconds":3}',
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["payload"]["server"], "fake")
        self.assertEqual(payload["payload"]["tools"][0]["name"], "echo")

    def test_run_tool_calls_enabled_mcp_server_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_script = write_fake_mcp_server(root / "fake_mcp.py")
            app_config = write_temp_mcp_app_config(root, server_script)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--run-tool",
                    "mcp.call_tool",
                    "--tool-input",
                    (
                        '{"server":"fake","tool_name":"echo","arguments":{"text":"hi"},'
                        '"confirm_process_execution":true,"confirm_tool_call":true,"timeout_seconds":3}'
                    ),
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["payload"]["tool_name"], "echo")
        self.assertEqual(payload["payload"]["content"][0]["text"], "echo:hi")


def _write_temp_openai_config(root: Path) -> Path:
    path = root / "moe.openai.json"
    path.write_text(
        json.dumps(
            {
                "routing": {"top_k": 1, "fallback_order": ["general"], "aggregation": "best"},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "base_url": "http://127.0.0.1:9999/v1",
                        "model": "local/model",
                        "role": "general",
                        "params": {"runtime_backend": "mlx_lm"},
                    }
                ],
                "rules": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_temp_app_config(root: Path, config_path: Path) -> Path:
    raw = json.loads((ROOT / "configs" / "app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = str(config_path)
    raw["runtime"]["work_dir"] = str(root / "runtime")
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
