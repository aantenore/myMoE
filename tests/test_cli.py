from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from local_moe.agent_tools import ApprovalRequest, arguments_sha256
from local_moe.cli import _agent_approval_handler
from local_moe.run_log import RunLogStore
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

    def test_agent_prompt_runs_openai_compatible_loop_with_metadata_only_trace(self) -> None:
        def respond(_request_payload: dict[str, object]) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>private reasoning</think>Agent result.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 21, "completion_tokens": 4},
            }

        with tempfile.TemporaryDirectory() as tmp, _serve_agent(respond) as server:
            root = Path(tmp)
            config_path = _write_temp_openai_config(
                root,
                base_url=f"http://127.0.0.1:{server.port}/v1",
            )
            app_config = _write_temp_app_config(root, config_path)
            private_prompt = "Private CLI agent prompt"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    str(config_path),
                    "--agent-prompt",
                    private_prompt,
                    "--agent-expert",
                    "general",
                    "--agent-tool",
                    "memory.search",
                    "--agent-max-model-turns",
                    "2",
                    "--agent-max-tool-calls",
                    "1",
                    "--agent-soft-wall-time-seconds",
                    "5",
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["mode"], "agent")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["final_answer"], "Agent result.")
        self.assertEqual(payload["trace_policy"], "metadata_only")
        self.assertNotIn(private_prompt, completed.stdout)
        self.assertNotIn("private reasoning", completed.stdout)
        self.assertEqual(len(server.requests), 1)
        request_payload = server.requests[0]
        self.assertEqual(request_payload["messages"][1]["content"], private_prompt)
        self.assertEqual(
            request_payload["tools"][0]["function"]["name"],
            "memory__search",
        )
        self.assertFalse(request_payload["parallel_tool_calls"])
        for event in payload["trace"]:
            self.assertNotIn("content", event)
            self.assertNotIn("arguments", event)
            self.assertNotIn("result", event)

    def test_agent_prompt_rejects_remote_endpoint_in_local_model_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_temp_openai_config(
                root,
                base_url="https://models.example.com/v1",
            )
            app_config = _write_temp_app_config(root, config_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    str(config_path),
                    "--agent-prompt",
                    "Do not send this private prompt off-device.",
                    "--agent-tool",
                    "memory.search",
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("local_model_required", completed.stderr)
        self.assertIn("blocked expert", completed.stderr)

    def test_deprecated_agent_max_wall_time_alias_warns(self) -> None:
        def respond(_request_payload: dict[str, object]) -> dict[str, object]:
            return {"choices": [{"message": {"content": "Done."}}]}

        with tempfile.TemporaryDirectory() as tmp, _serve_agent(respond) as server:
            root = Path(tmp)
            config_path = _write_temp_openai_config(
                root,
                base_url=f"http://127.0.0.1:{server.port}/v1",
            )
            app_config = _write_temp_app_config(root, config_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    str(config_path),
                    "--agent-prompt",
                    "Return a final answer.",
                    "--agent-tool",
                    "memory.search",
                    "--agent-max-wall-time-seconds",
                    "5",
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        self.assertIn("deprecated", completed.stderr.lower())
        self.assertEqual(json.loads(completed.stdout)["status"], "completed")

    def test_agent_prompt_requires_and_replays_exact_write_approval(self) -> None:
        tool_arguments = {
            "title": "CLI approval note",
            "content": "Stored only after the exact approval is replayed.",
        }

        def respond(request_payload: dict[str, object]) -> dict[str, object]:
            messages = request_payload["messages"]
            if messages[-1]["role"] == "tool":
                tool_result = json.loads(messages[-1]["content"])
                self.assertEqual(tool_result["status"], "success")
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Knowledge stored.",
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-write-1",
                                    "type": "function",
                                    "function": {
                                        "name": "knowledge__ingest",
                                        "arguments": json.dumps(tool_arguments),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp, _serve_agent(respond) as server:
            root = Path(tmp)
            config_path = _write_temp_openai_config(
                root,
                base_url=f"http://127.0.0.1:{server.port}/v1",
            )
            app_config = _write_temp_app_config(root, config_path)
            command = [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--app-config",
                str(app_config),
                "--config",
                str(config_path),
                "--agent-prompt",
                "Store this local note.",
                "--agent-tool",
                "knowledge.ingest",
                "--json",
            ]
            guarded = subprocess.run(
                command,
                cwd=ROOT,
                env=_env(),
                text=True,
                capture_output=True,
            )
            guarded_payload = json.loads(guarded.stdout)
            approval_token = guarded_payload["approval_requests"][0][
                "approval_token"
            ]
            memory_path = root / "runtime" / "memory.jsonl"
            self.assertFalse(memory_path.exists())

            approved = subprocess.run(
                [*command, "--agent-approve", approval_token],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            approved_payload = json.loads(approved.stdout)
            memory_records = memory_path.read_text(encoding="utf-8")

        self.assertEqual(guarded.returncode, 2)
        self.assertEqual(guarded_payload["status"], "approval_required")
        self.assertEqual(
            guarded_payload["approval_requests"][0]["tool_name"],
            "knowledge.ingest",
        )
        self.assertNotIn("confirm", guarded_payload["approval_requests"][0]["arguments"])
        self.assertEqual(approved_payload["status"], "completed")
        self.assertEqual(approved_payload["tool_results"][0]["status"], "success")
        self.assertTrue(approved_payload["grounded_in_tool_results"])
        self.assertIn("CLI approval note", memory_records)

    def test_agent_prompt_applies_app_process_execution_deny_policy(self) -> None:
        def respond(request_payload: dict[str, object]) -> dict[str, object]:
            messages = request_payload["messages"]
            if messages[-1]["role"] == "tool":
                tool_result = json.loads(messages[-1]["content"])
                self.assertEqual(tool_result["status"], "denied")
                self.assertEqual(tool_result["code"], "permission_denied")
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Process execution is disabled.",
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-process-1",
                                    "type": "function",
                                    "function": {
                                        "name": "mcp__list_tools",
                                        "arguments": '{"server":"filesystem"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp, _serve_agent(respond) as server:
            root = Path(tmp)
            config_path = _write_temp_openai_config(
                root,
                base_url=f"http://127.0.0.1:{server.port}/v1",
            )
            app_config = _write_temp_app_config(root, config_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    str(config_path),
                    "--agent-prompt",
                    "Inspect the filesystem MCP tools.",
                    "--agent-tool",
                    "mcp.list_tools",
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["tool_results"][0]["status"], "denied")
        self.assertFalse(payload["grounded_in_tool_results"])
        self.assertEqual(payload["approval_requests"], [])

    def test_agent_prompt_connector_deny_overrides_exact_approval(self) -> None:
        tool_arguments = {
            "surface": "mcp_server",
            "definition": {"name": "must-not-be-registered"},
        }

        def respond(request_payload: dict[str, object]) -> dict[str, object]:
            messages = request_payload["messages"]
            if messages[-1]["role"] == "tool":
                tool_result = json.loads(messages[-1]["content"])
                self.assertEqual(tool_result["status"], "denied")
                self.assertEqual(tool_result["code"], "permission_denied")
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "Connector configuration is denied.",
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-connector-1",
                                    "type": "function",
                                    "function": {
                                        "name": "extension__configure",
                                        "arguments": json.dumps(tool_arguments),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as tmp, _serve_agent(respond) as server:
            root = Path(tmp)
            config_path = _write_temp_openai_config(
                root,
                base_url=f"http://127.0.0.1:{server.port}/v1",
            )
            app_config = _write_temp_app_config(
                root,
                config_path,
                permission_overrides={"connector_install_policy": "deny"},
            )
            approval = (
                "extension.configure:"
                f"{arguments_sha256(tool_arguments)}"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    str(config_path),
                    "--agent-prompt",
                    "Register this connector.",
                    "--agent-tool",
                    "extension.configure",
                    "--agent-approve",
                    approval,
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["tool_results"][0]["status"], "denied")
        self.assertEqual(payload["tool_results"][0]["code"], "permission_denied")
        self.assertEqual(payload["approval_requests"], [])

    def test_agent_options_require_explicit_agent_prompt(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--agent-tool",
                "memory.search",
            ],
            cwd=ROOT,
            env=_env(),
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--agent-* options require --agent-prompt", completed.stderr)

    def test_agent_prompt_requires_explicit_tool_selection(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--agent-prompt",
                "Answer without tools.",
            ],
            cwd=ROOT,
            env=_env(),
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires at least one explicit --agent-tool", completed.stderr)

    def test_agent_approval_token_is_single_use_within_one_run(self) -> None:
        arguments_sha256 = "a" * 64
        handler = _agent_approval_handler(
            [f"knowledge.ingest:{arguments_sha256}"]
        )
        self.assertIsNotNone(handler)
        request = ApprovalRequest(
            call_id="call-1",
            tool_name="knowledge.ingest",
            arguments={"title": "note", "content": "body"},
            arguments_sha256=arguments_sha256,
            risk_class="write_local",
            side_effects="writes_memory_records",
        )

        self.assertTrue(handler(request).approved)
        self.assertFalse(handler(request).approved)

    def test_prompt_mode_can_persist_to_cli_chat_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            first = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--prompt",
                    "Remember this CLI detail: alpha.",
                    "--new-chat",
                    "--chat-title",
                    "CLI persisted",
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            first_payload = json.loads(first.stdout)
            second = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--prompt",
                    "Use the previous CLI detail.",
                    "--chat-session",
                    first_payload["session_id"],
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            second_payload = json.loads(second.stdout)
            chat_payload = json.loads((root / "runtime" / "chats.json").read_text(encoding="utf-8"))
            run_log_text = (root / "runtime" / "runs.jsonl").read_text(encoding="utf-8")

        self.assertEqual(first_payload["session"]["title"], "CLI persisted")
        self.assertEqual(second_payload["session_id"], first_payload["session_id"])
        self.assertGreater(second_payload["context"]["sections"]["recent_turns"], 0)
        self.assertEqual(len(chat_payload["sessions"]), 1)
        self.assertEqual(len(chat_payload["sessions"][0]["messages"]), 4)
        self.assertIn('"mode": "cli-prompt"', run_log_text)
        self.assertNotIn("Remember this CLI detail", run_log_text)
        self.assertNotIn("Use the previous CLI detail", run_log_text)

    def test_cli_searches_and_compacts_chat_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            created = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--prompt",
                    "Remember this compactable CLI session.",
                    "--new-chat",
                    "--chat-title",
                    "Compact Target",
                    "--json",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            session_id = json.loads(created.stdout)["session_id"]
            search = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--list-chats",
                    "--chat-query",
                    "compact",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            guarded = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--compact-chat",
                    session_id,
                ],
                cwd=ROOT,
                env=_env(),
                text=True,
                capture_output=True,
            )
            compacted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--compact-chat",
                    session_id,
                    "--chat-confirm",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            search_payload = json.loads(search.stdout)
            guarded_payload = json.loads(guarded.stderr)
            compacted_payload = json.loads(compacted.stdout)

        self.assertEqual(search_payload["count"], 1)
        self.assertEqual(search_payload["sessions"][0]["id"], session_id)
        self.assertEqual(guarded.returncode, 2)
        self.assertEqual(guarded_payload["error"], "confirmation_required")
        self.assertIn("[general:synthetic-general]", compacted_payload["summary"])
        self.assertEqual(compacted_payload["compaction"]["expert_id"], "general")
        self.assertTrue(compacted_payload["session"]["summary"])
        self.assertEqual(compacted_payload["session"]["message_count"], 3)

    def test_interactive_cli_uses_persistent_chat_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--interactive",
                    "--new-chat",
                    "--chat-title",
                    "CLI shell",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                input="Hello from the CLI shell.\nContinue from the previous turn.\n/summary\n/exit\n",
                capture_output=True,
            )
            chat_payload = json.loads((root / "runtime" / "chats.json").read_text(encoding="utf-8"))
            run_log_report = RunLogStore(root / "runtime" / "runs.jsonl").read_report(limit=10)

        self.assertIn("myMoE interactive shell", completed.stderr)
        self.assertIn("synthetic-", completed.stdout)
        self.assertEqual(len(chat_payload["sessions"]), 1)
        session = chat_payload["sessions"][0]
        self.assertEqual(session["title"], "CLI shell")
        self.assertEqual(len(session["messages"]), 4)
        self.assertGreater(session["messages"][3]["meta"]["context"]["sections"]["recent_turns"], 0)
        self.assertEqual(len(run_log_report.records), 2)
        self.assertEqual({record.mode for record in run_log_report.records}, {"cli-interactive"})

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

    def test_doctor_prints_markdown_report(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--doctor",
                "--doctor-format",
                "markdown",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("# myMoE System Doctor Report", completed.stdout)
        self.assertIn("## Checks", completed.stdout)
        self.assertIn("`hardware_fit`", completed.stdout)

    def test_about_prints_environment_snapshot(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--about",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["paths"]["moe_config"], "tests/fixtures/moe.synthetic.json")
        self.assertIn("python", payload)
        self.assertIn("packages", payload)
        self.assertEqual(payload["runtime"]["expert_count"], 3)

    def test_about_prints_markdown_snapshot(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--about",
                "--about-format",
                "markdown",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("# myMoE Environment Snapshot", completed.stdout)
        self.assertIn("## Experts", completed.stdout)
        self.assertIn("`synthetic-general`", completed.stdout)

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

    def test_recommend_profile_prints_local_decision(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--recommend-profile",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertGreaterEqual(payload["profiles_considered"], 1)
        self.assertIn(payload["recommendation"]["status"], {"ready", "needs_setup", "unavailable"})
        self.assertIn("profile_path", payload["recommendation"])

    def test_activate_profile_requires_confirmation_and_updates_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root, "configs/moe.live.fast-mlx.example.json")
            guarded = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config_path),
                    "--config",
                    "configs/moe.live.fast-mlx.example.json",
                    "--activate-profile",
                    "tests/fixtures/moe.synthetic.json",
                ],
                cwd=ROOT,
                env=_env(),
                text=True,
                capture_output=True,
            )
            confirmed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config_path),
                    "--config",
                    "configs/moe.live.fast-mlx.example.json",
                    "--activate-profile",
                    "tests/fixtures/moe.synthetic.json",
                    "--profile-confirm",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )
            raw = json.loads(app_config_path.read_text(encoding="utf-8"))

        self.assertEqual(guarded.returncode, 2)
        self.assertEqual(json.loads(guarded.stdout)["status"], "confirmation_required")
        payload = json.loads(confirmed.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["activated"])
        self.assertEqual(raw["default_moe_config"], "tests/fixtures/moe.synthetic.json")

    def test_prepare_profile_uses_requested_profile_and_confirmation_guard(self) -> None:
        guarded = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--prepare-profile",
                "tests/fixtures/moe.synthetic.json",
                "--prepare-execute",
                "--prepare-download-models",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )
        confirmed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--prepare-profile",
                "tests/fixtures/moe.synthetic.json",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        guarded_payload = json.loads(guarded.stdout)
        confirmed_payload = json.loads(confirmed.stdout)
        self.assertEqual(guarded_payload["status"], "confirmation_required")
        self.assertEqual(guarded_payload["profile_path"], "tests/fixtures/moe.synthetic.json")
        self.assertEqual(confirmed_payload["status"], "planned")
        self.assertEqual(confirmed_payload["profile_path"], "tests/fixtures/moe.synthetic.json")
        self.assertTrue(confirmed_payload["ok"])

    def test_prepare_profile_options_are_mutually_exclusive(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--prepare-profile",
                "tests/fixtures/moe.synthetic.json",
                "--prepare-recommended-profile",
            ],
            cwd=ROOT,
            env=_env(),
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("not allowed with argument", completed.stderr)

    def test_startup_preview_prints_readiness_runbook(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--startup",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["doctor"]["status"], "ready")
        self.assertEqual(payload["setup"]["status"], "ready")
        self.assertEqual(payload["model_processes"]["count"], 0)

    def test_startup_side_effects_require_confirmation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--startup",
                "--startup-prepare",
                "--startup-download-models",
                "--startup-start-models",
            ],
            cwd=ROOT,
            env=_env(),
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["status"], "confirmation_required")
        self.assertFalse(payload["ok"])

    def test_smoke_generate_prints_runtime_probe(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--smoke-generate",
                "--smoke-prompt",
                "Summarize this in one sentence.",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "pass")
        self.assertGreater(payload["content_chars"], 0)
        self.assertEqual(payload["route"]["selected"][0]["expert_id"], "general")

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
        self.assertIn("environment", payload)
        self.assertEqual(payload["environment"]["paths"]["moe_config"], "tests/fixtures/moe.synthetic.json")
        self.assertIn("security_audit", payload)

    def test_security_audit_prints_metadata_only_payload(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--security-audit",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["model_endpoints"]["remote_count"], 0)
        self.assertIn("environment variable names and values", " ".join(payload["privacy"]["excludes"]))

    def test_security_audit_can_render_markdown(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--security-audit",
                "--security-audit-format",
                "markdown",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("# myMoE Security Audit", completed.stdout)
        self.assertIn("## Checks", completed.stdout)

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
        self.assertEqual(
            payload["decision"]["primary_general"]["candidate_id"],
            "qwen3-4b-mlx-4bit",
        )
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

    def test_runtime_optimizer_prints_read_only_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--runtime-optimizer",
                    "--runtime-optimizer-runs-limit",
                    "5",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["mode"], "read_only")
        self.assertIn(payload["status"], {"ready", "watch", "attention"})
        self.assertEqual(payload["run_log"]["summary"]["record_count"], 0)
        self.assertIn("run_generation_smoke", {action["id"] for action in payload["actions"]})

    def test_runtime_optimizer_can_render_markdown(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_moe.cli",
                "--config",
                "tests/fixtures/moe.synthetic.json",
                "--runtime-optimizer",
                "--runtime-optimizer-format",
                "markdown",
            ],
            cwd=ROOT,
            env=_env(),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("# myMoE Runtime Optimizer Report", completed.stdout)
        self.assertIn("## Actions", completed.stdout)

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

    def test_runs_cli_lists_metadata_only_generation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            store = RunLogStore(root / "runtime" / "runs.jsonl")
            store.record_generation(
                mode="generate",
                prompt="Private CLI run prompt",
                response_payload={
                    "correlation_id": "corr-cli",
                    "route": {"selected": [{"expert_id": "general"}], "fallback_order": []},
                    "results": [{"model": "synthetic-general"}],
                    "errors": [],
                },
            )
            with store.path.open("a", encoding="utf-8") as handle:
                handle.write("{corrupt-cli-record\n")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--runs",
                ],
                cwd=ROOT,
                env=_env(),
                check=True,
                text=True,
                capture_output=True,
            )

        payload = json.loads(completed.stdout)
        rendered = completed.stdout
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["diagnostics"]["status"], "attention")
        self.assertEqual(payload["diagnostics"]["skipped_records"], 1)
        self.assertEqual(payload["summary"]["record_count"], 1)
        self.assertEqual(payload["summary"]["models"], [{"id": "synthetic-general", "count": 1}])
        self.assertEqual(payload["records"][0]["correlation_id"], "corr-cli")
        self.assertNotIn("Private CLI run prompt", rendered)

    def test_runs_cli_prune_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            guarded = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "local_moe.cli",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--runs-prune",
                ],
                cwd=ROOT,
                env=_env(),
                text=True,
                capture_output=True,
            )

        payload = json.loads(guarded.stdout)
        self.assertEqual(guarded.returncode, 2)
        self.assertEqual(payload["error"], "confirmation_required")

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


@dataclass(frozen=True)
class _AgentServer:
    port: int
    requests: list[dict[str, object]]


@contextmanager
def _serve_agent(responder):
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            request_payload = json.loads(self.rfile.read(content_length))
            requests.append(request_payload)
            response_payload = responder(request_payload)
            encoded = json.dumps(response_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _AgentServer(port=int(server.server_port), requests=requests)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _write_temp_openai_config(
    root: Path,
    *,
    base_url: str = "http://127.0.0.1:9999/v1",
) -> Path:
    path = root / "moe.openai.json"
    path.write_text(
        json.dumps(
            {
                "routing": {"top_k": 1, "fallback_order": ["general"], "aggregation": "best"},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "base_url": base_url,
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


def _write_temp_app_config(
    root: Path,
    config_path: Path | str,
    *,
    permission_overrides: dict[str, object] | None = None,
) -> Path:
    raw = json.loads((ROOT / "configs" / "app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = str(config_path)
    raw["runtime"]["work_dir"] = str(root / "runtime")
    raw["permissions"].update(permission_overrides or {})
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
