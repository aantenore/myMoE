from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from urllib.error import HTTPError
from urllib import request
import unittest

from local_moe.web import build_server
from tests.mcp_test_utils import write_fake_mcp_server, write_temp_mcp_app_config


class WebTests(unittest.TestCase):
    def test_serves_config_and_generates_with_synthetic_provider(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            config = _get_json(base_url + "/api/config")
            result = _post_json(
                base_url + "/api/generate",
                {"prompt": "Summarize this note into bullets."},
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(config["routing"]["aggregation"], "best")
        self.assertEqual(config["routing"]["strategy"], "rules")
        self.assertIn("semantic", config["routing"])
        self.assertIn("distilled", config["routing"])
        self.assertIn("[general:synthetic-general]", result["content"])
        self.assertEqual(result["route"]["selected"][0]["expert_id"], "general")
        self.assertIn("context", result)
        self.assertIn("token_estimate", result["context"])

    def test_runs_eval_endpoint(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            result = _post_json(
                base_url + "/api/evaluate",
                {"eval_path": "experiments/eval_set.jsonl"},
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["total"], 8)

    def test_serves_runtime_and_extensions(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            runtime = _get_json(base_url + "/api/runtime")
            setup = _get_json(base_url + "/api/setup")
            health = _get_json(base_url + "/api/health")
            extensions = _get_json(base_url + "/api/extensions")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertIn(runtime["backend"], {"mlx_lm", "ollama", "llama_cpp"})
        self.assertEqual(setup["status"], "ready")
        self.assertEqual(setup["models"], [])
        self.assertIn("download_command_display", setup)
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["experts"][0]["status"], "skipped")
        self.assertTrue(extensions["tools"])

    def test_serves_and_runs_cron_endpoint(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            status = _get_json(base_url + "/api/cron")
            result = _post_json(base_url + "/api/cron/run", {"dry_run": True})
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertIn("jobs", status)
        self.assertIn("auto", status)
        self.assertEqual(status["auto"]["policy"], "safe_jobs_only")
        self.assertFalse(status["auto"]["running"])
        self.assertIn("results", result)
        self.assertTrue(all(item["status"] == "dry_run" for item in result["results"]))

    def test_runs_tool_endpoint(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            result = _post_json(
                base_url + "/api/tools/run",
                {"name": "mcp.search_capabilities", "input": {"query": "filesystem"}},
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"]["servers"][0]["name"], "filesystem")

    def test_runs_mcp_list_tools_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_script = write_fake_mcp_server(root / "fake_mcp.py")
            app_config = write_temp_mcp_app_config(root, server_script)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                result = _post_json(
                    base_url + "/api/tools/run",
                    {
                        "name": "mcp.list_tools",
                        "input": {
                            "server": "fake",
                            "confirm_process_execution": True,
                            "timeout_seconds": 3,
                        },
                    },
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"]["server"], "fake")
        self.assertEqual(result["payload"]["tools"][0]["name"], "echo")

    def test_runs_mcp_call_tool_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_script = write_fake_mcp_server(root / "fake_mcp.py")
            app_config = write_temp_mcp_app_config(root, server_script)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                result = _post_json(
                    base_url + "/api/tools/run",
                    {
                        "name": "mcp.call_tool",
                        "input": {
                            "server": "fake",
                            "tool_name": "echo",
                            "arguments": {"text": "hi"},
                            "confirm_process_execution": True,
                            "confirm_tool_call": True,
                            "timeout_seconds": 3,
                        },
                    },
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"]["tool_name"], "echo")
        self.assertEqual(result["payload"]["content"][0]["text"], "echo:hi")

    def test_persists_chat_sessions_over_web_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                initial = _get_json(base_url + "/api/chats")
                first = _post_json(base_url + "/api/generate", {"prompt": "Summarize this note."})
                session_id = str(first["session_id"])
                listed = _get_json(base_url + "/api/chats")
                loaded = _get_json(base_url + f"/api/chats/{session_id}")
                second = _post_json(
                    base_url + "/api/generate",
                    {"prompt": "Continue the same chat.", "session_id": session_id},
                )
                renamed = _patch_json(
                    base_url + f"/api/chats/{session_id}",
                    {"title": "Local Session Notes"},
                )
                searched = _get_json(base_url + "/api/chats?query=local%20session")
                exported = _get_text(base_url + f"/api/chats/{session_id}/export.md")
                deleted = _delete_json(base_url + f"/api/chats/{session_id}")
                final = _get_json(base_url + "/api/chats")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(initial["count"], 0)
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["sessions"][0]["id"], session_id)
        self.assertEqual(loaded["messages"][0]["role"], "user")
        self.assertEqual(loaded["messages"][1]["role"], "assistant")
        self.assertIn("route", loaded["messages"][1]["meta"])
        self.assertIn("context", loaded["messages"][1]["meta"])
        self.assertEqual(second["session"]["message_count"], 4)
        self.assertIn("recent_turns", second["context"]["sections"])
        self.assertGreater(
            _prompt_chars(second["content"]),
            _prompt_chars(first["content"]) + len("Continue the same chat."),
        )
        self.assertEqual(renamed["title"], "Local Session Notes")
        self.assertEqual(searched["sessions"][0]["id"], session_id)
        self.assertIn("# Local Session Notes", exported)
        self.assertIn("## Assistant", exported)
        self.assertTrue(deleted["deleted"])
        self.assertEqual(final["count"], 0)

    def test_generate_rejects_missing_chat_session_before_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                with self.assertRaises(HTTPError) as raised:
                    _post_json(
                        base_url + "/api/generate",
                        {"prompt": "This should not generate.", "session_id": "missing"},
                    )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(raised.exception.code, 404)

    def test_generate_reports_context_compaction_when_budget_is_tight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "context-policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "default": {
                            "context_limit_tokens": 80,
                            "reserved_output_tokens": 20,
                            "compaction_trigger_ratio": 0.5,
                            "max_recent_turns": 20,
                        }
                    }
                ),
                encoding="utf-8",
            )
            app_config = _write_temp_app_config(root, context_policy_path=policy_path)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                first = _post_json(base_url + "/api/generate", {"prompt": "alpha " * 160})
                second = _post_json(
                    base_url + "/api/generate",
                    {"prompt": "beta " * 160, "session_id": first["session_id"]},
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertTrue(second["context"]["compaction_needed"])
        self.assertGreater(second["context"]["dropped_turns"], 0)

    def test_compacts_chat_session_and_uses_summary_in_next_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                first = _post_json(base_url + "/api/generate", {"prompt": "Discuss context compaction."})
                compacted = _post_json(
                    base_url + f"/api/chats/{first['session_id']}/compact",
                    {},
                )
                loaded = _get_json(base_url + f"/api/chats/{first['session_id']}")
                second = _post_json(
                    base_url + "/api/generate",
                    {"prompt": "Continue with the summary.", "session_id": first["session_id"]},
                )
                exported = _get_text(base_url + f"/api/chats/{first['session_id']}/export.md")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertIn("synthetic-general", compacted["summary"])
        self.assertEqual(compacted["compaction"]["expert_id"], "general")
        self.assertIn("summary_updated_at", compacted)
        self.assertTrue(loaded["summary"])
        self.assertEqual(loaded["messages"][-1]["role"], "system")
        self.assertEqual(loaded["messages"][-1]["meta"]["kind"], "summary_update")
        self.assertIn("summary", second["context"]["sections"])
        self.assertIn("## Summary", exported)

    def test_memory_api_and_context_retrieval_do_not_distort_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            server = build_server(
                "tests/fixtures/moe.synthetic.json",
                port=0,
                app_config_path=str(app_config),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                memory = _post_json(
                    base_url + "/api/memory",
                    {
                        "text": "Antonio preference: Python code examples in local AI apps.",
                        "scope": "default",
                        "kind": "preference",
                        "metadata": {"source": "test"},
                    },
                )
                searched = _get_json(base_url + "/api/memory?scope=default&query=Antonio%20preference")
                result = _post_json(
                    base_url + "/api/generate",
                    {"prompt": "Summarize Antonio preference."},
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(searched["records"][0]["id"], memory["id"])
        self.assertEqual(result["context"]["memory_ids"], [memory["id"]])
        self.assertIn("memory", result["context"]["sections"])
        self.assertEqual(result["route"]["selected"][0]["expert_id"], "general")
        self.assertEqual(result["results"][0]["expert_id"], "general")
        self.assertGreater(_prompt_chars(result["content"]), len("Summarize Antonio preference."))

    def test_ui_supports_markdown_rendering_and_enter_shortcut(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            html = _get_text(base_url + "/")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertIn("Enter sends / Alt Enter wraps", html)
        self.assertIn("function renderMarkdown", html)
        self.assertIn("^[*-] (.+)", html)
        self.assertIn("event.altKey", html)
        self.assertIn("What should we work on?", html)
        self.assertIn("advanced-panel", html)
        self.assertIn("runtime.model_commands || runtime.commands", html)
        self.assertIn("/api/setup", html)
        self.assertIn("renderSetup", html)
        self.assertIn("download_command_display", html)
        self.assertIn("experiments/eval_set_live_general.jsonl", html)
        self.assertIn("runCron", html)
        self.assertIn("/api/health", html)
        self.assertIn("renderHealth", html)
        self.assertIn("refreshHealth", html)
        self.assertIn("Refresh health", html)
        self.assertIn("/api/cron/run", html)
        self.assertIn("cron-confirm-writes", html)
        self.assertIn("runTool", html)
        self.assertIn("/api/tools/run", html)
        self.assertIn("saveMemory", html)
        self.assertIn("searchMemory", html)
        self.assertIn("/api/memory", html)
        self.assertIn("/api/chats", html)
        self.assertIn("session-list", html)
        self.assertIn("activeSessionId", html)
        self.assertIn("Search chats", html)
        self.assertIn("renameSession", html)
        self.assertIn("compactSession", html)
        self.assertIn("/compact", html)
        self.assertIn("exportSession", html)
        self.assertIn("deleteSession", html)
        self.assertIn("mcp.list_tools", html)
        self.assertIn("mcp.call_tool", html)
        self.assertIn("confirm_process_execution", html)
        self.assertIn("confirm_tool_call", html)
        self.assertIn("Confirm local write jobs", html)
        self.assertIn("config.routing?.strategy", html)
        self.assertIn("config.routing?.semantic?.enabled", html)
        self.assertIn("config.routing?.distilled?.enabled", html)
        self.assertIn("result.disagreement", html)
        self.assertIn("routeMeta(message.meta || {})", html)
        self.assertIn("hidden", html)


def _get_json(url: str) -> dict[str, object]:
    with request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    http_req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _patch_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    http_req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )
    with request.urlopen(http_req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _delete_json(url: str) -> dict[str, object]:
    http_req = request.Request(url, method="DELETE")
    with request.urlopen(http_req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _write_temp_app_config(root: Path, *, context_policy_path: Path | None = None) -> Path:
    raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = "tests/fixtures/moe.synthetic.json"
    raw["runtime"]["work_dir"] = str(root / "runtime")
    if context_policy_path is not None:
        raw["runtime"]["context_policy_config"] = str(context_policy_path)
        raw["runtime"]["context_policy_profile"] = "default"
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _prompt_chars(content: object) -> int:
    marker = "prompt_chars="
    text = str(content)
    return int(text.split(marker, 1)[1].split()[0])

if __name__ == "__main__":
    unittest.main()
