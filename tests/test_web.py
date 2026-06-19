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
            profiles = _get_json(base_url + "/api/config/profiles")
            processes = _get_json(base_url + "/api/models/processes")
            setup = _get_json(base_url + "/api/setup")
            doctor = _get_json(base_url + "/api/doctor")
            health = _get_json(base_url + "/api/health")
            extensions = _get_json(base_url + "/api/extensions")
            audit = _get_json(base_url + "/api/extensions/audit")
            support_bundle = _get_json(base_url + "/api/support-bundle")
            downloaded_bundle = json.loads(_get_text(base_url + "/api/support-bundle/download.json"))
            performance = _get_json(base_url + "/api/performance")
            performance_markdown = _get_text(base_url + "/api/performance/report.md")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertIn(runtime["backend"], {"mlx_lm", "ollama", "llama_cpp"})
        self.assertGreaterEqual(profiles["count"], 1)
        self.assertIn("tests/fixtures/moe.synthetic.json", {item["path"] for item in profiles["profiles"]})
        self.assertTrue(any(item["active"] for item in profiles["profiles"]))
        self.assertEqual(processes["count"], 0)
        self.assertEqual(processes["servers"], [])
        self.assertEqual(setup["status"], "ready")
        self.assertEqual(setup["models"], [])
        self.assertIn("download_command_display", setup)
        self.assertEqual(doctor["status"], "ready")
        self.assertEqual(doctor["summary"]["failed"], 0)
        self.assertIn("health", doctor)
        self.assertIn("extension_audit", doctor)
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["experts"][0]["status"], "skipped")
        self.assertTrue(extensions["tools"])
        self.assertEqual(audit["audit"]["issue_count"], 0)
        self.assertIn("extensions", audit)
        self.assertEqual(support_bundle["schema_version"], "1.0")
        self.assertEqual(support_bundle["doctor"]["status"], "ready")
        self.assertIn("chat transcripts", support_bundle["privacy"]["excludes"])
        self.assertEqual(downloaded_bundle["schema_version"], "1.0")
        self.assertEqual(performance["schema_version"], "1.0")
        self.assertIn(performance["status"], {"ready", "ready_partial"})
        self.assertEqual(performance["decision"]["primary_general"]["candidate_id"], "qwen3-30b-a3b-2507-mlx-4bit")
        self.assertNotIn("content_excerpt", json.dumps(performance))
        self.assertIn("# myMoE Performance Report", performance_markdown)

    def test_model_process_endpoints_are_confirmation_guarded(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            status = _get_json(base_url + "/api/models/processes")
            guarded_start = _post_json(base_url + "/api/models/start", {})
            confirmed_start = _post_json(base_url + "/api/models/start", {"confirm": True})
            guarded_stop = _post_json(base_url + "/api/models/stop", {})
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(status["count"], 0)
        self.assertEqual(guarded_start["status"], "confirmation_required")
        self.assertFalse(guarded_start["ok"])
        self.assertEqual(confirmed_start["status"], "no_commands")
        self.assertTrue(confirmed_start["ok"])
        self.assertEqual(guarded_stop["status"], "confirmation_required")
        self.assertFalse(guarded_stop["ok"])

    def test_runs_setup_endpoint_preview_and_confirmation_guard(self) -> None:
        server = build_server("tests/fixtures/moe.synthetic.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            preview = _post_json(base_url + "/api/setup/run", {})
            guarded = _post_json(base_url + "/api/setup/run", {"execute": True})
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(preview["status"], "planned")
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["setup_before"]["status"], "ready")
        self.assertEqual(guarded["status"], "confirmation_required")
        self.assertFalse(guarded["ok"])
        self.assertEqual(guarded["steps"][0]["status"], "confirmation_required")

    def test_creates_plugin_from_web_api_and_refreshes_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_plugin_app_config(root)
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
                    _post_json(base_url + "/api/plugins", {"plugin_id": "demo-plugin"})
                created = _post_json(
                    base_url + "/api/plugins",
                    {
                        "plugin_id": "demo-plugin",
                        "name": "Demo Plugin",
                        "description": "Adds demo behavior.",
                        "risk_class": "compute_only",
                        "confirm": True,
                    },
                )
                extensions = _get_json(base_url + "/api/extensions")
                audit = _get_json(base_url + "/api/extensions/audit")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

            manifest = root / "plugins" / "demo-plugin" / "plugin.json"
            skill = root / "plugins" / "demo-plugin" / "SKILL.md"
            manifest_exists = manifest.exists()
            skill_exists = skill.exists()
            created_plugin_ids = {plugin["id"] for plugin in created["extensions"]["plugins"]}
            created_skill_names = {item["name"] for item in created["extensions"]["skills"]}
            listed_plugin_ids = {plugin["id"] for plugin in extensions["plugins"]}
            audited_plugin_ids = {plugin["id"] for plugin in audit["extensions"]["plugins"]}

        self.assertEqual(raised.exception.code, 400)
        self.assertTrue(created["created"])
        self.assertEqual(created["audit"]["issue_count"], 0)
        self.assertTrue(manifest_exists)
        self.assertTrue(skill_exists)
        self.assertIn("demo-plugin", created_plugin_ids)
        self.assertIn("demo-plugin", created_skill_names)
        self.assertIn("demo-plugin", listed_plugin_ids)
        self.assertIn("demo-plugin", audited_plugin_ids)

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

    def test_extension_configure_tool_refreshes_web_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            mcp_path = root / "mcp.json"
            cron_path = root / "cron.json"
            mcp_path.write_text('{"servers": []}', encoding="utf-8")
            cron_path.write_text('{"jobs": []}', encoding="utf-8")
            raw = json.loads(app_config.read_text(encoding="utf-8"))
            raw["extensions"]["mcp_config"] = str(mcp_path)
            raw["extensions"]["cron_config"] = str(cron_path)
            app_config.write_text(json.dumps(raw), encoding="utf-8")
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
                        "name": "extension.configure",
                        "input": {
                            "surface": "cron_job",
                            "definition": {
                                "id": "daily-audit",
                                "description": "Run extension audit once per day.",
                                "enabled": True,
                                "schedule": {"type": "interval", "seconds": 86400},
                                "command": ["extension.audit"],
                                "risk_class": "compute_only",
                            },
                            "confirm": True,
                        },
                    },
                )
                cron = _get_json(base_url + "/api/cron")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"]["action"], "created")
        self.assertIn("daily-audit", [job["id"] for job in cron["jobs"]])

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

    def test_stream_generation_endpoint_persists_chat_session(self) -> None:
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
                raw = _post_text(
                    base_url + "/api/generate/stream",
                    {"prompt": "Summarize this streamed note."},
                )
                listed = _get_json(base_url + "/api/chats")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertIn("event: route", raw)
        self.assertIn("event: content", raw)
        self.assertIn("event: final", raw)
        self.assertIn('"session_id"', raw)
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["sessions"][0]["message_count"], 2)

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

    def test_memory_api_forget_requires_confirmation_and_valid_id(self) -> None:
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
                        "text": "Temporary local memory.",
                        "scope": "default",
                        "kind": "note",
                    },
                )
                with self.assertRaises(HTTPError) as delete_raised:
                    _delete_json(base_url + f"/api/memory/{memory['id']}")
                with self.assertRaises(HTTPError) as invalid_raised:
                    _delete_json(base_url + "/api/memory/?confirm=true")
                deleted = _delete_json(base_url + f"/api/memory/{memory['id']}?confirm=true")
                searched = _get_json(base_url + "/api/memory?scope=default&query=Temporary")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(delete_raised.exception.code, 400)
        self.assertEqual(invalid_raised.exception.code, 400)
        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["removed_ids"], [memory["id"]])
        self.assertEqual(searched["records"], [])

    def test_memory_maintenance_and_prune_expired_api(self) -> None:
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
                active = _post_json(
                    base_url + "/api/memory",
                    {"text": "Current local memory.", "scope": "default"},
                )
                expired = _post_json(
                    base_url + "/api/memory",
                    {
                        "text": "Expired local memory.",
                        "scope": "default",
                        "valid_until": "2026-01-01T00:00:00+00:00",
                    },
                )
                maintenance = _get_json(base_url + "/api/memory/maintenance")
                with self.assertRaises(HTTPError) as raised:
                    _post_json(base_url + "/api/memory/prune-expired", {})
                pruned = _post_json(
                    base_url + "/api/memory/prune-expired",
                    {"confirm": True},
                )
                listed = _get_json(base_url + "/api/memory?scope=default")
                audit = _get_json(base_url + "/api/audit?action=memory.prune_expired&limit=5")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(maintenance["total_records"], 2)
        self.assertEqual(maintenance["expired_records"], 1)
        self.assertEqual(pruned["removed_count"], 1)
        self.assertEqual(pruned["removed_ids"], [expired["id"]])
        self.assertEqual([record["id"] for record in listed["records"]], [active["id"]])
        self.assertEqual([event["status"] for event in audit["events"]], ["ok", "confirmation_required"])

    def test_data_export_import_api_is_confirmation_guarded(self) -> None:
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
                _post_json(base_url + "/api/chats", {"title": "Backup Test"})
                _post_json(
                    base_url + "/api/memory",
                    {
                        "text": "Local backup API memory.",
                        "scope": "default",
                        "kind": "fact",
                    },
                )
                with self.assertRaises(HTTPError) as export_raised:
                    _post_json(base_url + "/api/data/export", {})
                bundle = _post_json(base_url + "/api/data/export", {"confirm": True})
                with self.assertRaises(HTTPError) as import_raised:
                    _post_json(base_url + "/api/data/import", {"bundle": bundle})
                restored = _post_json(
                    base_url + "/api/data/import",
                    {"bundle": bundle, "mode": "merge", "confirm": True},
                )
                chats = _get_json(base_url + "/api/chats")
                memories = _get_json(base_url + "/api/memory?scope=default")
                export_audit = _get_json(base_url + "/api/audit?action=data.export&limit=10")
                import_audit = _get_json(base_url + "/api/audit?action=data.import&limit=10")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(export_raised.exception.code, 400)
        self.assertEqual(import_raised.exception.code, 400)
        self.assertEqual(bundle["schema_version"], "mymoe.local-data.v1")
        self.assertTrue(bundle["privacy"]["contains_user_content"])
        self.assertEqual(bundle["counts"]["chat_sessions"], 1)
        self.assertEqual(bundle["counts"]["memory_records"], 1)
        self.assertEqual(restored["mode"], "merge")
        self.assertEqual(restored["chats"]["updated_count"], 1)
        self.assertEqual(restored["memory"]["updated_count"], 1)
        self.assertEqual(chats["count"], 1)
        self.assertEqual(memories["count"], 1)
        self.assertIn("ok", {event["status"] for event in export_audit["events"]})
        self.assertIn("confirmation_required", {event["status"] for event in export_audit["events"]})
        self.assertIn("ok", {event["status"] for event in import_audit["events"]})
        self.assertIn("confirmation_required", {event["status"] for event in import_audit["events"]})
        self.assertNotIn("Local backup API memory.", json.dumps(export_audit))
        self.assertNotIn("Local backup API memory.", json.dumps(import_audit))

    def test_audit_prune_api_is_confirmation_guarded(self) -> None:
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
                for index in range(3):
                    _post_json(base_url + "/api/chats", {"title": f"Audit {index}"})
                with self.assertRaises(HTTPError) as raised:
                    _post_json(base_url + "/api/audit/prune", {"keep": 2})
                pruned = _post_json(
                    base_url + "/api/audit/prune",
                    {"keep": 2, "confirm": True},
                )
                audit = _get_json(base_url + "/api/audit?limit=10")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(pruned["keep"], 2)
        self.assertEqual(pruned["before_count"], 4)
        self.assertEqual(pruned["after_count"], 2)
        self.assertEqual(pruned["removed_count"], 3)
        self.assertEqual(audit["count"], 2)
        self.assertEqual([event["action"] for event in audit["events"]], ["audit.prune", "audit.prune"])
        self.assertEqual([event["status"] for event in audit["events"]], ["ok", "confirmation_required"])

    def test_knowledge_api_ingests_and_retrieves_context(self) -> None:
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
                        base_url + "/api/knowledge",
                        {
                            "title": "Local Routing Notes",
                            "content": "Semantic routing examples cover multilingual prompts.",
                        },
                    )
                imported = _post_json(
                    base_url + "/api/knowledge",
                    {
                        "title": "Local Routing Notes",
                        "content": "Semantic routing examples cover multilingual prompts.",
                        "scope": "default",
                        "confirm": True,
                    },
                )
                listed = _get_json(base_url + "/api/knowledge?scope=default")
                result = _post_json(
                    base_url + "/api/generate",
                    {"prompt": "What covers multilingual prompts?"},
                )
                with self.assertRaises(HTTPError) as delete_raised:
                    _delete_json(base_url + f"/api/knowledge/{imported['document_id']}")
                deleted = _delete_json(
                    base_url + f"/api/knowledge/{imported['document_id']}?confirm=true"
                )
                final = _get_json(base_url + "/api/knowledge?scope=default")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(delete_raised.exception.code, 400)
        self.assertEqual(imported["chunk_count"], 1)
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["records"][0]["kind"], "knowledge")
        self.assertEqual(result["context"]["memory_ids"], imported["record_ids"])
        self.assertIn("memory", result["context"]["sections"])
        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["removed_ids"], imported["record_ids"])
        self.assertEqual(final["count"], 0)

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
        self.assertIn("/api/generate/stream", html)
        self.assertIn("generateStream", html)
        self.assertIn("consumeSseEvents", html)
        self.assertIn("What should we work on?", html)
        self.assertIn("advanced-panel", html)
        self.assertIn("runtime.model_commands || runtime.commands", html)
        self.assertIn("/api/models/processes", html)
        self.assertIn("/api/models/start", html)
        self.assertIn("/api/models/stop", html)
        self.assertIn("renderModelProcesses", html)
        self.assertIn("Start models", html)
        self.assertIn("Stop managed", html)
        self.assertIn("/api/setup", html)
        self.assertIn("/api/setup/run", html)
        self.assertIn("/api/config/profiles", html)
        self.assertIn("renderSetup", html)
        self.assertIn("renderConfigProfiles", html)
        self.assertIn("Profiles", html)
        self.assertIn("runSetup", html)
        self.assertIn("setup-confirm", html)
        self.assertIn("extension.configure", html)
        self.assertIn("Knowledge", html)
        self.assertIn("/api/knowledge", html)
        self.assertIn("importKnowledge", html)
        self.assertIn("knowledge.ingest", html)
        self.assertIn("forgetKnowledge", html)
        self.assertIn("forgetMemory", html)
        self.assertIn("checkMemory", html)
        self.assertIn("pruneExpiredMemory", html)
        self.assertIn("/api/memory/maintenance", html)
        self.assertIn("/api/memory/prune-expired", html)
        self.assertIn("memory-prune-confirm", html)
        self.assertIn("/api/memory/", html)
        self.assertIn("/api/knowledge/", html)
        self.assertIn("Local Data", html)
        self.assertIn("/api/data/export", html)
        self.assertIn("/api/data/import", html)
        self.assertIn("exportData", html)
        self.assertIn("importData", html)
        self.assertIn("data.export", html)
        self.assertIn("data.import", html)
        self.assertIn("Audit Trail", html)
        self.assertIn("/api/audit", html)
        self.assertIn("/api/audit/prune", html)
        self.assertIn("refreshAudit", html)
        self.assertIn("pruneAudit", html)
        self.assertIn("audit-prune-confirm", html)
        self.assertIn("Prepare runtime", html)
        self.assertIn("download_command_display", html)
        self.assertIn("experiments/eval_set_live_general.jsonl", html)
        self.assertIn("runCron", html)
        self.assertIn("/api/health", html)
        self.assertIn("renderHealth", html)
        self.assertIn("refreshHealth", html)
        self.assertIn("Refresh health", html)
        self.assertIn("/api/cron/run", html)
        self.assertIn("cron-confirm-writes", html)
        self.assertIn("/api/doctor", html)
        self.assertIn("System Doctor", html)
        self.assertIn("runDoctor", html)
        self.assertIn("/api/support-bundle/download.json", html)
        self.assertIn("Download bundle", html)
        self.assertIn("downloadSupportBundle", html)
        self.assertIn("/api/performance", html)
        self.assertIn("/api/performance/report.md", html)
        self.assertIn("Performance", html)
        self.assertIn("renderPerformance", html)
        self.assertIn("refreshPerformance", html)
        self.assertIn("Download report", html)
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
        self.assertIn("/api/plugins", html)
        self.assertIn("/api/extensions/audit", html)
        self.assertIn("Plugin Studio", html)
        self.assertIn("Registry Audit", html)
        self.assertIn("runExtensionAudit", html)
        self.assertIn("createPlugin", html)
        self.assertIn("plugin-confirm", html)
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


def _post_text(url: str, payload: dict[str, object]) -> str:
    http_req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_req, timeout=5) as response:
        return response.read().decode("utf-8")


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


def _write_temp_plugin_app_config(root: Path) -> Path:
    path = _write_temp_app_config(root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["extensions"]["plugins_dir"] = str(root / "plugins")
    raw["extensions"]["skills_dir"] = str(root / "skills")
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _prompt_chars(content: object) -> int:
    marker = "prompt_chars="
    text = str(content)
    return int(text.split(marker, 1)[1].split()[0])

if __name__ == "__main__":
    unittest.main()
