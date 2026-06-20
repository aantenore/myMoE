from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from local_moe.app_config import load_app_config
from local_moe.chat_store import FileChatStore
from local_moe.config import load_config
from local_moe.extensions import (
    ExtensionRegistry,
    McpServerDefinition,
    ToolDefinition,
    load_cron_jobs,
    load_extension_registry,
    load_mcp_servers,
)
from local_moe.memory import FileMemoryStore
from local_moe.tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload
from tests.mcp_test_utils import write_fake_mcp_server


class ToolRunnerTests(unittest.TestCase):
    def test_memory_search_returns_scoped_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FileMemoryStore(memory_path)
            store.add("Local model routing should stay configurable.", scope="project")
            store.add("Hosted model note.", scope="other")
            runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)

            result = runner.run(
                "memory.search",
                {"query": "local routing configurable", "scope": "project", "limit": 5},
            )

        payload = tool_result_payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["payload"]["count"], 1)
        self.assertEqual(
            payload["payload"]["records"][0]["text"],
            "Local model routing should stay configurable.",
        )

    def test_knowledge_ingest_requires_confirmation_and_writes_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)

            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "knowledge.ingest",
                    {
                        "title": "Router Notes",
                        "content": "Local router labels should stay configurable.",
                    },
                )

            result = runner.run(
                "knowledge.ingest",
                {
                    "title": "Router Notes",
                    "content": "Local router labels should stay configurable.",
                    "scope": "project",
                    "metadata": {"source": "test"},
                    "confirm": True,
                },
            )
            search = FileMemoryStore(memory_path).search("router labels", scope="project")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["title"], "Router Notes")
        self.assertEqual(result.payload["chunk_count"], 1)
        self.assertEqual(search[0][0].kind, "knowledge")
        self.assertEqual(search[0][0].metadata["source"], "test")

    def test_memory_forget_requires_confirmation_and_deletes_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FileMemoryStore(memory_path)
            report = store.ingest_document(
                "Forgettable local knowledge.",
                title="Forgettable",
                scope="project",
            )
            runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)

            with self.assertRaises(ToolExecutionError):
                runner.run("memory.forget", {"document_id": report.document_id})

            result = runner.run(
                "memory.forget",
                {"document_id": report.document_id, "confirm": True},
            )
            remaining = FileMemoryStore(memory_path).list(scope="project")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["removed_count"], 1)
        self.assertEqual(result.payload["removed_ids"], list(report.record_ids))
        self.assertEqual(remaining, [])

    def test_memory_maintenance_and_prune_expired_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            store = FileMemoryStore(memory_path)
            active = store.add("Current fact.")
            expired = store.add("Expired fact.", valid_until="2026-01-01T00:00:00+00:00")
            runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)

            maintenance = runner.run(
                "memory.maintenance",
                {"now": "2026-06-20T00:00:00+00:00"},
            )
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "memory.prune_expired",
                    {"now": "2026-06-20T00:00:00+00:00"},
                )
            pruned = runner.run(
                "memory.prune_expired",
                {"now": "2026-06-20T00:00:00+00:00", "confirm": True},
            )
            remaining = FileMemoryStore(memory_path).list()

        self.assertEqual(maintenance.status, "ok")
        self.assertEqual(maintenance.payload["active_records"], 1)
        self.assertEqual(maintenance.payload["expired_records"], 1)
        self.assertEqual(pruned.payload["removed_count"], 1)
        self.assertEqual(pruned.payload["removed_ids"], [expired.id])
        self.assertEqual([record.id for record in remaining], [active.id])

    def test_data_export_import_requires_confirmation_and_restores_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_chat_path = root / "source" / "chats.json"
            source_memory_path = root / "source" / "memory.jsonl"
            target_chat_path = root / "target" / "chats.json"
            target_memory_path = root / "target" / "memory.jsonl"
            source_chats = FileChatStore(source_chat_path)
            source_memory = FileMemoryStore(source_memory_path)
            session = source_chats.append_exchange(
                session_id=None,
                user_content="Plan portable local data backups.",
                assistant_content="Use a confirmed JSON bundle.",
            )
            source_memory.add("Portable backups preserve memory.", scope="project")
            source_runner = LocalToolRunner(
                load_extension_registry(),
                chat_path=source_chat_path,
                memory_path=source_memory_path,
            )

            with self.assertRaises(ToolExecutionError):
                source_runner.run("data.export", {})

            exported = source_runner.run("data.export", {"confirm": True})
            target_runner = LocalToolRunner(
                load_extension_registry(),
                chat_path=target_chat_path,
                memory_path=target_memory_path,
            )
            with self.assertRaises(ToolExecutionError):
                target_runner.run("data.import", {"bundle": exported.payload["bundle"]})
            imported = target_runner.run(
                "data.import",
                {"bundle": exported.payload["bundle"], "mode": "merge", "confirm": True},
            )
            restored_session = FileChatStore(target_chat_path).get_session(session.id)
            restored_memory = FileMemoryStore(target_memory_path).search("portable backups", scope="project")

        self.assertEqual(exported.status, "ok")
        self.assertEqual(exported.payload["counts"]["chat_sessions"], 1)
        self.assertEqual(imported.status, "ok")
        self.assertEqual(imported.payload["chats"]["imported_count"], 1)
        self.assertIsNotNone(restored_session)
        self.assertEqual(restored_memory[0][0].text, "Portable backups preserve memory.")

    def test_context_compact_can_use_configured_model(self) -> None:
        runner = LocalToolRunner(
            load_extension_registry(),
            moe_config=load_config("tests/fixtures/moe.synthetic.json"),
        )

        result = runner.run(
            "context.compact",
            {
                "turns": [
                    {"role": "user", "content": "We chose a local-first app."},
                    {"role": "assistant", "content": "The router remains lightweight."},
                ],
                "existing_summary": "Existing decision: no cloud inference.",
                "use_model": True,
            },
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["turn_count"], 2)
        self.assertIn("Summarize this local-agent session", result.payload["prompt"])
        self.assertIn("synthetic-general", result.payload["summary"])

    def test_plugin_create_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = LocalToolRunner(load_extension_registry(), plugins_dir=tmp)

            with self.assertRaises(ToolExecutionError):
                runner.run("plugin.create", {"plugin_id": "demo-plugin"})

            result = runner.run(
                "plugin.create",
                {"plugin_id": "demo-plugin", "confirm": True},
            )

            created_path = Path(result.payload["path"])
            manifest_exists = (created_path / "plugin.json").exists()
            skill_exists = (created_path / "SKILL.md").exists()

        self.assertEqual(created_path.name, "demo-plugin")
        self.assertEqual(result.status, "ok")
        self.assertTrue(manifest_exists)
        self.assertTrue(skill_exists)

    def test_extension_audit_returns_registry_issues(self) -> None:
        runner = LocalToolRunner(load_extension_registry())

        result = runner.run("extension.audit", {})

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.payload["checked"])
        self.assertEqual(result.payload["issue_count"], 0)
        self.assertGreaterEqual(result.payload["plugin_count"], 1)

    def test_extension_configure_requires_confirmation_and_updates_mcp_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mcp_path = root / "mcp.json"
            cron_path = root / "cron.json"
            mcp_path.write_text('{"servers": []}', encoding="utf-8")
            cron_path.write_text('{"jobs": []}', encoding="utf-8")
            runner = LocalToolRunner(
                _extension_configure_registry(),
                app_config=_app_config_for_extensions(root, mcp_path, cron_path),
            )

            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "extension.configure",
                    {
                        "surface": "mcp_server",
                        "definition": {"name": "docs", "command": "npx"},
                    },
                )

            result = runner.run(
                "extension.configure",
                {
                    "surface": "mcp_server",
                    "definition": {
                        "name": "docs",
                        "description": "Read documentation files.",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "docs"],
                        "enabled": False,
                        "risk_class": "write_local",
                        "capabilities": ["resources", "tools"],
                        "transport": "stdio",
                        "allowed_tools": ["list_directory", "read_text_file"],
                    },
                    "confirm": True,
                },
            )

            configured = load_mcp_servers(mcp_path)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["action"], "created")
        self.assertEqual(result.payload["id"], "docs")
        self.assertEqual(configured[0].name, "docs")
        self.assertEqual(configured[0].allowed_tools, ("list_directory", "read_text_file"))
        self.assertEqual(result.payload["audit"]["issue_count"], 0)

    def test_extension_configure_updates_cron_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mcp_path = root / "mcp.json"
            cron_path = root / "cron.json"
            mcp_path.write_text('{"servers": []}', encoding="utf-8")
            cron_path.write_text('{"jobs": []}', encoding="utf-8")
            runner = LocalToolRunner(
                _extension_configure_registry(),
                app_config=_app_config_for_extensions(root, mcp_path, cron_path),
            )

            created = runner.run(
                "extension.configure",
                {
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
            )
            removed = runner.run(
                "extension.configure",
                {
                    "surface": "cron_job",
                    "mode": "remove",
                    "definition": {"id": "daily-audit"},
                    "confirm": True,
                },
            )
            remaining = load_cron_jobs(cron_path)

        self.assertEqual(created.payload["action"], "created")
        self.assertEqual(removed.payload["action"], "removed")
        self.assertEqual(remaining, [])

    def test_profile_activate_requires_confirmation_and_updates_app_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root, "configs/moe.live.fast-mlx.example.json")
            runner = LocalToolRunner(
                load_extension_registry(),
                app_config=load_app_config(app_config_path),
                app_config_path=str(app_config_path),
                active_config_path="configs/moe.live.fast-mlx.example.json",
            )

            with self.assertRaises(ToolExecutionError):
                runner.run("profile.activate", {"profile_path": "tests/fixtures/moe.synthetic.json"})
            result = runner.run(
                "profile.activate",
                {
                    "profile_path": "tests/fixtures/moe.synthetic.json",
                    "confirm": True,
                },
            )
            raw = json.loads(app_config_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["new_default_config"], "tests/fixtures/moe.synthetic.json")
        self.assertTrue(result.payload["restart_required"])
        self.assertEqual(raw["default_moe_config"], "tests/fixtures/moe.synthetic.json")

    def test_storage_inspect_reports_configured_paths_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            raw = json.loads(app_config_path.read_text(encoding="utf-8"))
            cache = root / "missing-cache"
            work = root / "missing-work"
            raw["runtime"]["model_cache_dir"] = str(cache)
            raw["runtime"]["work_dir"] = str(work)
            app_config_path.write_text(json.dumps(raw), encoding="utf-8")
            runner = LocalToolRunner(
                load_extension_registry(),
                app_config=load_app_config(app_config_path),
                app_config_path=str(app_config_path),
            )

            result = runner.run("storage.inspect", {"min_free_gib": 0})
            cache_exists = cache.exists()
            work_exists = work.exists()

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["schema_version"], "1.0")
        self.assertEqual(result.payload["status"], "ready")
        self.assertEqual(result.payload["min_free_gib"], 0.0)
        self.assertEqual({item["label"] for item in result.payload["paths"]}, {"model_cache_dir", "work_dir"})
        self.assertFalse(cache_exists)
        self.assertFalse(work_exists)

    def test_models_inventory_reports_configured_asset_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config_path = _write_temp_app_config(root, "tests/fixtures/moe.synthetic.json")
            runner = LocalToolRunner(
                load_extension_registry(),
                app_config=load_app_config(app_config_path),
                moe_config=load_config("tests/fixtures/moe.synthetic.json"),
                app_config_path=str(app_config_path),
                active_config_path="tests/fixtures/moe.synthetic.json",
            )

            result = runner.run("models.inventory", {"max_files": 10})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["schema_version"], "1.0")
        self.assertEqual(result.payload["summary"]["asset_count"], 0)
        self.assertEqual(result.payload["status"], "ready")

    def test_mcp_search_capabilities_returns_declared_servers(self) -> None:
        runner = LocalToolRunner(load_extension_registry())

        result = runner.run("mcp.search_capabilities", {"query": "filesystem"})

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["count"], 1)
        self.assertEqual(result.payload["servers"][0]["name"], "filesystem")
        self.assertFalse(result.payload["servers"][0]["enabled"])
        self.assertIn("allowed_tools", result.payload["servers"][0])

    def test_mcp_list_tools_requires_confirmation_and_lists_enabled_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_fake_mcp_server(Path(tmp) / "fake_mcp.py")
            registry = ExtensionRegistry(
                tools=(
                    ToolDefinition(
                        name="mcp.list_tools",
                        description="List MCP tools",
                        risk_class="process_execution",
                        side_effects="starts_process",
                        enabled=True,
                    ),
                ),
                skills=(),
                mcp_servers=(
                    McpServerDefinition(
                        name="fake",
                        description="Fake MCP server",
                        command=sys.executable,
                        args=(str(script),),
                        enabled=True,
                        risk_class="read_only",
                        capabilities=("tools",),
                        allowed_tools=("echo",),
                    ),
                ),
                cron_jobs=(),
                plugins=(),
            )
            blocked_runner = LocalToolRunner(registry)
            with self.assertRaises(ToolExecutionError):
                blocked_runner.run(
                    "mcp.list_tools",
                    {"server": "fake", "confirm_process_execution": True},
                )

            runner = LocalToolRunner(registry, allow_process_execution=True)
            with self.assertRaises(ToolExecutionError):
                runner.run("mcp.list_tools", {"server": "fake"})

            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.list_tools",
                    {
                        "server": "fake",
                        "confirm_process_execution": True,
                        "timeout_seconds": "slow",
                    },
                )

            result = runner.run(
                "mcp.list_tools",
                {"server": "fake", "confirm_process_execution": True, "timeout_seconds": 3},
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["server"], "fake")
        self.assertEqual(result.payload["tools"][0]["name"], "echo")

    def test_mcp_call_tool_requires_allowlist_and_confirmations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = write_fake_mcp_server(Path(tmp) / "fake_mcp.py")
            registry = ExtensionRegistry(
                tools=(
                    ToolDefinition(
                        name="mcp.call_tool",
                        description="Call MCP tool",
                        risk_class="process_execution",
                        side_effects="starts_process_and_calls_tool",
                        enabled=True,
                    ),
                ),
                skills=(),
                mcp_servers=(
                    McpServerDefinition(
                        name="fake",
                        description="Fake MCP server",
                        command=sys.executable,
                        args=(str(script),),
                        enabled=True,
                        risk_class="read_only",
                        capabilities=("tools",),
                        allowed_tools=("echo",),
                    ),
                ),
                cron_jobs=(),
                plugins=(),
            )
            blocked_runner = LocalToolRunner(registry)
            with self.assertRaises(ToolExecutionError):
                blocked_runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                        "confirm_process_execution": True,
                        "confirm_tool_call": True,
                    },
                )

            runner = LocalToolRunner(registry, allow_process_execution=True)
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                        "confirm_process_execution": True,
                    },
                )
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "not-allowed",
                        "arguments": {},
                        "confirm_process_execution": True,
                        "confirm_tool_call": True,
                    },
                )
            with self.assertRaises(ToolExecutionError):
                runner.run(
                    "mcp.call_tool",
                    {
                        "server": "fake",
                        "tool_name": "echo",
                        "arguments": [],
                        "confirm_process_execution": True,
                        "confirm_tool_call": True,
                    },
                )

            result = runner.run(
                "mcp.call_tool",
                {
                    "server": "fake",
                    "tool_name": "echo",
                    "arguments": {"text": "hello"},
                    "confirm_process_execution": True,
                    "confirm_tool_call": True,
                    "timeout_seconds": 3,
                },
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["server"], "fake")
        self.assertEqual(result.payload["tool_name"], "echo")
        self.assertEqual(result.payload["content"][0]["text"], "echo:hello")


def _extension_configure_registry() -> ExtensionRegistry:
    return ExtensionRegistry(
        tools=(
            ToolDefinition(
                name="extension.configure",
                description="Configure extension entries",
                risk_class="write_local",
                side_effects="writes_registry_files",
                enabled=True,
            ),
        ),
        skills=(),
        mcp_servers=(),
        cron_jobs=(),
        plugins=(),
    )


def _app_config_for_extensions(root: Path, mcp_path: Path, cron_path: Path) -> object:
    tools_path = root / "tools.json"
    tools_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "extension.configure",
                        "description": "Configure extension entries",
                        "risk_class": "write_local",
                        "side_effects": "writes_registry_files",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    skills_dir = root / "skills"
    plugins_dir = root / "plugins"
    skills_dir.mkdir()
    plugins_dir.mkdir()
    return SimpleNamespace(
        permissions=SimpleNamespace(allow_process_execution=False),
        extensions=SimpleNamespace(
            plugins_dir=str(plugins_dir),
            skills_dir=str(skills_dir),
            tools_config=str(tools_path),
            mcp_config=str(mcp_path),
            cron_config=str(cron_path),
        ),
    )


def _write_temp_app_config(root: Path, default_config: str) -> Path:
    raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = default_config
    raw["runtime"]["work_dir"] = str(root / "runtime")
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
