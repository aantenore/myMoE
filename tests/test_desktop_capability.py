from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any
import unittest

from local_moe.agent_tools import AgentToolRegistry, bound_tool_result
from local_moe.agent_types import AgentToolCall
from local_moe.desktop_capability import (
    CuaDriverDesktopProvider,
    DesktopCapabilityConfig,
    DesktopToolRunner,
    _desktop_session_policy,
    _desktop_user_policy,
    desktop_tool_specs,
    run_desktop_capability_canary,
)
from local_moe.extensions import (
    ExtensionRegistry,
    McpServerDefinition,
    ToolDefinition,
)
from local_moe.mcp_client import (
    McpClientError,
    McpTool,
    McpToolCallResult,
    McpToolList,
)
from local_moe.tool_runner import LocalToolRunner, ToolExecutionError


_GET_WINDOW_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "pid": {"type": "integer", "minimum": 1},
        "window_id": {"type": "integer", "minimum": 0},
        "include_screenshot": {"type": "boolean"},
        "max_elements": {"type": "integer", "minimum": 1},
        "max_depth": {"type": "integer", "minimum": 1},
    },
    "required": ["pid", "window_id", "include_screenshot"],
    "additionalProperties": False,
}

_PROCESS_IDENTITY = {
    "pid": 4242,
    "name": "Offline Editor",
    "started_at": "2026-07-21T08:00:00Z",
    "executable_sha256": "b" * 64,
}

_RUNTIME_RECEIPT = {
    "provider": "cua_driver",
    "version": "0.10.0",
    "executable_sha256": "a" * 64,
    "telemetry_enabled": False,
}

_FORBIDDEN_NODE_KEYS = {
    "bounds",
    "coordinates",
    "element_index",
    "element_token",
    "frame",
    "position",
    "raw_index",
    "screenshot",
    "x",
    "y",
}


class DesktopCapabilityTests(unittest.TestCase):
    def test_configuration_pins_provider_schema_and_exact_process_identity(self) -> None:
        config = DesktopCapabilityConfig.from_server(_server())

        self.assertEqual(config.provider, "cua_driver")
        self.assertEqual(config.version, "0.10.0")
        self.assertEqual(config.provider_executable_sha256, "a" * 64)
        self.assertEqual(
            config.tool_schema_sha256,
            {"get_window_state": _sha256_json(_GET_WINDOW_STATE_SCHEMA)},
        )
        self.assertEqual(config.target_id, "offline-editor")
        self.assertEqual(config.pid, 4242)
        self.assertEqual(config.window_id, 17)
        self.assertEqual(config.process_name, "Offline Editor")
        self.assertEqual(config.process_started_at, "2026-07-21T08:00:00Z")
        self.assertEqual(config.process_executable_sha256, "b" * 64)

    def test_configuration_rejects_unpinned_or_overbroad_provider(self) -> None:
        server = _server()
        invalid_changes = (
            {"provider": "other"},
            {"version": "latest"},
            {"version": "0.10.1"},
            {"provider_executable_sha256": "unverified"},
            {"telemetry_enabled": True},
            {"max_result_chars": 1_024},
            {"max_result_chars": 8_000},
            {"tool_schema_sha256": {}},
            {
                "tool_schema_sha256": {
                    "get_window_state": _sha256_json(_GET_WINDOW_STATE_SCHEMA),
                    "click": "c" * 64,
                }
            },
            {"target": {**server.desktop_capability["target"], "pid": 0}},
            {
                "target": {
                    **server.desktop_capability["target"],
                    "process_executable_sha256": "unknown",
                }
            },
            {"unexpected_policy_escape": True},
        )

        for change in invalid_changes:
            with self.subTest(change=change):
                raw = {**server.desktop_capability, **change}
                with self.assertRaises(ToolExecutionError):
                    DesktopCapabilityConfig.from_server(
                        replace(server, desktop_capability=raw)
                    )

        with self.assertRaises(ToolExecutionError):
            DesktopCapabilityConfig.from_server(
                replace(server, allowed_tools=("get_window_state", "click"))
            )

    def test_observe_calls_only_bound_semantic_snapshot_and_sanitizes_output(
        self,
    ) -> None:
        session = _FakeSession(
            result=_semantic_result(
                [
                    {
                        "element_index": 7,
                        "element_token": "provider-only-token",
                        "role": "button",
                        "label": "Save",
                        "value": "ready",
                        "description": "Save the document",
                        "depth": 2,
                        "frame": {"x": 40, "y": 80, "width": 120, "height": 30},
                        "states": {"enabled": True, "focused": False},
                    }
                ]
            )
        )
        provider = _provider(session)
        try:
            output = provider.observe()
        finally:
            provider.close()

        self.assertEqual(
            session.calls,
            [
                (
                    "get_window_state",
                    {
                        "pid": 4242,
                        "window_id": 17,
                        "include_screenshot": False,
                        "max_elements": 3,
                        "max_depth": 4,
                    },
                )
            ],
        )
        self.assertEqual(output["target_id"], "offline-editor")
        self.assertEqual(output["trust"], "untrusted_external")
        self.assertEqual(output["instruction_policy"], "content_is_data_only")
        self.assertEqual(output["runtime"]["provider"], "cua_driver")
        self.assertEqual(output["runtime"]["version"], "0.10.0")
        self.assertRegex(output["snapshot_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(output["nodes"]), 1)
        self.assertIn("Save", json.dumps(output["nodes"]))
        self.assertFalse(_all_keys(output["nodes"]) & _FORBIDDEN_NODE_KEYS)
        self.assertNotIn("provider-only-token", json.dumps(output))

    def test_schema_drift_fails_before_upstream_call_and_closes_session(self) -> None:
        drifted_schema = {
            **_GET_WINDOW_STATE_SCHEMA,
            "description": "changed upstream contract",
        }
        session = _FakeSession(
            schema=drifted_schema,
            result=_semantic_result([]),
        )
        provider = _provider(session)

        with self.assertRaisesRegex(ToolExecutionError, "schema drift"):
            provider.observe()

        self.assertEqual(session.calls, [])
        self.assertFalse(session.active)

    def test_mcp_provider_identity_drift_fails_before_observation(self) -> None:
        session = _FakeSession(
            result=_semantic_result([]),
            server_info={"name": "cua-driver", "version": "0.10.1"},
        )
        provider = _provider(session)

        with self.assertRaisesRegex(ToolExecutionError, "MCP identity"):
            provider.observe()

        self.assertEqual(session.calls, [])
        self.assertFalse(session.active)

    def test_image_content_is_rejected_and_session_is_closed(self) -> None:
        result = _semantic_result([])
        result = replace(
            result,
            content=(
                {
                    "type": "image",
                    "data": "not-a-real-image",
                    "mimeType": "image/png",
                },
            ),
        )
        session = _FakeSession(result=result)
        provider = _provider(session)

        with self.assertRaisesRegex(ToolExecutionError, "image|screenshot"):
            provider.observe()

        self.assertFalse(session.active)

    def test_missing_media_type_is_rejected_fail_closed(self) -> None:
        result = replace(
            _semantic_result(
                [{"role": "button", "label": "Save", "depth": 1}]
            ),
            content=({"text": "untyped provider content"},),
        )
        session = _FakeSession(result=result)
        provider = _provider(session)

        with self.assertRaisesRegex(ToolExecutionError, "media|unsupported"):
            provider.observe()

        self.assertFalse(session.active)

    def test_degraded_semantic_tree_is_rejected(self) -> None:
        result = _semantic_result(
            [{"role": "button", "label": "Unreliable", "depth": 1}]
        )
        result = replace(
            result,
            structured_content={**result.structured_content, "degraded": True},
        )
        session = _FakeSession(result=result)
        provider = _provider(session)

        with self.assertRaisesRegex(ToolExecutionError, "degraded|unauthoritative"):
            provider.observe()

        self.assertFalse(session.active)

    def test_empty_or_fully_invalid_semantic_tree_is_rejected(self) -> None:
        cases = (
            [],
            [{"role": "button", "label": "Too deep", "depth": 99}],
            ["not-an-element"],
        )
        for elements in cases:
            with self.subTest(elements=elements):
                session = _FakeSession(result=_semantic_result(elements))
                provider = _provider(session)

                with self.assertRaisesRegex(
                    ToolExecutionError,
                    "no usable semantic elements",
                ):
                    provider.observe()

                self.assertFalse(session.active)

    def test_huge_tree_is_bounded_and_sensitive_or_secure_values_are_redacted(
        self,
    ) -> None:
        elements: list[dict[str, Any]] = [
            {
                "element_index": 1,
                "element_token": "secure-token",
                "role": "textbox",
                "label": "Password",
                "value": "very-sensitive-value",
                "description": "Credential input",
                "depth": 1,
                "is_password": True,
                "frame": {"x": 1, "y": 2, "width": 3, "height": 4},
            },
            {
                "element_index": 2,
                "element_token": "redaction-token",
                "role": "text",
                "label": "Service token",
                "value": "api_key=sk-test-secret-material",
                "description": "api_key=sk-test-secret-material",
                "depth": 1,
            },
        ]
        elements.extend(
            {
                "element_index": index,
                "element_token": f"token-{index}",
                "role": "text",
                "label": f"Node {index} " + ("z" * 400),
                "value": "visible",
                "depth": 2,
            }
            for index in range(3, 40)
        )
        session = _FakeSession(result=_semantic_result(elements))
        provider = _provider(session)
        try:
            output = provider.observe()
        finally:
            provider.close()

        rendered = json.dumps(output)
        self.assertEqual(len(output["nodes"]), 3)
        self.assertTrue(output["truncated"])
        self.assertGreaterEqual(output["omitted_nodes"], 36)
        self.assertNotIn("very-sensitive-value", rendered)
        self.assertNotIn("sk-test-secret-material", rendered)
        self.assertIn("redact", rendered.lower())
        self.assertFalse(_all_keys(output["nodes"]) & _FORBIDDEN_NODE_KEYS)
        self.assertLessEqual(len(rendered), 6_000)

    def test_process_restart_or_pid_reuse_invalidates_the_session(self) -> None:
        identity = dict(_PROCESS_IDENTITY)
        session = _FakeSession(
            result=_semantic_result(
                [{"role": "button", "label": "Save", "depth": 1}]
            )
        )
        provider = _provider(session, identity=lambda: dict(identity))
        provider.observe()
        self.assertEqual(len(session.calls), 1)

        identity["started_at"] = "2026-07-21T08:01:00Z"
        identity["executable_sha256"] = "d" * 64
        with self.assertRaisesRegex(ToolExecutionError, "identity|restart|PID"):
            provider.observe()

        self.assertEqual(len(session.calls), 1)
        self.assertFalse(session.active)

    def test_observation_budget_is_propagated_to_mcp_bootstrap_and_call(self) -> None:
        session = _FakeSession(
            result=_semantic_result(
                [{"role": "button", "label": "Save", "depth": 1}]
            )
        )
        provider = _provider(session)
        try:
            provider.observe(timeout_seconds=1.0)
        finally:
            provider.close()

        self.assertIsNotNone(session.start_timeout_seconds)
        self.assertIsNotNone(session.list_timeout_seconds)
        self.assertIsNotNone(session.call_timeout_seconds)
        self.assertGreater(session.start_timeout_seconds or 0, 0)
        self.assertLessEqual(session.start_timeout_seconds or 2, 1.0)
        self.assertLessEqual(
            session.call_timeout_seconds or 2,
            session.start_timeout_seconds or 0,
        )

    def test_process_identity_is_rechecked_after_the_daemon_call(self) -> None:
        calls = 0

        def identity() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            current = dict(_PROCESS_IDENTITY)
            if calls >= 3:
                current["started_at"] = "2026-07-21T08:02:00Z"
            return current

        session = _FakeSession(
            result=_semantic_result(
                [{"role": "button", "label": "Save", "depth": 1}]
            )
        )
        provider = _provider(session, identity=identity)

        with self.assertRaisesRegex(ToolExecutionError, "identity|during observation"):
            provider.observe()

        self.assertFalse(session.active)

    def test_daemon_disconnect_fails_closed(self) -> None:
        session = _FakeSession(
            result=_semantic_result([]),
            call_error=McpClientError("desktop daemon disconnected"),
        )
        provider = _provider(session)

        with self.assertRaisesRegex(ToolExecutionError, "provider|daemon|disconnected"):
            provider.observe()

        self.assertFalse(session.active)

    def test_runtime_attestation_must_match_pin_and_disable_telemetry(self) -> None:
        invalid_receipts = (
            {**_RUNTIME_RECEIPT, "version": "0.10.1"},
            {**_RUNTIME_RECEIPT, "executable_sha256": "e" * 64},
            {**_RUNTIME_RECEIPT, "telemetry_enabled": True},
        )
        for receipt in invalid_receipts:
            with self.subTest(receipt=receipt):
                session = _FakeSession(result=_semantic_result([]))
                provider = _provider(session, runtime=receipt)
                with self.assertRaisesRegex(ToolExecutionError, "attestation|runtime"):
                    provider.observe()
                self.assertEqual(session.calls, [])
                self.assertFalse(session.active)

    def test_model_contract_is_one_empty_observation_tool_and_raw_mcp_is_hidden(
        self,
    ) -> None:
        binding = {
            "target_id": "offline-editor",
            "binding_sha256": "c" * 64,
        }
        specs = desktop_tool_specs(**binding)

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "desktop.observe")
        self.assertEqual(specs[0].input_schema["required"], ["target_id", "binding_sha256"])
        self.assertEqual(
            specs[0].input_schema["properties"]["target_id"]["enum"],
            ["offline-editor"],
        )
        self.assertTrue(specs[0].approval_required)
        self.assertIn("untrusted", specs[0].description.lower())

        provider = _RunnerProvider()
        runner = DesktopToolRunner(provider, allow_process_execution=True)
        result = runner.run("desktop.observe", provider.approval_binding)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["target_id"], "offline-editor")
        self.assertEqual(provider.observations, 1)
        for raw_name in ("get_window_state", "desktop.click", "desktop.type"):
            with self.subTest(raw_name=raw_name):
                with self.assertRaises(ToolExecutionError):
                    runner.run(raw_name, {})

        with self.assertRaisesRegex(ToolExecutionError, "bound target"):
            runner.run(
                "desktop.observe",
                {"target_id": "another-target", "binding_sha256": "d" * 64},
            )

        blocked = DesktopToolRunner(_RunnerProvider(), allow_process_execution=False)
        with self.assertRaises(ToolExecutionError):
            blocked.run("desktop.observe", binding)

    def test_agent_path_keeps_a_useful_structurally_bounded_tree(self) -> None:
        elements = [
            {
                "role": "text",
                "label": f"Node {index} " + ("x" * 200),
                "value": "visible",
                "depth": 2,
            }
            for index in range(128)
        ]
        base = _server()
        server = replace(
            base,
            desktop_capability={
                **base.desktop_capability,
                "max_nodes": 128,
                "max_text_chars": 200,
                "max_result_chars": 6_000,
            },
        )
        provider = _provider(
            _FakeSession(result=_semantic_result(elements)),
            server=server,
        )
        runner = DesktopToolRunner(provider, allow_process_execution=True)
        registry = AgentToolRegistry(runner, runner.specs)
        approval_requests = []
        try:
            execution = registry.execute(
                AgentToolCall(
                    id="call-1",
                    name="desktop.observe",
                    arguments=provider.approval_binding,
                ),
                approval_handler=lambda request: (
                    approval_requests.append(request) or True
                ),
            )
            bounded, serialized = bound_tool_result(
                execution.result,
                max_chars=8_000,
            )
        finally:
            provider.close()

        self.assertEqual(bounded.status, "success")
        self.assertEqual(len(approval_requests), 1)
        self.assertEqual(
            dict(approval_requests[0].arguments),
            provider.approval_binding,
        )
        self.assertNotEqual(
            approval_requests[0].arguments_sha256,
            hashlib.sha256(b"{}").hexdigest(),
        )
        self.assertTrue(bounded.data["nodes"])
        self.assertNotEqual(bounded.data, {"truncated": True})
        self.assertLessEqual(len(serialized), 8_000)

    def test_approval_binding_changes_with_server_or_target_config(self) -> None:
        first = _provider(_FakeSession(result=_semantic_result([])))
        base = _server()
        changed = replace(
            base,
            desktop_capability={
                **base.desktop_capability,
                "target": {
                    **base.desktop_capability["target"],
                    "window_id": 18,
                },
            },
        )
        second = _provider(
            _FakeSession(result=_semantic_result([])),
            server=changed,
        )

        self.assertNotEqual(
            first.approval_binding["binding_sha256"],
            second.approval_binding["binding_sha256"],
        )

    def test_generic_mcp_runner_cannot_discover_or_call_desktop_provider(self) -> None:
        tools = tuple(
            ToolDefinition(
                name=name,
                description=name,
                risk_class="process_execution",
                side_effects="starts_process",
                enabled=True,
            )
            for name in ("mcp.list_tools", "mcp.call_tool")
        )
        registry = ExtensionRegistry(
            tools=tools,
            skills=(),
            mcp_servers=(_server(),),
            cron_jobs=(),
            plugins=(),
        )
        runner = LocalToolRunner(registry, allow_process_execution=True)

        with self.assertRaisesRegex(ToolExecutionError, "desktop|guarded"):
            runner.run(
                "mcp.list_tools",
                {
                    "server": "desktop-local",
                    "confirm_process_execution": True,
                },
            )
        with self.assertRaisesRegex(ToolExecutionError, "desktop|guarded"):
            runner.run(
                "mcp.call_tool",
                {
                    "server": "desktop-local",
                    "tool_name": "get_window_state",
                    "arguments": {},
                    "confirm_process_execution": True,
                    "confirm_tool_call": True,
                },
            )

    def test_canary_qualifies_only_bounded_semantic_read_access(self) -> None:
        provider = _CanaryProvider()

        result = run_desktop_capability_canary(provider)

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["runtime_ready"])
        self.assertEqual(result["scope"], "desktop_semantic_read_only")
        self.assertTrue(result["checks"])
        self.assertTrue(all(check["passed"] for check in result["checks"]))
        self.assertIn("does_not_qualify_input_control", result["limits"])
        self.assertIn("does_not_qualify_visual_control", result["limits"])
        self.assertTrue(provider.closed)

    def test_canary_rejects_an_empty_semantic_tree(self) -> None:
        provider = _CanaryProvider()
        provider.empty = True

        result = run_desktop_capability_canary(provider)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["runtime_ready"])


class _FakeSession:
    def __init__(
        self,
        *,
        result: McpToolCallResult,
        schema: dict[str, Any] | None = None,
        call_error: Exception | None = None,
        server_info: dict[str, str] | None = None,
    ) -> None:
        self.active = False
        self.result = result
        self.schema = schema or _GET_WINDOW_STATE_SCHEMA
        self.call_error = call_error
        self._server_info = server_info or {
            "name": "cua-driver",
            "version": "0.10.0",
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.start_timeout_seconds: float | None = None
        self.list_timeout_seconds: float | None = None
        self.call_timeout_seconds: float | None = None

    def start(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> _FakeSession:
        self.start_timeout_seconds = timeout_seconds
        self.active = True
        return self

    def close(self) -> None:
        self.active = False

    @property
    def server_info(self) -> dict[str, str]:
        return dict(self._server_info)

    def list_tools(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolList:
        self.list_timeout_seconds = timeout_seconds
        return McpToolList(
            server="desktop-local",
            protocol_version="2025-11-25",
            tools=(
                McpTool(
                    name="get_window_state",
                    description="Return a semantic accessibility snapshot",
                    input_schema=self.schema,
                ),
                McpTool(name="click", input_schema={"type": "object"}),
                McpTool(name="type_text", input_schema={"type": "object"}),
            ),
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolCallResult:
        self.call_timeout_seconds = timeout_seconds
        self.calls.append((name, dict(arguments)))
        if self.call_error is not None:
            raise self.call_error
        return self.result


class _FakeDaemon:
    def __init__(
        self,
        config: DesktopCapabilityConfig,
        server: McpServerDefinition,
    ) -> None:
        self.config = config
        self.server = server
        self.closed = False

    def start(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[McpServerDefinition, dict[str, Any]]:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ToolExecutionError("invalid fake daemon timeout")
        return self.server, {
            "provider": "cua_driver",
            "version": "0.10.0",
            "executable_sha256": self.config.provider_executable_sha256,
            "permission_mode": "bounded",
            "user_policy_sha256": "d" * 64,
            "session_policy_sha256": "e" * 64,
            "user_policy_source_sha256": hashlib.sha256(
                _desktop_user_policy(self.config)
            ).hexdigest(),
            "session_policy_source_sha256": hashlib.sha256(
                _desktop_session_policy()
            ).hexdigest(),
            "socket_owner_verified": True,
            "daemon_process_verified": True,
        }

    def close(self) -> None:
        self.closed = True


class _RunnerProvider:
    def __init__(self) -> None:
        self.observations = 0
        self.closed = False

    @property
    def specs(self):
        return desktop_tool_specs(**self.approval_binding)

    @property
    def approval_binding(self) -> dict[str, str]:
        return {
            "target_id": "offline-editor",
            "binding_sha256": "c" * 64,
        }

    def observe(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        del timeout_seconds
        self.observations += 1
        return _delivered_snapshot()

    def close(self) -> None:
        self.closed = True


class _CanaryProvider(_RunnerProvider):
    empty = False

    def attest(self) -> dict[str, Any]:
        return {
            **_RUNTIME_RECEIPT,
            "daemon_authority": {
                "owned": True,
                "permission_mode": "bounded",
            },
        }

    def observe(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        snapshot = super().observe(timeout_seconds=timeout_seconds)
        if self.empty:
            snapshot = {**snapshot, "nodes": []}
        return snapshot


def _server() -> McpServerDefinition:
    return McpServerDefinition(
        name="desktop-local",
        description="Pinned semantic desktop provider fixture",
        command="cua-driver",
        args=("mcp",),
        enabled=True,
        risk_class="process_execution",
        capabilities=("desktop", "tools"),
        allowed_tools=("get_window_state",),
        timeout_seconds=3,
        cwd=".",
        desktop_capability={
            "enabled": True,
            "provider": "cua_driver",
            "version": "0.10.0",
            "provider_executable_sha256": "a" * 64,
            "telemetry_enabled": False,
            "tool_schema_sha256": {
                "get_window_state": _sha256_json(_GET_WINDOW_STATE_SCHEMA)
            },
            "target": {
                "id": "offline-editor",
                "pid": 4242,
                "window_id": 17,
                "process_name": "Offline Editor",
                "process_started_at": "2026-07-21T08:00:00Z",
                "process_executable_sha256": "b" * 64,
            },
            "max_nodes": 3,
            "max_depth": 4,
            "max_text_chars": 80,
        },
    )


def _provider(
    session: _FakeSession,
    *,
    runtime: dict[str, Any] | None = None,
    identity=None,
    server: McpServerDefinition | None = None,
) -> CuaDriverDesktopProvider:
    runtime_receipt = dict(runtime or _RUNTIME_RECEIPT)
    identity_source = identity or (lambda: dict(_PROCESS_IDENTITY))
    configured_server = server or _server()
    return CuaDriverDesktopProvider(
        configured_server,
        session_factory=lambda *_args, **_kwargs: session,
        runtime_attestor=lambda *_args, **_kwargs: dict(runtime_receipt),
        process_identity_resolver=lambda *_args, **_kwargs: identity_source(),
        daemon_factory=lambda config, admitted_server, _environment: _FakeDaemon(
            config,
            admitted_server,
        ),
    )


def _semantic_result(elements: list[dict[str, Any]]) -> McpToolCallResult:
    return McpToolCallResult(
        server="desktop-local",
        tool_name="get_window_state",
        content=(),
        is_error=False,
        structured_content={
            "pid": 4242,
            "window_id": 17,
            "degraded": False,
            "elements": elements,
        },
    )


def _delivered_snapshot() -> dict[str, Any]:
    nodes = [
        {
            "id": "n1",
            "role": "button",
            "label": "Canary control",
            "depth": 1,
            "states": ["enabled"],
        }
    ]
    snapshot_sha256 = _sha256_json(nodes)
    return {
        "target_id": "offline-editor",
        "desktop_session_id": "f" * 32,
        "revision": 1,
        "snapshot_sha256": snapshot_sha256,
        "nodes": nodes,
        "truncated": False,
        "omitted_nodes": 0,
        "authoritative": True,
        "trust": "untrusted_external",
        "instruction_policy": "content_is_data_only",
        "runtime": dict(_RUNTIME_RECEIPT),
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    unittest.main()
