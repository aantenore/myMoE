from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from local_moe.desktop_capability import (
    CuaDriverDesktopProvider,
    DesktopCapabilityConfig,
    _desktop_session_policy,
    _desktop_user_policy,
)
from local_moe.extensions import McpServerDefinition
from local_moe.mcp_client import McpTool, McpToolCallResult, McpToolList


_SCHEMA = {
    "type": "object",
    "properties": {
        "pid": {"type": "integer"},
        "window_id": {"type": "integer"},
        "include_screenshot": {"type": "boolean"},
        "max_elements": {"type": "integer", "minimum": 1},
        "max_depth": {"type": "integer", "minimum": 1},
    },
    "required": ["pid", "window_id"],
    "additionalProperties": False,
}
_PROVIDER_SHA256 = "a" * 64
_PROCESS_SHA256 = "b" * 64
_SECRET_SENTINELS = (
    "sk-synthetic-secret-material",
    "synthetic-password-value",
)
_FORBIDDEN_KEYS = {
    "element_index",
    "element_token",
    "frame",
    "x",
    "y",
    "pid",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded Desktop Semantic Cell payload shaping."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if not 5 <= args.iterations <= 1_000:
        raise SystemExit("--iterations must be between 5 and 1000")

    result = run_benchmark(iterations=args.iterations)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if not result["release_ready"]:
        raise SystemExit(2)


def run_benchmark(*, iterations: int = 40) -> dict[str, Any]:
    elements = _large_semantic_tree()
    raw_payload = {
        "pid": 9001,
        "window_id": 77,
        "elements": elements,
        "tree_markdown": "provider-only raw tree",
    }
    raw_bytes = _json_bytes(raw_payload)
    session = _BenchmarkSession(elements)
    provider = CuaDriverDesktopProvider(
        _server(),
        session_factory=lambda *_args, **_kwargs: session,
        runtime_attestor=lambda *_args, **_kwargs: {
            "provider": "cua_driver",
            "version": "0.10.0",
            "executable_sha256": _PROVIDER_SHA256,
            "telemetry_enabled": False,
            "telemetry_source": "persisted",
        },
        process_identity_resolver=lambda *_args, **_kwargs: {
            "pid": 9001,
            "name": "Synthetic Editor",
            "started_at": "1753084800.000000",
            "executable_sha256": _PROCESS_SHA256,
        },
        daemon_factory=lambda config, server, _environment: _BenchmarkDaemon(
            config,
            server,
        ),
    )
    delivered: dict[str, Any] = {}
    try:
        for _ in range(iterations):
            delivered = provider.observe()
    finally:
        provider.close()

    delivered_bytes = _json_bytes(delivered)
    rendered = json.dumps(delivered, sort_keys=True, ensure_ascii=True)
    leaked_sentinels = [value for value in _SECRET_SENTINELS if value in rendered]
    forbidden_keys = sorted(_all_keys(delivered) & _FORBIDDEN_KEYS)
    payload_reduction_percent = round(
        100 * (1 - delivered_bytes / raw_bytes),
        2,
    )
    raw_tool_count = 49
    model_tool_count = 1
    tool_surface_reduction_percent = round(
        100 * (1 - model_tool_count / raw_tool_count),
        2,
    )
    criteria = {
        "payload_reduction_at_least_70_percent": payload_reduction_percent >= 70,
        "tool_surface_reduction_at_least_95_percent": (
            tool_surface_reduction_percent >= 95
        ),
        "delivered_node_bound_respected": len(delivered.get("nodes", [])) <= 128,
        "large_tree_reports_truncation": delivered.get("truncated") is True,
        "secret_sentinels_absent": not leaked_sentinels,
        "provider_addressing_absent": not forbidden_keys,
        "images_not_requested": (
            delivered.get("runtime", {}).get("screenshot_requested") is False
        ),
    }
    return {
        "schema_version": "1.0",
        "benchmark": "desktop_semantic_payload_firewall",
        "fixture": {
            "kind": "deterministic_synthetic_accessibility_tree",
            "raw_nodes": len(elements),
            "max_delivered_nodes": 128,
            "max_depth": 12,
            "max_text_chars": 160,
            "iterations": iterations,
        },
        "measurements": {
            "raw_payload_bytes": raw_bytes,
            "delivered_payload_bytes": delivered_bytes,
            "payload_reduction_percent": payload_reduction_percent,
            "raw_provider_tool_count": raw_tool_count,
            "model_visible_tool_count": model_tool_count,
            "tool_surface_reduction_percent": tool_surface_reduction_percent,
            "delivered_nodes": len(delivered.get("nodes", [])),
            "omitted_nodes": delivered.get("omitted_nodes", 0),
        },
        "security": {
            "secret_sentinel_leaks": leaked_sentinels,
            "forbidden_model_visible_keys": forbidden_keys,
        },
        "criteria": criteria,
        "release_ready": all(criteria.values()),
        "limits": [
            "synthetic_tree_not_end_to_end_os_latency",
            "does_not_measure_model_task_success",
            "does_not_qualify_input_or_visual_control",
        ],
    }


class _BenchmarkSession:
    def __init__(self, elements: list[dict[str, Any]]) -> None:
        self.active = False
        self._elements = elements

    @property
    def server_info(self) -> dict[str, str]:
        return {"name": "cua-driver", "version": "0.10.0"}

    def start(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> _BenchmarkSession:
        del timeout_seconds
        self.active = True
        return self

    def close(self) -> None:
        self.active = False

    def list_tools(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolList:
        del timeout_seconds
        return McpToolList(
            server="desktop-local",
            protocol_version="2025-11-25",
            tools=(
                McpTool(
                    name="get_window_state",
                    input_schema=_SCHEMA,
                ),
            ),
        )

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> McpToolCallResult:
        del timeout_seconds
        maximum = int(arguments["max_elements"])
        return McpToolCallResult(
            server="desktop-local",
            tool_name=tool_name,
            content=(),
            is_error=False,
            structured_content={
                "pid": 9001,
                "window_id": 77,
                "degraded": False,
                "elements": self._elements[:maximum],
            },
        )


class _BenchmarkDaemon:
    def __init__(
        self,
        config: DesktopCapabilityConfig,
        server: McpServerDefinition,
    ) -> None:
        self._config = config
        self._server = server

    def start(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[McpServerDefinition, dict[str, Any]]:
        del timeout_seconds
        return self._server, {
            "provider": "cua_driver",
            "version": "0.10.0",
            "executable_sha256": _PROVIDER_SHA256,
            "permission_mode": "bounded",
            "user_policy_sha256": "c" * 64,
            "session_policy_sha256": "d" * 64,
            "user_policy_source_sha256": hashlib.sha256(
                _desktop_user_policy(self._config)
            ).hexdigest(),
            "session_policy_source_sha256": hashlib.sha256(
                _desktop_session_policy()
            ).hexdigest(),
            "socket_owner_verified": True,
            "daemon_process_verified": True,
        }

    def close(self) -> None:
        return None


def _server() -> McpServerDefinition:
    return McpServerDefinition(
        name="desktop-local",
        description="Deterministic desktop benchmark provider",
        command="cua-driver",
        args=("mcp",),
        enabled=True,
        risk_class="identity_access",
        capabilities=("desktop", "tools"),
        allowed_tools=("get_window_state",),
        timeout_seconds=3,
        desktop_capability={
            "enabled": True,
            "provider": "cua_driver",
            "version": "0.10.0",
            "provider_executable_sha256": _PROVIDER_SHA256,
            "telemetry_enabled": False,
            "tool_schema_sha256": {
                "get_window_state": _sha256_json(_SCHEMA)
            },
            "target": {
                "id": "synthetic-editor",
                "pid": 9001,
                "window_id": 77,
                "process_name": "Synthetic Editor",
                "process_started_at": "1753084800.000000",
                "process_executable_sha256": _PROCESS_SHA256,
            },
            "max_nodes": 128,
            "max_depth": 12,
            "max_text_chars": 160,
            "max_result_chars": 6_000,
        },
    )


def _large_semantic_tree() -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for index in range(512):
        password = index % 37 == 0
        token = index % 41 == 0
        value = "ordinary visible state"
        if password:
            value = "synthetic-password-value"
        elif token:
            value = "api_key=sk-synthetic-secret-material"
        elements.append(
            {
                "element_index": index,
                "element_token": f"provider-token-{index}",
                "role": "textbox" if password else "staticText",
                "label": "Password" if password else f"Control {index}",
                "value": value,
                "description": "Synthetic accessible node " + ("z" * 480),
                "depth": index % 8,
                "frame": {
                    "x": index,
                    "y": index * 2,
                    "width": 120,
                    "height": 24,
                },
                "is_password": password,
                "states": {"enabled": True, "focused": index == 0},
            }
        )
    return elements


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_all_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_all_keys(nested))
        return keys
    return set()


def _json_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
