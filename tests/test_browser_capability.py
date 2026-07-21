from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterator, Mapping
from dataclasses import replace
import hashlib
import http.client
import json
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from local_moe.agent_tools import AgentPermissionPolicy, AgentToolRegistry
from local_moe.agent_types import AgentToolCall
from local_moe.browser_capability import (
    PLAYWRIGHT_TOOLS,
    BrowserCapabilityConfig,
    PlaywrightMcpBrowserProvider,
    _CanaryServer,
    _ExactOriginProxy,
    _browser_process_environment,
    _filtered_response_headers,
    _parse_local_origin,
    _resolve_node_cli_launcher,
    _validated_response_status,
    _verify_browser_runtime,
    browser_tool_specs,
    run_browser_capability_canary,
)
from local_moe.extensions import McpServerDefinition
from local_moe.mcp_client import StdioMcpClient
from local_moe.tool_runner import ToolExecutionError, ToolRunResult


_SCHEMAS = {
    "browser_navigate": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    "browser_snapshot": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "browser_click": {
        "type": "object",
        "properties": {
            "element": {"type": "string"},
            "target": {"type": "string"},
        },
        "required": ["target"],
        "additionalProperties": False,
    },
    "browser_type": {
        "type": "object",
        "properties": {
            "element": {"type": "string"},
            "target": {"type": "string"},
            "text": {"type": "string"},
            "submit": {"type": "boolean"},
            "slowly": {"type": "boolean"},
        },
        "required": ["target", "text"],
        "additionalProperties": False,
    },
}


class BrowserCapabilityTests(unittest.TestCase):
    def test_persistent_local_browser_flow_returns_sticky_untrusted_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = _fake_server(Path(tmp))
            provider = _fake_provider(Path(tmp), server=server)
            try:
                navigated = provider.execute(
                    "navigate", {"url": "http://127.0.0.1:4312/app"}
                )
                observed = provider.observe()
                clicked = provider.execute(
                    "click",
                    {
                        **_binding(observed),
                        "target": "e1",
                        "target_label": _target_label(observed["snapshot"], "e1"),
                    },
                )
                typed = provider.execute(
                    "type",
                    {
                        **_binding(clicked),
                        "target": "e2",
                        "target_label": _target_label(clicked["snapshot"], "e2"),
                        "text": "offline test",
                    },
                )
            finally:
                provider.close()

        self.assertEqual(navigated["trust"], "untrusted_external")
        self.assertEqual(navigated["instruction_policy"], "content_is_data_only")
        self.assertEqual(navigated["revision"], 1)
        self.assertEqual(observed["revision"], 2)
        self.assertIn("initial", observed["snapshot"])
        self.assertEqual(clicked["revision"], 3)
        self.assertIn("clicked", clicked["snapshot"])
        self.assertEqual(typed["revision"], 4)
        self.assertIn("offline test", typed["snapshot"])
        self.assertEqual(
            typed["runtime"]["network_scope"], "exact_loopback_http_origin"
        )
        self.assertRegex(typed["runtime"]["executable_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            typed["runtime"]["node_executable_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            typed["runtime"]["configured_launch_arguments_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            typed["runtime"]["effective_launch_arguments_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_stale_snapshot_target_fails_before_upstream_interaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _fake_provider(Path(tmp))
            try:
                provider.execute("navigate", {"url": "http://localhost:4312/app"})
                observed = provider.observe()
                with self.assertRaisesRegex(ToolExecutionError, "snapshot revision is stale"):
                    provider.execute(
                        "click",
                        {
                            **_binding(observed),
                            "revision": 1,
                            "target": "e1",
                            "target_label": _target_label(observed["snapshot"], "e1"),
                        },
                    )
            finally:
                provider.close()

        with tempfile.TemporaryDirectory() as tmp:
            provider = _fake_provider(Path(tmp))
            try:
                provider.execute("navigate", {"url": "http://localhost:4312/app"})
                observed = provider.observe()
                with self.assertRaisesRegex(ToolExecutionError, "not present"):
                    provider.execute(
                        "click",
                        {
                            **_binding(observed),
                            "target": "e999",
                            "target_label": "missing",
                        },
                    )
            finally:
                provider.close()

    def test_external_and_file_urls_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _fake_provider(Path(tmp))
            try:
                for url in (
                    "https://example.com/",
                    "file:///tmp/page.html",
                    "data:text/html,hello",
                    "http://user:pass@localhost:4312/",
                ):
                    with self.subTest(url=url):
                        with self.assertRaises(ToolExecutionError):
                            provider.execute("navigate", {"url": url})
                self.assertFalse(provider._session)
            finally:
                provider.close()

    def test_redirect_to_external_origin_closes_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _fake_provider(Path(tmp))
            with self.assertRaises(ToolExecutionError):
                provider.execute(
                    "navigate", {"url": "http://127.0.0.1:4312/redirect"}
                )
            self.assertIsNone(provider._session)
            provider.close()

    def test_schema_drift_is_rejected_before_any_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = _fake_server(Path(tmp), expected_schema_drift=True)
            provider = _fake_provider(Path(tmp), server=server)
            with self.assertRaisesRegex(ToolExecutionError, "schema drift"):
                provider.execute(
                    "navigate",
                    {"url": "http://127.0.0.1:4312/app"},
                )
            self.assertIsNone(provider._session)
            provider.close()

    def test_model_sees_only_narrow_contracts_and_every_call_needs_approval(self) -> None:
        specs = browser_tool_specs()
        self.assertEqual({spec.name for spec in specs}, set(PLAYWRIGHT_TOOLS))
        self.assertTrue(all(spec.approval_required for spec in specs))
        self.assertNotIn("browser_run_code", {spec.name for spec in specs})

        registry = AgentToolRegistry(
            _SuccessfulRunner(),
            specs,
            permission_policy=AgentPermissionPolicy(
                auto_allow_risks=("process_execution",),
                approval_required_risks=(),
            ),
        )
        execution = registry.execute(
            AgentToolCall(
                id="call-1",
                name="browser__navigate",
                arguments={"url": "http://localhost:4312/app"},
            )
        )
        self.assertEqual(execution.result.status, "approval_required")
        self.assertTrue(execution.pause_required)

    def test_configuration_rejects_unsafe_launch_and_non_loopback_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unsafe = _fake_server(root, extra_args=("--extension",))
            with self.assertRaises(ToolExecutionError):
                BrowserCapabilityConfig.from_server(unsafe)

            nonlocal_config = dict(unsafe.browser_capability)
            nonlocal_config["allowed_hosts"] = ["192.168.1.20"]
            nonlocal_server = McpServerDefinition(
                **{**unsafe.__dict__, "browser_capability": nonlocal_config, "args": unsafe.args[:-1]}
            )
            with self.assertRaises(ToolExecutionError):
                BrowserCapabilityConfig.from_server(nonlocal_server)

            injected_environment = replace(
                _fake_server(root),
                env={"NODE_OPTIONS": "--require ./untrusted.js"},
            )
            with self.assertRaisesRegex(ToolExecutionError, "harness-owned"):
                BrowserCapabilityConfig.from_server(injected_environment)

            for timeout in (9, 181):
                with self.subTest(archive_verification_timeout_seconds=timeout):
                    server = _fake_server(root)
                    browser_config = dict(server.browser_capability)
                    browser_config["archive_verification_timeout_seconds"] = timeout
                    with self.assertRaisesRegex(
                        ToolExecutionError,
                        "archive_verification_timeout_seconds",
                    ):
                        BrowserCapabilityConfig.from_server(
                            replace(server, browser_capability=browser_config)
                        )

            for timeout in (10.5, "60", True):
                with self.subTest(archive_verification_timeout_type=type(timeout)):
                    server = _fake_server(root)
                    browser_config = dict(server.browser_capability)
                    browser_config["archive_verification_timeout_seconds"] = timeout
                    with self.assertRaisesRegex(
                        ToolExecutionError,
                        "must be an integer",
                    ):
                        BrowserCapabilityConfig.from_server(
                            replace(server, browser_capability=browser_config)
                        )

    def test_runtime_archive_verification_uses_configured_bounded_timeout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "npm-cache"
            cache.mkdir()
            server = _fake_server(root)
            browser_config = dict(server.browser_capability)
            browser_config["archive_verification_timeout_seconds"] = 73
            config = BrowserCapabilityConfig.from_server(
                replace(server, browser_capability=browser_config)
            )
            pack_timeouts: list[object] = []

            def fake_run(command, **kwargs):
                if "process.execPath" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=f"{sys.executable}\n",
                        stderr="",
                    )
                if "--version" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout="v24.0.0\n",
                        stderr="",
                    )
                pack_timeouts.append(kwargs.get("timeout"))
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

            with (
                patch(
                    "local_moe.browser_capability.shutil.which",
                    return_value=sys.executable,
                ),
                patch(
                    "local_moe.browser_capability.subprocess.run",
                    side_effect=fake_run,
                ),
                self.assertRaisesRegex(ToolExecutionError, "offline npm cache"),
            ):
                _verify_browser_runtime(
                    config,
                    {
                        "HOME": str(root),
                        "NPM_CONFIG_CACHE": str(cache),
                        "PATH": str(root),
                        "SystemRoot": str(root),
                    },
                )

            self.assertEqual(pack_timeouts, [73])
            self.assertNotEqual(
                config.digest,
                replace(config, archive_verification_timeout_seconds=60).digest,
            )

    def test_package_archive_integrity_is_computed_not_trusted_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "npm-cache"
            cache.mkdir()
            config = BrowserCapabilityConfig.from_server(_fake_server(root))

            def fake_run(command, **_kwargs):
                if "process.execPath" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=f"{sys.executable}\n",
                        stderr="",
                    )
                if "--version" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout="v20.12.0\n",
                        stderr="",
                    )
                destination = Path(command[command.index("--pack-destination") + 1])
                archive = destination / "playwright-mcp.tgz"
                archive.write_bytes(b"tampered archive")
                metadata = [
                    {
                        "name": config.package,
                        "version": config.version,
                        "integrity": config.package_integrity,
                        "filename": archive.name,
                    }
                ]
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(metadata),
                    stderr="",
                )

            with (
                patch(
                    "local_moe.browser_capability.shutil.which",
                    return_value=sys.executable,
                ),
                patch(
                    "local_moe.browser_capability.subprocess.run",
                    side_effect=fake_run,
                ),
                self.assertRaisesRegex(ToolExecutionError, "integrity verification failed"),
            ):
                _verify_browser_runtime(
                    config,
                    {
                        "HOME": str(root),
                        "NPM_CONFIG_CACHE": str(cache),
                        "PATH": str(root),
                    },
                )

    def test_windows_batch_shim_resolves_to_node_and_npx_cli_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Node & Tools" / "Program Files"
            cli = root / "node_modules" / "npm" / "bin" / "npx-cli.js"
            cli.parent.mkdir(parents=True)
            cli.write_text("// fixture\n", encoding="utf-8")
            node = root / "node.exe"
            node.write_bytes(b"node fixture")
            shim = root / "npx.cmd"
            shim.write_text("fixture\n", encoding="utf-8")

            def resolver(name: str) -> str | None:
                return {"npx": str(shim), "node": str(node)}.get(name)

            with patch(
                "local_moe.browser_capability.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [str(node), "-p", "process.execPath"],
                    0,
                    stdout=f"{node}\n",
                    stderr="",
                ),
            ):
                launcher = _resolve_node_cli_launcher(
                    "npx",
                    "npx-cli.js",
                    platform="win32",
                    source_environment={},
                    executable_resolver=resolver,
                )

            self.assertEqual(launcher.command, node.resolve())
            self.assertEqual(launcher.prefix_args, (str(cli.resolve()),))
            self.assertNotIn(".cmd", " ".join((str(launcher.command), *launcher.prefix_args)))

    def test_windows_environment_keeps_case_insensitive_system_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "npm-cache"
            cache.mkdir()
            source = _CaseInsensitiveEnvironment(
                {
                    "PATH": str(root),
                    "SYSTEMROOT": str(root / "Windows"),
                }
            )
            isolated_source = ChainMap(
                {"NPM_CONFIG_CACHE": str(cache)},
                source,
            )

            environment = _browser_process_environment(
                isolated_source,
                platform="win32",
                npm_user_config=(root / "browser.npmrc").resolve(),
            )

            self.assertEqual(environment["SystemRoot"], source["SYSTEMROOT"])
            self.assertEqual(environment["NPM_CONFIG_CACHE"], str(cache.resolve()))

    def test_runtime_rejects_node_18_for_the_pinned_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "npm-cache"
            cache.mkdir()
            config = BrowserCapabilityConfig.from_server(_fake_server(root))

            def fake_run(command, **_kwargs):
                if "process.execPath" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=f"{sys.executable}\n",
                        stderr="",
                    )
                if "--version" in command:
                    return subprocess.CompletedProcess(
                        command, 0, stdout="v18.20.0\n", stderr=""
                    )
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

            with (
                patch(
                    "local_moe.browser_capability.shutil.which",
                    return_value=sys.executable,
                ),
                patch(
                    "local_moe.browser_capability.subprocess.run",
                    side_effect=fake_run,
                ),
                self.assertRaisesRegex(ToolExecutionError, "Node.js 20 or newer"),
            ):
                _verify_browser_runtime(
                    config,
                    {
                        "HOME": str(root),
                        "NPM_CONFIG_CACHE": str(cache),
                        "PATH": str(root),
                    },
                )

    def test_prefetch_accepts_explicit_network_bootstrap_without_runtime_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "npm-cache"
            cache.mkdir()
            source = {
                "HOME": str(root),
                "PATH": str(root),
                "NPM_CONFIG_CACHE": str(cache),
                "HTTPS_PROXY": "http://proxy.internal:8080",
                "NPM_CONFIG_REGISTRY": "https://registry.internal/",
                "NODE_EXTRA_CA_CERTS": str(root / "enterprise-ca.pem"),
                "NODE_AUTH_TOKEN": "must-not-cross-the-boundary",
            }

            prefetch = _browser_process_environment(
                source,
                platform=sys.platform,
                npm_user_config=(root / "prefetch.npmrc").resolve(),
                offline=False,
                admission_network=True,
            )
            runtime = _browser_process_environment(
                source,
                platform=sys.platform,
                npm_user_config=(root / "runtime.npmrc").resolve(),
            )

            self.assertEqual(prefetch["HTTPS_PROXY"], source["HTTPS_PROXY"])
            self.assertEqual(
                prefetch["NPM_CONFIG_REGISTRY"], source["NPM_CONFIG_REGISTRY"]
            )
            self.assertNotIn("NODE_AUTH_TOKEN", prefetch)
            self.assertNotIn("HTTPS_PROXY", runtime)
            self.assertNotIn("NPM_CONFIG_REGISTRY", runtime)

    def test_exact_origin_proxy_forwards_one_port_and_blocks_another(self) -> None:
        with _CanaryServer(b"allowed") as allowed, _CanaryServer(b"forbidden") as forbidden:
            origin = _parse_local_origin(
                allowed.url,
                ("127.0.0.1",),
            )
            proxy = _ExactOriginProxy(origin)
            proxy.start()
            try:
                proxy_url = urlsplit(proxy.url)
                connection = http.client.HTTPConnection(
                    proxy_url.hostname,
                    proxy_url.port,
                    timeout=3,
                )
                connection.request("GET", allowed.url)
                allowed_response = connection.getresponse()
                self.assertEqual(allowed_response.status, 200)
                self.assertEqual(allowed_response.read(), b"allowed")
                connection.close()

                connection = http.client.HTTPConnection(
                    proxy_url.hostname,
                    proxy_url.port,
                    timeout=3,
                )
                connection.request("GET", forbidden.url)
                blocked_response = connection.getresponse()
                self.assertEqual(blocked_response.status, 403)
                blocked_response.read()
                connection.close()
            finally:
                proxy.close()

            self.assertGreaterEqual(allowed.hits, 1)
            self.assertEqual(forbidden.hits, 0)

    def test_response_headers_reject_splitting_controls_before_relay(self) -> None:
        unsafe_headers = (
            (("X-Test\r\nInjected", "value"),),
            (("X-Test\rInjected", "value"),),
            (("X-Test\nInjected", "value"),),
            (("X-Test:Injected", "value"),),
            (("X-Test", "value\r\nInjected: true"),),
            (("X-Test", "value\rInjected: true"),),
            (("X-Test", "value\nInjected: true"),),
            (("X-Test", "value\x00"),),
            (("X-Test", "value\x7f"),),
            (("X-Test", "value\u0100"),),
            ((object(), "value"),),
            (("X-Test", object()),),
        )
        for headers in unsafe_headers:
            with self.subTest(headers=headers), self.assertRaisesRegex(
                ToolExecutionError,
                "response header",
            ):
                _filtered_response_headers(headers)

        self.assertEqual(
            _filtered_response_headers(
                (
                    ("Content-Type", "text/plain; charset=utf-8"),
                    ("Connection", "X-Hop"),
                    ("X-Hop", "discarded"),
                    ("X-Safe", "visible\tvalue"),
                )
            ),
            (
                ("Content-Type", "text/plain; charset=utf-8"),
                ("X-Safe", "visible\tvalue"),
            ),
        )

    def test_proxy_rejects_malicious_upstream_headers_before_response_start(self) -> None:
        payload = (
            b"HTTP/1.1 200 UPSTREAM-SENTINEL\r\n"
            b"X-Test: safe\r\n"
            b" Injected: true\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n\r\n"
            b"ok"
        )

        class RawHandler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.recv(65_536)
                self.request.sendall(payload)

        class RawServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        with RawServer(("127.0.0.1", 0), RawHandler) as upstream:
            upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            upstream_thread.start()
            upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/"
            origin = _parse_local_origin(upstream_url, ("127.0.0.1",))
            proxy = _ExactOriginProxy(origin)
            proxy.start()
            try:
                proxy_url = urlsplit(proxy.url)
                with socket.create_connection(
                    (str(proxy_url.hostname), int(proxy_url.port or 0)),
                    timeout=3,
                ) as client:
                    client.sendall(
                        (
                            f"GET {upstream_url} HTTP/1.1\r\n"
                            f"Host: {origin.authority}\r\n"
                            "Connection: close\r\n\r\n"
                        ).encode("ascii")
                    )
                    chunks: list[bytes] = []
                    while True:
                        chunk = client.recv(65_536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                received = b"".join(chunks)
            finally:
                proxy.close()
                upstream.shutdown()
                upstream_thread.join(timeout=2)

        self.assertTrue(received.startswith(b"HTTP/1.1 403 "), received)
        self.assertEqual(received.count(b"HTTP/1.1 "), 1)
        self.assertNotIn(b"UPSTREAM-SENTINEL", received)
        self.assertNotIn(b"Injected", received)
        self.assertTrue(received.endswith(b"origin blocked\n"), received)

    def test_proxy_rejects_invalid_upstream_statuses(self) -> None:
        for status in (199, 600, True, "200", None):
            with self.subTest(status=status), self.assertRaisesRegex(
                ToolExecutionError,
                "response status",
            ):
                _validated_response_status(status)
        self.assertEqual(_validated_response_status(200), 200)
        self.assertEqual(_validated_response_status(599), 599)

    def test_proxy_never_relays_bodies_for_bodyless_statuses(self) -> None:
        for status in (204, 205, 304):
            with self.subTest(status=status):
                payload = (
                    f"HTTP/1.1 {status} UPSTREAM-SENTINEL\r\n"
                    "Content-Length: 13\r\n"
                    "Connection: close\r\n\r\n"
                    "body-sentinel"
                ).encode("ascii")

                class RawHandler(socketserver.BaseRequestHandler):
                    def handle(self) -> None:
                        self.request.recv(65_536)
                        self.request.sendall(payload)

                class RawServer(socketserver.ThreadingTCPServer):
                    allow_reuse_address = True
                    daemon_threads = True

                with RawServer(("127.0.0.1", 0), RawHandler) as upstream:
                    upstream_thread = threading.Thread(
                        target=upstream.serve_forever,
                        daemon=True,
                    )
                    upstream_thread.start()
                    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/"
                    origin = _parse_local_origin(upstream_url, ("127.0.0.1",))
                    proxy = _ExactOriginProxy(origin)
                    proxy.start()
                    try:
                        proxy_url = urlsplit(proxy.url)
                        with socket.create_connection(
                            (str(proxy_url.hostname), int(proxy_url.port or 0)),
                            timeout=3,
                        ) as client:
                            client.sendall(
                                (
                                    f"GET {upstream_url} HTTP/1.1\r\n"
                                    f"Host: {origin.authority}\r\n"
                                    "Connection: close\r\n\r\n"
                                ).encode("ascii")
                            )
                            chunks: list[bytes] = []
                            while True:
                                chunk = client.recv(65_536)
                                if not chunk:
                                    break
                                chunks.append(chunk)
                        received = b"".join(chunks)
                    finally:
                        proxy.close()
                        upstream.shutdown()
                        upstream_thread.join(timeout=2)

                response_headers, separator, response_body = received.partition(
                    b"\r\n\r\n"
                )
                self.assertEqual(separator, b"\r\n\r\n")
                self.assertTrue(
                    response_headers.startswith(f"HTTP/1.1 {status} ".encode("ascii")),
                    received,
                )
                self.assertNotIn(b"Content-Length:", response_headers)
                self.assertEqual(response_body, b"")

    def test_state_binding_tampering_invalidates_the_entire_session(self) -> None:
        mutations = {
            "browser_session_id": "f" * 32,
            "origin": "http://127.0.0.1:9999",
            "snapshot_sha256": "f" * 64,
            "target_label": "different target",
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                provider = _fake_provider(Path(tmp))
                observed = provider.execute(
                    "navigate",
                    {"url": "http://127.0.0.1:4312/app"},
                )
                payload = {
                    **_binding(observed),
                    "target": "e1",
                    "target_label": _target_label(observed["snapshot"], "e1"),
                }
                payload[field] = value
                with self.assertRaises(ToolExecutionError):
                    provider.execute("click", payload)
                self.assertIsNone(provider._session)

    def test_dom_change_between_approval_and_action_fails_before_click(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _fake_provider(root)
            observed = provider.execute(
                "navigate",
                {"url": "http://127.0.0.1:4312/mutate"},
            )
            with self.assertRaisesRegex(ToolExecutionError, "state changed"):
                provider.execute(
                    "click",
                    {
                        **_binding(observed),
                        "target": "e1",
                        "target_label": _target_label(observed["snapshot"], "e1"),
                    },
                )
            self.assertFalse((root / "browser_click.marker").exists())
            self.assertIsNone(provider._session)

    def test_upstream_action_and_post_observation_errors_close_the_session(self) -> None:
        for path in ("action-error", "post-error"):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp:
                provider = _fake_provider(Path(tmp))
                observed = provider.execute(
                    "navigate",
                    {"url": f"http://127.0.0.1:4312/{path}"},
                )
                with self.assertRaisesRegex(ToolExecutionError, "action error"):
                    provider.execute(
                        "click",
                        {
                            **_binding(observed),
                            "target": "e1",
                            "target_label": _target_label(observed["snapshot"], "e1"),
                        },
                    )
                self.assertIsNone(provider._session)

    def test_canary_receipt_qualifies_only_the_local_browser_scope(self) -> None:
        result = run_browser_capability_canary(_CanaryProvider())

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["runtime_ready"])
        self.assertEqual(result["scope"], "local_web_apps_only")
        self.assertEqual(
            [item["name"] for item in result["checks"]],
            ["navigate", "observe", "type", "click", "exact_origin_egress"],
        )
        self.assertIn("does_not_qualify_desktop_control", result["limits"])
        self.assertIn("does_not_qualify_host_network_containment", result["limits"])

    def test_canary_does_not_retry_failed_runtime_attestation(self) -> None:
        provider = _FailingAttestationCanaryProvider()

        result = run_browser_capability_canary(provider)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["checks"][0]["name"], "runtime")
        self.assertEqual(provider.attestation_attempts, 1)


class _SuccessfulRunner:
    def run(
        self,
        name: str,
        payload: dict[str, object] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolRunResult:
        return ToolRunResult(
            name=name,
            status="ok",
            risk_class="process_execution",
            side_effects="test",
            message="ok",
            payload={},
        )


class _CanaryProvider:
    def __init__(self) -> None:
        self.revision = 0
        self.url = ""

    @property
    def specs(self):
        return browser_tool_specs()

    def attest(self):
        return {
            "provider": "fake",
            "version": "1.0.0",
            "config_sha256": "a" * 64,
            "executable_sha256": "b" * 64,
        }

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def observe(self, *, timeout_seconds=None):
        self.revision += 1
        return self._payload(
            '- textbox "Name" [ref=e1]\n- button "Confirm" [ref=e2]\n- status: idle'
        )

    def execute(self, action, payload, *, timeout_seconds=None):
        self.revision += 1
        if action == "navigate":
            self.url = payload["url"]
            snapshot = '- textbox "Name" [ref=e1]\n- button "Confirm" [ref=e2]'
        elif action == "type":
            snapshot = '- textbox "Name" [ref=e1]\n- button "Confirm" [ref=e2]'
        else:
            snapshot = '- status: ready:offline\n- button "Confirm" [ref=e2]'
        return self._payload(snapshot)

    def _payload(self, snapshot):
        encoded = snapshot.encode("utf-8")
        return {
            "url": self.url,
            "origin": self.url.rstrip("/"),
            "browser_session_id": "a" * 32,
            "revision": self.revision,
            "snapshot": snapshot,
            "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
            "blocked_egress_attempts": 1,
            "runtime": self.attest(),
        }


class _FailingAttestationCanaryProvider(_CanaryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.attestation_attempts = 0

    def attest(self):
        self.attestation_attempts += 1
        raise ToolExecutionError("bounded runtime attestation failure")


def _fake_server(
    root: Path,
    *,
    expected_schema_drift: bool = False,
    extra_args: tuple[str, ...] = (),
) -> McpServerDefinition:
    script = root / "fake_browser_mcp.py"
    script.write_text(
        _fake_server_source(root / "browser_click.marker"),
        encoding="utf-8",
    )
    configured_schemas = dict(_SCHEMAS)
    if expected_schema_drift:
        configured_schemas["browser_click"] = {
            **configured_schemas["browser_click"],
            "description": "configured old schema",
        }
    digests = {
        name: _sha256_json(schema) for name, schema in configured_schemas.items()
    }
    return McpServerDefinition(
        name="browser-local",
        description="Deterministic browser provider fixture",
        command="npx",
        args=(
            "--offline",
            "-y",
            "@playwright/mcp@0.0.78",
            "--browser",
            "chrome",
            "--headless",
            "--isolated",
            "--block-service-workers",
            "--image-responses",
            "omit",
            "--output-mode",
            "stdout",
            "--codegen",
            "none",
            "--sandbox",
            *extra_args,
        ),
        enabled=True,
        risk_class="process_execution",
        capabilities=("browser", "tools"),
        allowed_tools=tuple(PLAYWRIGHT_TOOLS.values()),
        timeout_seconds=3,
        cwd=".",
        browser_capability={
            "enabled": True,
            "provider": "playwright_mcp",
            "package": "@playwright/mcp",
            "version": "0.0.78",
            "package_integrity": (
                "sha512-XLTUeA6mEN9sQ+hJ4dfG8EIkDbxS0K3Trc2RBkUJuf02TgE2FQRNTMtq/"
                "aJfhyRMINsRl/Ybc4sxcWLtFn4/TQ=="
            ),
            "allowed_hosts": ["localhost", "127.0.0.1", "::1"],
            "tool_schema_sha256": digests,
            "max_result_chars": 4_096,
        },
    )


def _fake_provider(
    root: Path,
    *,
    server: McpServerDefinition | None = None,
) -> PlaywrightMcpBrowserProvider:
    configured = server or _fake_server(root)
    script = root / "fake_browser_mcp.py"

    def session_factory(launched, environment):
        fake = replace(
            launched,
            command=sys.executable,
            args=(str(script),),
        )
        return StdioMcpClient(
            fake,
            timeout_seconds=fake.timeout_seconds,
            base_environment=environment,
        ).session()

    def runtime_attestor(_config, _environment):
        return {
            "package_archive_sha512": _config.package_integrity,
            "node_version": "v24.0.0",
            "node_executable_sha256": hashlib.sha256(
                Path(sys.executable).read_bytes()
            ).hexdigest(),
            "npm_cache_sha256": "c" * 64,
        }

    def fake_process_environment(source, platform, config):
        cache = config.with_name("npm-cache")
        cache.mkdir(exist_ok=True)
        isolated_source = ChainMap(
            {"NPM_CONFIG_CACHE": str(cache)},
            source,
        )
        return _browser_process_environment(
            isolated_source,
            platform=platform,
            npm_user_config=config,
        )

    return PlaywrightMcpBrowserProvider(
        configured,
        runtime_attestor=runtime_attestor,
        session_factory=session_factory,
        executable_resolver=lambda _command: sys.executable,
        environment_factory=fake_process_environment,
    )


def _binding(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in (
            "browser_session_id",
            "origin",
            "revision",
            "snapshot_sha256",
        )
    }


class _CaseInsensitiveEnvironment(Mapping[str, str]):
    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = {key.upper(): value for key, value in values.items()}

    def __getitem__(self, key: str) -> str:
        return self._values[key.upper()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _target_label(snapshot: object, target: str) -> str:
    if not isinstance(snapshot, str):
        raise AssertionError("snapshot must be text")
    for line in snapshot.splitlines():
        if f"[ref={target}]" in line:
            return " ".join(line.strip(" -").split())
    raise AssertionError(f"target not found: {target}")


def _fake_server_source(click_marker: Path) -> str:
    schemas = json.dumps(_SCHEMAS, sort_keys=True)
    return f'''import json
from pathlib import Path
import sys
schemas = json.loads({json.dumps(schemas)})
click_marker = Path({json.dumps(str(click_marker))})
current_url = ""
state = "initial"
mutate_on_snapshot = False
action_error = False
post_error_mode = False
post_error_next = False
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        result = {{"protocolVersion": "2025-11-25", "capabilities": {{"tools": {{}}}}, "serverInfo": {{"name": "fake-browser", "version": "1.0.0"}}}}
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        result = {{"tools": [{{"name": name, "description": name, "inputSchema": schema}} for name, schema in schemas.items()]}}
    elif method == "tools/call":
        params = message.get("params", {{}})
        name = params.get("name")
        arguments = params.get("arguments", {{}})
        is_error = False
        if name == "browser_navigate":
            requested = arguments.get("url", "")
            current_url = "https://example.com/escaped" if requested.endswith("/redirect") else requested
            state = "initial"
            mutate_on_snapshot = requested.endswith("/mutate")
            action_error = requested.endswith("/action-error")
            post_error_mode = requested.endswith("/post-error")
            post_error_next = False
        elif name == "browser_snapshot":
            if mutate_on_snapshot:
                state = "mutated"
                mutate_on_snapshot = False
            if post_error_next:
                is_error = True
                post_error_next = False
        elif name == "browser_click":
            click_marker.write_text("called", encoding="utf-8")
            if action_error:
                is_error = True
            else:
                state = "clicked"
                post_error_next = post_error_mode
        elif name == "browser_type":
            state = "typed:" + str(arguments.get("text", ""))
        snapshot = "- Page URL: " + current_url + "\\n- Page Title: Fixture\\n- button \\\"Safe action " + state + "\\\" [ref=e1]\\n- textbox \\\"Input\\\" [ref=e2]"
        result = {{"content": [{{"type": "text", "text": snapshot}}], "isError": is_error, "_meta": {{"must_not_escape": True}}}}
    else:
        result = {{}}
    sys.stdout.write(json.dumps({{"jsonrpc": "2.0", "id": message.get("id"), "result": result}}) + "\\n")
    sys.stdout.flush()
'''


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
