from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterator
from unittest import mock
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

from local_moe.app_config import GatewayPolicy, load_app_config
from local_moe.config import MoEConfig, parse_config
from local_moe.execution_scope import ScopePolicyError
from local_moe.openai_gateway import GatewayRequestError, OpenAIGatewayService
from local_moe.web import _safe_gateway_content_type, build_server


HTTP_TEST_TIMEOUT_SECONDS = 15


class OpenAIGatewayServiceTests(unittest.TestCase):
    def test_models_payload_exposes_auto_and_pinned_aliases(self) -> None:
        gateway = _service()

        payload = gateway.models_payload()

        self.assertEqual(
            [item["id"] for item in payload["data"]],
            ["mymoe", "mymoe/architect", "mymoe/coder"],
        )
        self.assertEqual(payload["data"][0]["mymoe"]["selection"], "deterministic_route")
        self.assertTrue(payload["data"][1]["mymoe"]["eligible"])
        self.assertEqual(payload["data"][2]["mymoe"]["upstream_model"], "local-coder")

    def test_auto_pinned_and_upstream_model_routing(self) -> None:
        gateway = _service()

        automatic = gateway.prepare_chat_completion(
            {"model": "mymoe", "messages": [{"role": "user", "content": "Fix Python code."}]}
        )
        pinned = gateway.prepare_chat_completion(
            {
                "model": "mymoe/architect",
                "messages": [{"role": "user", "content": "Fix Python code."}],
            }
        )
        upstream_name = gateway.prepare_chat_completion(
            {
                "model": "local-coder",
                "messages": [{"role": "user", "content": "Design an architecture."}],
            }
        )

        self.assertEqual(automatic.expert.id, "coder")
        self.assertEqual(automatic.route_selected, ("coder",))
        self.assertEqual(pinned.expert.id, "architect")
        self.assertEqual(pinned.route_selected, ("architect",))
        self.assertEqual(upstream_name.expert.id, "coder")
        with self.assertRaises(GatewayRequestError) as raised:
            gateway.prepare_chat_completion(
                {"model": "mymoe/missing", "messages": [{"role": "user", "content": "Hi"}]}
            )
        self.assertEqual(raised.exception.code, "model_not_found")
        self.assertEqual(raised.exception.status, 404)

    def test_content_parts_drive_routing_without_changing_the_request(self) -> None:
        gateway = _service()
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "Design the system architecture."},
        ]

        prepared = gateway.prepare_chat_completion(
            {"model": "mymoe", "messages": [{"role": "user", "content": content}]}
        )
        forwarded = json.loads(bytes(prepared.upstream_request.data or b"").decode("utf-8"))

        self.assertEqual(prepared.expert.id, "architect")
        self.assertEqual(forwarded["messages"][0]["content"], content)

    def test_tool_definitions_calls_and_results_are_forwarded_losslessly(self) -> None:
        gateway = _service()
        payload: dict[str, Any] = {
            "model": "mymoe/coder",
            "messages": [
                {"role": "system", "content": "Use tools when useful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect the repository."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA=="},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "text", "text": "# Local project"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a workspace file.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "response_format": {"type": "json_object"},
            "stream": False,
            "temperature": 0.2,
        }

        prepared = gateway.prepare_chat_completion(payload, correlation_id="session-42")
        forwarded = json.loads(bytes(prepared.upstream_request.data or b"").decode("utf-8"))
        expected = json.loads(json.dumps(payload))
        expected["model"] = "local-coder"

        self.assertEqual(forwarded, expected)
        self.assertEqual(prepared.correlation_id, "session-42")
        self.assertEqual(prepared.upstream_request.get_header("Accept"), "application/json")
        self.assertEqual(
            prepared.upstream_request.get_header("X-mymoe-correlation-id"),
            "session-42",
        )

    def test_blocks_provider_dereferenced_content_urls_but_preserves_text_urls(self) -> None:
        gateway = _service()
        for unsafe_url in (
            "file:///tmp/private.png",
            "http://127.0.0.1:9000/private.png",
            "https://metadata.example.test/token",
        ):
            with self.subTest(unsafe_url=unsafe_url), self.assertRaises(
                GatewayRequestError
            ) as raised:
                gateway.prepare_chat_completion(
                    {
                        "model": "mymoe/coder",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": unsafe_url},
                                    }
                                ],
                            }
                        ],
                    }
                )
            self.assertEqual(raised.exception.code, "unsafe_content_url")

        with self.assertRaises(GatewayRequestError) as mapped_content:
            gateway.prepare_chat_completion(
                {
                    "model": "mymoe/coder",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "input_image",
                                "image_url": "http://127.0.0.1/private.png",
                            },
                        }
                    ],
                }
            )
        self.assertEqual(mapped_content.exception.code, "unsafe_content_url")

        safe = gateway.prepare_chat_completion(
            {
                "model": "mymoe/coder",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain https://example.test without fetching it.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_url",
                                "type": "function",
                                "function": {
                                    "name": "inspect_url",
                                    "arguments": '{"url":"https://example.test"}',
                                },
                            }
                        ],
                    },
                ],
            }
        )
        forwarded = json.loads(bytes(safe.upstream_request.data or b"").decode("utf-8"))
        self.assertIn("https://example.test", forwarded["messages"][0]["content"])

    def test_rejects_excessive_json_depth_and_regenerates_unsafe_correlation_ids(self) -> None:
        gateway = _service()
        nested: object = "leaf"
        for _ in range(70):
            nested = [nested]

        with self.assertRaises(GatewayRequestError) as raised:
            gateway.prepare_chat_completion(
                {
                    "model": "mymoe/coder",
                    "messages": [{"role": "user", "content": "Code"}],
                    "metadata": nested,
                }
            )
        self.assertEqual(raised.exception.code, "invalid_json")

        prepared = gateway.prepare_chat_completion(
            {
                "model": "mymoe/coder",
                "messages": [{"role": "user", "content": "Code"}],
            },
            correlation_id="unsafe\x00header",
        )
        self.assertNotIn("\x00", prepared.correlation_id)
        self.assertNotEqual(prepared.correlation_id, "unsafe\x00header")

    def test_authorization_enforces_key_host_and_origin(self) -> None:
        key_policy = _policy(api_key_env="MYMOE_TEST_GATEWAY_KEY")
        gateway = OpenAIGatewayService(_moe_config(), key_policy)
        with mock.patch.dict(os.environ, {"MYMOE_TEST_GATEWAY_KEY": "local-secret"}):
            allowed = gateway.authorize(
                "127.0.0.1",
                "Bearer local-secret",
                host_header="localhost:8899",
                origin_header="http://127.0.0.1:3000",
            )
            bad_key = gateway.authorize(
                "127.0.0.1",
                "Bearer wrong",
                host_header="localhost:8899",
            )

        no_key_gateway = OpenAIGatewayService(_moe_config(), _policy())
        bad_host = no_key_gateway.authorize(
            "127.0.0.1",
            None,
            host_header="models.example.test",
        )
        bad_origin = no_key_gateway.authorize(
            "127.0.0.1",
            None,
            host_header="127.0.0.1:8899",
            origin_header="https://app.example.test",
        )

        self.assertTrue(allowed.allowed)
        self.assertEqual(bad_key.code, "invalid_api_key")
        self.assertEqual(bad_host.code, "invalid_host")
        self.assertEqual(bad_origin.code, "origin_forbidden")

    def test_request_size_limit_is_checked_before_opening_the_provider(self) -> None:
        opener = mock.Mock()
        gateway = OpenAIGatewayService(
            _moe_config(),
            _policy(max_request_bytes=256),
            opener=opener,
        )

        with self.assertRaises(GatewayRequestError) as raised:
            gateway.prepare_chat_completion(
                {"model": "mymoe", "messages": [{"role": "user", "content": "x" * 1024}]}
            )

        self.assertEqual(raised.exception.code, "request_too_large")
        self.assertEqual(raised.exception.status, 413)
        opener.assert_not_called()

    def test_execution_scope_blocks_remote_provider_before_open(self) -> None:
        opener = mock.Mock()
        gateway = OpenAIGatewayService(
            _moe_config(base_url="https://models.example.test/v1", single_expert=True),
            _policy(),
            opener=opener,
        )

        with self.assertRaises(ScopePolicyError):
            gateway.prepare_chat_completion(
                {"model": "mymoe/coder", "messages": [{"role": "user", "content": "Code"}]}
            )

        opener.assert_not_called()


class OpenAIGatewayHTTPTests(unittest.TestCase):
    def test_gateway_content_type_rejects_header_controls(self) -> None:
        default = "application/json; charset=utf-8"
        self.assertEqual(
            _safe_gateway_content_type("text/event-stream; charset=utf-8", default),
            "text/event-stream; charset=utf-8",
        )
        for unsafe in (
            "application/json\r\nX-Injected: true",
            "application/json\x00evil",
            "not-a-media-type",
            "x" * 201,
        ):
            with self.subTest(unsafe=unsafe):
                self.assertEqual(_safe_gateway_content_type(unsafe, default), default)

    def test_shared_control_plane_rejects_every_non_loopback_bind(self) -> None:
        for host in ("0.0.0.0", "::", "192.0.2.10", "models.example.test"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "loopback"):
                build_server(host=host, port=0)

    def test_non_streaming_proxy_uses_real_http_and_preserves_agent_payload(self) -> None:
        response_body = json.dumps(
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_next",
                                    "type": "function",
                                    "function": {"name": "run_tests", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(response_body) as upstream:
            root = Path(tmp)
            with _running_gateway(root, upstream.base_url) as gateway_url:
                payload = _tool_loop_payload(stream=False)
                response, body = _post_raw(
                    gateway_url + "/v1/chat/completions",
                    payload,
                    headers={"X-MyMoE-Correlation-ID": "http-session-7"},
                )

        self.assertEqual(body, response_body)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["X-MyMoE-Expert"], "coder")
        self.assertEqual(response.headers["X-MyMoE-Correlation-ID"], "http-session-7")
        self.assertEqual(len(upstream.requests), 1)
        forwarded = json.loads(upstream.requests[0]["body"].decode("utf-8"))
        expected = json.loads(json.dumps(payload))
        expected["model"] = "local-coder"
        self.assertEqual(forwarded, expected)
        self.assertEqual(upstream.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(upstream.requests[0]["headers"]["accept"], "application/json")

    def test_streaming_proxy_preserves_sse_bytes_and_headers(self) -> None:
        sse_body = (
            b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\r\n\r\n'
            b"data: [DONE]\n\n"
        )
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(
            sse_body,
            content_type="text/event-stream; charset=utf-8",
        ) as upstream:
            root = Path(tmp)
            with _running_gateway(root, upstream.base_url) as gateway_url:
                response, body = _post_raw(
                    gateway_url + "/v1/chat/completions",
                    {"model": "mymoe", "messages": [{"role": "user", "content": "Code"}], "stream": True},
                )

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers.get_content_type(), "text/event-stream")
        self.assertEqual(response.headers["Cache-Control"], "no-cache")
        self.assertEqual(response.headers["X-Accel-Buffering"], "no")
        self.assertEqual(body, sse_body)
        self.assertEqual(upstream.requests[0]["headers"]["accept"], "text/event-stream")

    def test_streaming_response_limit_closes_before_oversized_event(self) -> None:
        accepted_prefix = b'data: {"delta":"ok"}\n\n'
        oversized_event = b'data: {"delta":"' + (b"x" * (1024 * 1024))
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(
            accepted_prefix + oversized_event,
            content_type="text/event-stream; charset=utf-8",
            send_content_length=False,
        ) as upstream:
            root = Path(tmp)
            with _running_gateway(root, upstream.base_url, max_response_bytes=64) as gateway_url:
                response, body = _post_raw(
                    gateway_url + "/v1/chat/completions",
                    {"model": "mymoe", "messages": [{"role": "user", "content": "Code"}], "stream": True},
                )

        self.assertEqual(response.status, 200)
        self.assertEqual(body, accepted_prefix)
        self.assertNotIn(b"x" * 64, body)
        self.assertEqual(len(upstream.requests), 1)

    def test_provider_timeout_returns_502_and_records_terminal_audit_event(self) -> None:
        response_body = b'{"choices":[{"message":{"content":"late"}}]}'
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(
            response_body,
            delay_seconds=0.2,
        ) as upstream:
            root = Path(tmp)
            with _running_gateway(
                root,
                upstream.base_url,
                timeout_seconds=0.05,
            ) as gateway_url:
                with self.assertRaises(error.HTTPError) as raised:
                    _post_raw(
                        gateway_url + "/v1/chat/completions",
                        {
                            "model": "mymoe",
                            "messages": [{"role": "user", "content": "Code"}],
                        },
                    )
                body = json.loads(raised.exception.read().decode("utf-8"))
            events = [
                json.loads(line)
                for line in (root / "runtime" / "audit.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(raised.exception.code, 502)
        self.assertEqual(body["error"]["code"], "provider_response_error")
        self.assertEqual(events[-1]["status"], "provider_response_error")
        self.assertEqual(events[-1]["metadata"]["error_type"], "TimeoutError")

    def test_incomplete_provider_body_fails_closed_before_forwarding(self) -> None:
        response_body = b'{"choices":[{"message":{"content":"cut"}}]}'
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(
            response_body,
            declared_content_length=len(response_body) + 100,
        ) as upstream:
            root = Path(tmp)
            with _running_gateway(root, upstream.base_url) as gateway_url:
                with self.assertRaises(error.HTTPError) as raised:
                    _post_raw(
                        gateway_url + "/v1/chat/completions",
                        {
                            "model": "mymoe",
                            "messages": [{"role": "user", "content": "Code"}],
                        },
                    )
                body = json.loads(raised.exception.read().decode("utf-8"))

        self.assertEqual(raised.exception.code, 502)
        self.assertEqual(body["error"]["code"], "upstream_response_incomplete")

    def test_response_limit_fails_closed_before_non_streaming_body_is_sent(self) -> None:
        oversized = json.dumps({"choices": [{"message": {"content": "x" * 512}}]}).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(oversized) as upstream:
            root = Path(tmp)
            with _running_gateway(root, upstream.base_url, max_response_bytes=128) as gateway_url:
                with self.assertRaises(error.HTTPError) as raised:
                    _post_raw(
                        gateway_url + "/v1/chat/completions",
                        {"model": "mymoe", "messages": [{"role": "user", "content": "Code"}]},
                    )
                body = json.loads(raised.exception.read().decode("utf-8"))

        self.assertEqual(raised.exception.code, 502)
        self.assertEqual(body["error"]["code"], "upstream_response_too_large")
        self.assertEqual(len(upstream.requests), 1)

    def test_http_request_limit_rejects_body_before_provider_call(self) -> None:
        response_body = b'{"choices":[{"message":{"content":"unused"}}]}'
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(response_body) as upstream:
            root = Path(tmp)
            with _running_gateway(root, upstream.base_url, max_request_bytes=256) as gateway_url:
                with self.assertRaises(error.HTTPError) as raised:
                    _post_raw(
                        gateway_url + "/v1/chat/completions",
                        {
                            "model": "mymoe",
                            "messages": [{"role": "user", "content": "x" * 2048}],
                        },
                    )
                body = json.loads(raised.exception.read().decode("utf-8"))

        self.assertEqual(raised.exception.code, 413)
        self.assertEqual(body["error"]["code"], "request_too_large")
        self.assertEqual(upstream.requests, [])

    def test_http_auth_host_and_origin_fail_closed(self) -> None:
        response_body = b'{"choices":[{"message":{"content":"unused"}}]}'
        with tempfile.TemporaryDirectory() as tmp, _running_upstream(response_body) as upstream:
            root = Path(tmp)
            with mock.patch.dict(os.environ, {"MYMOE_TEST_GATEWAY_KEY": "local-secret"}):
                with _running_gateway(
                    root,
                    upstream.base_url,
                    key_environment_name="MYMOE_TEST_GATEWAY_KEY",
                ) as gateway_url:
                    with self.assertRaises(error.HTTPError) as missing_key:
                        _get_raw(gateway_url + "/v1/models")
                    with self.assertRaises(error.HTTPError) as bad_host:
                        _get_raw(
                            gateway_url + "/v1/models",
                            headers={"Authorization": "Bearer local-secret", "Host": "example.test"},
                        )
                    with self.assertRaises(error.HTTPError) as bad_origin:
                        _get_raw(
                            gateway_url + "/v1/models",
                            headers={
                                "Authorization": "Bearer local-secret",
                                "Origin": "https://app.example.test",
                            },
                        )
                    response, body = _get_raw(
                        gateway_url + "/v1/models",
                        headers={
                            "Authorization": "Bearer local-secret",
                            "Origin": "http://localhost:3000",
                        },
                    )

        self.assertEqual(missing_key.exception.code, 401)
        self.assertEqual(bad_host.exception.code, 403)
        self.assertEqual(bad_origin.exception.code, 403)
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "mymoe")
        self.assertEqual(upstream.requests, [])


def _policy(
    *,
    max_request_bytes: int = 8 * 1024 * 1024,
    max_response_bytes: int = 32 * 1024 * 1024,
    api_key_env: str = "",
) -> GatewayPolicy:
    return GatewayPolicy(
        enabled=True,
        model_alias="mymoe",
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        allow_non_loopback=False,
        api_key_env=api_key_env,
    )


def _service() -> OpenAIGatewayService:
    return OpenAIGatewayService(_moe_config(), _policy())


def _moe_config(
    *,
    base_url: str = "http://127.0.0.1:9999/v1",
    single_expert: bool = False,
) -> MoEConfig:
    experts = [
        {
            "id": "coder",
            "provider": "openai_compatible",
            "base_url": base_url,
            "model": "local-coder",
            "role": "coding",
        }
    ]
    rules = [{"expert_id": "coder", "keywords": ["python", "code", "test"], "weight": 3.0}]
    if not single_expert:
        experts.append(
            {
                "id": "architect",
                "provider": "openai_compatible",
                "base_url": base_url,
                "model": "local-architect",
                "role": "architecture",
            }
        )
        rules.append(
            {
                "expert_id": "architect",
                "keywords": ["architecture", "design", "system"],
                "weight": 3.0,
            }
        )
    return parse_config(
        {
            "routing": {"top_k": 1, "fallback_order": [], "aggregation": "best"},
            "experts": experts,
            "rules": rules,
        }
    )


def _tool_loop_payload(*, stream: bool) -> dict[str, Any]:
    return {
        "model": "mymoe",
        "messages": [
            {"role": "user", "content": "Inspect Python code."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "# myMoE"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": stream,
    }


class _RecordingUpstream(ThreadingHTTPServer):
    response_body: bytes
    response_status: int
    response_content_type: str
    response_delay_seconds: float
    declared_content_length: int | None
    send_content_length: bool
    requests: list[dict[str, Any]]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _OpenAIHandler(BaseHTTPRequestHandler):
    server: _RecordingUpstream

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.requests.append(
            {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": body,
            }
        )
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", self.server.response_content_type)
        declared_length = self.server.declared_content_length
        if self.server.send_content_length:
            self.send_header(
                "Content-Length",
                str(
                    declared_length
                    if declared_length is not None
                    else len(self.server.response_body)
                ),
            )
        self.end_headers()
        if self.server.response_delay_seconds:
            time.sleep(self.server.response_delay_seconds)
        try:
            self.wfile.write(self.server.response_body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _running_upstream(
    body: bytes,
    *,
    content_type: str = "application/json; charset=utf-8",
    status: int = 200,
    delay_seconds: float = 0.0,
    declared_content_length: int | None = None,
    send_content_length: bool = True,
) -> Iterator[_RecordingUpstream]:
    server = _RecordingUpstream(("127.0.0.1", 0), _OpenAIHandler)
    server.response_body = body
    server.response_status = status
    server.response_content_type = content_type
    server.response_delay_seconds = delay_seconds
    server.declared_content_length = declared_content_length
    server.send_content_length = send_content_length
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@contextmanager
def _running_gateway(
    root: Path,
    upstream_base_url: str,
    *,
    max_request_bytes: int = 8 * 1024 * 1024,
    max_response_bytes: int = 32 * 1024 * 1024,
    key_environment_name: str = "",
    timeout_seconds: float = 60.0,
) -> Iterator[str]:
    moe_path = root / "moe.gateway.json"
    moe_path.write_text(
        json.dumps(
            {
                "routing": {"top_k": 1, "fallback_order": [], "aggregation": "best"},
                "experts": [
                    {
                        "id": "coder",
                        "provider": "openai_compatible",
                        "base_url": upstream_base_url,
                        "model": "local-coder",
                        "role": "coding",
                        "timeout_seconds": timeout_seconds,
                    }
                ],
                "rules": [
                    {"expert_id": "coder", "keywords": ["python", "code", "test"], "weight": 3.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    app_path = root / "app.gateway.json"
    app = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
    app["default_moe_config"] = str(moe_path)
    app["runtime"]["work_dir"] = str(root / "runtime")
    app["gateway"] = {
        "enabled": True,
        "model_alias": "mymoe",
        "max_request_bytes": max_request_bytes,
        "max_response_bytes": max_response_bytes,
        "allow_non_loopback": False,
        "api_key_env": "",
    }
    app_path.write_text(json.dumps(app), encoding="utf-8")

    configured_app = load_app_config(app_path)
    configured_app = replace(
        configured_app,
        gateway=replace(
            configured_app.gateway,
            api_key_env=key_environment_name,
        ),
    )
    with mock.patch("local_moe.web.load_app_config", return_value=configured_app):
        server = build_server(str(moe_path), port=0, app_config_path=str(app_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _post_raw(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Any, bytes]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    http_request = request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    response = request.urlopen(http_request, timeout=HTTP_TEST_TIMEOUT_SECONDS)
    return response, response.read()


def _get_raw(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Any, bytes]:
    http_request = request.Request(url, headers=headers or {}, method="GET")
    response = request.urlopen(http_request, timeout=HTTP_TEST_TIMEOUT_SECONDS)
    return response, response.read()


if __name__ == "__main__":
    unittest.main()
