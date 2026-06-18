from __future__ import annotations

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from local_moe.config import ExpertConfig
from local_moe.providers import (
    GenerationRequest,
    OpenAICompatibleProvider,
    ProviderError,
    build_provider,
)


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    response_body: str = "{}"
    status_code: int = 200
    last_path: str | None = None
    last_payload: dict[str, object] | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        type(self).last_path = self.path
        type(self).last_payload = json.loads(raw_body)

        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).response_body.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _fake_openai_server(response: dict[str, object] | str, status: int = 200):
    _FakeOpenAIHandler.response_body = (
        response if isinstance(response, str) else json.dumps(response)
    )
    _FakeOpenAIHandler.status_code = status
    _FakeOpenAIHandler.last_path = None
    _FakeOpenAIHandler.last_payload = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _expert(base_url: str) -> ExpertConfig:
    return ExpertConfig(
        id="coder",
        provider="openai_compatible",
        model="fake-model",
        role="coding",
        base_url=base_url,
        timeout_seconds=1,
        params={"temperature": 0.2},
    )


class ProviderTests(unittest.TestCase):
    def test_builds_known_provider_types(self) -> None:
        self.assertIsNotNone(build_provider("mock"))
        self.assertIsInstance(build_provider("openai_compatible"), OpenAICompatibleProvider)

    def test_rejects_unknown_provider_type(self) -> None:
        with self.assertRaisesRegex(ProviderError, "Unsupported provider"):
            build_provider("missing")

    def test_openai_provider_parses_content_usage_and_timings(self) -> None:
        response = {
            "choices": [{"message": {"content": "ok from fake local model"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "timings": {"predicted_per_second": 123.5},
        }

        with _fake_openai_server(response) as server:
            host, port = server.server_address
            provider = OpenAICompatibleProvider()
            result = provider.generate(
                _expert(f"http://{host}:{port}/v1"),
                GenerationRequest(prompt="Write Python code", correlation_id="case-1"),
            )

        self.assertEqual(result.content, "ok from fake local model")
        self.assertEqual(result.prompt_tokens, 11)
        self.assertEqual(result.completion_tokens, 7)
        self.assertEqual(result.predicted_tokens_per_second, 123.5)
        self.assertEqual(_FakeOpenAIHandler.last_path, "/v1/chat/completions")
        self.assertEqual(_FakeOpenAIHandler.last_payload["model"], "fake-model")
        self.assertEqual(_FakeOpenAIHandler.last_payload["temperature"], 0.2)
        system = _FakeOpenAIHandler.last_payload["messages"][0]["content"]
        self.assertIn("response in the user's language", system)

    def test_openai_provider_does_not_send_local_runtime_params(self) -> None:
        response = {"choices": [{"message": {"content": "ok"}}]}
        with _fake_openai_server(response) as server:
            host, port = server.server_address
            provider = OpenAICompatibleProvider()
            expert = ExpertConfig(
                id="gemma",
                provider="openai_compatible",
                model="mlx-community/gemma-4-e4b-it-4bit",
                role="general",
                base_url=f"http://{host}:{port}/v1",
                timeout_seconds=1,
                params={
                    "runtime_backend": "mlx_lm",
                    "temperature": 0.1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            provider.generate(expert, GenerationRequest(prompt="hello", correlation_id="case-local"))

        self.assertNotIn("runtime_backend", _FakeOpenAIHandler.last_payload)
        self.assertEqual(_FakeOpenAIHandler.last_payload["temperature"], 0.1)
        self.assertEqual(
            _FakeOpenAIHandler.last_payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_openai_provider_rejects_invalid_payload_shape(self) -> None:
        with _fake_openai_server({"choices": []}) as server:
            host, port = server.server_address
            provider = OpenAICompatibleProvider()

            with self.assertRaisesRegex(ProviderError, "invalid payload"):
                provider.generate(
                    _expert(f"http://{host}:{port}/v1"),
                    GenerationRequest(prompt="hello", correlation_id="case-2"),
                )

    def test_openai_provider_rejects_invalid_json(self) -> None:
        with _fake_openai_server("{not-json}") as server:
            host, port = server.server_address
            provider = OpenAICompatibleProvider()

            with self.assertRaisesRegex(ProviderError, "invalid JSON"):
                provider.generate(
                    _expert(f"http://{host}:{port}/v1"),
                    GenerationRequest(prompt="hello", correlation_id="case-3"),
                )

    def test_openai_provider_wraps_transport_errors(self) -> None:
        provider = OpenAICompatibleProvider()

        with self.assertRaisesRegex(ProviderError, "request failed"):
            provider.generate(
                _expert("http://127.0.0.1:1/v1"),
                GenerationRequest(prompt="hello", correlation_id="case-4"),
            )


if __name__ == "__main__":
    unittest.main()
