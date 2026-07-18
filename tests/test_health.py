from __future__ import annotations

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from local_moe.config import ExpertConfig, MoEConfig, RoutingConfig
from local_moe.health import check_runtime_health, runtime_health_payload


class _ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/v1/models", "/api/v1/models"}:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "local-test-model"}]}).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _models_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class HealthTests(unittest.TestCase):
    def test_reports_ready_when_openai_endpoint_responds(self) -> None:
        with _models_server() as server:
            host, port = server.server_address
            config = _config(
                ExpertConfig(
                    id="general",
                    provider="openai_compatible",
                    model="local-test-model",
                    role="general",
                    base_url=f"http://{host}:{port}/v1",
                )
            )

            payload = runtime_health_payload(check_runtime_health(config, timeout_seconds=0.5))

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["experts"][0]["status"], "ok")
        self.assertIn("/v1/models", payload["experts"][0]["checked_url"])
        self.assertIn("1 models", payload["experts"][0]["message"])

    def test_preserves_path_prefix_when_probe_url_uses_v1_suffix(self) -> None:
        with _models_server() as server:
            host, port = server.server_address
            config = _config(
                ExpertConfig(
                    id="general",
                    provider="openai_compatible",
                    model="local-test-model",
                    role="general",
                    base_url=f"http://{host}:{port}/api/v1",
                )
            )

            payload = runtime_health_payload(check_runtime_health(config, timeout_seconds=0.5))

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["experts"][0]["status"], "ok")
        self.assertIn("/api/v1/models", payload["experts"][0]["checked_url"])

    def test_reports_degraded_when_required_endpoint_is_unreachable(self) -> None:
        config = _config(
            ExpertConfig(
                id="general",
                provider="openai_compatible",
                model="missing-model",
                role="general",
                base_url="http://127.0.0.1:9/v1",
            )
        )

        payload = runtime_health_payload(check_runtime_health(config, timeout_seconds=0.05))

        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["experts"][0]["status"], "unreachable")

    def test_reports_degraded_when_base_url_is_malformed(self) -> None:
        config = _config(
            ExpertConfig(
                id="general",
                provider="openai_compatible",
                model="missing-model",
                role="general",
                base_url="127.0.0.1:8101/v1",
            )
        )

        payload = runtime_health_payload(check_runtime_health(config, timeout_seconds=0.05))

        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["experts"][0]["status"], "malformed_base_url")

    def test_does_not_probe_endpoint_outside_execution_policy(self) -> None:
        config = _config(
            ExpertConfig(
                id="remote",
                provider="openai_compatible",
                model="remote-model",
                role="general",
                base_url="https://models.example.test/v1",
            )
        )

        def unexpected_opener(*_args, **_kwargs):
            raise AssertionError("blocked endpoint must not be probed")

        payload = runtime_health_payload(
            check_runtime_health(
                config,
                timeout_seconds=0.05,
                opener=unexpected_opener,
            )
        )

        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["experts"][0]["status"], "scope_blocked")

    def test_skips_non_http_test_provider(self) -> None:
        config = _config(
            ExpertConfig(
                id="synthetic",
                provider="synthetic",
                model="synthetic-model",
                role="general",
            )
        )

        payload = runtime_health_payload(check_runtime_health(config, timeout_seconds=0.05))

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["experts"][0]["status"], "skipped")


def _config(expert: ExpertConfig) -> MoEConfig:
    return MoEConfig(routing=RoutingConfig(), experts=(expert,), rules=())


if __name__ == "__main__":
    unittest.main()
