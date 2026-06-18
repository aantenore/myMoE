from __future__ import annotations

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from local_moe.runtime import LlamaServerSpec, wait_for_health


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _health_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class RuntimeTests(unittest.TestCase):
    def test_llama_server_spec_urls_and_command(self) -> None:
        spec = LlamaServerSpec(
            binary="/tmp/llama-server",
            model="/tmp/model.gguf",
            host="127.0.0.1",
            port=8101,
            ctx_size=8192,
            threads=6,
            gpu_layers=55,
            flash_attn="auto",
        )

        self.assertEqual(spec.base_url, "http://127.0.0.1:8101/v1")
        self.assertEqual(spec.health_url, "http://127.0.0.1:8101/health")
        self.assertEqual(
            spec.command(),
            [
                "/tmp/llama-server",
                "-m",
                "/tmp/model.gguf",
                "-ngl",
                "55",
                "-fa",
                "auto",
                "-c",
                "8192",
                "-t",
                "6",
                "--host",
                "127.0.0.1",
                "--port",
                "8101",
            ],
        )

    def test_wait_for_health_accepts_ok_response(self) -> None:
        with _health_server() as server:
            host, port = server.server_address
            parsed = wait_for_health(f"http://{host}:{port}/health", timeout_seconds=2)

        self.assertEqual(parsed["status"], "ok")


if __name__ == "__main__":
    unittest.main()
