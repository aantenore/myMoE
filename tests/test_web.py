from __future__ import annotations

import json
import threading
from urllib import request
import unittest

from local_moe.web import build_server


class WebTests(unittest.TestCase):
    def test_serves_config_and_generates_with_mock_provider(self) -> None:
        server = build_server("configs/moe.mock.json", port=0)
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
        self.assertIn("[general:mock-general]", result["content"])
        self.assertEqual(result["route"]["selected"][0]["expert_id"], "general")

    def test_runs_eval_endpoint(self) -> None:
        server = build_server("configs/moe.mock.json", port=0)
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
        server = build_server("configs/moe.mock.json", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            runtime = _get_json(base_url + "/api/runtime")
            extensions = _get_json(base_url + "/api/extensions")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertIn(runtime["backend"], {"mlx_lm", "ollama", "llama_cpp"})
        self.assertTrue(extensions["tools"])


def _get_json(url: str) -> dict[str, object]:
    with request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    http_req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
