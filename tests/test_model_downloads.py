from __future__ import annotations

import unittest

from local_moe.config import load_config, parse_config
from local_moe.model_downloads import (
    build_model_download_requests,
    validate_local_file_request,
)


class ModelDownloadTests(unittest.TestCase):
    def test_builds_mlx_huggingface_snapshot_requests(self) -> None:
        config = load_config("configs/moe.live.general-mlx.example.json")

        requests = build_model_download_requests(config, "mlx_lm")

        self.assertEqual({request.kind for request in requests}, {"huggingface_snapshot"})
        self.assertEqual({request.backend for request in requests}, {"mlx_lm"})
        self.assertTrue(any("Qwen3-30B-A3B" in request.repo_id for request in requests if request.repo_id))

    def test_builds_llama_cpp_quantized_gguf_snapshot_request(self) -> None:
        config = load_config("configs/moe.live.gemma-12b-agentic-gguf.example.json")

        requests = build_model_download_requests(config, "llama_cpp")

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.kind, "huggingface_snapshot")
        self.assertEqual(request.backend, "llama_cpp")
        self.assertEqual(request.repo_id, "yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF")
        self.assertEqual(request.allow_patterns, ("*Q4_K_M*.gguf",))

    def test_builds_ollama_pull_request(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "fast",
                        "provider": "openai_compatible",
                        "model": "qwen3:4b",
                        "role": "fast",
                        "params": {"runtime_backend": "ollama"},
                    }
                ],
                "rules": [],
            }
        )

        requests = build_model_download_requests(config, "ollama")

        self.assertEqual(requests[0].kind, "ollama_pull")
        self.assertEqual(requests[0].command, ("ollama", "pull", "qwen3:4b"))

    def test_treats_local_gguf_as_local_file(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "local",
                        "provider": "openai_compatible",
                        "model": "./models/local-model.Q4_K_M.gguf",
                        "role": "local",
                        "params": {"runtime_backend": "llama_cpp"},
                    }
                ],
                "rules": [],
            }
        )

        requests = build_model_download_requests(config, "llama_cpp")

        self.assertEqual(requests[0].kind, "local_file")
        with self.assertRaises(FileNotFoundError):
            validate_local_file_request(requests[0])

    def test_rejects_missing_backend_in_mixed_download_plan(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "generic",
                        "provider": "openai_compatible",
                        "model": "owner/model",
                        "role": "generic",
                    }
                ],
                "rules": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "Mixed runtime downloads"):
            build_model_download_requests(config, "mixed")


if __name__ == "__main__":
    unittest.main()
