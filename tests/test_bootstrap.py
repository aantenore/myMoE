from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

from local_moe.bootstrap import RuntimePlan, build_runtime_plan, runtime_plan_payload
from local_moe.config import parse_config
from local_moe.model_servers import build_model_server_specs


class RuntimePlanTests(unittest.TestCase):
    def test_legacy_model_commands_initializer_remains_supported(self) -> None:
        commands = (
            ("llama-server", "-hf", "owner/cpp-model"),
            ("ollama", "pull", "owner/ollama-model"),
        )

        keyword_plan = RuntimePlan(
            platform_key="linux",
            backend="mixed",
            install_commands=(),
            model_commands=commands,
            notes=("legacy",),
        )
        positional_plan = RuntimePlan("linux", "mixed", (), commands, ("legacy",))

        self.assertEqual(keyword_plan, positional_plan)
        self.assertEqual(keyword_plan.model_commands, commands)
        self.assertEqual(keyword_plan.expert_commands, ())
        self.assertEqual(
            runtime_plan_payload(keyword_plan)["expert_commands"],
            [],
        )

        specs = build_model_server_specs(
            _mixed_runtime_config(),
            keyword_plan,
            work_dir=".",
        )
        self.assertEqual(
            tuple(spec.command for spec in specs),
            commands,
        )

        replaced = replace(keyword_plan, model_commands=(("replacement",),))
        self.assertEqual(replaced.model_commands, (("replacement",),))
        self.assertEqual(replaced.expert_commands, ())

        generated = build_runtime_plan(_mixed_runtime_config())
        with self.assertRaisesRegex(ValueError, "with_model_commands"):
            replace(generated, model_commands=(("replacement",),))
        replaced_generated = generated.with_model_commands((("replacement",),))
        self.assertEqual(replaced_generated.model_commands, (("replacement",),))
        self.assertEqual(replaced_generated.expert_commands, ())

        cosmetic = replace(generated, notes=("cosmetic",))
        self.assertEqual(cosmetic.expert_commands, generated.expert_commands)

    def test_expert_commands_are_canonical_and_payload_remains_compatible(self) -> None:
        config = _mixed_runtime_config()
        with mock.patch(
            "local_moe.bootstrap.detect_platform_key",
            return_value="linux",
        ):
            plan = build_runtime_plan(config)

        payload = runtime_plan_payload(plan)

        self.assertEqual(
            plan.model_commands,
            tuple(command.argv for command in plan.expert_commands),
        )
        self.assertEqual(
            payload["model_commands"],
            [list(command.argv) for command in plan.expert_commands],
        )
        self.assertEqual(
            payload["expert_commands"],
            [
                {
                    "expert_id": command.expert_id,
                    "backend": command.backend,
                    "argv": list(command.argv),
                }
                for command in plan.expert_commands
            ],
        )

    def test_mixed_plan_records_backend_and_argv_for_each_expert(self) -> None:
        config = _mixed_runtime_config()
        with mock.patch(
            "local_moe.bootstrap.detect_platform_key",
            return_value="linux",
        ):
            plan = build_runtime_plan(config)

        commands = {command.expert_id: command for command in plan.expert_commands}

        self.assertEqual(plan.backend, "mixed")
        self.assertEqual(set(commands), {"cpp-expert", "ollama-expert"})
        self.assertEqual(commands["cpp-expert"].backend, "llama_cpp")
        self.assertEqual(commands["cpp-expert"].argv[0], "llama-server")
        self.assertIn("owner/cpp-model", commands["cpp-expert"].argv)
        self.assertEqual(commands["ollama-expert"].backend, "ollama")
        self.assertEqual(
            commands["ollama-expert"].argv,
            ("ollama", "pull", "owner/ollama-model"),
        )

    def test_local_gguf_uses_a_file_argument_and_rejects_unknown_source(self) -> None:
        raw = _mixed_runtime_config_payload()
        raw["experts"] = [raw["experts"][0]]
        raw["routing"]["fallback_order"] = ["cpp-expert"]
        raw["experts"][0]["model"] = "models/coder.gguf"
        raw["experts"][0]["params"]["runtime_model_source"] = "local"
        raw["experts"][0]["params"]["runtime_executable"] = "runtime/llama-server"
        config = parse_config(raw)
        with mock.patch(
            "local_moe.bootstrap.detect_platform_key",
            return_value="linux",
        ):
            command = build_runtime_plan(config).expert_commands[0]

        self.assertIn("-m", command.argv)
        self.assertEqual(command.argv[0], "runtime/llama-server")
        self.assertNotIn("-hf", command.argv)
        self.assertEqual(
            command.argv[command.argv.index("-m") + 1], "models/coder.gguf"
        )

        raw["experts"][0]["params"]["runtime_model_source"] = "mutable"
        with self.assertRaisesRegex(ValueError, "runtime_model_source"):
            build_runtime_plan(parse_config(raw))

    def test_process_bound_profile_emits_only_the_hardened_direct_flags(self) -> None:
        raw = _mixed_runtime_config_payload()
        raw["experts"] = [raw["experts"][0]]
        raw["routing"]["fallback_order"] = ["cpp-expert"]
        raw["experts"][0]["model"] = "models/coder.gguf"
        raw["experts"][0]["params"].update(
            {
                "runtime_model_source": "local",
                "runtime_executable": "runtime/llama-server",
                "runtime_security_profile": "process_bound_v1",
            }
        )

        command = build_runtime_plan(parse_config(raw)).expert_commands[0].argv

        self.assertEqual(
            command[3:],
            (
                "--alias",
                "cpp-expert",
                "--host",
                "127.0.0.1",
                "--port",
                "8201",
                "--offline",
                "--no-ui",
                "--no-ui-mcp-proxy",
                "--no-agent",
                "--no-slots",
                "--fit",
                "off",
                "--ctx-size",
                "4096",
                "--parallel",
                "1",
            ),
        )
        self.assertNotIn("-hf", command)
        self.assertNotIn("--model-url", command)

        raw["experts"][0]["params"]["runtime_security_profile"] = "future"
        with self.assertRaisesRegex(ValueError, "runtime_security_profile"):
            build_runtime_plan(parse_config(raw))


def _mixed_runtime_config():
    return parse_config(_mixed_runtime_config_payload())


def _mixed_runtime_config_payload():
    return {
        "routing": {
            "top_k": 1,
            "fallback_order": ["cpp-expert", "ollama-expert"],
            "aggregation": "best",
        },
        "experts": [
            {
                "id": "cpp-expert",
                "provider": "openai_compatible",
                "model": "owner/cpp-model",
                "role": "code",
                "base_url": "http://127.0.0.1:8201/v1",
                "params": {"runtime_backend": "llama_cpp"},
            },
            {
                "id": "ollama-expert",
                "provider": "openai_compatible",
                "model": "owner/ollama-model",
                "role": "general",
                "base_url": "http://127.0.0.1:8202/v1",
                "params": {"runtime_backend": "ollama"},
            },
        ],
        "rules": [],
    }


if __name__ == "__main__":
    unittest.main()
