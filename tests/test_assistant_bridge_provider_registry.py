from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import local_moe.assistant_bridge as assistant_bridge
from local_moe.assistant_bridge import (
    AssistantBridgeError,
    AssistantBridgeRunner,
    build_assistant_task,
    default_provider_adapter_registry,
    load_assistant_bridge_config,
)
from local_moe.assistant_bridge_provider_registry import (
    ProviderAdapterRegistry,
    ProviderAdapterRegistryError,
)


@dataclass(frozen=True)
class _Adapter:
    adapter_id: str


class _ProbeAdapter:
    adapter_id = "probe_cli"
    ephemeral_environment_keys = ("PROBE_HOME",)

    def __init__(self) -> None:
        self.runtime_calls = 0

    def validate_provider(self, provider) -> None:
        if provider.adapter != self.adapter_id:
            raise AssistantBridgeError("Probe adapter identity mismatch.")

    def runtime_descriptor(
        self,
        provider,
        task,
        *,
        local_provider_override=None,
    ):
        self.runtime_calls += 1
        authorized = (
            assistant_bridge.RISK_LEVELS[task.capability_demand.risk_class]
            <= assistant_bridge.RISK_LEVELS["write_local"]
        )
        workspace_access = (
            assistant_bridge._effective_workspace_access(
                provider,
                task.capability_demand,
                allow_remote_workspace=task.allow_remote_workspace,
            )
            if authorized
            else "not_authorized"
        )
        runtime = {
            "provider_id": provider.id,
            "adapter": provider.adapter,
            "execution_scope": provider.execution_scope,
            "model": provider.model,
            "sandbox": (
                assistant_bridge._effective_sandbox(
                    provider,
                    task.capability_demand,
                )
                if authorized
                else "not_authorized"
            ),
            "workspace_access": workspace_access,
            "agent_tool_network_access": False,
            "web_search_materialized": False,
            "user_config_ignored": True,
            "rules_ignored": True,
            "environment_keys": list(provider.environment_allowlist),
            "ephemeral_environment_keys": list(self.ephemeral_environment_keys),
            "runtime_override": local_provider_override,
        }
        runtime["runtime_sha256"] = assistant_bridge._sha256_text(
            assistant_bridge._canonical_json(runtime)
        )
        return runtime

    def build_command_plan(self, *args, **kwargs):
        raise AssertionError("blocked route must not build a command")

    def attest_remote_binding(self, provider):
        raise AssertionError("blocked route must not bind remote authority")

    def execute_command(self, *args, **kwargs):
        raise AssertionError("blocked route must not execute a command")


class ProviderAdapterRegistryTests(unittest.TestCase):
    def test_registry_is_immutable_and_has_deterministic_ids(self) -> None:
        first = _Adapter("z-adapter")
        second = _Adapter("a-adapter")
        registry = ProviderAdapterRegistry((first, second))

        self.assertEqual(registry.ids, ("a-adapter", "z-adapter"))
        self.assertEqual(len(registry), 2)
        self.assertIn("a-adapter", registry)
        self.assertIs(registry.require("a-adapter"), second)
        with self.assertRaises(TypeError):
            registry._adapters["new"] = _Adapter("new")  # type: ignore[index]

    def test_registry_rejects_empty_duplicate_and_invalid_composition(self) -> None:
        cases = (
            (),
            (_Adapter("same"), _Adapter("same")),
            (_Adapter("../invalid"),),
        )
        for adapters in cases:
            with self.subTest(adapters=adapters):
                with self.assertRaises(ProviderAdapterRegistryError):
                    ProviderAdapterRegistry(adapters)

    def test_unknown_adapter_lookup_is_explicit(self) -> None:
        registry = ProviderAdapterRegistry((_Adapter("known"),))

        with self.assertRaisesRegex(
            ProviderAdapterRegistryError,
            "not registered",
        ):
            registry.require("missing")

    def test_default_registry_exposes_only_the_production_codex_adapter(self) -> None:
        registry = default_provider_adapter_registry()

        self.assertEqual(registry.ids, ("codex_cli",))
        self.assertEqual(len(registry), 1)

    def test_provider_spec_is_composable_but_default_runner_fails_closed(
        self,
    ) -> None:
        config = load_assistant_bridge_config(Path("configs") / "assistant-bridge.json")
        custom = replace(
            config,
            local=replace(
                config.local,
                adapter="probe_cli",
                local_provider="",
            ),
            premium=replace(
                config.premium,
                adapter="probe_cli",
            ),
        )

        with self.assertRaisesRegex(AssistantBridgeError, "not registered"):
            AssistantBridgeRunner(custom, state_ledger=Mock())

        adapter = _ProbeAdapter()
        runner = AssistantBridgeRunner.with_provider_adapters(
            custom,
            adapter_registry=ProviderAdapterRegistry((adapter,)),
            state_ledger=Mock(),
        )
        task = build_assistant_task(
            "Inspect an unsupported authority request.",
            profile="offline",
            risk_class="destructive",
        )
        with tempfile.TemporaryDirectory() as temporary:
            receipt = runner.inspect_route(task, workspace=temporary)

        self.assertEqual(receipt.route, "blocked")
        self.assertEqual(receipt.local_runtime["adapter"], "probe_cli")
        self.assertEqual(
            receipt.local_runtime["ephemeral_environment_keys"],
            ("PROBE_HOME",),
        )
        self.assertEqual(adapter.runtime_calls, 2)


if __name__ == "__main__":
    unittest.main()
