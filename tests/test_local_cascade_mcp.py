from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_moe.local_cascade_mcp import (
    CASCADE_CONFIG_ENV,
    MOE_CONFIG_ENV,
    CascadeAdapterUnavailable,
    DelegatePlanRequest,
    DelegateRunRequest,
    LazyCascadeAdapter,
    LocalCascadeCoreAdapter,
    LocalCascadeToolSurface,
    build_server,
)
from local_moe.local_cascade_contracts import (
    LocalCascadeConfigV1,
    LocalCascadeTierV1,
    LocalCascadeVerifierV1,
)
from local_moe.hardware import HardwareProfile


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "plugins" / "mymoe-local-cascade"


class _FakeAdapter:
    def __init__(self) -> None:
        self.plan_request: DelegatePlanRequest | None = None
        self.run_request: DelegateRunRequest | None = None

    def inspect_machine(self) -> object:
        return {
            "status": "ready",
            "machine": "arm64",
            "memory_gib": 24,
            "runtimes": [{"name": "mlx", "status": "ready"}],
            "environment": {"MODEL_API_KEY": "do-not-return"},
            "api_key": "do-not-return",
        }

    def plan(self, request: DelegatePlanRequest) -> object:
        self.plan_request = request
        return {
            "status": "ready",
            "plan_id": "plan-1",
            "plan_sha256": "a" * 64,
            "selected_tier": "small",
            "reason_codes": ["bounded_task", "local_fit"],
            "installation": {
                "model": "local-model",
                "requires_confirmation": True,
                "download_performed": False,
            },
            "task": request.task,
            "reasoning": "hidden working",
        }

    def run(self, request: DelegateRunRequest) -> object:
        self.run_request = request
        return {
            "status": "complete",
            "content": "abcdefghij",
            "model": "local-model",
            "receipt": {
                "receipt_id": "receipt-1",
                "task_sha256": request.task_sha256,
                "task": request.task,
                "secret": "do-not-return",
            },
        }

    def inspect_receipt(self, receipt_id: str) -> object:
        return {
            "status": "complete",
            "receipt_id": receipt_id,
            "task_sha256": "b" * 64,
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            "content": "private result",
            "raw_task": "private task",
        }


class _ExplodingAdapter(_FakeAdapter):
    def inspect_machine(self) -> object:
        raise RuntimeError("private/path secret-value")


class _FakeFastMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *, name: str):
        def register(function):
            self.tools[name] = function
            return function

        return register


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _RecordingOpener:
    def __init__(self, *, completion_tokens: int = 8) -> None:
        self.calls: list[tuple[object, float]] = []
        self.completion_tokens = completion_tokens

    def __call__(self, target, *, timeout: float):
        self.calls.append((target, timeout))
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "A concise local result from one expert."
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": self.completion_tokens,
                },
            }
        )


def _write_default_adapter_configs(
    root: Path,
    *,
    host: str = "127.0.0.1",
    missing_model_ref: bool = False,
) -> dict[str, str]:
    tiers = (
        LocalCascadeTierV1(
            tier_id="small",
            cost_rank=1,
            model_ref="missing" if missing_model_ref else "small",
            max_input_tokens=4_096,
            max_output_tokens=512,
        ),
        LocalCascadeTierV1(
            tier_id="strong",
            cost_rank=2,
            model_ref="strong",
            max_input_tokens=8_192,
            max_output_tokens=1_024,
        ),
    )
    cascade = LocalCascadeConfigV1(
        cascade_id="plugin-test",
        tiers=tiers,
        verifier=LocalCascadeVerifierV1(
            output_format="text",
            min_characters=1,
            max_characters=2_000,
        ),
        max_attempts=2,
    )
    cascade_path = root / "cascade.json"
    cascade_path.write_text(json.dumps(cascade.payload()), encoding="utf-8")
    moe_path = root / "moe.json"
    moe_path.write_text(
        json.dumps(
            {
                "execution": {
                    "max_scope": "device_only",
                    "allowed_scopes": ["device_only"],
                    "allow_scope_widening": False,
                },
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "small",
                        "provider": "openai_compatible",
                        "model": "small-model",
                        "role": "general",
                        "base_url": f"http://{host}:8101/v1",
                        "execution": {
                            "scope": "device_only",
                            "transport": "direct_local",
                        },
                    },
                    {
                        "id": "strong",
                        "provider": "openai_compatible",
                        "model": "strong-model",
                        "role": "general",
                        "base_url": f"http://{host}:8102/v1",
                        "execution": {
                            "scope": "device_only",
                            "transport": "direct_local",
                        },
                    },
                ],
                "rules": [],
            }
        ),
        encoding="utf-8",
    )
    return {
        CASCADE_CONFIG_ENV: str(cascade_path),
        MOE_CONFIG_ENV: str(moe_path),
    }


def _hardware_probe() -> HardwareProfile:
    return HardwareProfile(
        machine="arm64",
        cpu_brand="Local test CPU",
        memory_bytes=24 * 1024**3,
        memory_gib=24.0,
        recommended_strategy="small_single_expert",
        rationale=("test",),
    )


def _resource_probe() -> dict[str, object]:
    return {
        "system": "Darwin",
        "accelerator_kind": "integrated",
        "accelerator_memory_available_bytes": None,
        "source_sha256": "c" * 64,
    }


class LocalCascadeMcpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _FakeAdapter()
        self.surface = LocalCascadeToolSurface(self.adapter)

    def test_machine_inspection_is_read_only_and_does_not_echo_secrets(self) -> None:
        payload = self.surface.machine_inspect()

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["scope"], "local_read_only")
        self.assertEqual(payload["privacy"], "metadata_only")
        rendered = json.dumps(payload)
        self.assertNotIn("MODEL_API_KEY", rendered)
        self.assertNotIn("do-not-return", rendered)

    def test_plan_passes_raw_task_only_to_adapter_and_forces_plan_only_installation(
        self,
    ) -> None:
        task = "Summarize this bounded diff."

        payload = self.surface.delegate_plan(
            task,
            efficiency_profile="economy",
            max_steps=3,
        )

        self.assertIsNotNone(self.adapter.plan_request)
        assert self.adapter.plan_request is not None
        self.assertEqual(self.adapter.plan_request.task, task)
        self.assertEqual(self.adapter.plan_request.installation_mode, "plan_only")
        self.assertEqual(payload["task_sha256"], sha256(task.encode()).hexdigest())
        self.assertEqual(payload["installation_mode"], "plan_only")
        self.assertFalse(payload["installation_executed"])
        self.assertNotIn(task, json.dumps(payload))
        self.assertNotIn("hidden working", json.dumps(payload))

    def test_run_returns_direct_content_and_metadata_only_receipt(self) -> None:
        task = "Classify this message."

        payload = self.surface.delegate_run(
            task,
            plan_id="plan-1",
            plan_sha256="A" * 64,
            max_output_chars=256,
        )

        self.assertIsNotNone(self.adapter.run_request)
        assert self.adapter.run_request is not None
        self.assertEqual(self.adapter.run_request.task, task)
        self.assertEqual(self.adapter.run_request.plan_sha256, "a" * 64)
        self.assertEqual(payload["content"], "abcdefghij")
        self.assertFalse(payload["content_truncated"])
        rendered_receipt = json.dumps(payload["receipt"])
        self.assertNotIn(task, rendered_receipt)
        self.assertNotIn("do-not-return", rendered_receipt)

    def test_receipt_inspection_never_returns_raw_task_or_result(self) -> None:
        payload = self.surface.receipt_inspect("receipt-1")

        self.assertEqual(payload["privacy"], "metadata_only")
        rendered = json.dumps(payload)
        self.assertNotIn("private task", rendered)
        self.assertNotIn("private result", rendered)

    def test_errors_are_stable_and_do_not_echo_internal_exception_text(self) -> None:
        payload = LocalCascadeToolSurface(_ExplodingAdapter()).machine_inspect()

        self.assertEqual(payload["error"]["code"], "inspection_failed")
        self.assertNotIn("private/path", json.dumps(payload))
        self.assertNotIn("secret-value", json.dumps(payload))

    def test_installation_side_effect_is_rejected(self) -> None:
        class _UnsafeAdapter(_FakeAdapter):
            def plan(self, request: DelegatePlanRequest) -> object:
                return {
                    "status": "ready",
                    "plan_id": "plan-1",
                    "installation": {"download_performed": True},
                }

        payload = LocalCascadeToolSurface(_UnsafeAdapter()).delegate_plan(
            "A small task"
        )

        self.assertEqual(payload["error"]["code"], "installation_side_effect_rejected")

    def test_run_rejects_side_effect_outside_the_public_projection(self) -> None:
        class _UnsafeAdapter(_FakeAdapter):
            def run(self, request: DelegateRunRequest) -> object:
                return {
                    "status": "complete",
                    "content": "result",
                    "receipt": {"receipt_id": "receipt-1"},
                    "installation_executed": True,
                }

        payload = LocalCascadeToolSurface(_UnsafeAdapter()).delegate_run(
            "A small task",
            plan_id="plan-1",
            plan_sha256="a" * 64,
        )

        self.assertEqual(payload["error"]["code"], "installation_side_effect_rejected")

    def test_invalid_inputs_do_not_reach_adapter(self) -> None:
        payload = self.surface.delegate_run(
            "Small task",
            plan_id="bad id",
            plan_sha256="not-a-digest",
        )

        self.assertEqual(payload["error"]["code"], "invalid_plan_id")
        self.assertIsNone(self.adapter.run_request)

    def test_lazy_adapter_loads_once_on_first_tool_call(self) -> None:
        loads: list[int] = []

        def factory() -> _FakeAdapter:
            loads.append(1)
            return self.adapter

        lazy = LazyCascadeAdapter(factory)
        self.assertEqual(loads, [])
        lazy.inspect_machine()
        lazy.inspect_machine()
        self.assertEqual(loads, [1])

    def test_missing_core_is_reported_without_import_details(self) -> None:
        def unavailable():
            raise CascadeAdapterUnavailable("private import detail")

        payload = LocalCascadeToolSurface(
            LazyCascadeAdapter(unavailable)
        ).machine_inspect()

        self.assertEqual(payload["error"]["code"], "adapter_unavailable")
        self.assertNotIn("private import detail", json.dumps(payload))

    def test_server_registers_only_the_four_public_tools(self) -> None:
        fake_server = _FakeFastMCP()
        with patch(
            "local_moe.local_cascade_mcp._new_fastmcp", return_value=fake_server
        ):
            server = build_server(self.adapter)

        self.assertIs(server, fake_server)
        self.assertEqual(
            set(fake_server.tools),
            {"machine_inspect", "delegate_plan", "delegate_run", "receipt_inspect"},
        )

    def test_default_adapter_policies_and_one_local_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opener = _RecordingOpener()
            adapter = LocalCascadeCoreAdapter.from_environment(
                _write_default_adapter_configs(Path(tmp)),
                opener=opener,
                hardware_probe=_hardware_probe,
                resource_probe=_resource_probe,
            )
            surface = LocalCascadeToolSurface(adapter)

            machine = surface.machine_inspect()
            economy = surface.delegate_plan(
                "Summarize the local note.",
                task_kind="summarization",
                efficiency_profile="economy",
                max_steps=2,
            )
            balanced = surface.delegate_plan(
                "Summarize the local note.",
                task_kind="summarization",
                efficiency_profile="balanced",
                max_steps=2,
            )
            quality = surface.delegate_plan(
                "Summarize the local note.",
                task_kind="summarization",
                efficiency_profile="quality",
                max_steps=2,
            )
            outcome = surface.delegate_run(
                "Summarize the local note.",
                plan_id=quality["plan_id"],
                plan_sha256=quality["plan_sha256"],
            )

            self.assertEqual(machine["status"], "configured")
            self.assertEqual(
                machine["models"][0]["runtime_status"],
                "configured_not_probed",
            )
            self.assertEqual(economy["route"]["tier_ids"], ["small"])
            self.assertEqual(balanced["route"]["tier_ids"], ["small", "strong"])
            self.assertEqual(quality["route"]["tier_ids"], ["strong"])
            self.assertEqual(
                quality["route"]["execution_scope_attestation"],
                "adapter_declared_unverified",
            )
            self.assertEqual(outcome["status"], "passed")
            self.assertEqual(
                outcome["content"],
                "A concise local result from one expert.",
            )
            self.assertEqual(len(opener.calls), 1)
            self.assertIn(":8102/", opener.calls[0][0].full_url)
            self.assertEqual(
                outcome["receipt"]["token_totals"]["actual_output_tokens"],
                8,
            )
            self.assertEqual(outcome["receipt"]["schema_version"], "1.1")
            self.assertEqual(
                outcome["receipt"]["requested_execution_scope"],
                "offline_local",
            )
            self.assertEqual(
                outcome["receipt"]["execution_scope_attestation"],
                "adapter_declared_unverified",
            )
            self.assertRegex(
                outcome["receipt"]["run_id"],
                r"^cascade-run-[0-9a-f]{32}$",
            )
            self.assertRegex(
                outcome["receipt"]["evidence_sha256"],
                r"^[0-9a-f]{64}$",
            )

            receipt = surface.receipt_inspect(outcome["receipt"]["run_id"])
            rendered = json.dumps(receipt)
            self.assertEqual(receipt["privacy"], "metadata_only")
            self.assertNotIn("Summarize the local note.", rendered)
            self.assertNotIn("A concise local result", rendered)

    def test_default_adapter_rejects_task_changes_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opener = _RecordingOpener()
            adapter = LocalCascadeCoreAdapter.from_environment(
                _write_default_adapter_configs(Path(tmp)),
                opener=opener,
                hardware_probe=_hardware_probe,
                resource_probe=_resource_probe,
            )
            surface = LocalCascadeToolSurface(adapter)
            plan = surface.delegate_plan(
                "Summarize alpha.",
                task_kind="summarization",
            )

            outcome = surface.delegate_run(
                "Summarize beta.",
                plan_id=plan["plan_id"],
                plan_sha256=plan["plan_sha256"],
            )

            self.assertEqual(outcome["error"]["code"], "task_binding_mismatch")
            self.assertEqual(opener.calls, [])

    def test_default_adapter_preserves_core_token_ceiling_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opener = _RecordingOpener(completion_tokens=2_000)
            adapter = LocalCascadeCoreAdapter.from_environment(
                _write_default_adapter_configs(Path(tmp)),
                opener=opener,
                hardware_probe=_hardware_probe,
                resource_probe=_resource_probe,
            )
            surface = LocalCascadeToolSurface(adapter)
            plan = surface.delegate_plan(
                "Summarize the local note.",
                task_kind="summarization",
                efficiency_profile="quality",
            )

            outcome = surface.delegate_run(
                "Summarize the local note.",
                plan_id=plan["plan_id"],
                plan_sha256=plan["plan_sha256"],
            )

            self.assertEqual(outcome["status"], "exhausted")
            self.assertEqual(outcome["content"], "")
            self.assertEqual(outcome["reason_codes"], ["output_token_limit_exceeded"])
            self.assertEqual(len(opener.calls), 1)

    def test_default_adapter_requires_numeric_loopback_and_exact_model_refs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CascadeAdapterUnavailable) as non_numeric:
                LocalCascadeCoreAdapter.from_environment(
                    _write_default_adapter_configs(Path(tmp), host="localhost")
                )
            self.assertEqual(non_numeric.exception.code, "numeric_loopback_required")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CascadeAdapterUnavailable) as missing_ref:
                LocalCascadeCoreAdapter.from_environment(
                    _write_default_adapter_configs(
                        Path(tmp),
                        missing_model_ref=True,
                    )
                )
            self.assertEqual(missing_ref.exception.code, "model_ref_not_configured")

    def test_default_adapter_missing_configuration_is_actionable_and_private(
        self,
    ) -> None:
        payload = LocalCascadeToolSurface(
            LazyCascadeAdapter(lambda: LocalCascadeCoreAdapter.from_environment({}))
        ).machine_inspect()

        self.assertEqual(payload["error"]["code"], "configuration_required")
        rendered = json.dumps(payload)
        self.assertIn(CASCADE_CONFIG_ENV, rendered)
        self.assertIn(MOE_CONFIG_ENV, rendered)
        self.assertNotIn("/Users/", rendered)

    def test_plugin_manifest_and_stdio_server_are_repo_local(self) -> None:
        manifest = json.loads(
            (PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
        )
        mcp = json.loads((PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8"))
        server = mcp["mcpServers"]["mymoe-local-cascade"]

        self.assertEqual(manifest["name"], "mymoe-local-cascade")
        self.assertEqual(manifest["mcpServers"], "./.mcp.json")
        self.assertEqual(server["command"], "python3")
        self.assertEqual(server["cwd"], ".")
        self.assertNotIn("env", server)
        self.assertNotIn("download", json.dumps(mcp).lower())

    def test_standalone_plugin_launcher_uses_explicit_project_root_offline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "cached-plugin"
            shutil.copytree(PLUGIN_ROOT, copied)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "MYMOE_PROJECT_ROOT": str(ROOT),
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    str(copied / "scripts" / "launch_mcp.py"),
                    "--dry-run",
                ],
                cwd=copied,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload, {"status": "ready", "mode": "project_root_offline"})
        self.assertNotIn(str(ROOT), completed.stdout)


if __name__ == "__main__":
    unittest.main()
