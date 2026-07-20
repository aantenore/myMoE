from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_moe.assistant_bridge import (
    AssistantBridgeError,
    CapabilityDemand,
    CommandResult,
    build_assistant_task,
    build_codex_command_plan,
    load_assistant_bridge_config,
    plan_assistant_route,
)
from local_moe.assistant_bridge_provider_registry import ProviderAdapterRegistry
from local_moe.assistant_provider_probe import (
    AssistantProviderProbeOperationalError,
    AssistantProviderProbeError,
    EXIT_COMPATIBLE,
    EXIT_CONTRACT,
    EXIT_INCOMPATIBLE,
    EXIT_OPERATIONAL,
    _capture_model_identity,
    _reconcile_model_identity,
    main,
    run_local_provider_probe,
    write_probe_report,
)


class _ProbeAdapter:
    adapter_id = "codex_cli"
    ephemeral_environment_keys = ("CODEX_HOME", "HOME")

    def __init__(self, *, mode: str = "pass") -> None:
        self.mode = mode
        self.workspace: Path | None = None
        self.plan_kwargs: dict[str, object] | None = None
        self.provider = None

    def build_command_plan(self, provider, **kwargs):
        self.provider = provider
        self.workspace = Path(kwargs["workspace"])
        self.plan_kwargs = dict(kwargs)
        prompt = kwargs["prompt"]
        output_path = Path(kwargs["output_path"])
        payload = {
            "provider_id": provider.id,
            "adapter_id": provider.adapter,
            "mode": provider.mode,
            "argv_sha256": "b" * 64,
            "argv_shape": ["<executable>", "exec", "-"],
            "stdin": {
                "transport": "stdin",
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "characters": len(prompt),
                "content_in_argv": False,
            },
            "workspace_sha256": hashlib.sha256(
                str(self.workspace.resolve()).encode("utf-8")
            ).hexdigest(),
            "output_path_sha256": hashlib.sha256(
                str(output_path.resolve()).encode("utf-8")
            ).hexdigest(),
            "command_sha256": "a" * 64,
            "sandbox": "read-only",
            "permission_profile": "mymoe_workspace_read",
            "permission_profile_effective_attested": False,
            "permission_workspace_rule": "read",
            "network_access": False,
            "shell_network_access": False,
            "web_search_mode": "disabled",
            "workspace_access": "read_only",
            "model": provider.model,
            "local_provider": provider.local_provider,
            "environment_keys": list(provider.environment_allowlist),
            "ephemeral_environment_keys": ["CODEX_HOME", "HOME"],
            "executable": {"sha256": "c" * 64},
            "environment_sha256": "d" * 64,
            "runtime": {
                "schema_version": "assistant-bridge-runtime-capabilities/v1",
                "strict_tree_supported": True,
            },
            "runtime_policy": {"require_tree_isolation": True},
            "launcher_chain": {
                "schema_version": "assistant-bridge-launcher-chain/v1",
                "fingerprint": "e" * 64,
                "strict": True,
            },
            "launcher_authority_sha256": "f" * 64,
            "launcher_artifact_sha256": [],
        }
        return SimpleNamespace(
            command_sha256="a" * 64,
            payload=lambda: payload,
        )

    def execute_command(self, provider, plan, **kwargs):
        assert self.workspace is not None
        marker = (self.workspace / "MYMOE_PROBE_MARKER.txt").read_text(
            encoding="ascii"
        ).strip()
        if self.mode == "pass":
            output = f"MYMOE_TOOL_OK:{marker}"
            status = "completed"
            code = "launcher_completed"
            returncode = 0
        elif self.mode == "mismatch":
            output = "I cannot access the workspace."
            status = "completed"
            code = "launcher_completed"
            returncode = 0
        elif self.mode == "short_mismatch":
            output = "a"
            status = "completed"
            code = "launcher_completed"
            returncode = 0
        else:
            output = ""
            status = "failed"
            code = "launcher_timeout"
            returncode = None
        return CommandResult(
            provider_id=provider.id,
            status=status,
            code=code,
            returncode=returncode,
            duration_ms=321,
            output=output,
            stdout_sha256="b" * 64,
            stdout_bytes=123,
            stderr_sha256="c" * 64,
            stderr_bytes=45,
            command_sha256=plan.command_sha256,
            prompt_tokens=100,
            completion_tokens=20,
        )


class AssistantProviderProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        identity = {
            "status": "mutable_reference_unverified",
            "reference": "qwen3:4b",
            "reason_code": "test_runtime_has_no_model_identity",
        }
        model_patch = patch(
            "local_moe.assistant_provider_probe._capture_model_identity",
            return_value=identity,
        )
        model_patch.start()
        self.addCleanup(model_patch.stop)

    def test_shipped_config_routes_unproven_tool_work_away_from_local(self) -> None:
        config = load_assistant_bridge_config(
            Path("configs") / "assistant-bridge.json"
        )
        task = build_assistant_task(
            "Repair a parser and run its tests.",
            profile="balanced",
            required_capabilities=("code", "tests"),
            required_tools=("filesystem", "shell"),
            risk_class="write_local",
            allow_remote=True,
            allow_remote_workspace=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            receipt = plan_assistant_route(task, config, workspace=temporary)

        self.assertEqual(receipt.route, "premium")
        self.assertIn("capability:code", receipt.local_gaps)
        self.assertIn("tool:filesystem", receipt.local_gaps)
        self.assertIn("risk:write_local", receipt.local_gaps)
        self.assertEqual(config.local.sandbox, "read-only")
        self.assertEqual(config.local.workspace_access, "read_only")

    def test_primary_cli_dispatches_probe_without_a_second_console_script(self) -> None:
        from local_moe import cli

        with (
            patch.object(sys, "argv", ["mymoe", "assistant-probe", "--json"]),
            patch(
                "local_moe.assistant_provider_probe.main",
                return_value=EXIT_COMPATIBLE,
            ) as probe_main,
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main()

        self.assertEqual(raised.exception.code, EXIT_COMPATIBLE)
        probe_main.assert_called_once_with(["--json"])

    def test_direct_command_plan_cannot_exceed_provider_declarations(self) -> None:
        config = load_assistant_bridge_config(
            Path("configs") / "assistant-bridge.json"
        )
        provider = config.local
        demands = (
            CapabilityDemand(required=("code",)),
            CapabilityDemand(required=("analysis",), tools=("shell",)),
            CapabilityDemand(required=("analysis",), risk_class="write_local"),
            CapabilityDemand(required=("web",), tools=("web",)),
        )
        for demand in demands:
            with self.subTest(demand=demand), self.assertRaisesRegex(
                AssistantBridgeError, "declared authority"
            ):
                build_codex_command_plan(
                    provider,
                    prompt="bounded",
                    workspace=Path.cwd(),
                    demand=demand,
                )

        executable_provider = replace(provider, executable=sys.executable)
        with self.assertRaisesRegex(AssistantBridgeError, "workspace ceiling"):
            build_codex_command_plan(
                executable_provider,
                prompt="bounded",
                workspace=Path.cwd(),
                demand=CapabilityDemand(required=("analysis",)),
                workspace_access="read_write",
            )

    def test_pass_proves_workspace_read_without_leaking_marker(self) -> None:
        adapter = _ProbeAdapter()
        marker = "mymoe-" + "d" * 32

        report = run_local_provider_probe(
            Path("configs") / "assistant-bridge.json",
            timeout_seconds=12,
            adapter_registry=ProviderAdapterRegistry((adapter,)),
            now=lambda: datetime(2026, 7, 20, 3, 0, tzinfo=timezone.utc),
            marker_factory=lambda: marker,
        )

        self.assertEqual(report["status"], "compatible")
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["authorizes_routing"])
        self.assertEqual(
            report["observed_capabilities"],
            ["codex_tool_protocol", "filesystem_read"],
        )
        self.assertEqual(report["reason_codes"], ["workspace_marker_recovered"])
        self.assertNotIn(marker, json.dumps(report, sort_keys=True))
        assert adapter.plan_kwargs is not None
        self.assertEqual(adapter.plan_kwargs["workspace_access"], "read_only")
        self.assertEqual(adapter.plan_kwargs["demand"].risk_class, "read_only")
        self.assertEqual(adapter.provider.capabilities, ("filesystem",))
        self.assertEqual(adapter.provider.tools, ("shell",))
        self.assertEqual(adapter.provider.max_risk, "read_only")
        self.assertEqual(adapter.provider.sandbox, "read-only")
        self.assertEqual(adapter.provider.workspace_access, "read_only")
        self.assertEqual(report["provider"]["declared_capabilities"], ["analysis"])
        self.assertEqual(report["provider"]["declared_tools"], [])
        declared = (Path("configs") / "assistant-bridge.json").read_bytes()
        self.assertEqual(
            report["binding"]["bridge_config_declared_bytes_sha256"],
            hashlib.sha256(declared).hexdigest(),
        )
        self.assertEqual(
            report["execution_identity"]["command_plan"]["command_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            report["execution_identity"]["model"]["status"],
            "mutable_reference_unverified",
        )

    def test_completed_response_without_marker_is_incompatible(self) -> None:
        report = run_local_provider_probe(
            Path("configs") / "assistant-bridge.json",
            adapter_registry=ProviderAdapterRegistry(
                (_ProbeAdapter(mode="mismatch"),)
            ),
            marker_factory=lambda: "mymoe-" + "e" * 32,
        )

        self.assertEqual(report["status"], "incompatible")
        self.assertEqual(report["observed_capabilities"], [])
        self.assertEqual(
            report["reason_codes"], ["workspace_marker_not_recovered"]
        )

    def test_short_mismatch_is_incompatible_without_serializing_raw_output(self) -> None:
        report = run_local_provider_probe(
            Path("configs") / "assistant-bridge.json",
            adapter_registry=ProviderAdapterRegistry(
                (_ProbeAdapter(mode="short_mismatch"),)
            ),
            marker_factory=lambda: "mymoe-" + "1" * 32,
        )

        self.assertEqual(report["status"], "incompatible")
        self.assertEqual(report["result"]["output_chars"], 1)
        self.assertEqual(
            report["result"]["output_sha256"],
            hashlib.sha256(b"a").hexdigest(),
        )
        self.assertNotIn("output", report["result"])
        with patch(
            "local_moe.assistant_provider_probe.run_local_provider_probe",
            return_value=report,
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--json"]), EXIT_INCOMPATIBLE)

    def test_result_metadata_rejects_raw_content_fields(self) -> None:
        class _LeakyResult(CommandResult):
            def metadata_payload(self):
                payload = super().metadata_payload()
                payload["output"] = self.output
                return payload

        class _LeakyAdapter(_ProbeAdapter):
            def execute_command(self, provider, plan, **kwargs):
                return _LeakyResult(
                    provider_id=provider.id,
                    status="completed",
                    code="launcher_completed",
                    returncode=0,
                    duration_ms=1,
                    output="raw provider response",
                    command_sha256=plan.command_sha256,
                )

        with self.assertRaisesRegex(
            AssistantProviderProbeError, "invalid result contract"
        ):
            run_local_provider_probe(
                Path("configs") / "assistant-bridge.json",
                adapter_registry=ProviderAdapterRegistry((_LeakyAdapter(),)),
                marker_factory=lambda: "mymoe-" + "2" * 32,
            )

    def test_result_must_bind_exact_command_and_provider(self) -> None:
        class _MissingCommandAdapter(_ProbeAdapter):
            def execute_command(self, provider, plan, **kwargs):
                result = super().execute_command(provider, plan, **kwargs)
                return replace(result, command_sha256=None)

        class _WrongProviderAdapter(_ProbeAdapter):
            def execute_command(self, provider, plan, **kwargs):
                result = super().execute_command(provider, plan, **kwargs)
                return replace(result, provider_id="another-provider")

        for adapter, message in (
            (_MissingCommandAdapter(), "inspected command plan"),
            (_WrongProviderAdapter(), "inspected provider"),
        ):
            with self.subTest(adapter=type(adapter).__name__), self.assertRaisesRegex(
                AssistantProviderProbeError,
                message,
            ):
                run_local_provider_probe(
                    Path("configs") / "assistant-bridge.json",
                    adapter_registry=ProviderAdapterRegistry((adapter,)),
                    marker_factory=lambda: "mymoe-" + "5" * 32,
                )

    def test_completed_result_requires_success_code_and_returncode(self) -> None:
        class _ContradictoryCompletionAdapter(_ProbeAdapter):
            def execute_command(self, provider, plan, **kwargs):
                result = super().execute_command(provider, plan, **kwargs)
                return replace(
                    result,
                    code="runtime_attestation_failed",
                    returncode=7,
                )

        with self.assertRaisesRegex(
            AssistantProviderProbeError,
            "invalid result contract",
        ):
            run_local_provider_probe(
                Path("configs") / "assistant-bridge.json",
                adapter_registry=ProviderAdapterRegistry(
                    (_ContradictoryCompletionAdapter(),)
                ),
                marker_factory=lambda: "mymoe-" + "6" * 32,
            )

    def test_plain_output_includes_reason_and_json_hint(self) -> None:
        report = run_local_provider_probe(
            Path("configs") / "assistant-bridge.json",
            adapter_registry=ProviderAdapterRegistry(
                (_ProbeAdapter(mode="timeout"),)
            ),
            marker_factory=lambda: "mymoe-" + "7" * 32,
        )
        stdout = io.StringIO()
        with patch(
            "local_moe.assistant_provider_probe.run_local_provider_probe",
            return_value=report,
        ), redirect_stdout(stdout):
            self.assertEqual(main([]), EXIT_OPERATIONAL)

        self.assertIn("launcher_timeout", stdout.getvalue())
        self.assertIn("Use --json for evidence", stdout.getvalue())

    def test_command_plan_metadata_rejects_prompt_fields(self) -> None:
        class _LeakyPlanAdapter(_ProbeAdapter):
            def build_command_plan(self, provider, **kwargs):
                plan = super().build_command_plan(provider, **kwargs)
                return SimpleNamespace(
                    command_sha256=plan.command_sha256,
                    payload=lambda: {
                        **plan.payload(),
                        "prompt": kwargs["prompt"],
                    },
                )

        with self.assertRaisesRegex(
            AssistantProviderProbeError, "public identity schema"
        ):
            run_local_provider_probe(
                Path("configs") / "assistant-bridge.json",
                adapter_registry=ProviderAdapterRegistry((_LeakyPlanAdapter(),)),
                marker_factory=lambda: "mymoe-" + "3" * 32,
            )

    def test_command_plan_requires_complete_bound_authority_metadata(self) -> None:
        class _IncompletePlanAdapter(_ProbeAdapter):
            def build_command_plan(self, provider, **kwargs):
                plan = super().build_command_plan(provider, **kwargs)
                payload = plan.payload()
                payload.pop("permission_profile")
                return SimpleNamespace(
                    command_sha256=plan.command_sha256,
                    payload=lambda: payload,
                )

        class _MismatchedPlanAdapter(_ProbeAdapter):
            def build_command_plan(self, provider, **kwargs):
                plan = super().build_command_plan(provider, **kwargs)
                payload = plan.payload()
                payload["shell_network_access"] = True
                return SimpleNamespace(
                    command_sha256=plan.command_sha256,
                    payload=lambda: payload,
                )

        for adapter, message in (
            (_IncompletePlanAdapter(), "public identity schema"),
            (_MismatchedPlanAdapter(), "authority metadata"),
        ):
            with self.subTest(adapter=type(adapter).__name__), self.assertRaisesRegex(
                AssistantProviderProbeError,
                message,
            ):
                run_local_provider_probe(
                    Path("configs") / "assistant-bridge.json",
                    adapter_registry=ProviderAdapterRegistry((adapter,)),
                    marker_factory=lambda: "mymoe-" + "4" * 32,
                )

    def test_launcher_failure_is_preserved_as_a_reason_code(self) -> None:
        report = run_local_provider_probe(
            Path("configs") / "assistant-bridge.json",
            adapter_registry=ProviderAdapterRegistry(
                (_ProbeAdapter(mode="timeout"),)
            ),
            marker_factory=lambda: "mymoe-" + "f" * 32,
        )

        self.assertEqual(report["status"], "indeterminate")
        self.assertEqual(report["reason_codes"], ["launcher_timeout"])
        self.assertEqual(report["result"]["duration_ms"], 321)

    def test_adapter_contract_failure_maps_to_contract_exit(self) -> None:
        class _ContractFailureAdapter(_ProbeAdapter):
            def execute_command(self, provider, plan, **kwargs):
                raise AssistantBridgeError("provider/plan contract mismatch")

        registry = ProviderAdapterRegistry((_ContractFailureAdapter(),))
        with self.assertRaises(AssistantProviderProbeError):
            run_local_provider_probe(
                Path("configs") / "assistant-bridge.json",
                adapter_registry=registry,
                marker_factory=lambda: "mymoe-" + "8" * 32,
            )

        with patch(
            "local_moe.assistant_provider_probe.default_provider_adapter_registry",
            return_value=registry,
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "--bridge-config",
                        "configs/assistant-bridge.json",
                        "--json",
                    ]
                ),
                EXIT_CONTRACT,
            )

    def test_report_write_is_private_atomic_and_rejects_links(self) -> None:
        report = {"schema_version": "test/v1", "status": "compatible"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = write_probe_report(root / "reports" / "probe.json", report)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), report)
            if os.name != "nt":
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)

            linked = root / "linked.json"
            try:
                linked.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"links unavailable: {exc}")
            with self.assertRaises(AssistantProviderProbeOperationalError):
                write_probe_report(linked, report)

            outside = root / "outside"
            outside.mkdir()
            redirected_parent = root / "redirected-parent"
            try:
                redirected_parent.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory links unavailable: {exc}")
            with self.assertRaises(AssistantProviderProbeOperationalError):
                write_probe_report(redirected_parent / "probe.json", report)
            self.assertFalse((outside / "probe.json").exists())

    def test_cli_exit_codes_distinguish_result_and_error_classes(self) -> None:
        provider = {"id": "local", "model": "test"}
        cases = (
            ("compatible", EXIT_COMPATIBLE),
            ("incompatible", EXIT_INCOMPATIBLE),
            ("indeterminate", EXIT_OPERATIONAL),
        )
        for status, expected in cases:
            with self.subTest(status=status), patch(
                "local_moe.assistant_provider_probe.run_local_provider_probe",
                return_value={"status": status, "provider": provider},
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--json"]), expected)

        with patch(
            "local_moe.assistant_provider_probe.run_local_provider_probe",
            side_effect=AssistantProviderProbeError("invalid contract"),
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--json"]), EXIT_CONTRACT)

    def test_rejects_unbounded_timeout_and_invalid_marker(self) -> None:
        registry = ProviderAdapterRegistry((_ProbeAdapter(),))
        for timeout in (0, 301):
            with self.subTest(timeout=timeout):
                with self.assertRaises(AssistantProviderProbeError):
                    run_local_provider_probe(
                        Path("configs") / "assistant-bridge.json",
                        timeout_seconds=timeout,
                        adapter_registry=registry,
                    )
        with self.assertRaises(AssistantProviderProbeError):
            run_local_provider_probe(
                Path("configs") / "assistant-bridge.json",
                adapter_registry=registry,
                marker_factory=lambda: "predictable",
            )


class AssistantProviderModelIdentityTests(unittest.TestCase):
    class _Response:
        def __init__(self, payload: bytes, *, status: int = 200) -> None:
            self.payload = payload
            self.status = status

        def read(self, limit: int) -> bytes:
            return self.payload[:limit]

    class _Connection:
        def __init__(self, response) -> None:
            self.response = response
            self.request_args = None
            self.closed = False

        def request(self, *args, **kwargs) -> None:
            self.request_args = (args, kwargs)

        def getresponse(self):
            return self.response

        def close(self) -> None:
            self.closed = True

    def _provider(self):
        return load_assistant_bridge_config(
            Path("configs") / "assistant-bridge.json"
        ).local

    def test_ollama_identity_is_content_addressed_and_loopback_bounded(self) -> None:
        payload = json.dumps(
            {
                "models": [
                    {
                        "name": "qwen3:4b",
                        "model": "qwen3:4b",
                        "digest": "a" * 64,
                        "size": 123,
                    }
                ]
            }
        ).encode("utf-8")
        connection = self._Connection(self._Response(payload))
        factory_args = None

        def factory(*args, **kwargs):
            nonlocal factory_args
            factory_args = (args, kwargs)
            return connection

        with patch(
            "local_moe.assistant_provider_probe.http.client.HTTPConnection",
            side_effect=factory,
        ):
            identity = _capture_model_identity(self._provider(), 45)

        self.assertEqual(identity["status"], "content_addressed")
        self.assertEqual(identity["digest"], "sha256:" + "a" * 64)
        self.assertEqual(identity["size_bytes"], 123)
        self.assertEqual(factory_args, (("127.0.0.1", 11434), {"timeout": 1.0}))
        self.assertEqual(
            connection.request_args,
            (("GET", "/api/tags"), {"headers": {"Accept": "application/json"}}),
        )
        self.assertTrue(connection.closed)

    def test_ollama_identity_fails_closed_on_oversized_or_ambiguous_data(self) -> None:
        oversized = self._Connection(self._Response(b"x" * 9))
        with patch(
            "local_moe.assistant_provider_probe.http.client.HTTPConnection",
            return_value=oversized,
        ), patch("local_moe.assistant_provider_probe._MAX_OLLAMA_TAGS_BYTES", 8):
            identity = _capture_model_identity(self._provider(), 1)
        self.assertEqual(identity["status"], "mutable_reference_unverified")
        self.assertEqual(identity["reason_code"], "ollama_tags_response_too_large")

        ambiguous_payload = json.dumps(
            {
                "models": [
                    {"name": "qwen3:4b", "digest": "a" * 64, "size": 123},
                    {"name": "qwen3:4b", "digest": "b" * 64, "size": 456},
                ]
            }
        ).encode("utf-8")
        ambiguous = self._Connection(self._Response(ambiguous_payload))
        with patch(
            "local_moe.assistant_provider_probe.http.client.HTTPConnection",
            return_value=ambiguous,
        ):
            identity = _capture_model_identity(self._provider(), 1)
        self.assertEqual(identity["status"], "mutable_reference_unverified")
        self.assertEqual(
            identity["reason_code"],
            "ollama_model_reference_missing_or_ambiguous",
        )

    def test_model_identity_reconciliation_requires_one_stable_digest(self) -> None:
        stable = {
            "status": "content_addressed",
            "reference": "qwen3:4b",
            "digest": "sha256:" + "a" * 64,
            "size_bytes": 123,
            "source": "ollama_loopback_tags_api",
        }
        reconciled = _reconcile_model_identity(stable, stable)
        self.assertTrue(reconciled["stable_during_probe"])

        changed = dict(stable, digest="sha256:" + "b" * 64)
        reconciled = _reconcile_model_identity(stable, changed)
        self.assertEqual(reconciled["status"], "mutable_reference_unverified")
        self.assertEqual(
            reconciled["reason_code"], "model_identity_changed_during_probe"
        )


if __name__ == "__main__":
    unittest.main()
