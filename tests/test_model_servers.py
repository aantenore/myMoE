from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.model_servers import (
    ModelServerManager,
    ModelServerSpec,
    build_model_server_specs,
    model_server_action_payload,
)
from local_moe.bootstrap import build_runtime_plan
from local_moe.config import parse_config
from local_moe.execution_scope import (
    ExecutionAttestation,
    ExecutionScope,
    ExecutionScopeGuard,
    ExecutionTransport,
    ScopePolicyError,
)


class ModelServerManagerTests(unittest.TestCase):
    def test_start_requires_confirmation(self) -> None:
        manager = ModelServerManager((_spec(),), reachability_checker=lambda _url: False)

        action = manager.start(confirm=False)

        payload = model_server_action_payload(action)
        self.assertEqual(payload["status"], "confirmation_required")
        self.assertFalse(payload["ok"])

    def test_start_skips_when_endpoint_is_already_reachable(self) -> None:
        manager = ModelServerManager((_spec(),), reachability_checker=lambda _url: True)

        action = manager.start(confirm=True)

        payload = model_server_action_payload(action)
        self.assertEqual(payload["status"], "skipped")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["results"][0]["status"], "external_running")

    def test_start_and_stop_managed_process(self) -> None:
        process = _FakeProcess(pid=1234)
        created: list[tuple[tuple[str, ...], Path]] = []

        def factory(command: tuple[str, ...], log_path: Path) -> _FakeProcess:
            created.append((command, log_path))
            return process

        manager = ModelServerManager(
            (_spec(),),
            reachability_checker=lambda _url: False,
            process_factory=factory,
        )

        start = manager.start(confirm=True)
        status = manager.status()
        stop_guard = manager.stop(confirm=False)
        stop = manager.stop(confirm=True)

        self.assertEqual(created[0][0], ("python", "-m", "model_server"))
        self.assertEqual(model_server_action_payload(start)["results"][0]["status"], "managed_running")
        self.assertEqual(status["servers"][0]["pid"], 1234)
        self.assertEqual(model_server_action_payload(stop_guard)["status"], "confirmation_required")
        self.assertEqual(model_server_action_payload(stop)["results"][0]["status"], "stopped")
        self.assertTrue(process.terminated)

    def test_builds_specs_from_runtime_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = parse_config(
                {
                    "routing": {"top_k": 1, "fallback_order": ["general"], "aggregation": "best"},
                    "experts": [
                        {
                            "id": "general",
                            "provider": "openai_compatible",
                            "model": "owner/model",
                            "role": "general",
                            "base_url": "http://127.0.0.1:8199/v1",
                            "params": {"runtime_backend": "llama_cpp"},
                        }
                    ],
                    "rules": [],
                }
            )
            plan = build_runtime_plan(config, {"fallback": "llama_cpp"})
            specs = build_model_server_specs(config, plan, work_dir=tmp)

        self.assertEqual(specs[0].expert_id, "general")
        self.assertEqual(specs[0].base_url, "http://127.0.0.1:8199/v1")
        self.assertEqual(specs[0].log_path.endswith("model-1.log"), True)
        self.assertIn("llama-server", specs[0].command[0])

    def test_scope_blocked_server_is_not_probed_or_started(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "remote",
                        "provider": "openai_compatible",
                        "model": "remote-model",
                        "role": "general",
                        "base_url": "https://models.example.test/v1",
                        "execution": {
                            "scope": "paid_remote",
                            "transport": "gateway",
                        },
                    }
                ],
                "rules": [],
            }
        )
        probes = 0
        starts = 0

        def probe(_url: str) -> bool:
            nonlocal probes
            probes += 1
            return True

        def start(_command: tuple[str, ...], _log_path: Path) -> _FakeProcess:
            nonlocal starts
            starts += 1
            return _FakeProcess(pid=9999)

        manager = ModelServerManager.from_config(
            config,
            reachability_checker=probe,
            process_factory=start,
        )

        status = manager.status()
        action = manager.start(confirm=True)

        self.assertEqual(status["servers"][0]["status"], "scope_blocked")
        self.assertEqual(action.status, "scope_blocked")
        self.assertFalse(action.ok)
        self.assertEqual(probes, 0)
        self.assertEqual(starts, 0)

    def test_status_and_start_use_fresh_execution_attestation(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "model": "local-model",
                        "role": "general",
                        "base_url": "http://127.0.0.1:8199/v1",
                    }
                ],
                "rules": [],
            }
        )
        attestor = _RevocableAttestor()
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        probes = 0
        starts = 0

        def probe(_url: str) -> bool:
            nonlocal probes
            probes += 1
            return False

        def start(_command: tuple[str, ...], _log_path: Path) -> _FakeProcess:
            nonlocal starts
            starts += 1
            return _FakeProcess(pid=9999)

        manager = ModelServerManager.from_config(
            config,
            reachability_checker=probe,
            process_factory=start,
            execution_guard=guard,
        )
        self.assertEqual(attestor.calls, 0)

        first_status = manager.status()
        self.assertEqual(first_status["servers"][0]["status"], "stopped")
        self.assertEqual(attestor.calls, 1)
        self.assertEqual(probes, 1)

        attestor.allowed = False
        blocked_status = manager.status()
        blocked_start = manager.start(confirm=True)

        self.assertEqual(blocked_status["servers"][0]["status"], "scope_blocked")
        self.assertEqual(blocked_start.status, "scope_blocked")
        self.assertEqual(attestor.calls, 3)
        self.assertEqual(probes, 1)
        self.assertEqual(starts, 0)

        attestor.allowed = True
        recovered_start = manager.start(confirm=True)

        self.assertEqual(recovered_start.status, "started")
        self.assertEqual(attestor.calls, 5)
        self.assertEqual(starts, 1)

    def test_start_rechecks_after_probe_before_launching_local_process(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "model": "local-model",
                        "role": "general",
                        "base_url": "http://127.0.0.1:8199/v1",
                    }
                ],
                "rules": [],
            }
        )
        attestor = _RevocableAttestor()
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        starts = 0

        def probe(_url: str) -> bool:
            attestor.allowed = False
            return False

        def start(_command: tuple[str, ...], _log_path: Path) -> _FakeProcess:
            nonlocal starts
            starts += 1
            return _FakeProcess(pid=9999)

        manager = ModelServerManager.from_config(
            config,
            reachability_checker=probe,
            process_factory=start,
            execution_guard=guard,
        )

        action = manager.start(confirm=True)

        self.assertEqual(action.status, "scope_blocked")
        self.assertEqual(attestor.calls, 2)
        self.assertEqual(starts, 0)

    def test_start_does_not_probe_again_after_authorized_process_launch(self) -> None:
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "model": "local-model",
                        "role": "general",
                        "base_url": "http://127.0.0.1:8199/v1",
                    }
                ],
                "rules": [],
            }
        )
        attestor = _RevocableAttestor()
        guard = ExecutionScopeGuard(config.execution_policy, attestor=attestor)
        probes = 0

        def probe(_url: str) -> bool:
            nonlocal probes
            probes += 1
            return False

        def start(_command: tuple[str, ...], _log_path: Path) -> _FakeProcess:
            attestor.allowed = False
            return _FakeProcess(pid=9999)

        manager = ModelServerManager.from_config(
            config,
            reachability_checker=probe,
            process_factory=start,
            execution_guard=guard,
        )

        action = manager.start(confirm=True)

        self.assertEqual(action.status, "started")
        self.assertEqual(attestor.calls, 2)
        self.assertEqual(probes, 1)
        self.assertEqual(action.results[0].status, "managed_running")

    def test_external_transport_is_probed_but_never_started_locally(self) -> None:
        config = parse_config(
            {
                "execution": {
                    "max_scope": "private_mesh",
                    "allowed_scopes": ["device_only", "private_mesh"],
                },
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "mesh",
                        "provider": "openai_compatible",
                        "model": "mesh-model",
                        "role": "general",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "execution": {
                            "scope": "private_mesh",
                            "transport": "mesh_llm",
                        },
                    }
                ],
                "rules": [],
            }
        )
        guard = ExecutionScopeGuard(
            config.execution_policy,
            attestor=_FixedMeshAttestor(),
        )
        probes = 0
        starts = 0

        def probe(_url: str) -> bool:
            nonlocal probes
            probes += 1
            return False

        def start(_command: tuple[str, ...], _log_path: Path) -> _FakeProcess:
            nonlocal starts
            starts += 1
            return _FakeProcess(pid=9999)

        manager = ModelServerManager.from_config(
            config,
            reachability_checker=probe,
            process_factory=start,
            execution_guard=guard,
        )

        action = manager.start(confirm=True)

        self.assertEqual(action.status, "external_unreachable")
        self.assertFalse(action.ok)
        self.assertEqual(action.results[0].status, "external_unreachable")
        self.assertEqual(probes, 1)
        self.assertEqual(starts, 0)

    def test_reads_sanitized_model_server_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "model-1.log"
            log_path.write_text(
                "\n".join(
                    [
                        "startup",
                        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
                        "hf_token=hf_abcdefghijklmnopqrstuvwxyz123456",
                        "ready",
                    ]
                ),
                encoding="utf-8",
            )
            manager = ModelServerManager(
                (
                    ModelServerSpec(
                        expert_id="general",
                        model="local/model",
                        base_url="http://127.0.0.1:9999/v1",
                        command=("python", "-m", "model_server"),
                        log_path=str(log_path),
                    ),
                ),
                reachability_checker=lambda _url: False,
            )

            payload = manager.logs(max_lines=3)

        self.assertEqual(payload["count"], 1)
        log = payload["logs"][0]
        self.assertEqual(log["status"], "ready")
        self.assertEqual(log["line_count"], 3)
        self.assertTrue(log["sanitized"])
        joined = "\n".join(log["lines"])
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", joined)
        self.assertIn("[REDACTED_TOKEN]", joined)
        self.assertIn("[REDACTED_SECRET]", joined)
        self.assertIn("ready", joined)

    def test_model_server_logs_report_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ModelServerManager(
                (
                    ModelServerSpec(
                        expert_id="general",
                        model="local/model",
                        base_url="http://127.0.0.1:9999/v1",
                        command=("python", "-m", "model_server"),
                        log_path=str(Path(tmp) / "missing.log"),
                    ),
                ),
                reachability_checker=lambda _url: False,
            )

            payload = manager.logs(expert_id="general")

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["expert_id"], "general")
        self.assertEqual(payload["logs"][0]["status"], "missing")
        self.assertEqual(payload["logs"][0]["lines"], [])


class _FakeProcess:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid
        self.terminated = False
        self.killed = False
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0

    def terminate(self) -> None:
        self.terminated = True
        self._running = False

    def kill(self) -> None:
        self.killed = True
        self._running = False

    def wait(self, timeout: float | None = None) -> int:
        self._running = False
        return 0


class _RevocableAttestor:
    def __init__(self) -> None:
        self.allowed = True
        self.calls = 0

    def attest(self, target) -> ExecutionAttestation:
        self.calls += 1
        if not self.allowed:
            raise ScopePolicyError("fresh model-server attestation was revoked.")
        return ExecutionAttestation(
            expert_id=target.expert_id,
            scope=ExecutionScope.DEVICE_ONLY,
            transport=ExecutionTransport.DIRECT_LOCAL,
            authority="test",
        )


class _FixedMeshAttestor:
    def attest(self, target) -> ExecutionAttestation:
        return ExecutionAttestation(
            expert_id=target.expert_id,
            scope=ExecutionScope.PRIVATE_MESH,
            transport=ExecutionTransport.MESH_LLM,
            authority="test",
        )


def _spec() -> ModelServerSpec:
    return ModelServerSpec(
        expert_id="general",
        model="local/model",
        base_url="http://127.0.0.1:9999/v1",
        command=("python", "-m", "model_server"),
        log_path="work/runtime/model-1.log",
    )


if __name__ == "__main__":
    unittest.main()
