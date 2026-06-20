from __future__ import annotations

from pathlib import Path
import unittest

from local_moe.model_servers import ModelServerManager, ModelServerSpec
from local_moe.startup import run_startup_readiness


class StartupReadinessTests(unittest.TestCase):
    def test_preview_is_read_only_and_reports_doctor_state(self) -> None:
        payload = run_startup_readiness(config_path="tests/fixtures/moe.synthetic.json")

        self.assertEqual(payload["status"], "planned")
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["confirmed"])
        self.assertEqual(payload["setup"]["status"], "ready")
        self.assertEqual(payload["doctor"]["status"], "ready")
        self.assertEqual(payload["model_processes"]["count"], 0)

    def test_side_effects_require_confirmation(self) -> None:
        payload = run_startup_readiness(
            config_path="tests/fixtures/moe.synthetic.json",
            prepare=True,
            download_models=True,
            start_models=True,
            confirm=False,
        )

        self.assertEqual(payload["status"], "confirmation_required")
        self.assertFalse(payload["ok"])
        self.assertIsNone(payload["setup_run"])
        self.assertIsNone(payload["model_action"])
        self.assertTrue(any(step["status"] == "confirmation_required" for step in payload["steps"]))

    def test_confirmed_start_uses_model_manager_contract(self) -> None:
        process = _FakeProcess(pid=4321)
        created: list[tuple[tuple[str, ...], Path]] = []

        def factory(command: tuple[str, ...], log_path: Path) -> _FakeProcess:
            created.append((command, log_path))
            return process

        manager = ModelServerManager(
            (
                ModelServerSpec(
                    expert_id="general",
                    model="local/model",
                    base_url="http://127.0.0.1:9999/v1",
                    command=("python", "-m", "model_server"),
                    log_path="work/runtime/model-1.log",
                ),
            ),
            reachability_checker=lambda _url: False,
            process_factory=factory,
        )

        payload = run_startup_readiness(
            config_path="tests/fixtures/moe.synthetic.json",
            start_models=True,
            confirm=True,
            model_start_wait_seconds=0,
            model_manager=manager,
        )

        self.assertEqual(created[0][0], ("python", "-m", "model_server"))
        self.assertEqual(payload["status"], "needs_attention")
        self.assertEqual(payload["model_action"]["status"], "started")
        self.assertEqual(payload["model_processes"]["servers"][0]["pid"], 4321)


class _FakeProcess:
    def __init__(self, *, pid: int) -> None:
        self.pid = pid
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0


if __name__ == "__main__":
    unittest.main()
