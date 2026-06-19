from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from local_moe.extensions import CronJobDefinition
from local_moe.memory import FileMemoryStore
from local_moe.scheduler import cron_status, cron_summary_payload, run_due_jobs


class SchedulerTests(unittest.TestCase):
    def test_runs_memory_maintenance_and_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            state_path = Path(tmp) / "cron-state.json"
            store = FileMemoryStore(memory_path)
            store.add("Current fact")
            store.add("Expired fact", valid_until="2026-01-01T00:00:00+00:00")
            job = CronJobDefinition(
                id="memory-maintenance",
                description="Memory maintenance",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=("memory.maintenance", "--memory-path", str(memory_path)),
                risk_class="compute_only",
            )

            summary = run_due_jobs((job,), state_path=state_path, now_epoch=120)

        payload = cron_summary_payload(summary)
        self.assertEqual(payload["results"][0]["status"], "ok")
        self.assertEqual(payload["results"][0]["payload"]["total_records"], 2)
        self.assertEqual(payload["last_run_epoch"]["memory-maintenance"], 120)

    def test_dry_run_does_not_persist_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "cron-state.json"
            job = CronJobDefinition(
                id="memory-maintenance",
                description="Memory maintenance",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=("memory.maintenance",),
                risk_class="compute_only",
            )

            summary = run_due_jobs((job,), state_path=state_path, now_epoch=120, dry_run=True)
            status = cron_status((job,), state_path=state_path, now_epoch=120)

        self.assertEqual(summary.results[0].status, "dry_run")
        self.assertFalse(state_path.exists())
        self.assertTrue(status["jobs"][0]["due"])

    def test_rejects_unsupported_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = CronJobDefinition(
                id="bad",
                description="Bad job",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=("python", "script.py"),
                risk_class="process_execution",
            )

            summary = run_due_jobs((job,), state_path=Path(tmp) / "state.json", now_epoch=120)

        self.assertEqual(summary.results[0].status, "error")
        self.assertIn("Unsupported cron action", summary.results[0].message)

    def test_runs_extension_audit_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = CronJobDefinition(
                id="extension-audit",
                description="Extension audit",
                enabled=True,
                schedule={"type": "startup"},
                command=("extension.audit",),
                risk_class="compute_only",
            )

            summary = run_due_jobs((job,), state_path=Path(tmp) / "cron-state.json", now_epoch=120)

        self.assertEqual(summary.results[0].status, "ok")
        self.assertTrue(summary.results[0].payload["checked"])

    def test_runs_router_distillation_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eval_path = Path(tmp) / "eval.jsonl"
            labels_path = Path(tmp) / "labels.jsonl"
            artifact_path = Path(tmp) / "router.json"
            eval_path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "prompt": "Summarize this note",
                        "expected_expert": "general",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            job = CronJobDefinition(
                id="router-distillation-refresh",
                description="Router distillation",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=(
                    "router.distill",
                    "--eval",
                    str(eval_path),
                    "--labels",
                    str(labels_path),
                    "--artifact",
                    str(artifact_path),
                ),
                risk_class="write_local",
            )

            summary = run_due_jobs((job,), state_path=Path(tmp) / "cron-state.json", now_epoch=120)
            labels_exists = labels_path.exists()
            artifact_exists = artifact_path.exists()

        self.assertEqual(summary.results[0].status, "ok")
        self.assertEqual(summary.results[0].payload["labels"], 1)
        self.assertTrue(labels_exists)
        self.assertTrue(artifact_exists)


if __name__ == "__main__":
    unittest.main()
