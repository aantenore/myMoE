from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from local_moe.extensions import CronJobDefinition
from local_moe.memory import FileMemoryStore
from local_moe.scheduler import (
    BackgroundCronRunner,
    auto_runnable_jobs,
    cron_status,
    cron_summary_payload,
    run_due_jobs,
)


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
        self.assertEqual(payload["results"][0]["payload"]["expired_records"], 1)
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

    def test_write_local_jobs_require_confirmation(self) -> None:
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

        self.assertEqual(summary.results[0].status, "needs_confirmation")
        self.assertFalse(labels_path.exists())
        self.assertFalse(artifact_path.exists())

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

    def test_runs_runtime_optimizer_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            run_log_path = root / "runtime" / "runs.jsonl"
            job = CronJobDefinition(
                id="runtime-optimizer",
                description="Runtime optimizer",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=(
                    "runtime.optimizer",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--run-log-path",
                    str(run_log_path),
                    "--run-limit",
                    "5",
                ),
                risk_class="compute_only",
            )

            summary = run_due_jobs((job,), state_path=root / "cron-state.json", now_epoch=120)

        payload = summary.results[0].payload
        self.assertEqual(summary.results[0].status, "ok")
        self.assertEqual(summary.results[0].message, "Runtime optimizer report completed.")
        self.assertEqual(payload["mode"], "read_only")
        self.assertIn(payload["status"], {"ready", "watch", "attention"})
        self.assertEqual(payload["run_log"]["summary"]["record_count"], 0)
        self.assertIn("run_generation_smoke", {action["id"] for action in payload["actions"]})
        self.assertEqual(summary.last_run_epoch["runtime-optimizer"], 120)

    def test_runtime_optimizer_output_requires_write_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_config = _write_temp_app_config(root)
            report_path = root / "optimizer.md"
            compute_only = CronJobDefinition(
                id="runtime-optimizer",
                description="Runtime optimizer",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=(
                    "runtime.optimizer",
                    "--app-config",
                    str(app_config),
                    "--config",
                    "tests/fixtures/moe.synthetic.json",
                    "--out",
                    str(report_path),
                    "--format",
                    "markdown",
                ),
                risk_class="compute_only",
            )
            write_local = CronJobDefinition(
                id="runtime-optimizer-write",
                description="Runtime optimizer write",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=compute_only.command,
                risk_class="write_local",
            )

            rejected = run_due_jobs((compute_only,), state_path=root / "state-1.json", now_epoch=120)
            guarded = run_due_jobs((write_local,), state_path=root / "state-2.json", now_epoch=120)
            confirmed = run_due_jobs(
                (write_local,),
                state_path=root / "state-3.json",
                now_epoch=120,
                confirm_writes=True,
            )
            report_exists = report_path.exists()
            report_text = report_path.read_text(encoding="utf-8") if report_exists else ""

        self.assertEqual(rejected.results[0].status, "error")
        self.assertIn("requires a write-local risk class", rejected.results[0].message)
        self.assertEqual(guarded.results[0].status, "needs_confirmation")
        self.assertEqual(confirmed.results[0].status, "ok")
        self.assertTrue(report_exists)
        self.assertIn("# myMoE Runtime Optimizer Report", report_text)

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

            summary = run_due_jobs(
                (job,),
                state_path=Path(tmp) / "cron-state.json",
                now_epoch=120,
                confirm_writes=True,
            )
            labels_exists = labels_path.exists()
            artifact_exists = artifact_path.exists()

        self.assertEqual(summary.results[0].status, "ok")
        self.assertEqual(summary.results[0].payload["labels"], 1)
        self.assertTrue(labels_exists)
        self.assertTrue(artifact_exists)

    def test_memory_prune_expired_requires_write_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.jsonl"
            state_path = Path(tmp) / "cron-state.json"
            store = FileMemoryStore(memory_path)
            store.add("Current fact")
            expired = store.add("Expired fact", valid_until="2026-01-01T00:00:00+00:00")
            job = CronJobDefinition(
                id="memory-prune-expired",
                description="Memory prune expired",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=("memory.prune_expired", "--memory-path", str(memory_path)),
                risk_class="write_local",
            )

            guarded = run_due_jobs((job,), state_path=state_path, now_epoch=120)
            confirmed = run_due_jobs(
                (job,),
                state_path=state_path,
                now_epoch=121,
                confirm_writes=True,
            )
            remaining = FileMemoryStore(memory_path).list()

        self.assertEqual(guarded.results[0].status, "needs_confirmation")
        self.assertEqual(confirmed.results[0].status, "ok")
        self.assertEqual(confirmed.results[0].payload["removed_ids"], [expired.id])
        self.assertEqual([record.text for record in remaining], ["Current fact"])

    def test_auto_runnable_jobs_skip_write_risk_without_confirmation(self) -> None:
        safe_job = CronJobDefinition(
            id="memory-maintenance",
            description="Memory maintenance",
            enabled=True,
            schedule={"type": "interval", "seconds": 60},
            command=("memory.maintenance",),
            risk_class="compute_only",
        )
        write_job = CronJobDefinition(
            id="router-distillation-refresh",
            description="Router distillation",
            enabled=True,
            schedule={"type": "interval", "seconds": 60},
            command=("router.distill",),
            risk_class="write_local",
        )
        disabled_job = CronJobDefinition(
            id="disabled-safe",
            description="Disabled safe job",
            enabled=False,
            schedule={"type": "interval", "seconds": 60},
            command=("extension.audit",),
            risk_class="compute_only",
        )

        self.assertEqual(
            tuple(job.id for job in auto_runnable_jobs((safe_job, write_job, disabled_job))),
            ("memory-maintenance",),
        )
        self.assertEqual(
            tuple(
                job.id
                for job in auto_runnable_jobs(
                    (safe_job, write_job, disabled_job),
                    confirm_writes=True,
                )
            ),
            ("memory-maintenance", "router-distillation-refresh"),
        )

    def test_background_runner_runs_safe_jobs_and_reports_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "cron-state.json"
            safe_job = CronJobDefinition(
                id="memory-maintenance",
                description="Memory maintenance",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=("memory.maintenance", "--memory-path", str(Path(tmp) / "memory.jsonl")),
                risk_class="compute_only",
            )
            write_job = CronJobDefinition(
                id="router-distillation-refresh",
                description="Router distillation",
                enabled=True,
                schedule={"type": "interval", "seconds": 60},
                command=("router.distill",),
                risk_class="write_local",
            )
            runner = BackgroundCronRunner(
                (safe_job, write_job),
                state_path=state_path,
                enabled=True,
                confirm_writes=False,
                now_func=lambda: 120,
            )

            summary = runner.run_once()
            status = runner.status_payload()

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.due, ("memory-maintenance",))
        self.assertEqual(summary.results[0].status, "ok")
        self.assertEqual(status["policy"], "safe_jobs_only")
        self.assertEqual(status["auto_job_ids"], ["memory-maintenance"])
        self.assertEqual(status["skipped_job_ids"], ["router-distillation-refresh"])
        self.assertEqual(status["run_count"], 1)
        self.assertEqual(status["last_error"], None)


def _write_temp_app_config(root: Path) -> Path:
    raw = json.loads(Path("configs/app.json").read_text(encoding="utf-8"))
    raw["default_moe_config"] = "tests/fixtures/moe.synthetic.json"
    raw["runtime"]["work_dir"] = str(root / "runtime")
    path = root / "app.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
