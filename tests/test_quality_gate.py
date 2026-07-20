from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "experiments" / "run_quality_gate.py"
    spec = importlib.util.spec_from_file_location("run_quality_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiments/run_quality_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_clean_runtime_dependency(runner):
    """Keep Git provenance fixtures hermetic while preserving production checks."""

    original_returncode = runner._git_returncode

    def returncode(root, *args):
        if args == (
            "diff",
            "--quiet",
            "--no-ext-diff",
            "HEAD",
            "--",
            "src/local_moe/config.py",
        ):
            return 0
        return original_returncode(root, *args)

    return mock.patch.object(runner, "_git_returncode", side_effect=returncode)


def _write_routing_eval_fixture(tmp_path: Path, runner):
    config_path = ROOT / "tests" / "fixtures" / "moe.synthetic.json"
    eval_path = ROOT / "experiments" / "eval_set_extended.jsonl"
    result = runner.evaluate_router(
        runner.load_config(config_path),
        runner.load_eval_cases(eval_path),
    )
    result["provenance"] = {
        "config_path": str(config_path),
        "config_sha256": runner._file_sha256(config_path),
        "eval_path": str(eval_path),
        "eval_sha256": runner._file_sha256(eval_path),
    }
    result_path = tmp_path / "routing-eval.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    config = {
        "result_path": str(result_path),
        "config_path": str(config_path),
        "eval_path": str(eval_path),
        "min_accuracy": 0.9,
        "min_total": 50,
        "required_complexities": [
            "simple",
            "medium",
            "complex",
            "very_complex",
        ],
    }
    return config, result, result_path


def _write_quality_benchmark_fixture(tmp_path: Path, runner):
    source_config_path = tmp_path / "moe.json"
    source_config_path.write_text(
        json.dumps(
            {
                "routing": {
                    "top_k": 1,
                    "fallback_order": ["general", "fast"],
                    "aggregation": "best",
                    "strategy": "rules",
                },
                "experts": [
                    {
                        "id": "general",
                        "provider": "synthetic",
                        "model": "synthetic-general",
                        "role": "general",
                    },
                    {
                        "id": "fast",
                        "provider": "synthetic",
                        "model": "synthetic-fast",
                        "role": "fast",
                    },
                ],
                "rules": [],
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "cases.jsonl"
    dataset_records = [
        {
            "id": "case-a",
            "prompt": "Give a safe answer for case A.",
            "category": "test",
            "complexity": "simple",
            "task_checks": [{"id": "nonempty", "type": "nonempty"}],
            "quality_rubric": [
                {
                    "id": "safe",
                    "type": "contains_any",
                    "values": ["safe"],
                    "weight": 1.0,
                }
            ],
        },
        {
            "id": "case-b",
            "prompt": "Give a safe answer for case B.",
            "category": "test",
            "complexity": "simple",
            "task_checks": [{"id": "nonempty", "type": "nonempty"}],
            "quality_rubric": [
                {
                    "id": "safe",
                    "type": "contains_any",
                    "values": ["safe"],
                    "weight": 1.0,
                }
            ],
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(item) for item in dataset_records) + "\n",
        encoding="utf-8",
    )
    cases_by_id = {
        case.id: case for case in runner.load_benchmark_cases(dataset_path)
    }
    benchmark_implementation_path = tmp_path / "quality_benchmark.py"
    benchmark_implementation_path.write_text("# benchmark\n", encoding="utf-8")
    evaluator_implementation_path = tmp_path / "deterministic_evaluator.py"
    evaluator_implementation_path.write_text("# evaluator\n", encoding="utf-8")

    variants = ["single_general", "moe_top1", "moe_top2"]
    repetitions = 2
    decision = {
        "baseline_variant": "single_general",
        "value_variant": "moe_top1",
        "diagnostic_variants": ["moe_top2"],
        "required_operational_variants": ["single_general", "moe_top1"],
        "minimum_task_success_rate": 1.0,
        "minimum_quality_pass_rate": 1.0,
        "minimum_quality_score": 0.7,
        "maximum_failure_rate": 0.0,
        "minimum_top1_quality_delta": 0.0,
        "minimum_top1_task_success_delta": 0.0,
        "maximum_top1_failure_rate_delta": 0.0,
        "maximum_top1_latency_ratio": 0.8,
        "maximum_top1_routed_latency_ratio": 0.8,
        "minimum_top1_non_general_route_rate": 0.5,
        "minimum_top2_complete_compare_rate": 1.0,
        "minimum_top2_disagreement_report_rate": 1.0,
        "maximum_top2_response_error_rate": 0.0,
    }
    manifest_path = tmp_path / "quality-benchmark.json"
    manifest = {
        "schema_version": 1,
        "source_config": str(source_config_path),
        "dataset": str(dataset_path),
        "general_expert_id": "general",
        "variants": variants,
        "repetitions": repetitions,
        "generation_overrides": {"temperature": 0.0},
        "evaluator": {
            "type": "deterministic_rubric",
            "quality_pass_threshold": 0.7,
        },
        "decision": decision,
        "store_outputs": True,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    records = []
    for repetition in range(1, repetitions + 1):
        for case in dataset_records:
            for variant in variants:
                benchmark_case = cases_by_id[case["id"]]
                selected = ["fast"] if variant == "moe_top1" else ["general"]
                actual = ["general", "fast"] if variant == "moe_top2" else selected
                output = f"A safe answer for {case['id']}."
                task_validation, quality_judgment = runner.evaluate_case_output(
                    benchmark_case,
                    output,
                    quality_pass_threshold=0.7,
                )
                latency = {
                    "single_general": 10.0,
                    "moe_top1": 5.0,
                    "moe_top2": 8.0,
                }[variant]
                records.append(
                    {
                        "variant": variant,
                        "case_id": case["id"],
                        "category": benchmark_case.category,
                        "complexity": benchmark_case.complexity,
                        "repetition": repetition,
                        "correlation_id": f"{variant}-{case['id']}-{repetition}",
                        "execution": {
                            "status": "ok",
                            "latency_seconds": latency,
                            "selected_experts": selected,
                            "actual_experts": actual,
                            "errors": [],
                            "disagreement_reported": variant == "moe_top2",
                            "completion_tokens": 10,
                            "output": output,
                        },
                        "task_validation": task_validation,
                        "quality_judgment": quality_judgment,
                    }
                )
    metrics = runner.summarize_records(
        records,
        variants,
        general_expert_id="general",
    )
    comparisons = runner.compare_to_baseline(metrics, decision)
    gate = runner.evaluate_benchmark_gate(metrics, comparisons, decision)
    git_root = runner._git_repository_root()
    if git_root is None:
        raise RuntimeError("Tests require the repository checkout")
    git_commit = runner._git_text(git_root, "rev-parse", "HEAD")
    if git_commit is None:
        raise RuntimeError("Tests require a readable git HEAD")
    result = {
        "schema_version": 1,
        "status": "complete",
        "provenance": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": runner._file_sha256(manifest_path),
            "source_config_path": str(source_config_path),
            "source_config_sha256": runner._file_sha256(source_config_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": runner._file_sha256(dataset_path),
            "case_ids_sha256": runner._sha256_json(
                [item["id"] for item in dataset_records]
            ),
            "case_count": len(dataset_records),
            "variants": variants,
            "repetitions": repetitions,
            "generation_overrides": manifest["generation_overrides"],
            "benchmark_implementation_sha256": runner._file_sha256(
                benchmark_implementation_path
            ),
            "evaluator_implementation_sha256": runner._file_sha256(
                evaluator_implementation_path
            ),
            "git_commit": git_commit,
            "git_dirty": False,
        },
        "deterministic_validation": {"status": "passed"},
        "readiness": {"status": "ready"},
        "execution": {
            "status": "complete",
            "planned_records": len(records),
            "records": records,
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "gate": gate,
    }
    result_path = tmp_path / "quality-benchmark-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    config = {
        "mode": "required",
        "result_path": str(result_path),
        "manifest_path": str(manifest_path),
        "benchmark_implementation_path": str(benchmark_implementation_path),
        "evaluator_implementation_path": str(evaluator_implementation_path),
        "runtime_dependency_paths": ["src/local_moe/config.py"],
    }
    return config, result, result_path


def _blocked_quality_benchmark_result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "blocked",
        "created_at": "2026-07-10T00:00:00+00:00",
        "provenance": {"manifest_sha256": "evidence"},
        "deterministic_validation": {"status": "passed", "checks": []},
        "readiness": {"status": "blocked", "experts": []},
        "execution": {"status": "not_run", "planned_records": 0, "records": []},
        "metrics": {},
        "comparisons": {"status": "not_run"},
        "gate": {
            "status": "blocked",
            "passed": False,
            "reason": "local model endpoints are unavailable",
        },
    }


class QualityGateTests(unittest.TestCase):
    def test_routing_eval_recomputes_and_rejects_hollow_report(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_routing_eval_fixture(
                Path(tmp),
                runner,
            )
            coherent = runner._check_routing_eval(config)
            result.update(
                {
                    "accuracy": 1.0,
                    "total": 50,
                    "results": [],
                }
            )
            result_path.write_text(json.dumps(result), encoding="utf-8")
            hollow = runner._check_routing_eval(config)

        self.assertTrue(coherent["passed"])
        self.assertTrue(coherent["report_matches_recomputed"])
        self.assertFalse(hollow["passed"])
        self.assertFalse(hollow["report_matches_recomputed"])

    def test_routing_eval_rejects_stale_input_provenance(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_routing_eval_fixture(
                Path(tmp),
                runner,
            )
            result["provenance"]["eval_sha256"] = "stale"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_routing_eval(config)

        self.assertFalse(check["passed"])
        self.assertTrue(check["report_matches_recomputed"])
        self.assertFalse(check["provenance_matches"]["eval_content"])

    def test_release_and_offline_ci_profiles_are_distinct(self) -> None:
        runner = _load_runner()

        release = runner._load_gate_config(ROOT / "configs" / "quality-gate.json")
        ci = runner._load_gate_config(ROOT / "configs" / "quality-gate-ci.json")

        self.assertEqual(release["profile"], "release")
        self.assertEqual(release["quality_benchmark"]["mode"], "required")
        self.assertEqual(ci["profile"], "ci_offline")
        self.assertEqual(ci["quality_benchmark"]["mode"], "offline_optional")
        self.assertEqual(ci["routing_holdout"], release["routing_holdout"])

    def test_offline_overlay_passes_without_quality_artifact_but_is_not_release_ready(
        self,
    ) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routing_config, _, _ = _write_routing_eval_fixture(root, runner)
            ci = runner._load_gate_config(ROOT / "configs" / "quality-gate-ci.json")
            ci["routing_eval"] = routing_config
            ci["quality_benchmark"]["result_path"] = str(
                root / "missing-quality-benchmark.json"
            )
            ci["forbidden_listeners"] = []

            checks = runner._run_gate_checks(ci)
            report = runner._summarize_gate(ci["profile"], checks)

        quality = next(item for item in checks if item["name"] == "quality_benchmark")
        required = next(item for item in checks if item["name"] == "required_files")
        self.assertTrue(required["passed"])
        self.assertEqual(quality["status"], "skipped")
        self.assertEqual(quality["reason"], "artifact_missing")
        self.assertTrue(report["passed"])
        self.assertFalse(report["release_ready"])

    def test_forbidden_listener_gate_covers_both_live_model_ports(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "quality-gate.json").read_text(encoding="utf-8")
        )

        ports = {item["port"] for item in config["forbidden_listeners"]}

        self.assertEqual(ports, {8101, 8102})

    def test_quality_benchmark_gate_accepts_coherent_complete_evidence(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = _write_quality_benchmark_fixture(Path(tmp), runner)

            with _mock_clean_runtime_dependency(runner):
                check = runner._check_quality_benchmark(config)

        self.assertTrue(check["passed"])
        self.assertTrue(check["release_eligible"])
        self.assertEqual(check["expected_record_count"], 12)
        self.assertTrue(all(check["checks"].values()))
        self.assertTrue(all(check["provenance_matches"].values()))

    def test_quality_benchmark_gate_rejects_forged_execution_count(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            result["execution"]["records"].pop()
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["checks"]["record_count_matches"])
        self.assertFalse(check["checks"]["record_coverage_matches"])

    def test_quality_benchmark_gate_rejects_stale_provenance_and_failed_gate(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            result["provenance"]["dataset_sha256"] = "stale"
            result["gate"] = {"status": "failed", "passed": False}
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["checks"]["provenance_matches"])
        self.assertFalse(check["provenance_matches"]["dataset_content"])
        self.assertFalse(check["checks"]["gate_passed"])

    def test_quality_benchmark_gate_recomputes_metrics_comparisons_and_gate(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            result["metrics"]["moe_top1"]["quality_score"] = 0.9
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertTrue(check["checks"]["evidence_recomputed"])
        self.assertFalse(check["checks"]["metrics_match_recomputed"])
        self.assertTrue(check["checks"]["comparisons_match_recomputed"])
        self.assertTrue(check["checks"]["gate_matches_recomputed"])

    def test_quality_benchmark_gate_rejects_forged_record_judgment_even_when_aggregates_align(
        self,
    ) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            record = next(
                item
                for item in result["execution"]["records"]
                if item["variant"] == "single_general"
            )
            record["quality_judgment"]["score"] = 0.8
            manifest = json.loads(
                Path(config["manifest_path"]).read_text(encoding="utf-8")
            )
            result["metrics"] = runner.summarize_records(
                result["execution"]["records"],
                manifest["variants"],
                general_expert_id=manifest["general_expert_id"],
            )
            result["comparisons"] = runner.compare_to_baseline(
                result["metrics"],
                manifest["decision"],
            )
            result["gate"] = runner.evaluate_benchmark_gate(
                result["metrics"],
                result["comparisons"],
                manifest["decision"],
            )
            self.assertTrue(result["gate"]["passed"])
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["checks"]["record_judgments_match"])
        validation = check["judgment_validation"]
        self.assertEqual(validation["mismatch_count"], 1)
        self.assertIn("quality_judgment", validation["mismatches"][0]["fields"])

    def test_release_quality_benchmark_requires_stored_outputs(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            manifest_path = Path(config["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["store_outputs"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result["provenance"]["manifest_sha256"] = runner._file_sha256(
                manifest_path
            )
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["checks"]["release_outputs_stored"])
        self.assertTrue(check["checks"]["record_judgments_match"])

    def test_quality_benchmark_gate_rejects_hollow_records_with_valid_keys(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            result["execution"]["records"] = [
                {
                    "variant": item["variant"],
                    "case_id": item["case_id"],
                    "repetition": item["repetition"],
                }
                for item in result["execution"]["records"]
            ]
            result_path.write_text(json.dumps(result), encoding="utf-8")

            check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertTrue(check["checks"]["record_coverage_matches"])
        self.assertFalse(check["checks"]["evidence_recomputed"])
        self.assertFalse(check["checks"]["record_judgments_match"])
        self.assertIn("record judgments", check["recomputation_error"])

    def test_quality_benchmark_gate_rejects_dirty_or_unknown_git_evidence(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            result["provenance"]["git_dirty"] = True
            result_path.write_text(json.dumps(result), encoding="utf-8")
            dirty = runner._check_quality_benchmark(config)

            result["provenance"]["git_dirty"] = False
            result["provenance"]["git_commit"] = "0" * 40
            result_path.write_text(json.dumps(result), encoding="utf-8")
            unknown = runner._check_quality_benchmark(config)

        self.assertFalse(dirty["passed"])
        self.assertFalse(dirty["git_evidence"]["git_dirty_is_false"])
        self.assertFalse(unknown["passed"])
        self.assertFalse(unknown["git_evidence"]["commit_available"])

    def test_quality_benchmark_gate_rejects_runtime_dependency_drift(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _ = _write_quality_benchmark_fixture(Path(tmp), runner)
            original_git_blob = runner._git_blob

            def stale_benchmark_blob(root, commit, relative_path):
                if commit == "HEAD":
                    return original_git_blob(root, commit, relative_path)
                return b"stale"

            with (
                _mock_clean_runtime_dependency(runner),
                mock.patch.object(
                    runner,
                    "_git_blob",
                    side_effect=stale_benchmark_blob,
                ),
            ):
                check = runner._check_quality_benchmark(config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["provenance_matches"]["git_evidence"])
        dependencies = check["git_evidence"]["runtime_dependencies"]
        dependency = dependencies["src/local_moe/config.py"]
        self.assertTrue(dependency["working_tree_clean"])
        self.assertFalse(dependency["matches"])
        self.assertNotEqual(
            dependency["current_head_sha256"],
            dependency["benchmark_commit_sha256"],
        )

    def test_release_runtime_provenance_is_verified_and_rejects_package_drift(
        self,
    ) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, result, result_path = _write_quality_benchmark_fixture(
                Path(tmp),
                runner,
            )
            manifest = json.loads(
                Path(config["manifest_path"]).read_text(encoding="utf-8")
            )
            source = runner.load_config(manifest["source_config"])
            result["provenance"]["runtime_environment"] = (
                runner.collect_runtime_environment_provenance()
            )
            result["provenance"]["model_snapshots"] = (
                runner.collect_model_snapshot_provenance(source)
            )
            config["runtime_provenance"] = {
                "required_packages": ["local-moe-orchestrator"],
                "require_model_snapshot_revision": True,
                "verify_current_environment_in_release": True,
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with _mock_clean_runtime_dependency(runner):
                coherent = runner._check_quality_benchmark(config)

                result["provenance"]["runtime_environment"]["packages"][
                    "local-moe-orchestrator"
                ]["version"] = "forged"
                result_path.write_text(json.dumps(result), encoding="utf-8")
                drifted = runner._check_quality_benchmark(config)

        self.assertTrue(coherent["passed"])
        self.assertTrue(coherent["runtime_provenance"]["current_runtime_matches"])
        self.assertFalse(drifted["passed"])
        self.assertFalse(drifted["runtime_provenance"]["current_runtime_matches"])

    def test_offline_mode_skips_blocked_live_artifact_but_is_not_release_eligible(
        self,
    ) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            config, _, result_path = _write_quality_benchmark_fixture(
                Path(tmp), runner
            )
            result = _blocked_quality_benchmark_result()
            result_path.write_text(json.dumps(result), encoding="utf-8")

            release_check = runner._check_quality_benchmark(config)
            config["mode"] = "offline_optional"
            check = runner._check_quality_benchmark(config)

        self.assertFalse(release_check["passed"])
        self.assertFalse(release_check["release_eligible"])
        self.assertTrue(check["passed"])
        self.assertEqual(check["status"], "skipped")
        self.assertFalse(check["release_eligible"])

    def test_offline_mode_rejects_failed_unknown_and_malformed_blocked_artifacts(
        self,
    ) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "quality.json"
            config = {"mode": "offline_optional", "result_path": str(result_path)}
            payloads = {
                "failed": {"status": "failed", "gate": {"passed": False}},
                "unknown": {},
                "malformed_blocked": {
                    "status": "blocked",
                    "gate": {"status": "blocked", "passed": False},
                },
            }
            for label, payload in payloads.items():
                with self.subTest(label=label):
                    result_path.write_text(json.dumps(payload), encoding="utf-8")
                    check = runner._check_quality_benchmark(config)
                    self.assertFalse(check["passed"])
                    self.assertEqual(check["status"], "failed")

    def test_offline_mode_skips_only_an_absent_artifact(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "absent.json"
            check = runner._check_quality_benchmark(
                {"mode": "offline_optional", "result_path": str(result_path)}
            )

        self.assertTrue(check["passed"])
        self.assertEqual(check["status"], "skipped")
        self.assertEqual(check["reason"], "artifact_missing")

    def test_gate_profiles_are_enumerated_and_release_success_requires_eligibility(
        self,
    ) -> None:
        runner = _load_runner()

        invalid = runner._check_gate_profile(
            {
                "profile": "relese",
                "quality_benchmark": {"mode": "required"},
            }
        )
        mismatched = runner._check_gate_profile(
            {
                "profile": "release",
                "quality_benchmark": {"mode": "offline_optional"},
            }
        )
        non_eligible = [{"name": "quality", "passed": True, "release_eligible": False}]
        release = runner._summarize_gate("release", non_eligible)
        offline = runner._summarize_gate("ci_offline", non_eligible)

        self.assertFalse(invalid["passed"])
        self.assertFalse(mismatched["passed"])
        self.assertTrue(release["checks_passed"])
        self.assertFalse(release["passed"])
        self.assertFalse(release["release_ready"])
        self.assertTrue(offline["passed"])
        self.assertFalse(offline["release_ready"])

    def test_release_config_tracks_runtime_dependencies(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "quality-gate.json").read_text(encoding="utf-8")
        )

        dependencies = set(config["quality_benchmark"]["runtime_dependency_paths"])

        self.assertTrue(
            {
                ".gitattributes",
                "pyproject.toml",
                "uv.lock",
                "experiments/run_quality_benchmark.py",
                "src/local_moe/orchestrator.py",
                "src/local_moe/providers.py",
                "src/local_moe/router.py",
                "src/local_moe/quality_benchmark.py",
                "outputs/router-distilled-live-general.json",
            }.issubset(dependencies)
        )
        policy = config["quality_benchmark"]["runtime_provenance"]
        self.assertTrue(policy["require_model_snapshot_revision"])
        self.assertTrue(policy["verify_current_environment_in_release"])

    def test_live_holdout_gate_passes_with_current_evidence(self) -> None:
        runner = _load_runner()
        config = json.loads(
            (ROOT / "configs" / "quality-gate.json").read_text(encoding="utf-8")
        )

        check = runner._check_routing_holdout(config["routing_holdout"])

        self.assertTrue(check["passed"])
        self.assertTrue(check["integrity"]["passed"])
        self.assertTrue(check["artifact_matches_training"])
        self.assertTrue(all(check["provenance_matches"].values()))

    def test_live_holdout_gate_rejects_stale_result_provenance(self) -> None:
        runner = _load_runner()
        config = json.loads(
            (ROOT / "configs" / "quality-gate.json").read_text(encoding="utf-8")
        )["routing_holdout"]
        result = json.loads(
            (ROOT / "outputs" / "live-general-routing-holdout.json").read_text(
                encoding="utf-8"
            )
        )
        result["provenance"]["holdout_data_sha256"] = "stale"

        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            stale_config = dict(config)
            stale_config["result_path"] = str(result_path)

            check = runner._check_routing_holdout(stale_config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["provenance_matches"]["holdout"])

    def test_live_holdout_gate_rejects_forged_metrics(self) -> None:
        runner = _load_runner()
        config = json.loads(
            (ROOT / "configs" / "quality-gate.json").read_text(encoding="utf-8")
        )["routing_holdout"]
        result = json.loads(
            (ROOT / "outputs" / "live-general-routing-holdout.json").read_text(
                encoding="utf-8"
            )
        )
        result["accuracy"] = 1.0
        result["total"] = 999
        result["results"] = []

        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            forged_config = dict(config)
            forged_config["result_path"] = str(result_path)

            check = runner._check_routing_holdout(forged_config)

        self.assertFalse(check["passed"])
        self.assertFalse(check["report_matches_recomputed"])

    def test_live_holdout_gate_binds_full_artifact_content(self) -> None:
        runner = _load_runner()
        gate = json.loads(
            (ROOT / "configs" / "quality-gate.json").read_text(encoding="utf-8")
        )["routing_holdout"]
        config = json.loads(
            (ROOT / gate["config_path"]).read_text(encoding="utf-8")
        )
        artifact = json.loads(
            (ROOT / gate["artifact_path"]).read_text(encoding="utf-8")
        )
        result = json.loads(
            (ROOT / gate["result_path"]).read_text(encoding="utf-8")
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = tmp_path / "artifact.json"
            artifact["untrusted_metadata"] = "content changed"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

            config_path = tmp_path / "config.json"
            config["routing"]["distilled"]["artifact_path"] = str(artifact_path)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result_path = tmp_path / "result.json"
            result["provenance"]["config_path"] = str(config_path)
            result["provenance"]["config_sha256"] = runner._file_sha256(config_path)
            result["provenance"]["artifact_path"] = str(artifact_path)
            result_path.write_text(json.dumps(result), encoding="utf-8")

            changed_config = dict(gate)
            changed_config["config_path"] = str(config_path)
            changed_config["artifact_path"] = str(artifact_path)
            changed_config["result_path"] = str(result_path)
            check = runner._check_routing_holdout(changed_config)

        self.assertFalse(check["passed"])
        self.assertTrue(check["artifact_path_matches_config"])
        self.assertFalse(check["provenance_matches"]["artifact_content"])


if __name__ == "__main__":
    unittest.main()
