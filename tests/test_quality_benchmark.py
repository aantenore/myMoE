from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest import mock

from local_moe.config import ExpertConfig, MoEConfig, RoutingConfig, load_config
from local_moe.orchestrator import LocalMoE
from local_moe.quality_benchmark import (
    BenchmarkCase,
    BenchmarkSpec,
    QualityBenchmarkError,
    build_variant_config,
    check_benchmark_readiness,
    collect_model_snapshot_provenance,
    compare_to_baseline,
    evaluate_case_output,
    evaluate_benchmark_gate,
    evaluate_host_memory_gate,
    load_benchmark_cases,
    load_benchmark_spec,
    run_quality_benchmark,
    summarize_records,
)


class _Response:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class QualityBenchmarkTests(unittest.TestCase):
    def test_loads_manifest_and_cases(self) -> None:
        spec = load_benchmark_spec("configs/quality-benchmark.json")
        cases = load_benchmark_cases(spec.dataset_path)

        self.assertEqual(spec.variants, ("single_general", "moe_top1", "moe_top2"))
        self.assertEqual(spec.generation_overrides["max_tokens"], 768)
        self.assertEqual(spec.decision["maximum_truncation_rate"], 0.0)
        self.assertEqual(spec.decision["maximum_top1_truncation_rate"], 0.0)
        self.assertEqual(spec.repetitions, 3)
        self.assertEqual(spec.decision["minimum_top1_routed_case_count"], 6)
        self.assertEqual(
            spec.decision["maximum_operational_mean_latency_seconds"],
            30.0,
        )
        self.assertTrue(spec.decision["host_memory"]["required"])
        self.assertEqual(
            spec.decision["host_memory"]["minimum_sample_coverage"],
            0.9,
        )
        self.assertGreaterEqual(len(cases), 8)
        self.assertTrue(all(case.task_checks and case.quality_rubric for case in cases))

    def test_rejects_duplicate_case_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            case = {
                "id": "same",
                "prompt": "Prompt",
                "task_checks": [{"id": "nonempty", "type": "nonempty"}],
                "quality_rubric": [
                    {"id": "nonempty", "type": "nonempty", "weight": 1.0}
                ],
            }
            path.write_text(
                json.dumps(case) + "\n" + json.dumps(case) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(QualityBenchmarkError, "Duplicate"):
                load_benchmark_cases(path)

    def test_rejects_vacuous_and_misshaped_deterministic_checks(self) -> None:
        invalid_checks = (
            {"id": "empty", "type": "contains_all", "values": []},
            {"id": "empty", "type": "contains_all_groups", "groups": []},
            {"id": "typed", "type": "min_words", "value": True},
            {"id": "unknown", "type": "nonempty", "unexpected": "value"},
        )

        for check in invalid_checks:
            with self.subTest(check=check), TemporaryDirectory() as tmp:
                path = Path(tmp) / "cases.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "id": "invalid",
                            "prompt": "Prompt",
                            "task_checks": [check],
                            "quality_rubric": [
                                {
                                    "id": "quality",
                                    "type": "nonempty",
                                    "weight": 1.0,
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                with self.assertRaises(QualityBenchmarkError):
                    load_benchmark_cases(path)

    def test_builds_isolated_single_and_moe_variants(self) -> None:
        source = load_config("tests/fixtures/moe.synthetic.json")

        single = build_variant_config(
            source,
            "single_general",
            general_expert_id="general",
            generation_overrides={"temperature": 0.0},
        )
        top1 = build_variant_config(source, "moe_top1", general_expert_id="general")
        top2 = build_variant_config(source, "moe_top2", general_expert_id="general")

        self.assertEqual([expert.id for expert in single.experts], ["general"])
        self.assertEqual(single.routing.strategy, "rules")
        self.assertFalse(single.routing.semantic.enabled)
        self.assertFalse(single.routing.distilled.enabled)
        self.assertEqual(single.experts[0].params["temperature"], 0.0)
        self.assertEqual(top1.routing.top_k, 1)
        self.assertEqual(top1.routing.aggregation, "best")
        self.assertEqual(top2.routing.top_k, 2)
        self.assertEqual(top2.routing.aggregation, "compare")

    def test_readiness_requires_configured_model_not_just_endpoint(self) -> None:
        config = _http_config(model="required-model")

        missing = check_benchmark_readiness(
            config,
            timeout_seconds=0.1,
            model_match="exact",
            opener=lambda *args, **kwargs: _Response({"data": [{"id": "other-model"}]}),
        )
        present = check_benchmark_readiness(
            config,
            timeout_seconds=0.1,
            model_match="exact",
            opener=lambda *args, **kwargs: _Response(
                {"data": [{"id": "required-model"}]}
            ),
        )

        self.assertEqual(missing["status"], "blocked")
        self.assertFalse(missing["experts"][0]["model_available"])
        self.assertEqual(present["status"], "ready")
        self.assertTrue(present["experts"][0]["model_available"])

    def test_readiness_does_not_probe_endpoint_outside_execution_policy(self) -> None:
        config = _http_config(model="remote-model")
        remote = replace(
            config.experts[0],
            base_url="https://models.example.test/v1",
        )
        config = replace(config, experts=(remote,))

        def unexpected_opener(*_args, **_kwargs):
            raise AssertionError("blocked endpoint must not be probed")

        result = check_benchmark_readiness(
            config,
            timeout_seconds=0.1,
            model_match="exact",
            opener=unexpected_opener,
        )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["experts"][0]["reason_code"], "scope_blocked")

    def test_model_snapshot_revision_comes_from_local_huggingface_cache(self) -> None:
        revision = "a" * 40
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            repository = cache / "models--example--model"
            (repository / "refs").mkdir(parents=True)
            (repository / "snapshots" / revision).mkdir(parents=True)
            (repository / "refs" / "main").write_text(revision, encoding="ascii")
            config = _http_config(model="example/model")

            with mock.patch.dict(
                "os.environ",
                {"HF_HUB_CACHE": str(cache), "HF_HOME": str(cache / "unused")},
                clear=False,
            ):
                snapshots = collect_model_snapshot_provenance(config)

        self.assertEqual(snapshots[0]["status"], "resolved")
        self.assertEqual(snapshots[0]["revision"], revision)

    def test_required_host_memory_gate_rejects_growth_peak_and_unavailable_swap(
        self,
    ) -> None:
        policy = {
            "required": True,
            "minimum_sample_coverage": 0.9,
            "maximum_swap_growth_bytes": 50,
            "maximum_peak_ram_used_percent": 95.0,
        }
        observation = {
            "status": "available",
            "before": _host_memory_sample(800, 10),
            "after": _host_memory_sample(900, 20),
            "peak_observed": {
                "memory": {"status": "available", "used_bytes": 940},
                "swap": {"status": "available", "used_bytes": 20},
            },
        }

        passing = evaluate_host_memory_gate(observation, policy)
        decision = _gate_decision()
        decision["host_memory"] = policy
        metrics = _gate_metrics()
        comparisons = compare_to_baseline(
            metrics,
            {"baseline_variant": "single_general"},
        )
        integrated_passing = evaluate_benchmark_gate(
            metrics,
            comparisons,
            decision,
            host_memory=observation,
        )
        observation["after"] = _host_memory_sample(900, 100)
        swap_growth = evaluate_host_memory_gate(observation, policy)
        observation["after"] = _host_memory_sample(900, 20)
        observation["peak_observed"]["memory"]["used_bytes"] = 960
        ram_peak = evaluate_host_memory_gate(observation, policy)
        observation["before"]["swap"] = {"status": "unavailable"}
        unavailable = evaluate_host_memory_gate(observation, policy)
        integrated_unavailable = evaluate_benchmark_gate(
            metrics,
            comparisons,
            decision,
            host_memory=observation,
        )

        self.assertTrue(passing["passed"])
        self.assertTrue(integrated_passing["passed"])
        self.assertEqual(passing["swap_growth_bytes"], 10)
        self.assertEqual(passing["peak_ram_used_percent"], 94.0)
        self.assertFalse(swap_growth["passed"])
        self.assertFalse(ram_peak["passed"])
        self.assertFalse(unavailable["passed"])
        self.assertFalse(unavailable["counters_available"])
        self.assertFalse(integrated_unavailable["passed"])
        self.assertFalse(integrated_unavailable["host_memory_check"]["passed"])

    def test_required_host_memory_gate_tolerates_only_configured_sample_gaps(
        self,
    ) -> None:
        policy = {
            "required": True,
            "minimum_sample_coverage": 0.9,
            "maximum_swap_growth_bytes": 50,
            "maximum_peak_ram_used_percent": 95.0,
        }
        observation = {
            "status": "partial",
            "sample_count": 100,
            "available_sample_count": 96,
            "before": _host_memory_sample(800, 10),
            "after": _host_memory_sample(900, 20),
            "peak_observed": {
                "memory": {"status": "available", "used_bytes": 940},
                "swap": {"status": "available", "used_bytes": 20},
            },
        }

        sufficient = evaluate_host_memory_gate(observation, policy)
        observation["available_sample_count"] = 89
        insufficient = evaluate_host_memory_gate(observation, policy)

        self.assertTrue(sufficient["passed"])
        self.assertEqual(sufficient["sample_coverage"], 0.96)
        self.assertFalse(insufficient["passed"])
        self.assertEqual(insufficient["sample_coverage"], 0.89)

    def test_evaluator_separates_task_validation_and_quality_judgment(self) -> None:
        case = BenchmarkCase(
            id="case",
            prompt="Explain",
            category="reasoning",
            complexity="simple",
            task_checks=({"id": "length", "type": "min_words", "value": 3},),
            quality_rubric=(
                {
                    "id": "risk",
                    "type": "contains_any",
                    "values": ["risk"],
                    "weight": 0.75,
                },
                {
                    "id": "concise",
                    "type": "max_words",
                    "value": 10,
                    "weight": 0.25,
                },
            ),
        )

        task, quality = evaluate_case_output(
            case,
            "A short answer without the expected topic.",
            quality_pass_threshold=0.7,
        )

        self.assertTrue(task["passed"])
        self.assertFalse(quality["passed"])
        self.assertEqual(quality["score"], 0.25)
        self.assertIn("evidence", quality["criteria"][0])

    def test_summary_evidence_limit_accepts_negative_semantic_paraphrases(
        self,
    ) -> None:
        cases = load_benchmark_cases("experiments/quality_benchmark_cases.jsonl")
        case = next(item for item in cases if item.id == "three_bullet_summary")
        valid_outputs = (
            "- Routes between a general model and fallback.\n"
            "- Keeps memory, health diagnostics, and tools local.\n"
            "- Current evidence shows no improvement in answer quality over a single model.",
            "- Routes between a general model and fallback.\n"
            "- Keeps memory, health diagnostics, and tools local.\n"
            "- It has not yet demonstrated superior answer quality compared to a single model.",
        )
        positive_claim = (
            "- Routes between a general model and fallback.\n"
            "- Keeps memory, health diagnostics, and tools local.\n"
            "- It demonstrates superior answer quality compared to a single model."
        )

        for output in valid_outputs:
            _, quality = evaluate_case_output(
                case,
                output,
                quality_pass_threshold=0.7,
            )
            self.assertTrue(quality["passed"])
        _, positive_quality = evaluate_case_output(
            case,
            positive_claim,
            quality_pass_threshold=0.7,
        )
        self.assertFalse(positive_quality["passed"])

    def test_unavailable_runtime_is_blocked_without_execution(self) -> None:
        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))
            memory_calls: list[bool] = []

            payload = run_quality_benchmark(
                spec,
                readiness_checker=lambda *args, **kwargs: {
                    "status": "blocked",
                    "experts": [{"expert_id": "general", "status": "blocked"}],
                },
                memory_sampler=lambda: (
                    memory_calls.append(True) or _host_memory_sample(1, 1)
                ),
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertFalse(payload["gate"]["passed"])
        self.assertEqual(payload["execution"]["records"], [])
        self.assertEqual(payload["metrics"], {})
        self.assertEqual(memory_calls, [])

    def test_runs_all_variants_and_emits_provenance_rich_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))
            sampler = _SequenceMemorySampler(
                [
                    _host_memory_sample(100, 10),
                    _host_memory_sample(110, 11),
                    _host_memory_sample(130, 13),
                    _host_memory_sample(120, 12),
                    _host_memory_sample(160, 16),
                    _host_memory_sample(140, 14),
                    _host_memory_sample(200, 20),
                    _host_memory_sample(180, 18),
                ]
            )

            payload = run_quality_benchmark(
                spec,
                memory_sampler=sampler,
                memory_sample_interval_seconds=0.0,
            )

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["deterministic_validation"]["status"], "passed")
        self.assertEqual(payload["readiness"]["status"], "ready")
        self.assertEqual(payload["execution"]["planned_records"], 3)
        self.assertEqual(len(payload["execution"]["records"]), 3)
        self.assertEqual(
            set(payload["metrics"]), {"single_general", "moe_top1", "moe_top2"}
        )
        self.assertIn("manifest_sha256", payload["provenance"])
        self.assertIn("source_config_sha256", payload["provenance"])
        self.assertIn("dataset_sha256", payload["provenance"])
        self.assertIn("quality_score", payload["metrics"]["single_general"])
        self.assertIn("latency_seconds", payload["metrics"]["moe_top2"])
        self.assertEqual(payload["metrics"]["moe_top1"]["route_fulfillment_rate"], 1.0)
        self.assertEqual(payload["metrics"]["moe_top1"]["truncation_rate"], 0.0)
        self.assertEqual(
            payload["metrics"]["moe_top1"]["finish_reason_counts"],
            {"unknown": 1},
        )
        self.assertEqual(sampler.calls, 8)
        memory_provenance = payload["provenance"]["host_memory_sampling"]
        self.assertEqual(memory_provenance["scope"], "host")
        self.assertEqual(memory_provenance["content"], "metadata_only")
        self.assertEqual(memory_provenance["record_sample_interval_seconds"], 0.0)

        run_memory = payload["execution"]["host_memory"]
        self.assertEqual(run_memory["status"], "available")
        self.assertEqual(run_memory["sample_count"], 8)
        self.assertEqual(run_memory["before"]["memory"]["used_bytes"], 100)
        self.assertEqual(run_memory["after"]["memory"]["used_bytes"], 180)
        self.assertEqual(run_memory["peak_observed"]["memory"]["used_bytes"], 200)
        self.assertEqual(run_memory["peak_observed"]["swap"]["used_bytes"], 20)

        records = payload["execution"]["records"]
        self.assertEqual(
            records[0]["execution"]["host_memory"]["peak_observed"]["memory"][
                "used_bytes"
            ],
            130,
        )
        self.assertEqual(
            payload["metrics"]["single_general"]["host_memory"]["memory_used_bytes"][
                "peak_observed_max"
            ],
            130,
        )
        self.assertEqual(
            payload["metrics"]["moe_top2"]["host_memory"]["swap_used_bytes"][
                "peak_observed_max"
            ],
            20,
        )

    def test_memory_sampler_failure_is_explicit_and_does_not_break_benchmark(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))

            def unavailable_sampler() -> dict[str, object]:
                raise OSError("host counters unavailable")

            payload = run_quality_benchmark(
                spec,
                memory_sampler=unavailable_sampler,
                memory_sample_interval_seconds=0.0,
            )

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["execution"]["host_memory"]["status"], "unavailable")
        self.assertEqual(
            payload["execution"]["host_memory"]["peak_observed"]["memory"],
            {"status": "unavailable"},
        )
        for metrics in payload["metrics"].values():
            self.assertEqual(metrics["host_memory"]["status"], "unavailable")
            self.assertEqual(
                metrics["host_memory"]["memory_used_bytes"],
                {"status": "unavailable"},
            )

    def test_run_records_length_finish_reason_and_fails_operational_gate(self) -> None:
        class _LengthFinishRunner:
            def __init__(self, config: MoEConfig) -> None:
                self._runner = LocalMoE(config)

            def generate(self, *args: object, **kwargs: object):
                response = self._runner.generate(*args, **kwargs)
                return replace(
                    response,
                    results=tuple(
                        replace(result, finish_reason="length")
                        for result in response.results
                    ),
                )

        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))
            payload = run_quality_benchmark(
                spec,
                moe_factory=_LengthFinishRunner,
                memory_sampler=lambda: _host_memory_sample(100, 10),
                memory_sample_interval_seconds=0.0,
            )

        self.assertTrue(
            all(
                record["execution"]["truncated"]
                for record in payload["execution"]["records"]
            )
        )
        self.assertTrue(
            all(
                item["finish_reason"] == "length"
                for record in payload["execution"]["records"]
                for item in record["execution"]["finish_reasons"]
            )
        )
        self.assertEqual(
            payload["metrics"]["single_general"]["truncation_rate"],
            1.0,
        )
        self.assertTrue(
            all(not check["passed"] for check in payload["gate"]["operational_checks"])
        )

    def test_periodic_sampler_captures_peak_between_record_endpoints(self) -> None:
        peak_sampled = threading.Event()

        def sampler() -> dict[str, object]:
            if threading.current_thread() is threading.main_thread():
                return _host_memory_sample(100, 10)
            peak_sampled.set()
            return _host_memory_sample(900, 90)

        class _WaitForPeakRunner:
            def __init__(self, config: MoEConfig) -> None:
                self._runner = LocalMoE(config)

            def generate(self, *args: object, **kwargs: object):
                peak_sampled.clear()
                if not peak_sampled.wait(timeout=1.0):
                    raise RuntimeError("periodic memory sample was not observed")
                return self._runner.generate(*args, **kwargs)

        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))
            payload = run_quality_benchmark(
                spec,
                moe_factory=_WaitForPeakRunner,
                memory_sampler=sampler,
                memory_sample_interval_seconds=0.001,
            )

        for record in payload["execution"]["records"]:
            observation = record["execution"]["host_memory"]
            self.assertEqual(observation["before"]["memory"]["used_bytes"], 100)
            self.assertEqual(observation["after"]["memory"]["used_bytes"], 100)
            self.assertEqual(
                observation["peak_observed"]["memory"]["used_bytes"],
                900,
            )

    def test_top1_non_regression_and_latency_win_passes_value_gate(self) -> None:
        metrics = _gate_metrics()
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertTrue(gate["passed"])
        self.assertEqual(metrics["moe_top1"]["finish_reason_counts"], {"unknown": 8})
        self.assertTrue(gate["moe_value_checks"][0]["passed"])
        self.assertTrue(gate["diagnostic_checks"][0]["passed"])
        self.assertTrue(
            all(
                check["absolute_latency_passed"] for check in gate["operational_checks"]
            )
        )

    def test_absolute_operational_latency_cap_fails_even_when_relative_value_passes(
        self,
    ) -> None:
        metrics = _gate_metrics()
        metrics["single_general"]["latency_seconds"] = {"mean": 40.0}
        metrics["moe_top1"]["latency_seconds"] = {"mean": 36.0}
        comparisons = compare_to_baseline(
            metrics,
            {"baseline_variant": "single_general"},
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertTrue(gate["moe_value_checks"][0]["passed"])
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["operational_thresholds"]["maximum_operational_mean_latency_seconds"],
            30.0,
        )
        self.assertTrue(
            all(
                not check["absolute_latency_passed"]
                for check in gate["operational_checks"]
            )
        )

    def test_route_fulfillment_compares_selected_and_actual_experts(self) -> None:
        records = [
            _summary_record(
                case_id="fulfilled",
                selected_experts=["fast_fallback"],
                actual_experts=["fast_fallback"],
            ),
            _summary_record(
                case_id="fallback",
                selected_experts=["fast_fallback"],
                actual_experts=["general"],
                errors=["fast_fallback: unavailable"],
            ),
        ]

        metrics = summarize_records(
            records,
            ("moe_top1",),
            general_expert_id="general",
        )["moe_top1"]

        self.assertEqual(metrics["route_fulfillment_rate"], 0.5)
        self.assertEqual(metrics["response_error_rate"], 0.5)

    def test_top1_unfulfilled_route_cannot_claim_value(self) -> None:
        metrics = _gate_metrics()
        metrics["moe_top1"]["route_fulfillment_rate"] = 0.875
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["moe_value_checks"][0]["passed"])
        self.assertEqual(gate["moe_value_checks"][0]["route_fulfillment_rate"], 0.875)

    def test_top1_response_error_cannot_claim_value(self) -> None:
        metrics = _gate_metrics()
        metrics["moe_top1"]["response_error_rate"] = 0.125
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["moe_value_checks"][0]["passed"])
        self.assertEqual(gate["moe_value_checks"][0]["response_error_rate"], 0.125)

    def test_finish_reason_metrics_count_truncation_and_unknown(self) -> None:
        records = [
            _summary_record(
                case_id="truncated",
                selected_experts=["fast_fallback"],
                actual_experts=["fast_fallback"],
                finish_reasons=["length"],
            ),
            _summary_record(
                case_id="legacy",
                selected_experts=["fast_fallback"],
                actual_experts=["fast_fallback"],
            ),
        ]

        metrics = summarize_records(
            records,
            ("moe_top1",),
            general_expert_id="general",
        )["moe_top1"]

        self.assertEqual(
            metrics["finish_reason_counts"],
            {"length": 1, "unknown": 1},
        )
        self.assertEqual(metrics["truncation_rate"], 0.5)

    def test_top1_truncation_fails_value_and_operational_checks(self) -> None:
        metrics = _gate_metrics()
        metrics["moe_top1"]["finish_reason_counts"] = {"length": 1, "stop": 7}
        metrics["moe_top1"]["truncation_rate"] = 0.125
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        top1_operational = next(
            item for item in gate["operational_checks"] if item["variant"] == "moe_top1"
        )
        self.assertFalse(top1_operational["passed"])
        self.assertEqual(top1_operational["truncation_rate"], 0.125)
        self.assertFalse(gate["moe_value_checks"][0]["passed"])
        self.assertEqual(gate["moe_value_checks"][0]["truncation_rate"], 0.125)

    def test_routed_latency_uses_median_of_paired_case_ratios(self) -> None:
        metrics = _gate_metrics()
        metrics["single_general"]["latency_seconds_by_case"] = {
            "long#1": 100.0,
            "short#1": 1.0,
        }
        metrics["moe_top1"]["latency_seconds_by_case"] = {
            "long#1": 50.0,
            "short#1": 2.0,
        }
        metrics["moe_top1"]["non_general_case_keys"] = ["long#1", "short#1"]

        comparison = compare_to_baseline(
            metrics,
            {"baseline_variant": "single_general"},
        )["candidates"]["moe_top1"]

        self.assertEqual(
            comparison["routed_latency_ratios_by_case"],
            {"long#1": 0.5, "short#1": 2.0},
        )
        self.assertEqual(comparison["routed_latency_ratio"], 1.25)
        self.assertEqual(
            comparison["routed_latency_ratio_statistic"],
            "median_of_per_case_ratios",
        )
        gate = evaluate_benchmark_gate(
            metrics,
            {"status": "complete", "candidates": {"moe_top1": comparison}},
            _gate_decision(),
        )
        self.assertFalse(gate["moe_value_checks"][0]["passed"])

    def test_too_few_routed_cases_cannot_claim_latency_value(self) -> None:
        metrics = _gate_metrics()
        metrics["moe_top1"]["non_general_case_keys"] = ["fast#1"]
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertEqual(gate["moe_value_checks"][0]["routed_case_count"], 1)
        self.assertFalse(gate["moe_value_checks"][0]["passed"])

    def test_top2_cannot_hide_a_top1_regression(self) -> None:
        metrics = _gate_metrics()
        metrics["moe_top1"]["quality_score"] = 0.79
        metrics["moe_top2"]["quality_score"] = 1.0
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["moe_value_checks"][0]["passed"])
        self.assertEqual(gate["moe_value_checks"][0]["variant"], "moe_top1")

    def test_incomplete_top2_compare_fails_only_diagnostic_contract(self) -> None:
        metrics = _gate_metrics()
        metrics["moe_top2"]["complete_compare_rate"] = 0.875
        comparisons = compare_to_baseline(
            metrics, {"baseline_variant": "single_general"}
        )

        gate = evaluate_benchmark_gate(metrics, comparisons, _gate_decision())

        self.assertFalse(gate["passed"])
        self.assertTrue(gate["moe_value_checks"][0]["passed"])
        self.assertFalse(gate["diagnostic_checks"][0]["passed"])


class _SequenceMemorySampler:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self._samples = samples
        self.calls = 0

    def __call__(self) -> dict[str, object]:
        sample = self._samples[self.calls]
        self.calls += 1
        return sample


def _host_memory_sample(memory_used: int, swap_used: int) -> dict[str, object]:
    memory_total = 1_000
    swap_total = 500
    return {
        "status": "available",
        "scope": "host",
        "source": "deterministic_test_sampler",
        "memory": {
            "status": "available",
            "total_bytes": memory_total,
            "available_bytes": memory_total - memory_used,
            "used_bytes": memory_used,
        },
        "swap": {
            "status": "available",
            "total_bytes": swap_total,
            "free_bytes": swap_total - swap_used,
            "used_bytes": swap_used,
        },
    }


def _gate_metrics() -> dict[str, dict[str, object]]:
    base = {
        "planned": 8,
        "completed": 8,
        "failures": 0,
        "failure_rate": 0.0,
        "task_success_rate": 1.0,
        "quality_pass_rate": 1.0,
        "quality_score": 0.8,
        "non_general_route_rate": 0.0,
        "complete_compare_rate": 0.0,
        "disagreement_report_rate": 0.0,
        "response_error_rate": 0.0,
        "route_fulfillment_rate": 1.0,
        "finish_reason_counts": {"unknown": 8},
        "truncation_rate": 0.0,
        "latency_seconds": {"mean": 10.0},
        "latency_seconds_by_case": {"fast#1": 10.0, "summary#1": 20.0},
        "non_general_case_keys": [],
    }
    return {
        "single_general": dict(base),
        "moe_top1": {
            **base,
            "non_general_route_rate": 0.25,
            "latency_seconds": {"mean": 9.0},
            "latency_seconds_by_case": {"fast#1": 8.0, "summary#1": 12.0},
            "non_general_case_keys": ["fast#1", "summary#1"],
        },
        "moe_top2": {
            **base,
            "quality_score": 0.1,
            "complete_compare_rate": 1.0,
            "disagreement_report_rate": 1.0,
            "latency_seconds": {"mean": 15.0},
        },
    }


def _gate_decision() -> dict[str, object]:
    return {
        "baseline_variant": "single_general",
        "value_variant": "moe_top1",
        "diagnostic_variants": ["moe_top2"],
        "required_operational_variants": ["single_general", "moe_top1"],
        "minimum_task_success_rate": 1.0,
        "minimum_quality_pass_rate": 1.0,
        "minimum_quality_score": 0.7,
        "maximum_failure_rate": 0.0,
        "maximum_truncation_rate": 0.0,
        "maximum_operational_mean_latency_seconds": 30.0,
        "minimum_top1_quality_delta": 0.0,
        "minimum_top1_task_success_delta": 0.0,
        "maximum_top1_failure_rate_delta": 0.0,
        "maximum_top1_latency_ratio": 0.9,
        "maximum_top1_routed_latency_ratio": 0.8,
        "minimum_top1_routed_case_count": 2,
        "minimum_top1_non_general_route_rate": 0.2,
        "minimum_top1_route_fulfillment_rate": 1.0,
        "maximum_top1_response_error_rate": 0.0,
        "maximum_top1_truncation_rate": 0.0,
        "minimum_top2_complete_compare_rate": 1.0,
        "minimum_top2_disagreement_report_rate": 1.0,
        "maximum_top2_response_error_rate": 0.0,
    }


def _summary_record(
    *,
    case_id: str,
    selected_experts: list[str],
    actual_experts: list[str],
    errors: list[str] | None = None,
    finish_reasons: list[str] | None = None,
) -> dict[str, object]:
    execution: dict[str, object] = {
        "status": "ok",
        "latency_seconds": 1.0,
        "selected_experts": selected_experts,
        "actual_experts": actual_experts,
        "errors": errors or [],
        "disagreement_reported": False,
        "completion_tokens": 10,
    }
    if finish_reasons is not None:
        execution["finish_reasons"] = [
            {"expert_id": expert_id, "finish_reason": finish_reason}
            for expert_id, finish_reason in zip(actual_experts, finish_reasons)
        ]
    return {
        "variant": "moe_top1",
        "case_id": case_id,
        "category": "summary",
        "complexity": "simple",
        "repetition": 1,
        "execution": execution,
        "task_validation": {"passed": True},
        "quality_judgment": {"passed": True, "score": 1.0},
    }


def _http_config(*, model: str) -> MoEConfig:
    expert = ExpertConfig(
        id="general",
        provider="openai_compatible",
        model=model,
        role="general",
        base_url="http://127.0.0.1:8101/v1",
    )
    return MoEConfig(routing=RoutingConfig(), experts=(expert,), rules=())


def _write_synthetic_spec(root: Path) -> BenchmarkSpec:
    manifest = root / "manifest.json"
    dataset = root / "cases.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    dataset.write_text(
        json.dumps(
            {
                "id": "one",
                "prompt": "Design Python architecture",
                "category": "reasoning",
                "complexity": "simple",
                "task_checks": [{"id": "nonempty", "type": "nonempty"}],
                "quality_rubric": [
                    {"id": "nonempty", "type": "nonempty", "weight": 1.0}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return BenchmarkSpec(
        manifest_path=manifest,
        source_config_path=Path("tests/fixtures/moe.synthetic.json"),
        dataset_path=dataset,
        general_expert_id="general",
        variants=("single_general", "moe_top1", "moe_top2"),
        repetitions=1,
        endpoint_timeout_seconds=0.1,
        model_match="exact",
        generation_overrides={"temperature": 0.0},
        evaluator={"type": "deterministic_rubric", "quality_pass_threshold": 0.7},
        decision={
            "baseline_variant": "single_general",
            "minimum_task_success_rate": 0.0,
            "minimum_quality_score": 0.0,
            "maximum_failure_rate": 1.0,
            "minimum_moe_quality_delta": 0.0,
            "maximum_moe_latency_ratio": 100.0,
        },
        store_outputs=True,
    )


if __name__ == "__main__":
    unittest.main()
