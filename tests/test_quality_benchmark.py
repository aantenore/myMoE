from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from local_moe.config import ExpertConfig, MoEConfig, RoutingConfig, load_config
from local_moe.quality_benchmark import (
    BenchmarkCase,
    BenchmarkSpec,
    QualityBenchmarkError,
    build_variant_config,
    check_benchmark_readiness,
    evaluate_case_output,
    load_benchmark_cases,
    load_benchmark_spec,
    run_quality_benchmark,
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
            opener=lambda *args, **kwargs: _Response({"data": [{"id": "required-model"}]}),
        )

        self.assertEqual(missing["status"], "blocked")
        self.assertFalse(missing["experts"][0]["model_available"])
        self.assertEqual(present["status"], "ready")
        self.assertTrue(present["experts"][0]["model_available"])

    def test_evaluator_separates_task_validation_and_quality_judgment(self) -> None:
        case = BenchmarkCase(
            id="case",
            prompt="Explain",
            category="reasoning",
            complexity="simple",
            task_checks=(
                {"id": "length", "type": "min_words", "value": 3},
            ),
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

    def test_unavailable_runtime_is_blocked_without_execution(self) -> None:
        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))

            payload = run_quality_benchmark(
                spec,
                readiness_checker=lambda *args, **kwargs: {
                    "status": "blocked",
                    "experts": [{"expert_id": "general", "status": "blocked"}],
                },
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertFalse(payload["gate"]["passed"])
        self.assertEqual(payload["execution"]["records"], [])
        self.assertEqual(payload["metrics"], {})

    def test_runs_all_variants_and_emits_provenance_rich_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            spec = _write_synthetic_spec(Path(tmp))

            payload = run_quality_benchmark(spec)

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["deterministic_validation"]["status"], "passed")
        self.assertEqual(payload["readiness"]["status"], "ready")
        self.assertEqual(payload["execution"]["planned_records"], 3)
        self.assertEqual(len(payload["execution"]["records"]), 3)
        self.assertEqual(set(payload["metrics"]), {"single_general", "moe_top1", "moe_top2"})
        self.assertIn("manifest_sha256", payload["provenance"])
        self.assertIn("source_config_sha256", payload["provenance"])
        self.assertIn("dataset_sha256", payload["provenance"])
        self.assertIn("quality_score", payload["metrics"]["single_general"])
        self.assertIn("latency_seconds", payload["metrics"]["moe_top2"])


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
