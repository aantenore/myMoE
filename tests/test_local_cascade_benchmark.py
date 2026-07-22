from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.benchmark_local_cascade import (
    DEFAULT_CONFIG,
    load_config,
    render_report,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class LocalCascadeBenchmarkTests(unittest.TestCase):
    def test_report_is_deterministic_and_matches_checked_artifact(self) -> None:
        first = run_benchmark()
        second = run_benchmark()

        self.assertEqual(first, second)
        self.assertTrue(first["contract_checks_passed"])
        self.assertEqual(
            (ROOT / "outputs" / "local-cascade-contract-benchmark.json").read_text(
                encoding="utf-8"
            ),
            render_report(first),
        )
        self.assertEqual(json.loads(render_report(first)), first)

    def test_configuration_uses_replaceable_roles_and_no_authority(self) -> None:
        config = load_config(DEFAULT_CONFIG)

        self.assertEqual(
            [tier.model_ref for tier in config.ordered_tiers],
            [
                "local_cascade_utility",
                "local_cascade_resident",
                "local_cascade_specialist",
            ],
        )
        self.assertEqual(config.execution_scope, "offline_local")
        self.assertFalse(config.allow_network)
        self.assertFalse(config.allow_tools)
        self.assertFalse(config.allow_writes)
        self.assertEqual(config.parallel_attempts, 1)

    def test_metrics_do_not_collapse_unlike_token_categories(self) -> None:
        report = run_benchmark()
        tokens = report["local_token_observations"]

        self.assertEqual(tokens["actual_input_tokens"], 173)
        self.assertEqual(tokens["actual_output_tokens"], 37)
        self.assertEqual(tokens["estimated_input_tokens"], 210)
        self.assertEqual(tokens["estimated_output_tokens"], 40)
        self.assertEqual(tokens["unknown_input_attempts"], 2)
        self.assertEqual(tokens["unknown_output_attempts"], 2)
        self.assertNotIn("total_tokens", tokens)
        self.assertNotIn("savings_percentage", report)

    def test_escalation_and_counterfactual_are_explicit(self) -> None:
        report = run_benchmark()

        self.assertEqual(report["local_attempt_observations"]["total"], 8)
        self.assertEqual(report["verifier_observations"]["passed_attempts"], 3)
        self.assertEqual(
            report["verifier_observations"]["failed_completed_attempts"],
            4,
        )
        self.assertEqual(report["verifier_observations"]["exhausted_runs"], 1)
        self.assertEqual(report["premium_counterfactual"]["actual_premium_calls"], 0)
        self.assertEqual(
            report["premium_counterfactual"]["simulated_premium_calls_avoided"],
            3,
        )

    def test_reductions_are_scoped_to_bytes_not_presented_as_token_savings(
        self,
    ) -> None:
        reductions = run_benchmark()["context_and_tool_output_reduction"]

        for key in ("context_selection", "command_aware_tool_output_filter"):
            self.assertEqual(reductions[key]["measurement_unit"], "utf8_bytes")
            self.assertGreater(reductions[key]["reduction_bytes"], 0)
        self.assertEqual(
            reductions["aggregation_policy"],
            "not_aggregated_across_surfaces",
        )


if __name__ == "__main__":
    unittest.main()
