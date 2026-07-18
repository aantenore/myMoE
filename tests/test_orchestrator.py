from __future__ import annotations

import unittest

from local_moe.config import load_config, parse_config
from local_moe.orchestrator import LocalMoE
from local_moe.providers import ProviderError


class FailingProvider:
    def generate(self, expert, req):
        raise ProviderError("endpoint unavailable")


class OrchestratorTests(unittest.TestCase):
    def test_preserves_correlation_id(self) -> None:
        moe = LocalMoE(load_config("tests/fixtures/moe.synthetic.json"))
        response = moe.generate("Write Python code", correlation_id="case-1")
        self.assertEqual(response.correlation_id, "case-1")
        self.assertEqual(response.results[0].correlation_id, "case-1")

    def test_parallel_compare_calls_top_k(self) -> None:
        raw = load_config("tests/fixtures/moe.synthetic.json")
        config = type(raw)(
            routing=type(raw.routing)(
                top_k=2,
                fallback_order=raw.routing.fallback_order,
                aggregation="compare",
            ),
            experts=raw.experts,
            rules=raw.rules,
        )
        moe = LocalMoE(config)
        response = moe.generate("Design Python architecture", correlation_id="case-2")
        self.assertEqual(len(response.results), 2)
        self.assertIsNotNone(response.disagreement)
        self.assertIn("Deterministic disagreement report", response.content)
        self.assertEqual(len(response.disagreement.pairwise_overlaps), 1)
        self.assertIn(response.disagreement.status, {"agreement_likely", "review_recommended"})

    def test_selected_fast_expert_can_fall_back_to_resident_general(self) -> None:
        config = parse_config(
            {
                "routing": {
                    "top_k": 1,
                    "fallback_order": ["general", "fast_fallback"],
                    "aggregation": "best",
                },
                "experts": [
                    {
                        "id": "general",
                        "provider": "synthetic",
                        "model": "general-model",
                        "role": "general",
                        "weight": 1.0,
                    },
                    {
                        "id": "fast_fallback",
                        "provider": "synthetic",
                        "model": "fast-model",
                        "role": "summary",
                        "weight": 0.4,
                    },
                ],
                "rules": [
                    {
                        "expert_id": "fast_fallback",
                        "keywords": ["summarize"],
                        "weight": 2.0,
                    }
                ],
            }
        )
        moe = LocalMoE(config)
        moe._providers["fast_fallback"] = FailingProvider()

        response = moe.generate("Summarize this note.")

        self.assertEqual(response.route.selected[0].expert_id, "fast_fallback")
        self.assertEqual(response.results[0].expert_id, "general")
        self.assertIn("endpoint unavailable", response.errors[0])

    def test_route_prompt_can_differ_from_generation_prompt(self) -> None:
        moe = LocalMoE(load_config("tests/fixtures/moe.synthetic.json"))

        response = moe.generate(
            "Relevant memory: write Python code.\n\nCurrent user message: summarize this note.",
            route_prompt="summarize this note",
            correlation_id="case-route-prompt",
        )

        self.assertEqual(response.route.selected[0].expert_id, "general")
        self.assertEqual(response.results[0].expert_id, "general")
        self.assertIn("prompt_chars=", response.content)

    def test_generate_stream_emits_route_content_and_final_response(self) -> None:
        moe = LocalMoE(load_config("tests/fixtures/moe.synthetic.json"))

        events = list(moe.generate_stream("Summarize this note.", correlation_id="case-stream"))

        self.assertEqual(events[0].kind, "route")
        self.assertEqual(events[0].route.selected[0].expert_id, "general")
        self.assertTrue(any(event.kind == "content" for event in events))
        self.assertEqual(events[-1].kind, "final")
        self.assertEqual(events[-1].response.correlation_id, "case-stream")
        self.assertEqual(events[-1].response.results[0].correlation_id, "case-stream")

    def test_parallel_stream_reuses_the_emitted_route_decision(self) -> None:
        raw = load_config("tests/fixtures/moe.synthetic.json")
        config = type(raw)(
            routing=type(raw.routing)(
                top_k=2,
                fallback_order=raw.routing.fallback_order,
                aggregation="compare",
            ),
            experts=raw.experts,
            rules=raw.rules,
        )
        moe = LocalMoE(config)

        events = list(moe.generate_stream("Design Python architecture."))

        self.assertIs(events[0].route, events[-1].response.route)


if __name__ == "__main__":
    unittest.main()
