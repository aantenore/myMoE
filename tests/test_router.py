from __future__ import annotations

import unittest

from local_moe.config import load_config
from local_moe.router import RuleRouter


class RouterTests(unittest.TestCase):
    def test_routes_coding_prompt_to_coder(self) -> None:
        config = load_config("tests/fixtures/moe.synthetic.json")
        router = RuleRouter(config)
        decision = router.route("Write Python code and tests for a class.")
        self.assertEqual(decision.selected[0].expert_id, "coder")

    def test_routes_architecture_prompt_to_architect(self) -> None:
        config = load_config("tests/fixtures/moe.synthetic.json")
        router = RuleRouter(config)
        decision = router.route("Design a scalable gateway architecture.")
        self.assertEqual(decision.selected[0].expert_id, "architect")

    def test_routes_general_prompt_to_general(self) -> None:
        config = load_config("tests/fixtures/moe.synthetic.json")
        router = RuleRouter(config)
        decision = router.route("Summarize this note into bullets.")
        self.assertEqual(decision.selected[0].expert_id, "general")

    def test_hybrid_router_routes_italian_summary_to_fast_fallback(self) -> None:
        config = load_config("configs/moe.live.general-mlx.example.json")
        router = RuleRouter(config)

        decision = router.route("Riassumi questa nota in tre punti brevi.")

        self.assertEqual(decision.selected[0].expert_id, "fast_fallback")
        self.assertTrue(any(item.startswith("semantic:") for item in decision.selected[0].matched_keywords))

    def test_hybrid_router_routes_italian_one_sentence_summary_to_fast_fallback(self) -> None:
        config = load_config("configs/moe.live.general-mlx.example.json")
        router = RuleRouter(config)

        decision = router.route("Riassumi in una frase il risultato del download automatico dei modelli GGUF.")

        self.assertEqual(decision.selected[0].expert_id, "fast_fallback")

    def test_hybrid_router_routes_italian_analysis_to_general(self) -> None:
        config = load_config("configs/moe.live.general-mlx.example.json")
        router = RuleRouter(config)

        decision = router.route("Analizza i rischi e le opportunita di una memoria locale.")

        self.assertEqual(decision.selected[0].expert_id, "general")
        self.assertTrue(any(item.startswith("semantic:") for item in decision.selected[0].matched_keywords))

    def test_hybrid_router_routes_spanish_comparison_to_general(self) -> None:
        config = load_config("configs/moe.live.general-mlx.example.json")
        router = RuleRouter(config)

        decision = router.route("Compara estas opciones para una aplicacion de escritorio local.")

        self.assertEqual(decision.selected[0].expert_id, "general")


if __name__ == "__main__":
    unittest.main()
