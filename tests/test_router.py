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


if __name__ == "__main__":
    unittest.main()
