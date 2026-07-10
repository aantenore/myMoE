from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from local_moe.config import DistilledRoutingConfig
from local_moe.distilled_router import (
    RouteLabel,
    load_distilled_router_artifact,
    train_distilled_router_artifact,
    write_distilled_router_artifact,
)


class DistilledRouterTests(unittest.TestCase):
    def test_trains_and_loads_centroid_artifact(self) -> None:
        labels = [
            RouteLabel(prompt_id="a", prompt="Write Python tests", primary="coder"),
            RouteLabel(prompt_id="b", prompt="Design system architecture", primary="architect"),
            RouteLabel(prompt_id="c", prompt="Summarize this note", primary="general"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.json"
            artifact = train_distilled_router_artifact(labels)
            write_distilled_router_artifact(artifact, path)
            loaded = load_distilled_router_artifact(
                DistilledRoutingConfig(enabled=True, artifact_path=str(path))
            )

        expert_id, confidence = loaded.predict("Please write Python unit tests")

        self.assertEqual(loaded.training_cases, 3)
        self.assertTrue(loaded.training_data_sha256)
        self.assertEqual(loaded.training_prompt_ids, ("a", "b", "c"))
        self.assertEqual(len(loaded.training_prompt_hashes), 3)
        self.assertEqual(expert_id, "coder")
        self.assertGreater(confidence, 0)

    def test_live_artifact_predicts_summary_route(self) -> None:
        loaded = load_distilled_router_artifact(
            DistilledRoutingConfig(
                enabled=True,
                artifact_path="outputs/router-distilled-live-general.json",
            )
        )

        expert_id, confidence = loaded.predict("Riassumi in una frase questo aggiornamento.")

        self.assertEqual(expert_id, "fast_fallback")
        self.assertGreater(confidence, 0)
        self.assertEqual(loaded.training_cases, 52)
        self.assertTrue(loaded.training_data_sha256)


if __name__ == "__main__":
    unittest.main()
