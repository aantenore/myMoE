from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "experiments" / "run_quality_gate.py"
    spec = importlib.util.spec_from_file_location("run_quality_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load experiments/run_quality_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QualityGateTests(unittest.TestCase):
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
