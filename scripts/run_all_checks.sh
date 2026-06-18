#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m compileall src tests experiments scripts
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 experiments/run_smoke_eval.py \
  --config configs/moe.mock.json \
  --eval experiments/eval_set.jsonl \
  --out outputs/smoke-eval.json
PYTHONPATH=src python3 experiments/run_smoke_eval.py \
  --config configs/moe.mock.json \
  --eval experiments/eval_set_extended.jsonl \
  --out outputs/smoke-eval-extended.json
PYTHONPATH=src python3 experiments/run_quality_gate.py \
  --config configs/quality-gate.json \
  --out outputs/quality-gate.json
PYTHONPATH=src python3 scripts/hardware_report.py
