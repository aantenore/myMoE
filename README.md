# Local MoE Orchestrator

Goal: design and prototype a local-first, general-purpose Mixture-of-Experts system that can run on a workstation without requiring cloud inference.

This project does not try to train a monolithic MoE from scratch. That would be expensive and brittle for local hardware. The first viable architecture is a system-level MoE:

1. run one strong resident local expert plus smaller or cold-loaded experts,
2. route each request with a lightweight configurable router,
3. optionally synthesize multiple expert answers,
4. distill routing decisions and/or expert outputs later.

## Quick Start

Run the full local quality gate:

```bash
./scripts/run_all_checks.sh
```

Run the offline smoke experiment with mock experts:

```bash
PYTHONPATH=src python3 experiments/run_smoke_eval.py \
  --config configs/moe.mock.json \
  --eval experiments/eval_set.jsonl \
  --out outputs/smoke-eval.json
```

Ask the orchestrator directly:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --config configs/moe.mock.json \
  --prompt "Write Python code for a retry policy with exponential backoff"
```

## Recommended Local Model Path

For Antonio's machine class (Apple Silicon, 24 GB RAM), this is a general-purpose app, so the default is not a coder model.

- `primary general`: Qwen3-30B-A3B-Instruct-2507 MLX 4-bit.
- `stretch general`: Qwen3.6-35B-A3B MLX 4-bit with tighter context caps.
- `multimodal alternative`: Gemma 4 26B-A4B MLX 4-bit.
- `fast fallback`: Gemma 4 E4B or similar small local model.
- `optional specialist`: Qwen3-Coder-30B-A3B only for coding-heavy workflows.
- `judge/router-teacher`: use Codex/GPT-class teacher offline during dataset creation, not in runtime.

The runtime must remain local. Distillation data can be created with a stronger teacher, then used to train a local router or small student.

## Project Layout

```text
configs/
  moe.mock.json            # deterministic local mock experts
  moe.local.example.json   # template for real llama.cpp/Ollama/LM Studio endpoints
  moe.live.general-mlx.example.json
  quality-gate.json        # thresholds and project artifact checks
docs/
  architecture.md
  context-architecture.md
  distillation-plan.md
  evaluation.md
  model-selection.md
experiments/
  eval_set.jsonl
  eval_set_extended.jsonl
  run_quality_gate.py
  run_smoke_eval.py
src/local_moe/
  config.py
  cli.py
  evaluator.py
  context.py
  hardware.py
  memory.py
  router.py
  providers.py
  orchestrator.py
  runtime.py
tests/
  test_cli.py
  test_config.py
  test_evaluator.py
  test_orchestrator.py
  test_providers.py
  test_runtime.py
  test_router.py
```

## Current Experiment

The first experiment validates the routing harness, not model quality. It checks that prompts are routed to the intended expert from configuration alone. This is the right first gate because model downloads are large, while a broken router wastes every later run.

The live experiment now plugs in one real local GGUF endpoint and compares:

- single general model,
- system-level MoE top-1 routing,
- top-2 routing with synthesis.

On the detected Apple M5 Pro / 24 GB machine, the current recommendation is:

1. Use one strong resident general expert first.
2. Keep MoE as routing, context, memory, fallback, and cold-load specialist harness.
3. Keep only small fallback/compaction experts resident alongside the heavy model.
4. Add large specialist models only if evals beat the general baseline enough to justify memory and latency.

The current quality gate compiles source/tests/scripts, runs 24 unit and contract tests, evaluates 34 deterministic routing cases across the base and extended sets, checks required files, and verifies no live eval server remains on `127.0.0.1:8101`.
