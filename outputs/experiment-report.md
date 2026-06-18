# Local MoE Experiment Report

Date: 2026-06-18

## What Changed

- Removed the previous Gemma local-test project artifacts.
- Created a new local-first system-level MoE prototype.
- Added config-driven routing, mock experts, OpenAI-compatible local provider boundary, CLI and eval harness.
- Added design docs and distillation plan.

## Experiment

Command:

```bash
PYTHONPATH=src python3 experiments/run_smoke_eval.py \
  --config configs/moe.mock.json \
  --eval experiments/eval_set.jsonl \
  --out outputs/smoke-eval.json
```

Initial result:

```json
{"accuracy": 0.875, "total": 8}
```

Failure:

```json
{
  "id": "mixed_code_arch",
  "expected_expert": "coder",
  "selected_expert": "architect"
}
```

Fix:

- Added `interface`, `interfaces`, `plugin` and `runner` to coder routing keywords.
- Kept the core unchanged.

Final result:

```json
{"accuracy": 1.0, "total": 8}
```

By complexity:

```json
{
  "medium": 1.0,
  "complex": 1.0,
  "simple": 1.0
}
```

## Interpretation

The harness works. The first failure was useful: ambiguous architecture+code prompts need explicit routing policy. This confirms that the system should learn or configure routing separately from provider implementation.

This does not prove model quality yet. It proves that the local MoE shell can route, preserve correlation ids, swap providers by config and evaluate routing changes.

## Recommended Next Experiment

Download or attach one real local expert and keep the others mocked:

1. Start one local `llama-server` on a dedicated port.
2. Copy `configs/moe.local.example.json` to `configs/moe.local.json`.
3. Point `coder.base_url` to that server.
4. Run the CLI against coding prompts.
5. Compare latency and quality against mock and single-model baseline.

