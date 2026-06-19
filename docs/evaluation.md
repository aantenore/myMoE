# Evaluation

## Baselines

Always compare at least:

1. single local general model,
2. MoE top-1 routing,
3. MoE top-2 routing with synthesis or disagreement reporting.

## Metrics

| Dimension | Measurement | Gate |
| --- | --- | --- |
| Routing accuracy | expected expert vs selected expert | >= 0.90 on curated eval |
| Latency | wall-clock seconds and tokens/sec | must be acceptable interactively |
| Completeness | rubric per task family | no regression vs single model |
| Failure transparency | errors preserve correlation id and expert id | required |
| Compare transparency | top-k compare responses expose deterministic disagreement metadata | required for `aggregation = compare` |
| Local footprint | RAM/VRAM/disk | must fit target machine |

## Current Eval Sets

The base eval set is intentionally small and deterministic. It covers:

- coding,
- architecture,
- general language,
- mixed coding/architecture prompts.

The extended eval set adds broader router pressure:

- coding provider and test requests,
- architecture and gateway decisions,
- general writing and summarization,
- mixed prompts where keywords intentionally collide.

The live general eval set validates the default app profile:

- `general` for analysis, comparison, decisions, planning, and research-style prompts,
- `fast_fallback` for summarization, compression, rewriting, and translation prompts.
- multilingual route coverage across English, Italian, Spanish, French, German, and Portuguese prompts.

Do not overfit to it. The next set should have at least 50 examples stratified by simple, medium, complex and very complex tasks.

## Current Results

- Deterministic routing fixture eval: `8/8` after config adjustment.
- Live small-model benchmark: `196.47 tok/s` generation on Qwen2.5-Coder-1.5B Q4_K_M.
- Live single-expert eval: 6 real calls, average latency `0.585 s`, average server-reported generation speed `194.93 tok/s`.
- Live distilled routing eval: `19/19` route cases across the default general/fallback live profile.

Interpretation: the harness is working, and the machine is fast enough for local inference. The current small model is not quality-optimal; it is a smoke expert. The next decision should be based on a stronger single expert before real multi-model MoE.

## Quality Gate

Before adding another model download, the project must pass:

```bash
./scripts/run_all_checks.sh
```

The automated gate checks:

- compileability for source, tests, scripts, and experiments,
- unit tests for config, router, provider contracts, runtime, evaluator, CLI, and orchestrator,
- base and extended routing eval,
- minimum extended routing accuracy of `0.90`,
- required project files,
- no leftover local live eval listener on `127.0.0.1:8101`.
