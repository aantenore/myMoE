# Evaluation

## Baselines

Always compare at least:

1. single local general model,
2. MoE top-1 routing,
3. MoE top-2 routing with comparison/concatenation and disagreement reporting.

## Metrics

| Dimension | Measurement | Gate |
| --- | --- | --- |
| Training-fit routing accuracy | expected expert vs selected expert | diagnostic only; never a generalization claim |
| Holdout routing accuracy | expected expert vs selected expert | >= 0.90 for the committed live routing gate |
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

The live routing data has separate roles:

- `eval_set_live_general.jsonl` is the curated source used to create the 52
  training labels. Its score measures training fit only.
- `eval_set_live_general_holdout_v5.jsonl` is a frozen, independently authored
  52-case holdout with 26 cases per expert, 13 per complexity level, and all 13
  supported routing languages.
- The quality gate requires zero shared ids and zero shared normalized prompt
  hashes, then binds config, training labels, artifact, holdout, and report with
  SHA-256 provenance.

The committed holdout is a regression suite once observed. Rotate or add a new
unseen set before making a fresh generalization claim.

## Current Results

- Deterministic routing fixture eval: `8/8` after config adjustment.
- Extended deterministic routing eval: `56/56` across coding, architecture, general writing, and mixed prompts.
- Live small-model benchmark: `196.47 tok/s` generation on Qwen2.5-Coder-1.5B Q4_K_M.
- Live single-expert eval: 6 real calls, average latency `0.585 s`, average server-reported generation speed `194.93 tok/s`.
- Training-fit routing diagnostic: `52/52`; this is expected because those
  cases generated the labels and must not be presented as generalization.
- Leakage-free live routing holdout: `52/52` (`100%`), 95% Wilson interval
  `93.1%-100%`, with zero id or prompt-hash overlap. Evidence lives in
  `outputs/live-general-routing-holdout.json`.

Interpretation: the harness works and the current router clears the committed
multilingual routing gate. The bidirectional runtime fallback preserves
availability when a fast route misses or its endpoint is offline. The next
product gate is a full single-general vs routed top-1/top-2 answer-quality
comparison with latency, memory, and failures.

The deterministic answer-quality benchmark scaffold for that product gate lives
in `configs/quality-benchmark.json`, `experiments/quality_benchmark_cases.jsonl`,
and `experiments/run_quality_benchmark.py`. Its committed artifact is allowed to
be `blocked` when local endpoints are not running; it must be rerun against live
Qwen/Gemma endpoints before claiming answer-quality advantage.

## Quality Gate

Before adding another model download, the project must pass:

```bash
./scripts/run_all_checks.sh
```

The automated gate checks:

- compileability for source, tests, scripts, and experiments,
- unit tests for config, router, provider contracts, runtime, evaluator, CLI, and orchestrator,
- base and extended routing eval,
- a disjoint live routing holdout regenerated on every run,
- artifact and report provenance freshness,
- zero train/holdout id or normalized prompt-hash overlap,
- minimum holdout routing accuracy of `0.90` across at least `50` cases,
- minimum extended routing accuracy of `0.90`,
- minimum extended routing eval size of `50`,
- required project files,
- no leftover local live eval listener on `127.0.0.1:8101`.
