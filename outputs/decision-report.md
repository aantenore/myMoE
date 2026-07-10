# Local MoE Decision Report

Date: 2026-06-18

> Historical harness-validation report. Its measurements remain experiment
> evidence, but its candidate recommendation predates the current Qwen3 4B plus
> Qwen3 1.7B default and the isolated Qwen3 30B quality-first profile.

## Machine

- CPU/GPU: Apple M5 Pro
- Architecture: arm64
- Memory: 24 GiB unified
- llama.cpp backend: Metal + BLAS
- Recommended strategy from harness: `single_strong_expert_first`

## Decision

Use a **single strong local expert first**, with the MoE harness kept as orchestration/fallback infrastructure.

Do not start by keeping several real models resident at the same time. On this 24 GiB machine, multi-model MoE residency will compete with IDE/browser/context memory. The better engineering path is:

1. benchmark one strong local expert,
2. compare it against the MoE router,
3. add second/third real experts only if eval quality beats the single expert enough to justify latency and memory.

## Runtime Installed

- llama.cpp `b9700`
- `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf`

## llama-bench

Command:

```bash
./work/tools/llama-b9700/llama-bench \
  -m work/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 -fa auto -p 512 -n 128 -r 3 -t 8 -o md
```

Results:

| Test | Speed |
| --- | ---: |
| prompt processing `pp512` | 5668.93 +/- 53.48 tok/s |
| token generation `tg128` | 196.47 +/- 0.47 tok/s |

Maximum resident set size observed: about 1.1 GB.

## Live Eval: Hybrid MoE

Config: `configs/moe.live.qwen25-coder.json`

- Real expert: `coder`
- Deterministic fixture experts: `architect`, `general`

Summary:

| Metric | Value |
| --- | ---: |
| records | 6 |
| average latency | 0.208 s |
| real local calls | 2 |
| average local generation speed | 198.92 tok/s |

This validates mixed real/fixture routing but is not a quality comparison because non-coder experts used deterministic fixtures.

## Live Eval: Single Real Expert

Config: `configs/single.live.qwen25-coder.json`

Summary:

| Metric | Value |
| --- | ---: |
| records | 6 |
| average latency | 0.585 s |
| real local calls | 6 |
| average generation speed | 194.93 tok/s |

The small coder model is fast and stable, but quality is limited. It asks for missing input on underspecified bug/summarization prompts, which is reasonable. It is not the final model; it is the live harness proof.

## Historical Next Model Candidate

First serious candidate:

```text
unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL
```

Expected risk: likely runs on 24 GiB only with careful quant/context settings and less room for multitasking. Test it as a single expert before introducing real multi-expert residency.

## Acceptance Evidence

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 experiments/run_smoke_eval.py \
  --config tests/fixtures/moe.synthetic.json \
  --eval experiments/eval_set.jsonl \
  --out outputs/smoke-eval.json
PYTHONPATH=src python3 experiments/run_live_eval.py \
  --config configs/single.live.qwen25-coder.json \
  --eval experiments/eval_set.jsonl \
  --llama-server work/tools/llama-b9700/llama-server \
  --model work/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --port 8101 \
  --out outputs/single-live-eval.json \
  --limit 6
```
