# Performance Benchmarking

`experiments/benchmark_models.py` benchmarks MLX candidates from `configs/model-benchmark.json`.

It measures:

- model load time,
- per-prompt latency,
- prompt tokens per second,
- generation tokens per second,
- peak MLX memory,
- prompt failures,
- a risk-adjusted score combining local performance with a documented quality prior.

The score intentionally rewards memory headroom. On a 24 GB Apple Silicon machine, the best app model is not simply the largest model that starts; it must leave enough room for the OS, web UI, context compaction, memory retrieval, and optional cold-loaded specialists.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ".[mlx]"
```

The `.[mlx]` extra pins the MLX package versions validated locally:

- `mlx==0.31.1`
- `mlx-metal==0.31.1`
- `mlx-lm==0.31.2`

This matters for Gemma 4 E4B. Newer MLX packages loaded during testing reproduced the upstream `Received 126 parameters not in model` failure. The pinned profile loads and generates successfully.

## Run

Small smoke pass:

```bash
make benchmark-small
```

Full candidate pass:

```bash
PYTHONPATH=src .venv/bin/python experiments/benchmark_models.py
```

Outputs:

- `outputs/performance-benchmark.json`
- `outputs/performance-decision.md`

## Current Candidate Set

- `mlx-community/Qwen3-1.7B-4bit`
- `mlx-community/Qwen3-4B-4bit`
- `mlx-community/gemma-4-e4b-it-4bit`
- `lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`
- `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` as a rejected 24 GB stretch candidate after Metal OOM at 8192 and 2048 KV cache sizes
- `lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit`
- `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit`

The current policy is to choose one heavy resident general expert and one small resident fallback/compaction expert. Other large specialists should be cold-loaded only after they win evals for their task class.

Qwen3.6 OptiQ is intentionally kept in the benchmark manifest as negative evidence: it is a better-looking current candidate on paper, but it did not run reliably on the tested 24 GB Apple Silicon machine. The tight retry is stored in:

- `outputs/qwen36-optiq-low-kv-benchmark.json`
- `outputs/qwen36-optiq-low-kv-decision.md`

## GGUF Specialist Benchmarking

The current MLX benchmark does not directly score GGUF files. GGUF candidates are still first-class runtime profiles through llama.cpp:

- `configs/moe.live.gemma-12b-coder-gguf.example.json`
- `configs/moe.live.gemma-12b-agentic-gguf.example.json`

The v2 agentic profile is the preferred successor for local coding, terminal, and tool-use tests. It should be benchmarked with a llama.cpp-backed eval slice before being enabled as a default coding route.

## Gemma E4B Regression

Run the focused Gemma E4B benchmark with:

```bash
make benchmark-gemma
```

The current Gemma E4B result is stored with the broader model benchmark output:

- `outputs/performance-benchmark.json`
- `outputs/performance-decision.md`
