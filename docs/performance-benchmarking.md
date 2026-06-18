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
- `lmstudio-community/gemma-4-E4B-it-MLX-4bit`
- `lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`
- `lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit`
- `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit`

The current policy is to choose one heavy resident general expert and one small resident fallback/compaction expert. Other large specialists should be cold-loaded only after they win evals for their task class.
