# Tested Performance

This file records the local machine and benchmark results used to choose the default model profile.

## Tested Machine

- Machine: Apple M5 Pro, `arm64`
- Memory: 24.0 GiB unified memory
- OS/runtime target: macOS Apple Silicon with MLX
- Python: 3.12 virtual environment through `uv`
- MLX runtime: `mlx-lm`

## Decision

Default primary model:

- `lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`

Fast fallback / first-run model:

- `mlx-community/Qwen3-4B-4bit`

Why:

- Qwen3 30B-A3B passed the local MLX benchmark and produced the best risk-adjusted score.
- Qwen3 4B is dramatically smaller, starts quickly, and is the best practical fallback/compaction model tested.
- Gemma 4 E4B failed to load with the current `mlx-lm` runtime/artifact combination, so it is not selected.
- Gemma 26B variants remain stretch candidates; one quick run was too slow for the default benchmark loop and should be evaluated separately before making them defaults.

## Latest Benchmark Snapshot

See:

- `outputs/performance-benchmark.json`
- `outputs/performance-decision.md`

The benchmark uses short prompts and `96` output tokens per prompt. It is a performance and viability gate, not a full semantic quality benchmark.

| Candidate | Status | Avg generation tok/s | Peak memory | Avg latency |
| --- | --- | ---: | ---: | ---: |
| Qwen3 30B-A3B Instruct 2507 MLX 4-bit | ok | 96.95 | 17.31 GB | 9.47 s |
| Qwen3 4B MLX 4-bit | ok | 91.13 | 2.54 GB | 1.25 s |
| Qwen3 1.7B MLX 4-bit | ok | 209.87 | 1.17 GB | 0.58 s |
| Gemma 4 E4B it MLX 4-bit | failed to load | - | - | - |

## Practical Requirements

Minimum for first-run demo:

- 8-12 GiB RAM class machine
- Qwen3 4B or similar local model
- Ollama on Windows/Linux or MLX on Apple Silicon

Recommended for the default myMoE profile:

- Apple Silicon or a strong Linux/Windows local inference setup
- 24 GiB RAM or more
- Enough disk for model cache; this test used about 40 GiB after downloading several candidates
- One heavy resident model at a time, plus optional small fallback if memory remains comfortable

Preferred production-like local profile:

- 32-48 GiB RAM if you want multiple large specialists resident
- 24 GiB is viable with one heavy model plus a small fallback/compaction model
