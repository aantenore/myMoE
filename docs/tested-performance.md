# Tested Performance

This file records the local machine and benchmark results used to choose the default model profile.

## Tested Machine

- Machine: Apple M5 Pro, `arm64`
- Memory: 24.0 GiB unified memory
- OS/runtime target: macOS Apple Silicon with MLX
- Python: 3.12 virtual environment through `uv`
- MLX runtime: pinned `.[mlx]` profile (`mlx==0.31.2`, `mlx-metal==0.31.2`, `mlx-lm==0.31.3`, `transformers==5.12.1`)

## Decision

Default primary model:

- `lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`

Fast fallback / first-run model:

- `mlx-community/gemma-4-e4b-it-4bit`

Smallest fast demo model:

- `mlx-community/Qwen3-4B-4bit`

Optional GGUF coding/agentic specialist:

- `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`

Why:

- Qwen3 30B-A3B passed the local MLX benchmark and produced the best risk-adjusted score.
- Gemma 4 E4B and Qwen 30B both start with the pinned MLX profile; the explicit Transformers pin avoids the observed `mlx_lm.server` import break in newer Transformers.
- Qwen3 4B is dramatically smaller and remains the best practical low-memory first-run model.
- Gemma 26B variants remain stretch candidates; one quick run was too slow for the default benchmark loop and should be evaluated separately before making them defaults.
- The Gemma 12B GGUF specialist is documented and launchable through llama.cpp, but it was not selected as the general default because it is coding/agentic-specialized and has not yet beaten the Qwen general baseline on Antonio's general-purpose eval set.

## Latest Benchmark Snapshot

See:

- `outputs/performance-benchmark.json`
- `outputs/performance-decision.md`

The app also exposes the latest sanitized decision through CLI `--performance-report`, web `/api/performance`, and web `/api/performance/report.md`, so operators can confirm the selected model policy without reading raw benchmark artifacts.

The benchmark uses short prompts and `96` output tokens per prompt. It is a performance and viability gate, not a full semantic quality benchmark.

| Candidate | Status | Avg generation tok/s | Peak memory | Avg latency |
| --- | --- | ---: | ---: | ---: |
| Qwen3 30B-A3B Instruct 2507 MLX 4-bit | ok | 79.68 | 17.29 GB | 10.56 s |
| Gemma 4 E4B it MLX 4-bit | ok | 70.47 | 4.39 GB | 1.66 s |
| Qwen3 4B MLX 4-bit | ok | 93.89 | 2.49 GB | 1.03 s |
| Qwen3 1.7B MLX 4-bit | ok | 191.18 | 1.09 GB | 0.54 s |
| Qwen3.6 35B-A3B OptiQ MLX 4-bit | failed | - | Metal OOM | - |

Qwen3.6 OptiQ was retried with a tighter `2048` KV cache and still failed with Metal OOM. It is not selected for this 24 GB hardware class. See `outputs/qwen36-optiq-low-kv-benchmark.json` and `outputs/qwen36-optiq-low-kv-decision.md`.

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
