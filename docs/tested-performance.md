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

- `mlx-community/Qwen3-4B-4bit`

Fast fallback / first-run model:

- `mlx-community/Qwen3-1.7B-4bit`

Optional Gemma regression profile:

- `mlx-community/gemma-4-e4b-it-4bit`

Quality-first isolated profile:

- `lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit`

Optional GGUF coding/agentic specialist:

- `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`

Why:

- Qwen3 30B-A3B produced the best isolated candidate score, but the active desktop live test showed that its `17.29 GB` model footprint leaves insufficient headroom for a second resident expert and normal OS/app memory.
- Qwen3 4B is the default primary because it remains responsive under joint residency with the 1.7B fallback; Qwen3 30B is preserved as an explicit quality-first isolated profile.
- Qwen3 1.7B is the resident fallback because its measured `1.09 GB` peak leaves ample headroom beside the default Qwen3 4B primary, measured at `2.49 GB` in the isolated performance run.
- A live joint-residency smoke with Qwen 30B plus Gemma E4B drove the 24 GiB machine beyond `22 GB` of swap and produced an invalid Gemma payload during top-2 generation. Gemma remains supported for isolated regression runs through the pinned MLX profile.
- In the earlier 30B joint-residency experiment, replacing Gemma with Qwen3 4B fixed the model payload but still drove the already-loaded desktop host to about `24 GB` of swap. That experiment is not the default topology: the default uses Qwen3 4B plus the smaller 1.7B fallback and treats host memory as a release metric.
- The release benchmark requires observable RAM and swap counters, rejects more than `4 GiB` of host-wide swap growth during the run, and rejects observed peak RAM use above `95%`. The 4 GiB allowance separates ordinary active-desktop noise from the greater-than-10-GiB growth observed with the rejected heavy topology.
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

## Latest Answer-Quality Release Evidence

The provenance-bound 2026-07-19 run used the pinned Qwen3 4B and 1.7B MLX
snapshots and completed all 72 planned executions. Single-general and routed
top-1 both reached `1.0` task success, `1.0` quality pass rate, and `0.975`
deterministic quality score.
Top-1 routed 25% of cases to the smaller expert, reduced overall mean latency
from `3.1942 s` to `3.0137 s`, and reached a median routed-pair latency ratio of
`0.4354`. Host sampling covered every observation, peak RAM use was `86.2605%`,
and swap grew by `753,076,798` bytes. The release quality gate passed and marked
the artifact release-ready.

Top-2 is intentionally diagnostic and did not outperform top-1: its task
success was `0.75`, deterministic quality score `0.9188`, and mean latency
`3.6044 s`. These results are bounded to the committed eight-case rubric and
three repetitions; they do not establish general semantic superiority.

## Practical Requirements

Minimum for first-run demo:

- 8-12 GiB RAM class machine
- Qwen3 4B or similar local model
- Ollama on Windows/Linux or MLX on Apple Silicon

Recommended for the default myMoE profile:

- Apple Silicon or a strong Linux/Windows local inference setup
- 24 GiB RAM or more
- Enough disk for model cache; this test used about 40 GiB after downloading several candidates
- Qwen3 4B as the resident primary plus the measured Qwen3 1.7B fallback; monitor host memory during longer top-2 generations

Preferred production-like local profile:

- 32-48 GiB RAM if you want multiple large specialists resident
- 24 GiB is viable for the measured Qwen3 4B plus Qwen3 1.7B default pair
