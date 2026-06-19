# Performance Decision

Hardware budget: `24.0 GiB` unified memory.
Tested machine: `Apple M5 Pro` / `arm64` / `24.0 GiB RAM`.

## Decision

- Primary general expert: `Qwen3 30B-A3B Instruct 2507 MLX 4-bit`
- Fast fallback/compaction expert: `Gemma 4 E4B it MLX 4-bit`
- Architecture: one resident heavy general expert plus one small resident fallback/compaction expert; cold-load specialists only after eval wins.

## Ranked Results

| Rank | Candidate | Role | Status | Score | Tok/s | Peak GB | Load s |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `Qwen3 30B-A3B Instruct 2507 MLX 4-bit` | `primary_general` | `ok` | 0.903 | 79.68 | 17.29 | 6.99 |
| 2 | `Gemma 4 E4B it MLX 4-bit` | `fast_compaction_or_fallback` | `ok` | 0.828 | 70.47 | 4.39 | 2.38 |
| 3 | `Qwen3 4B MLX 4-bit` | `fast_compaction_or_fallback` | `ok` | 0.797 | 93.89 | 2.49 | 1.90 |
| 4 | `Qwen3 1.7B MLX 4-bit` | `fast_compaction_or_fallback` | `ok` | 0.739 | 191.18 | 1.09 | 1.25 |
| 5 | `Qwen3.6 35B-A3B OptiQ MLX 4-bit` | `primary_general_stretch` | `failed` | 0.000 | - | - | - |
| 6 | `Gemma 4 26B-A4B it MLX 4-bit` | `primary_general_alternative` | `not_run` | 0.000 | - | - | - |
| 7 | `Gemma 4 26B-A4B it OptiQ MLX 4-bit` | `primary_general_alternative` | `not_run` | 0.000 | - | - | - |

## Notes

- Scores combine measured local performance with a documented quality prior.
- A failed or skipped model gets zero reliability and cannot be selected.
- The score intentionally rewards memory headroom because this app must remain usable while the OS, UI, and context/memory layers are active.
- Qwen3.6 OptiQ was retried with `max_kv_size=2048` and still failed with Metal OOM; see `outputs/qwen36-optiq-low-kv-decision.md`.
