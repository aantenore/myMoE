# Performance Decision

Hardware budget: `24.0 GiB` unified memory.
Tested machine: `Apple M5 Pro` / `arm64` / `24.0 GiB RAM`.

## Decision

- Primary general expert: `none`
- Fast fallback/compaction expert: `Gemma 4 E4B it MLX 4-bit`
- Architecture: one resident heavy general expert plus one small resident fallback/compaction expert; cold-load specialists only after eval wins.

## Ranked Results

| Rank | Candidate | Role | Status | Score | Tok/s | Peak GB | Load s |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `Gemma 4 E4B it MLX 4-bit` | `fast_compaction_or_fallback` | `ok` | 0.829 | 72.76 | 4.39 | 1.38 |
| 2 | `Qwen3 1.7B MLX 4-bit` | `fast_compaction_or_fallback` | `not_run` | 0.000 | - | - | - |
| 3 | `Qwen3 4B MLX 4-bit` | `fast_compaction_or_fallback` | `not_run` | 0.000 | - | - | - |
| 4 | `Qwen3 30B-A3B Instruct 2507 MLX 4-bit` | `primary_general` | `not_run` | 0.000 | - | - | - |
| 5 | `Gemma 4 26B-A4B it MLX 4-bit` | `primary_general_alternative` | `not_run` | 0.000 | - | - | - |
| 6 | `Gemma 4 26B-A4B it OptiQ MLX 4-bit` | `primary_general_alternative` | `not_run` | 0.000 | - | - | - |

## Notes

- Scores combine measured local performance with a documented quality prior.
- A failed or skipped model gets zero reliability and cannot be selected.
- The score intentionally rewards memory headroom because this app must remain usable while the OS, UI, and context/memory layers are active.
