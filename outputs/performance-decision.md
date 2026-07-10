# Performance Decision

Hardware budget: `24.0 GiB` unified memory.
Tested machine: `Apple M5 Pro` / `arm64` / `24.0 GiB RAM`.

## Decision

- Default resident general expert: `Qwen3 4B MLX 4-bit`
- Quality-first isolated expert: `Qwen3 30B-A3B Instruct 2507 MLX 4-bit`
- Fast fallback/compaction expert: `Qwen3 1.7B MLX 4-bit`
- Architecture: one resident 4B general expert plus one 1.7B fallback/compaction expert; run the 30B expert alone when explicitly selecting the quality-first profile.

## Ranked Results

This table preserves the isolated candidate ranking and benchmark roles recorded
at measurement time. Those historical roles are not the current runtime
selection; the decision above is authoritative for the default profile.

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
- The ranked table measures candidates in isolation. Later joint-residency smoke tests overrode its runtime selection: Qwen 30B + Gemma E4B pushed the host above `22 GB` of swap, and Qwen 30B + Qwen3 4B still reached about `24 GB` during the full benchmark. Qwen3 1.7B is the memory-bounded resident fallback at `1.09 GB` measured peak; Qwen3 4B is the default primary.
- Qwen3.6 OptiQ was retried with `max_kv_size=2048` and still failed with Metal OOM; see `outputs/qwen36-optiq-low-kv-decision.md`.
