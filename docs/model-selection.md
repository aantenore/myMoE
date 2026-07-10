# Model Selection

## Hardware Assumption

Target class: Apple Silicon laptop/workstation with about 24 GB unified memory.

## Current Decision

This app is general-purpose, not coding-first. The local architecture should therefore be a system-level MoE with:

1. one strong resident general expert,
2. one small resident expert for summarization, compaction, classification, and fallback,
3. optional cold-loaded specialists for coding, vision, research, or domain tasks,
4. deterministic and later trainable routing,
5. evals that compare against a single strong general baseline.

The current 1.5B Qwen coder model is only a smoke-test model. It proves the harness and llama.cpp runtime; it should not drive product decisions.

## Measured Local Result

On the tested Apple M5 Pro / 24 GiB machine, the current benchmark selects:

- default resident general expert: `Qwen3-4B-4bit`,
- fast fallback/compaction expert: `Qwen3-1.7B-4bit`.

Measured short-generation snapshot:

| Candidate | Status | Avg generation tok/s | Peak memory |
| --- | --- | ---: | ---: |
| Qwen3 30B-A3B Instruct 2507 MLX 4-bit | ok | 79.68 | 17.29 GB |
| Gemma 4 E4B it MLX 4-bit | ok | 70.47 | 4.39 GB |
| Qwen3 4B MLX 4-bit | ok | 93.89 | 2.49 GB |
| Qwen3 1.7B MLX 4-bit | ok | 191.18 | 1.09 GB |
| Qwen3.6 35B-A3B OptiQ MLX 4-bit | failed | - | Metal OOM |

The isolated candidate benchmark ranked the 30B model highest, but it did not model the actual desktop working set. Live joint-residency attempts with the 30B candidate caused severe swap pressure even after reducing the fallback to 1.7B. The default therefore uses Qwen3 4B plus Qwen3 1.7B: both are measured, cached, and small enough to preserve OS/app headroom. Qwen3 30B remains available as a quality-first isolated profile, and Gemma remains an explicitly supported isolated regression profile.
Qwen3.6 OptiQ was tested as the newest stretch candidate and rejected for this 24 GB machine after Metal OOM with both the normal 8192 KV cache benchmark and a tighter 2048 KV retry.

## Quality-First Isolated Profile

Use `Qwen3-30B-A3B-Instruct-2507` in MLX 4-bit only when answer quality is worth running a single heavy model and the rest of the desktop working set is controlled.

Why:

- It is a general MoE model, not a coder-only model.
- The MLX 4-bit artifact is listed at about `17.2 GB`, which leaves some headroom on a 24 GB Mac.
- LM Studio describes it as improved in instruction following, logical reasoning, text comprehension, math, science, coding, and tool usage.
- It supports up to `262,144` context tokens, but this app should cap runtime context much lower at first.
- Its model card explicitly describes this release as non-thinking mode, so myMoE does not enable thinking for this primary expert.

Start it with:

```bash
./scripts/start_mlx_general_expert.sh
```

Then point the orchestrator to its isolated profile:

```bash
configs/moe.live.qwen30-mlx.example.json
```

## Stretch Candidates

| Role | Candidate | Why | 24 GB Risk |
| --- | --- | --- | --- |
| Primary general default | `Qwen3 4B MLX 4-bit` | Responsive resident general model with room for a small fallback | Lower semantic ceiling than the isolated 30B profile |
| Quality-first isolated | `Qwen3-30B-A3B-Instruct-2507-MLX-4bit` | Best isolated candidate quality/performance score | Do not co-reside on the tested 24 GB desktop workload |
| Multimodal general | `Gemma 4 26B-A4B-it` OptiQ MLX 4-bit | Vision, reasoning, tool use, good speed/quality tradeoff | Great alternative, but compare on Antonio-specific evals |
| Fast fallback | `Qwen3 1.7B MLX 4-bit` | Summarization, compaction, routing, cheap fallback | Memory-bounded resident choice for the 24 GiB profile |
| Optional fallback regression | `Gemma 4 E4B it MLX 4-bit` | Isolated Gemma compatibility and quality checks | Do not co-reside with Qwen 30B on the tested 24 GiB host |
| Optional coding specialist | `Qwen3-Coder-30B-A3B` MLX/GGUF | Use only for explicitly coding-heavy workflows | Not default for this app |
| Optional GGUF coding/agentic specialist | `Gemma 4 12B Agentic Fable5 Composer 2.5 v2` GGUF | Local coding, terminal, and tool-use experiments through llama.cpp | Newly published; benchmark locally before enabling as default route |
| Legacy GGUF coding specialist | `Gemma 4 12B Coder Fable5 Composer 2.5 v1` GGUF | Python/coding specialist requested during research | Superseded by the same author's v2 for agentic/coding tasks |
| Rejected on tested 24 GB | `Qwen3.6-35B-A3B` OptiQ MLX 4-bit | Newer general/agentic direction | Failed with Metal OOM at 8192 and 2048 KV cache sizes |
| Rejected for this machine | `Qwen3-Coder-Next` | Strong but too large | 4-bit wants >45 GB; even good low-bit paths want >30 GB |

## Why The Linked Gemma 12B Coder GGUF Is Not The Default

The linked `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF` model is not automatically worse than the selected default. It is simply optimized for a different job:

- The card describes it as a focused fine-tune for verifiable Python coding tasks.
- It is English-centric, while this app is meant to be general-purpose and multilingual.
- It is GGUF-first, so it runs through llama.cpp/LM Studio/Ollama rather than the current MLX benchmark harness.
- The same model card now points to a v2 agentic/coding successor published on 2026-06-19.

myMoE therefore adds both:

- `configs/moe.live.gemma-12b-coder-gguf.example.json` for the v1 model,
- `configs/moe.live.gemma-12b-agentic-gguf.example.json` for the preferred v2 successor.

Neither replaces the general-purpose default until it wins local evals on its task slice. For coding/terminal/tool-use slices, the v2 GGUF profile is the more interesting candidate to benchmark next.

## MoE Runtime Policy

On 24 GB, keep the default topology explicit:

- resident general expert: Qwen3 4B,
- resident fallback/compaction expert: Qwen3 1.7B,
- quality-first isolated expert: Qwen3 30B, selected instead of the default pair,
- other specialists: manual and isolated unless a dedicated eval justifies them,
- max initial context: `16K-32K`,
- compaction trigger: around `70-75%`,
- routing eval before enabling a specialist by default.

This gives application-level MoE behavior without implying that the machine can
host several 17-22 GB models simultaneously or that automatic cold-loading is
already implemented.

## Sources Checked

- Qwen3 30B A3B 2507 MLX 4-bit: https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit
- Qwen3 30B A3B 2507 LM Studio page: https://lmstudio.ai/models/qwen/qwen3-30b-a3b-2507
- Qwen3.6 35B A3B official page: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6 Apple Silicon quant reference: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit
- Qwen3.6 OptiQ MLX reference: https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit
- Gemma 4 local guide: https://unsloth.ai/docs/models/gemma-4
- Gemma 4 E4B MLX artifact: https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit
- Gemma E4B MLX compatibility issue: https://github.com/ml-explore/mlx-lm/issues/1242
- Gemma 4 26B A4B MLX 4-bit: https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit
- Gemma 4 26B A4B OptiQ MLX 4-bit: https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit
- Qwen3-Coder 30B A3B official/GGUF reference: https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
- Gemma 4 12B Coder Fable5 Composer 2.5 v1 GGUF: https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF
- Gemma 4 12B Agentic Fable5 Composer 2.5 v2 GGUF: https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF
- Qwen3-Coder-Next GGUF requirements: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
