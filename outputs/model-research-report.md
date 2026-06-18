# Model Research Report

Generated: 2026-06-18

## Correction

The target app is general-purpose, not coding-first. The prior Qwen2.5-Coder 1.5B model remains useful only as a smoke-test endpoint.

## Recommendation

Default first serious local model:

```text
lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit
```

Why:

- general MoE model, not a coder-only model,
- listed as MLX 4-bit and about `17.2 GB`,
- suitable headroom for a 24 GB Apple Silicon machine if context is capped,
- broad instruction following, logical reasoning, multilingual, long-context and tool-use positioning.

Stretch model:

```text
Qwen3.6-35B-A3B MLX 4-bit
```

Use it only with tighter context caps and memory monitoring.

Multimodal alternative:

```text
lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit
```

Good candidate for chat, vision, reasoning, and general multimodal workflows.

## Runtime Shape

For 24 GB unified memory:

- keep one heavy general expert resident,
- keep one small fallback/summary/compaction expert resident,
- cold-load coding or multimodal specialists only when needed,
- use context compaction and memory retrieval before increasing context window,
- benchmark specialist routes against the primary general model before promoting them.

## Sources

- https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit
- https://lmstudio.ai/models/qwen/qwen3-30b-a3b-2507
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://unsloth.ai/docs/models/gemma-4
- https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit
- https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
