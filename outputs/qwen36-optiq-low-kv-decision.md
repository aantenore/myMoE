# Qwen3.6 OptiQ Low-KV Retry

> Historical benchmark decision. The Qwen3.6 OOM evidence remains valid, while
> the old runtime-selection conclusion has been superseded by later
> joint-residency testing.

Tested machine: `Apple M5 Pro` / `arm64` / `24.0 GiB RAM`.

## Result

`mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` is not viable as a myMoE default on the tested 24 GB Apple Silicon machine.

It failed twice:

- standard benchmark profile: `max_kv_size = 8192`,
- tight retry profile: `max_kv_size = 2048`.

Both attempts ended with:

```text
[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

## Command

```bash
PYTHONPATH=src .venv/bin/python experiments/benchmark_models.py \
  --include qwen36-35b-a3b-optiq-mlx-4bit \
  --max-tokens 96 \
  --max-kv-size 2048 \
  --timeout-seconds 1800 \
  --out outputs/qwen36-optiq-low-kv-benchmark.json \
  --report outputs/qwen36-optiq-low-kv-decision.md
```

## Decision Impact

Qwen3 4B MLX 4-bit is now the default primary for this hardware class, paired
with the Qwen3 1.7B fallback. Qwen3 30B-A3B Instruct 2507 MLX 4-bit remains the
quality-first isolated profile. Qwen3.6 OptiQ should only be retried on a
larger-memory Apple Silicon machine or with a substantially smaller quantization
profile that still passes quality evals.
