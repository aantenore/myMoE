# Gemma 4 E4B Runtime

Gemma 4 E4B is supported, but it must use the validated MLX profile documented here.

## What Failed

The latest MLX stack tested during this work:

- `mlx==0.31.2`
- `mlx-lm==0.31.3`
- `mlx-metal==0.31.2`
- `mlx-vlm==0.6.3`

failed to load `mlx-community/gemma-4-e4b-it-4bit` with:

```text
ValueError: Received 126 parameters not in model
```

This is not a myMoE routing failure. It is a runtime/artifact compatibility issue in the MLX stack. The same version break is documented upstream in `ml-explore/mlx-lm` issue 1242.

## Validated Fix

The validated local profile is:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ".[mlx]"
```

The `.[mlx]` extra pins:

- `mlx==0.31.1`
- `mlx-metal==0.31.1`
- `mlx-lm==0.31.2`

With that profile, this command loads and generates:

```bash
PYTHONPATH=src .venv/bin/python experiments/benchmark_models.py \
  --include gemma4-e4b-it-mlx-4bit \
  --prompt-limit 2 \
  --max-tokens 96
```

Use this live config to run Gemma E4B through the same OpenAI-compatible provider used by the rest of the app:

```bash
PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py \
  --config configs/moe.live.gemma-e4b-mlx.example.json \
  --download-models

PYTHONPATH=src .venv/bin/python scripts/start_local_models.py \
  --config configs/moe.live.gemma-e4b-mlx.example.json
```

Then run the UI:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.web \
  --config configs/moe.live.gemma-e4b-mlx.example.json \
  --port 8089
```

## Thinking Output

Gemma 4 can emit a thinking channel. myMoE uses a model-level policy:

```json
"supports_thinking": true,
"thinking_policy": "auto"
```

When the prompt is simple, the provider sends `chat_template_kwargs.enable_thinking = false`. When the prompt asks for analysis, planning, debugging, comparison, or similarly complex work, it sends `enable_thinking = true`.

The provider strips `<think>...</think>` and Gemma channel tokens such as `<|channel>thought ... <channel|>` before returning content to the UI/CLI. This keeps reasoning support available without leaking raw thinking markup into normal chat.

The benchmark runner still disables thinking for deterministic performance measurements when the tokenizer supports that flag.

## Tested Result

On the tested Apple M5 Pro / 24 GiB machine:

| Model | Runtime profile | Status | Avg generation tok/s | Peak memory |
| --- | --- | --- | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | pinned `.[mlx]` | ok | 72.11 | 4.39 GB |

See:

- `outputs/gemma-e4b-benchmark.json`
- `outputs/gemma-e4b-decision.md`

## Source Notes

- Upstream compatibility issue: https://github.com/ml-explore/mlx-lm/issues/1242
- Gemma with MLX guide: https://ai.google.dev/gemma/docs/integrations/mlx
- MLX Community Gemma E4B artifact: https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit
- `mlx-vlm` Gemma 4 support notes: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md
