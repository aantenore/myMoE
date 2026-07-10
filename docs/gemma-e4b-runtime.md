# Gemma 4 E4B Runtime

Gemma 4 E4B is supported as an optional isolated profile, but it must use the validated MLX profile documented here. It is not the default resident fallback on the tested 24 GiB machine.

## What Failed Previously

An unpinned MLX/Transformers stack tested during this work:

- `mlx==0.31.2`
- `mlx-lm==0.31.3`
- `mlx-metal==0.31.2`
- `transformers==5.13.0`

failed before server startup because `mlx_lm.server` could not import:

```text
AttributeError: 'str' object has no attribute '__module__'
```

This is not a myMoE routing failure. It is a runtime package compatibility
issue, and setup readiness now checks that the configured MLX server module can
actually import before reporting `ready`.

## Validated Fix

The validated local profile is:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ".[mlx]"
```

The `.[mlx]` extra pins:

- `mlx==0.31.2`
- `mlx-metal==0.31.2`
- `mlx-lm==0.31.3`
- `transformers==5.12.1`

With that profile, both configured live servers import, start, and generate
short smoke responses locally. The performance benchmark can be run with:

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

In `auto` mode the provider sends `chat_template_kwargs.enable_thinking = true` only for explicit security/threat or formal-proof work. Routine analysis, architecture, planning, and comparison stay non-thinking so a small resident model remains interactive. A dedicated reasoning profile can use `thinking_policy = on` when that latency is intentional.

The provider strips `<think>...</think>` and Gemma channel tokens such as `<|channel>thought ... <channel|>` before returning content to the UI/CLI. This keeps reasoning support available without leaking raw thinking markup into normal chat.

The benchmark runner still disables thinking for deterministic performance measurements when the tokenizer supports that flag.

## Tested Result

On the tested Apple M5 Pro / 24 GiB machine:

| Model | Runtime profile | Status | Avg generation tok/s | Peak memory |
| --- | --- | --- | ---: | ---: |
| `mlx-community/gemma-4-e4b-it-4bit` | pinned `.[mlx]` | ok | 70.47 | 4.39 GB |

This isolated result is valid, but it is not a joint-residency result. When Gemma E4B and Qwen3 30B were loaded together for the answer-quality top-2 smoke, host swap exceeded `22 GB` and the Gemma request returned an invalid payload. The default 24 GiB profile therefore uses Qwen3 1.7B as its memory-bounded resident fallback; keep this Gemma profile for isolated compatibility and regression checks.

See:

- `outputs/performance-benchmark.json`
- `outputs/performance-decision.md`

## Source Notes

- Upstream compatibility issue: https://github.com/ml-explore/mlx-lm/issues/1242
- Gemma with MLX guide: https://ai.google.dev/gemma/docs/integrations/mlx
- MLX Community Gemma E4B artifact: https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit
- `mlx-vlm` Gemma 4 support notes: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md
