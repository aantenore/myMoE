# Installation

myMoE is local-first and requires a real local model for normal CLI/UI usage.
The public configs under `configs/` are live local-model profiles or templates for live local-model profiles.

## Supported Platforms

| Platform | Preferred backend | Notes |
| --- | --- | --- |
| macOS Apple Silicon | MLX (`mlx-lm`) | Fastest path for this repo's tested machine class. The default `.[mlx]` extra pins the stack validated with Qwen and Gemma E4B. |
| Windows | Ollama | Cross-platform local daemon with OpenAI-compatible API. |
| Linux | Ollama | Good default; llama.cpp can be configured manually for GGUF. |
| Fallback | llama.cpp | Use when you need direct GGUF control. |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ".[mlx]"
PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py --download-models
```

Validated Apple Silicon MLX package profile:

```text
mlx==0.31.1
mlx-metal==0.31.1
mlx-lm==0.31.2
```

This pin is intentional. Newer MLX packages were observed to reject the current Gemma 4 E4B MLX artifact with `Received 126 parameters not in model`.

Optional extras:

- `.[gguf]`: installs the Python-side downloader dependencies for llama.cpp/GGUF profiles. The `llama-server` binary is still installed from llama.cpp releases.
- `.[mlx-current]`: tracks the latest `mlx-lm` stack for experiments.
- `.[mlx-vlm]`: installs `mlx-vlm` for future multimodal server experiments. Do not use it as the default Gemma E4B path until the upstream compatibility issue is resolved.

Or let the bootstrap script run the safe install commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py --execute --download-models
```

The same guarded flow is available through the app CLI:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --prepare-runtime
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --prepare-runtime \
  --prepare-execute \
  --prepare-download-models \
  --prepare-confirm
```

`--prepare-runtime` without side-effect flags is a preview. Installs and model downloads require `--prepare-confirm`, and the web UI uses the same confirmation policy in the Advanced Setup panel.

Before or after bootstrap, inspect setup readiness:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --setup
```

The setup report lists the selected runtime backend, model cache path, model asset status, and the exact bootstrap command for the active config. Hugging Face snapshots are checked in the local cache, local GGUF paths are validated, and Ollama profiles report the required pull command.

To combine setup inspection, runtime preparation, model startup, and System Doctor verification in one guarded flow:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --startup \
  --startup-prepare \
  --startup-download-models \
  --startup-start-models \
  --startup-confirm
```

`--startup` without side-effect flags is a read-only preview. `--startup-start-models` can also be combined with `--startup-only-first` when a machine should launch only the primary local expert.

To choose the best local profile for the detected machine and current model cache without editing config files:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --recommend-profile
```

The same read-only decision is exposed by `/api/config/recommendation` and embedded in `/api/config/profiles` for the Advanced Profiles panel.

To prepare the recommended profile's runtime dependencies and model assets before activation:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --prepare-recommended-profile \
  --prepare-execute \
  --prepare-download-models \
  --prepare-confirm
```

To set the recommended profile as the app default for the next start:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --activate-recommended-profile \
  --profile-confirm
```

Profile activation validates the target profile, writes only `default_moe_config` in the app config file, and returns a restart command. The running process keeps using the profile it started with.

## Start Models

```bash
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py
```

For constrained machines, start only the first configured model:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py --only-first
```

Inspect model process status from the app CLI:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --models-status
```

Inspect sanitized model server log tails from the app CLI:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --models-logs --models-log-lines 80
```

Start models from the app CLI in the foreground:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --start-models \
  --models-confirm
```

The web UI exposes equivalent guarded start/stop controls in Advanced Runtime. It skips endpoints that already respond, stops only processes started by the current web server, and shows bounded sanitized log tails from runtime-plan-generated log files.

Fast MLX config for first-run demos:

```bash
PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py \
  --config configs/moe.live.fast-mlx.example.json \
  --download-models
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py \
  --config configs/moe.live.fast-mlx.example.json
```

Gemma 4 E4B config:

```bash
PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py \
  --config configs/moe.live.gemma-e4b-mlx.example.json \
  --download-models
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py \
  --config configs/moe.live.gemma-e4b-mlx.example.json
```

Gemma configs declare `supports_thinking = true` and `thinking_policy = auto`. myMoE enables thinking only for prompts that look complex enough to benefit from it, then strips Gemma thinking/channel tokens from the user-visible response.

Optional Gemma 4 12B GGUF coding/agentic specialist:

```bash
# Install llama.cpp first:
# https://github.com/ggml-org/llama.cpp/releases
uv pip install --python .venv/bin/python ".[gguf]"
PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py \
  --config configs/moe.live.gemma-12b-agentic-gguf.example.json \
  --download-models
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py \
  --config configs/moe.live.gemma-12b-agentic-gguf.example.json
```

For Hugging Face GGUF specs such as `owner/repo:Q4_K_M`, bootstrap downloads only matching `*.gguf` files instead of cloning the whole repository. Local `.gguf` file paths are validated and reused.

The older `configs/moe.live.gemma-12b-coder-gguf.example.json` profile is retained for the v1 model that was evaluated during research. Prefer the v2 agentic profile for new coding/tool-use experiments.

## Start UI

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.web --port 8089
```

Open `http://127.0.0.1:8089`.

## Cross-Platform Ollama Config

Use `configs/moe.live.ollama.example.json` when running through Ollama:

```bash
ollama pull qwen3:4b
ollama serve
PYTHONPATH=src .venv/bin/python -m local_moe.web --config configs/moe.live.ollama.example.json
```

## Doctor

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --doctor
PYTHONPATH=src .venv/bin/python -m local_moe.cli --doctor --doctor-format markdown
```

The doctor output reports an overall `ready`, `attention`, or `blocked` status with normalized checks, recommendations, platform, backend choice, install commands, model commands, setup readiness, active-profile hardware fit, runtime health, model process state, extension audit, configured tools, skills, plugins, MCP servers, and cron jobs. The same report is available in the web UI through `/api/doctor` and Advanced System Doctor. The Markdown form is available from CLI and `/api/doctor/report.md` for issue reports and handoffs.

To capture a metadata-only environment snapshot for reproducibility or issue reports:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --about
PYTHONPATH=src .venv/bin/python -m local_moe.cli --about --about-format markdown
```

The snapshot includes app mode, config paths, platform, Python version, selected package versions, git revision, hardware summary, routing policy, and configured local model identities. The same data is exposed by the web UI through `/api/about`, `/api/about/report.md`, and Advanced Environment.

To verify that the configured local model endpoint can produce visible content, not only answer health probes:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --smoke-generate
```

The same generation smoke test is exposed in the web UI through `/api/smoke/generate` and Advanced Runtime.

To inspect the latest local model benchmark decision without starting a new benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --performance-report
PYTHONPATH=src .venv/bin/python -m local_moe.cli --performance-report --performance-report-format markdown
```

The web UI exposes the same sanitized data through `/api/performance` and a Markdown handoff report through `/api/performance/report.md`.

To combine recent run metadata, profile recommendation, and benchmark status into read-only next actions:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --runtime-optimizer
PYTHONPATH=src .venv/bin/python -m local_moe.cli --runtime-optimizer --runtime-optimizer-format markdown
```

The web UI exposes the same advisory report through `/api/runtime/optimizer` and `/api/runtime/optimizer/report.md`.

For issue reports or handoffs, generate a privacy-safe support bundle:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli --support-bundle-out outputs/support-bundle.json
```

The bundle includes the Doctor report, quality gate status, sanitized performance report, hardware profile, runtime file paths, and model log paths. It excludes chat transcripts, memory records, environment variables, benchmark response excerpts, API keys, and log contents. The web UI exposes the same artifact through Advanced System Doctor.

## Background Maintenance

The default `configs/app.json` enables `runtime.cron_auto_run=true`. When the web UI starts, it also starts a lightweight in-process scheduler that polls every `runtime.cron_poll_seconds` seconds and runs due safe jobs. By default, `runtime.cron_confirm_writes=false`, so write-local jobs such as router distillation remain manual and require the CLI `--cron-confirm-writes` flag or the UI confirmation checkbox.

This design stays cross-platform because it does not require launchd, systemd, Windows Task Scheduler, or a separate daemon. Operators who prefer OS-level scheduling can still call the same CLI commands from their scheduler of choice.
