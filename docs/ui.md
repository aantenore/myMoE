# UI And CLI

## Web UI

Run the local UI with the default live config:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.web \
  --port 8089
```

Then open:

```text
http://127.0.0.1:8089
```

Start the configured local model server first:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py --only-first
PYTHONPATH=src .venv/bin/python -m local_moe.web --port 8089
```

The UI is a dependency-free shadcn/new-york inspired chat surface. The default view is intentionally simple for non-technical users:

- left rail with a new chat action and starter prompts,
- central chat transcript,
- sticky composer,
- concise model status,
- Advanced drawer hidden by default.

The Advanced drawer contains runtime commands, configured models, last routing metadata, extension registry, the allowlisted tool runner, cron controls, and the deterministic router eval button. Users who only want to chat do not need to see backend details.

The Tools section exposes only configured local tools. It accepts JSON input and returns JSON output from `/api/tools/run`. The default examples are safe to inspect; `plugin.create` still requires `confirm: true` before it writes a plugin scaffold.

MCP tool discovery is available through `mcp.list_tools`. It starts an enabled stdio MCP server and lists its declared tools, but only when the app config has `permissions.allow_process_execution=true` and the tool input includes `confirm_process_execution: true`. The default app config blocks process execution, so the UI can show the tool contract without silently launching processes.

The bundled MCP-enabled example uses the local filesystem MCP server. It is useful for verifying integration, but it is marked `write_local` because the upstream server advertises write/edit tools.

The Cron section runs due allowlisted jobs through `/api/cron/run`. Write-local jobs require the "Confirm local write jobs" checkbox, matching the CLI `--cron-confirm-writes` flag.

Chat responses are rendered with a small safe Markdown renderer. It supports bold, emphasis, inline code, fenced code blocks, links, blockquotes, headings, and bullet lists while escaping model-provided HTML before formatting.

Keyboard behavior:

- `Enter` sends the current prompt.
- `Alt+Enter` inserts a newline.

## Screenshots

Chat-first empty state:

![myMoE chat](screenshots/dashboard.png)

Composer with multiline prompt support before sending:

![myMoE response](screenshots/composer.png)

Advanced runtime, model, routing, extension, MCP, tools, cron, and eval drawer. Cron includes a local "Run due jobs" action backed by the allowlisted scheduler runner:

![myMoE advanced runtime](screenshots/extensions.png)

Live generation through a local Gemma 4 E4B model, including Markdown rendering and route metadata:

![myMoE live generation](screenshots/live-generation.png)

Mobile layout check for the same chat-first flow:

![myMoE mobile layout](screenshots/mobile.png)

## CLI

Single prompt:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --prompt "Analyze the tradeoff between a single local model and a routed MoE."
```

JSON output:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --prompt "Summarize this plan." \
  --json
```

Interactive shell:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --interactive
```

Eval mode:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --config configs/moe.live.fast-mlx.example.json \
  --eval experiments/eval_set_extended.jsonl
```

Eval mode can run against any live local config. Normal UI and CLI usage should use a live local model config.

Run an allowlisted tool:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --run-tool mcp.search_capabilities \
  --tool-input '{"query":"filesystem"}'
```

List tools from an enabled MCP server after explicitly enabling process execution in a separate app config:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --app-config configs/app.mcp-enabled.local.example.json \
  --run-tool mcp.list_tools \
  --tool-input '{"server":"filesystem","confirm_process_execution":true}'
```

Run cron jobs that write local artifacts:

```bash
PYTHONPATH=src .venv/bin/python -m local_moe.cli \
  --run-cron \
  --cron-confirm-writes
```
