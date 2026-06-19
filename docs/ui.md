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

The Advanced drawer contains runtime commands, configured models, last routing metadata, extension registry, and the deterministic router eval button. Users who only want to chat do not need to see backend details.

Chat responses are rendered with a small safe Markdown renderer. It supports bold, emphasis, inline code, fenced code blocks, links, blockquotes, headings, and bullet lists while escaping model-provided HTML before formatting.

Keyboard behavior:

- `Enter` sends the current prompt.
- `Alt+Enter` inserts a newline.

## Screenshots

Chat-first empty state:

![myMoE chat](screenshots/dashboard.png)

Composer with multiline prompt support before sending:

![myMoE response](screenshots/composer.png)

Advanced runtime, model, routing, extension, MCP, cron, and eval drawer:

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
