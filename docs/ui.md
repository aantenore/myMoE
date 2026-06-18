# UI And CLI

## Web UI

Run the local UI with the default live config:

```bash
PYTHONPATH=src python3 -m local_moe.web \
  --port 8089
```

Then open:

```text
http://127.0.0.1:8089
```

For a live local model, start the model server first and pass a live config:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_local_models.py --only-first
PYTHONPATH=src python3 -m local_moe.web --port 8089
```

The UI is a dependency-free shadcn/new-york inspired surface: zinc dark theme, thin borders, small radius, badges, cards, and compact operator panels.

It now exposes runtime and extension status so you can see the selected backend, model commands, tools, skills, plugins, MCP servers, and cron jobs.

## Screenshots

Dashboard with live model-required config and extension registry:

![myMoE dashboard](screenshots/dashboard.png)

Composer view with an Italian prompt and runtime panels visible:

![myMoE composer](screenshots/composer.png)

Extension registry tab:

![myMoE extensions](screenshots/extensions.png)

Live generation through `Qwen3-4B-4bit`:

![myMoE live generation](screenshots/live-generation.png)

The UI exposes:

- prompt generation,
- selected route,
- provider errors,
- expert list,
- router eval against the extended eval set.

## CLI

Single prompt:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --prompt "Analyze the tradeoff between a single local model and a routed MoE."
```

JSON output:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --prompt "Summarize this plan." \
  --json
```

Interactive shell:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --interactive
```

Eval mode:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --config configs/moe.mock.json \
  --eval experiments/eval_set_extended.jsonl
```

Eval mode intentionally uses the mock fixture because it validates deterministic routing rather than model quality.
