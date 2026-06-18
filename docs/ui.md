# UI And CLI

## Web UI

Run the local UI with the mock config:

```bash
PYTHONPATH=src python3 -m local_moe.web \
  --config configs/moe.mock.json \
  --port 8089
```

Then open:

```text
http://127.0.0.1:8089
```

For a live local model, start the model server first and pass a live config:

```bash
./scripts/start_mlx_general_expert.sh

PYTHONPATH=src python3 -m local_moe.web \
  --config configs/moe.live.general-mlx.example.json \
  --port 8089
```

The UI is a dependency-free shadcn/new-york inspired surface: zinc dark theme, thin borders, small radius, badges, cards, and compact operator panels.

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
  --config configs/moe.mock.json \
  --prompt "Analyze the tradeoff between a single local model and a routed MoE."
```

JSON output:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --config configs/moe.mock.json \
  --prompt "Summarize this plan." \
  --json
```

Interactive shell:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --config configs/moe.mock.json \
  --interactive
```

Eval mode:

```bash
PYTHONPATH=src python3 -m local_moe.cli \
  --config configs/moe.mock.json \
  --eval experiments/eval_set_extended.jsonl
```
