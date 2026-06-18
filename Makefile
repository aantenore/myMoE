.PHONY: check test eval ui cli doctor setup-models start-models benchmark-small

check:
	./scripts/run_all_checks.sh

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

eval:
	PYTHONPATH=src python3 experiments/run_smoke_eval.py \
		--config configs/moe.mock.json \
		--eval experiments/eval_set_extended.jsonl \
		--out outputs/smoke-eval-extended.json

ui:
	PYTHONPATH=src python3 -m local_moe.web \
		--port 8089

cli:
	PYTHONPATH=src python3 -m local_moe.cli \
		--interactive

doctor:
	PYTHONPATH=src python3 -m local_moe.cli --doctor

setup-models:
	PYTHONPATH=src .venv/bin/python scripts/bootstrap_runtime.py --execute --download-models

start-models:
	PYTHONPATH=src .venv/bin/python scripts/start_local_models.py

benchmark-small:
	PYTHONPATH=src .venv/bin/python experiments/benchmark_models.py \
		--include qwen3-1.7b-mlx-4bit,qwen3-4b-mlx-4bit \
		--prompt-limit 2 \
		--max-tokens 96
