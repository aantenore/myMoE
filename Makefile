.PHONY: check test eval distill-router prepare-runtime models-status models-logs cron-status run-cron run-cron-writes ui cli doctor setup-models start-models benchmark-small benchmark-gemma

check:
	python3 scripts/run_ci_checks.py

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

eval:
	PYTHONPATH=src python3 experiments/run_smoke_eval.py \
		--config tests/fixtures/moe.synthetic.json \
		--eval experiments/eval_set_extended.jsonl \
		--out outputs/smoke-eval-extended.json

distill-router:
	PYTHONPATH=src python3 experiments/build_route_label_dataset.py \
		--eval experiments/eval_set_live_general.jsonl \
		--out experiments/route_labels_live_general.jsonl \
		--teacher-source curated_live_eval
	PYTHONPATH=src python3 experiments/train_distilled_router.py \
		--labels experiments/route_labels_live_general.jsonl \
		--out outputs/router-distilled-live-general.json

prepare-runtime:
	@PYTHONPATH=src python3 -m local_moe.cli --prepare-runtime --prepare-execute --prepare-download-models --prepare-confirm

models-status:
	@PYTHONPATH=src python3 -m local_moe.cli --models-status

models-logs:
	@PYTHONPATH=src python3 -m local_moe.cli --models-logs

cron-status:
	@PYTHONPATH=src python3 -m local_moe.cli --cron-status

run-cron:
	@PYTHONPATH=src python3 -m local_moe.cli --run-cron

run-cron-writes:
	@PYTHONPATH=src python3 -m local_moe.cli --run-cron --cron-confirm-writes

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
		--include qwen3-1.7b-mlx-4bit,qwen3-4b-mlx-4bit,gemma4-e4b-it-mlx-4bit \
		--prompt-limit 2 \
		--max-tokens 96

benchmark-gemma:
	PYTHONPATH=src .venv/bin/python experiments/benchmark_models.py \
		--include gemma4-e4b-it-mlx-4bit \
		--prompt-limit 2 \
		--max-tokens 96
