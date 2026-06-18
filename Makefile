.PHONY: check test eval ui cli

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
		--config configs/moe.mock.json \
		--port 8089

cli:
	PYTHONPATH=src python3 -m local_moe.cli \
		--config configs/moe.mock.json \
		--interactive
