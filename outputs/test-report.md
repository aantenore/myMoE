# Test Report

Generated: 2026-06-18

## Scope

The test hardening pass added coverage for configuration validation, routing evaluation, OpenAI-compatible provider contracts, runtime server specs, CLI behavior, web UI endpoints, context assembly, file-backed memory, and orchestrator correlation behavior.

## New Test Surface

- `tests/test_config.py`: duplicate expert ids, missing experts, invalid `top_k`, unsupported aggregation, unknown rule/fallback experts.
- `tests/test_providers.py`: fake OpenAI-compatible HTTP server, usage/timing parsing, invalid payload handling, invalid JSON handling, transport error wrapping.
- `tests/test_evaluator.py`: JSONL eval loading and accuracy/complexity aggregation.
- `tests/test_runtime.py`: llama-server command/URL construction and health probing.
- `tests/test_cli.py`: eval mode and prompt mode through the public CLI.
- `tests/test_web.py`: web config, generation, and eval endpoints over a local HTTP server.
- `tests/test_context.py`: cache-friendly context section ordering, budget truncation, memory snippet ranking, compaction prompt requirements.
- `tests/test_memory.py`: append-only local memory writes, scoped listing, temporal validity, keyword retrieval.
- `experiments/eval_set_extended.jsonl`: 26 router cases across coding, architecture, general writing, and mixed prompts.
- `experiments/run_quality_gate.py`: project-level quality gate.

## Verification

Command:

```bash
./scripts/run_all_checks.sh
```

Result:

- compileall: passed
- unit/contract tests: `33/33` passed
- base routing eval: `8/8`, accuracy `1.0`
- extended routing eval: `26/26`, accuracy `1.0`
- quality gate: passed
- live eval listener check on `127.0.0.1:8101`: passed

## Notes

The router remains intentionally configurable and deterministic. During the extended eval, a broad `implement` keyword created a false positive by matching `implementation`; it was removed from config and replaced with more specific coding signals such as `client` and `adapter`.

The hardware recommendation is now general-purpose MoE with one strong resident general expert, one small resident fallback/compaction expert, and cold-loaded specialists only when evals justify them.
