# Test Report

Generated: 2026-06-19

## Scope

The test hardening pass covers configuration validation, routing evaluation, OpenAI-compatible provider contracts, runtime server specs, runtime health checks, CLI behavior, web UI endpoints, persisted local chat sessions, context assembly, file-backed memory, MCP stdio discovery and guarded tool calls, allowlisted local tools, cron permission policy, and orchestrator correlation behavior.

## New Test Surface

- `tests/test_config.py`: duplicate expert ids, missing experts, invalid `top_k`, unsupported aggregation, unknown rule/fallback experts.
- `tests/test_providers.py`: fake OpenAI-compatible HTTP server, usage/timing parsing, invalid payload handling, invalid JSON handling, transport error wrapping.
- `tests/test_evaluator.py`: JSONL eval loading and accuracy/complexity aggregation.
- `tests/test_runtime.py`: llama-server command/URL construction and health probing.
- `tests/test_health.py`: runtime health status for reachable, unreachable, malformed, path-prefixed, and skipped expert providers.
- `tests/test_cli.py`: eval mode and prompt mode through the public CLI.
- `tests/test_chat_store.py`: local chat session create, append, reload, list, search, rename, export, and delete behavior.
- `tests/test_web.py`: web config, generation, persisted chat sessions, chat management APIs, and eval endpoints over a local HTTP server.
- `tests/test_tools.py`: allowlisted local tool execution, write confirmation, MCP capability search, MCP process confirmation, and guarded MCP `tools/call` execution.
- `tests/test_mcp_client.py`: raw stdio MCP `initialize`, `tools/list`, and `tools/call` behavior against a fake MCP server.
- `tests/test_scheduler.py`: cron dry runs, allowlisted actions, unsupported command rejection, and write-local confirmation.
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
- unit/contract tests: `114/114` passed
- base routing eval: `8/8`, accuracy `1.0`
- extended routing eval: `26/26`, accuracy `1.0`
- quality gate: passed
- live eval listener check on `127.0.0.1:8101`: passed
- real MCP filesystem discovery through `npx -y @modelcontextprotocol/server-filesystem .`: passed, `14` tools listed
- real MCP filesystem `tools/call` through `list_allowed_directories`: passed
- Playwright browser smoke for persisted chat sessions: passed
- Playwright browser smoke for chat rename, search, and delete controls: passed
- Playwright browser smoke for runtime health panel: passed
- live local-model dashboard screenshot regenerated with `Qwen3-30B-A3B-Instruct-2507-MLX-4bit`: passed

## Notes

The router remains intentionally configurable and deterministic. During the extended eval, a broad `implement` keyword created a false positive by matching `implementation`; it was removed from config and replaced with more specific coding signals such as `client` and `adapter`.

The hardware recommendation is now general-purpose MoE with one strong resident general expert, one small resident fallback/compaction expert, and cold-loaded specialists only when evals justify them.

MCP execution is intentionally narrow. The filesystem MCP server advertises write/edit tools, so the example enabled profile is guarded by `allow_process_execution=true`, per-call `confirm_process_execution=true`, per-call `confirm_tool_call=true`, and a server-level `allowed_tools` list that keeps the example profile read-oriented.
