# Test Report

Generated: 2026-06-20

## Scope

The test hardening pass covers configuration validation, routing evaluation, multilingual routing coverage, OpenAI-compatible provider contracts, streaming provider contracts, runtime server specs, runtime setup readiness, System Doctor readiness reporting, sanitized performance decision reporting, privacy-safe support bundle export, guarded runtime preparation, guarded model process management, plugin-local skill discovery, manual extension registry auditing, local audit trail logging, guarded extension self-configuration, runtime health checks, CLI behavior, web UI endpoints, streamed chat generation, persisted local chat sessions, context assembly, file-backed memory, local knowledge ingestion, guarded local memory deletion, confirmed local data backup and restore, MCP stdio discovery and guarded tool calls, allowlisted local tools, cron permission policy, background cron automation, and orchestrator correlation behavior.

## New Test Surface

- `tests/test_config.py`: duplicate expert ids, missing experts, invalid `top_k`, unsupported aggregation, unknown rule/fallback experts.
- `tests/test_audit.py`: local JSONL audit event writes, latest-first listing, action/status filtering, and metadata truncation.
- `tests/test_providers.py`: fake OpenAI-compatible HTTP server, streaming SSE parsing, usage/timing parsing, invalid payload handling, invalid JSON handling, reasoning-channel stripping, transport error wrapping.
- `tests/test_evaluator.py`: JSONL eval loading, minimum coverage guards, and accuracy/complexity aggregation.
- `tests/test_runtime.py`: llama-server command/URL construction and health probing.
- `tests/test_setup_status.py`: side-effect-free setup readiness for Hugging Face cache hits, missing local files, Ollama pull commands, and no-model fixture profiles.
- `tests/test_health.py`: runtime health status for reachable, unreachable, malformed, path-prefixed, and skipped expert providers.
- `tests/test_doctor.py`: unified setup, health, process, extension audit, and cron readiness report.
- `tests/test_performance_report.py`: sanitized benchmark decision payload, missing benchmark status, and Markdown report rendering.
- `tests/test_support_bundle.py`: privacy-safe diagnostic bundle content and exclusions.
- `tests/test_cli.py`: eval mode, setup readiness, performance report output, guarded runtime preparation preview, model process status, doctor output, and prompt mode through the public CLI.
- `tests/test_chat_store.py`: local chat session create, append, reload, list, search, rename, durable summary, export, and delete behavior.
- `tests/test_data_bundle.py`: portable local data export and restore for chat sessions plus memory records, including merge and replace behavior.
- `tests/test_web.py`: web config, generation, streamed generation, persisted chat sessions, chat compaction APIs, memory APIs, guarded memory and knowledge deletion APIs, confirmed local data backup APIs, local audit APIs, knowledge import APIs, chat management APIs, setup preparation APIs, System Doctor APIs, performance report APIs, support bundle APIs, plugin creation APIs, extension audit APIs, extension self-configuration refresh, cron status APIs, and eval endpoints over a local HTTP server.
- `tests/test_setup_runner.py`: runtime preparation preview, confirmation guard, injected install runner, and local-file model validation without network access.
- `tests/test_tools.py`: allowlisted local tool execution, knowledge ingestion, guarded memory/document deletion, confirmed local data export/import, extension audit, guarded extension self-configuration for MCP/cron registries, write confirmation, MCP capability search, MCP process confirmation, and guarded MCP `tools/call` execution.
- `tests/test_mcp_client.py`: raw stdio MCP `initialize`, `tools/list`, and `tools/call` behavior against a fake MCP server.
- `tests/test_model_servers.py`: managed model server specs, confirmation guards, reachable-endpoint skips, and managed start/stop lifecycle with fake processes.
- `tests/test_extensions.py`: registry loading, plugin-local skill discovery, plugin audit, and runtime plan coverage.
- `tests/test_scheduler.py`: cron dry runs, allowlisted actions, unsupported command rejection, write-local confirmation, auto-runnable filtering, and background runner status.
- `tests/test_context.py`: cache-friendly context section ordering, policy loading, budget truncation, memory snippet ranking, compaction prompt requirements.
- `tests/test_orchestrator.py`: correlation propagation, compare mode, streamed route/content/final events, and separated routing/generation prompts.
- `tests/test_memory.py`: local memory writes, guarded record/document deletion, knowledge document chunking, scoped listing, temporal validity, keyword retrieval.
- `experiments/eval_set_extended.jsonl`: 56 router cases across coding, architecture, general writing, and mixed prompts.
- `experiments/eval_set_live_general.jsonl`: 52 live general-purpose routing cases across English, Italian, Spanish, French, German, Portuguese, Dutch, Polish, Arabic, Hindi, Japanese, Korean, and Chinese prompts.
- `experiments/route_labels_extended.jsonl`: 56 regenerated distilled router labels from the curated extended eval.
- `experiments/route_labels_live_general.jsonl`: 52 regenerated distilled router labels for the live general-purpose router.
- `experiments/run_quality_gate.py`: project-level quality gate.

## Verification

Command:

```bash
./scripts/run_all_checks.sh
```

Result:

- compileall: passed
- unit/contract tests: `169/169` passed
- base routing eval: `8/8`, accuracy `1.0`
- extended routing eval: `56/56`, accuracy `1.0`
- live general routing eval: `52/52`, accuracy `1.0`
- quality gate: passed
- live setup readiness for `configs/moe.live.general-mlx.example.json`: passed, Qwen and Gemma MLX snapshots cached
- forbidden listener check on `127.0.0.1:8101`: passed, no active listener during quality gate
- real MCP filesystem discovery through `npx -y @modelcontextprotocol/server-filesystem .`: passed, `14` tools listed
- real MCP filesystem `tools/call` through `list_allowed_directories`: passed
- Playwright browser smoke for persisted chat sessions: passed
- Playwright browser smoke for chat rename, search, and delete controls: passed
- Playwright browser smoke for runtime health panel: passed
- Playwright browser smoke for System Doctor panel: passed
- Playwright browser smoke for support bundle download control: passed
- Playwright browser smoke for Performance report panel: passed
- Playwright browser smoke for model process controls: passed
- Playwright browser smoke for Plugin Studio confirmation guard: passed
- Playwright browser smoke for extension registry audit: passed
- Playwright browser smoke for setup readiness panel: passed
- Playwright browser smoke for guarded setup preparation controls: passed
- Playwright browser smoke for chat compaction action: passed
- Playwright browser smoke for memory panel and context retrieval: passed
- Playwright browser smoke for cron background automation status: passed
- live local-model dashboard screenshot regenerated with `Qwen3-30B-A3B-Instruct-2507-MLX-4bit`: passed

## Notes

The router remains intentionally configurable and deterministic. During the extended eval, a broad `implement` keyword created a false positive by matching `implementation`; it was removed from config and replaced with more specific coding signals such as `client` and `adapter`.

The live general-purpose router now has a balanced multilingual fixture for `general` and `fast_fallback` decisions. Its generated report is stored in `outputs/live-general-routing-eval.json` and is required by the quality gate.

`extension.configure` now lets operators add, update, or remove MCP server and cron job registry entries through the allowlisted tool runner. It writes only to app-configured registry paths, validates entries before writing, requires confirmation, and refreshes the running web registry and cron runner.

Streaming generation is now covered from provider SSE parsing through orchestrator events and the web `/api/generate/stream` endpoint. The UI updates the assistant bubble while content arrives and falls back to `/api/generate` if streaming is unavailable before content starts.

Local knowledge import now chunks pasted documents into `knowledge` memory records with document metadata. It is exposed through `knowledge.ingest`, `/api/knowledge`, and the Advanced Knowledge panel, with confirmation required before local records are written. Guarded forget controls now remove a single memory record or all chunks for one imported document id only when confirmation is supplied.

Local data backup now exports chat sessions and memory records into a schema-versioned JSON bundle through `data.export`, `/api/data/export`, and the Advanced Local Data panel. Restore through `data.import` or `/api/data/import` supports `merge` and `replace` modes and requires confirmation before writing local stores.

Local audit trail logging now records sensitive host-side actions in `<runtime.work_dir>/audit.jsonl` and exposes recent events through `/api/audit` plus the Advanced Audit Trail panel. Audit events intentionally store metadata only, not prompt text, chat transcripts, memory text, environment variables, or model log bodies.

The hardware recommendation is now general-purpose MoE with one strong resident general expert, one small resident fallback/compaction expert, and cold-loaded specialists only when evals justify them.

MCP execution is intentionally narrow. The filesystem MCP server advertises write/edit tools, so the example enabled profile is guarded by `allow_process_execution=true`, per-call `confirm_process_execution=true`, per-call `confirm_tool_call=true`, and a server-level `allowed_tools` list that keeps the example profile read-oriented.
