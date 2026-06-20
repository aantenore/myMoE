# Test Report

Generated: 2026-06-20

## Scope

The test hardening pass covers configuration validation, runtime profile discovery with hardware fit, profile recommendation, guarded profile preparation, guarded profile activation, guarded startup runbook orchestration, and copyable launch hints, routing evaluation, multilingual routing coverage, OpenAI-compatible provider contracts, streaming provider contracts, runtime server specs, cross-platform quality gate orchestration, generation smoke testing, metadata-only generation run logging, analytics, and retention pruning, sanitized model log diagnostics, runtime setup readiness, System Doctor readiness reporting with active-profile hardware fit, storage capacity diagnostics, and Markdown export, metadata-only Environment Snapshot reporting, sanitized performance decision reporting, privacy-safe support bundle export, guarded runtime preparation, guarded model process management, plugin-local skill discovery, manual extension registry auditing, guided Extension Studio configuration, local audit trail logging and retention pruning, guarded extension self-configuration, runtime health checks, CLI behavior, web UI endpoints, streamed chat generation, persisted local chat sessions, context assembly, file-backed memory, local knowledge ingestion, guarded local memory deletion, read-only memory maintenance, guarded expired-memory pruning, confirmed local data backup and restore, MCP stdio discovery and guarded tool calls, allowlisted local tools, cron permission policy, background cron automation, and orchestrator correlation behavior.

## New Test Surface

- `tests/test_config.py`: duplicate expert ids, missing experts, invalid `top_k`, unsupported aggregation, unknown rule/fallback experts.
- `tests/test_config_profiles.py`: read-only runtime profile discovery, setup readiness summary, active/default/recommended flags, hardware-fit payloads, recommendation scoring, launch hint payloads, and active profile inclusion outside `configs/`.
- `tests/test_profile_activation.py`: confirmation guard, validated app-config default profile writes, restart-required reporting, and recommended profile activation.
- `tests/test_audit.py`: local JSONL audit event writes, latest-first listing, action/status filtering, metadata truncation, and latest-event retention pruning.
- `tests/test_run_log.py`: metadata-only generation run writes, prompt/answer exclusion, latest-first listing, aggregate latency/token/context/error summaries, recommendations, and latest-run retention pruning.
- `tests/test_providers.py`: fake OpenAI-compatible HTTP server, streaming SSE parsing, usage/timing parsing, invalid payload handling, invalid JSON handling, reasoning-channel stripping, transport error wrapping.
- `tests/test_evaluator.py`: JSONL eval loading, minimum coverage guards, and accuracy/complexity aggregation.
- `tests/test_doctor.py`: normalized setup, health, extension, cron, hardware-fit, and storage checks, required failure for profiles that exceed the detected machine, optional storage pressure warnings, and metadata-only Markdown rendering.
- `tests/test_environment.py`: metadata-only environment snapshot generation and Markdown rendering for platform, Python, package, git, hardware, storage, config, and configured local model identity handoffs.
- `tests/test_storage.py`: read-only runtime storage capacity diagnostics for configured model cache and work directories, including missing-path probing without creating directories and low-free-space recommendations.
- `tests/test_smoke.py`: generation smoke pass/fail reports, including explicit failure for blank visible output.
- `tests/test_ci_runner.py`: cross-platform quality gate command plan, `PYTHONPATH` environment construction, and JSON dry-run output.
- `tests/test_runtime.py`: llama-server command/URL construction and health probing.
- `tests/test_setup_status.py`: side-effect-free setup readiness for Hugging Face cache hits, missing local files, Ollama pull commands, and no-model fixture profiles.
- `tests/test_health.py`: runtime health status for reachable, unreachable, malformed, path-prefixed, and skipped expert providers.
- `tests/test_doctor.py`: unified setup, health, process, extension audit, and cron readiness report.
- `tests/test_performance_report.py`: sanitized benchmark decision payload, missing benchmark status, and Markdown report rendering.
- `tests/test_support_bundle.py`: privacy-safe diagnostic bundle content and exclusions, including embedded environment snapshot and storage metadata.
- `tests/test_cli.py`: eval mode, setup readiness, profile recommendation, profile preparation, profile activation, startup runbook preview and confirmation guard, performance report output, environment snapshot output, generation run log listing with metadata-only summary and confirmation-guarded pruning, guarded runtime preparation preview, model process status, sanitized model log tail output, doctor output, and prompt mode through the public CLI.
- `tests/test_chat_store.py`: local chat session create, append, reload, list, search, rename, durable summary, export, and delete behavior.
- `tests/test_data_bundle.py`: portable local data export and restore for chat sessions plus memory records, including merge and replace behavior.
- `tests/test_web.py`: web config, environment snapshot APIs, startup runbook APIs, runtime profile discovery, recommendation, preparation, and activation APIs, generation smoke APIs, generation, streamed generation, persisted chat sessions, metadata-only run log APIs, summaries, and guarded pruning, chat compaction APIs, memory APIs, memory maintenance and expired pruning APIs, guarded memory and knowledge deletion APIs, confirmed local data backup APIs, local audit and audit pruning APIs, knowledge import APIs, chat management APIs, setup preparation APIs, System Doctor APIs, performance report APIs, support bundle APIs, model log diagnostics APIs, plugin creation APIs, extension audit APIs, guided extension templates and configure APIs, extension self-configuration refresh, cron status APIs, and eval endpoints over a local HTTP server.
- `tests/test_setup_runner.py`: runtime preparation preview, confirmation guard, injected install runner, and local-file model validation without network access.
- `tests/test_startup.py`: read-only startup preview, side-effect confirmation guard, and model-manager based startup action orchestration.
- `tests/test_tools.py`: allowlisted local tool execution, knowledge ingestion, memory maintenance, guarded expired-memory pruning, guarded memory/document deletion, confirmed local data export/import, extension audit, guarded extension self-configuration for MCP/cron registries, guarded profile activation, write confirmation, MCP capability search, MCP process confirmation, and guarded MCP `tools/call` execution.
- `tests/test_mcp_client.py`: raw stdio MCP `initialize`, `tools/list`, and `tools/call` behavior against a fake MCP server.
- `tests/test_model_servers.py`: managed model server specs, confirmation guards, reachable-endpoint skips, sanitized bounded log tail diagnostics, missing log reporting, and managed start/stop lifecycle with fake processes.
- `tests/test_extensions.py`: registry loading, guided extension templates, plugin-local skill discovery, plugin audit, and runtime plan coverage.
- `tests/test_scheduler.py`: cron dry runs, allowlisted actions, unsupported command rejection, write-local confirmation, expired-memory pruning jobs, auto-runnable filtering, and background runner status.
- `tests/test_context.py`: cache-friendly context section ordering, policy loading, budget truncation, memory snippet ranking, compaction prompt requirements.
- `tests/test_orchestrator.py`: correlation propagation, compare mode, streamed route/content/final events, and separated routing/generation prompts.
- `tests/test_memory.py`: local memory writes, guarded record/document deletion, knowledge document chunking, scoped listing, temporal validity, keyword retrieval.
- `experiments/eval_set_extended.jsonl`: 56 router cases across coding, architecture, general writing, and mixed prompts.
- `experiments/eval_set_live_general.jsonl`: 52 live general-purpose routing cases across English, Italian, Spanish, French, German, Portuguese, Dutch, Polish, Arabic, Hindi, Japanese, Korean, and Chinese prompts.
- `experiments/route_labels_extended.jsonl`: 56 regenerated distilled router labels from the curated extended eval.
- `experiments/route_labels_live_general.jsonl`: 52 regenerated distilled router labels for the live general-purpose router.
- `experiments/run_quality_gate.py`: project-level quality gate.
- `scripts/run_ci_checks.py`: cross-platform Python quality gate runner used by `make check`, the shell compatibility wrapper, and CI templates.

## Verification

Command:

```bash
python3 scripts/run_ci_checks.py
```

Result:

- compileall: passed
- unit/contract tests: `233/233` passed
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
- Playwright browser screenshot for sanitized Model Logs panel: passed
- Playwright browser smoke for Plugin Studio confirmation guard: passed
- Playwright browser screenshot for guided Extension Studio controls: passed
- Playwright browser smoke for extension registry audit: passed
- Playwright browser screenshot for Audit Trail pruning controls: passed
- Playwright browser screenshot for Run Log pruning controls: passed
- Playwright browser smoke for setup readiness panel: passed
- Playwright browser screenshot for runtime profile discovery: passed
- Playwright browser smoke for guarded setup preparation controls: passed
- Playwright browser smoke for chat compaction action: passed
- Playwright browser smoke for memory panel and context retrieval: passed
- Playwright browser smoke for cron background automation status: passed
- live local-model dashboard screenshot regenerated with `Qwen3-30B-A3B-Instruct-2507-MLX-4bit`: passed

## Notes

The router remains intentionally configurable and deterministic. During the extended eval, a broad `implement` keyword created a false positive by matching `implementation`; it was removed from config and replaced with more specific coding signals such as `client` and `adapter`.

The live general-purpose router now has a balanced multilingual fixture for `general` and `fast_fallback` decisions. Its generated report is stored in `outputs/live-general-routing-eval.json` and is required by the quality gate.

Extension Studio now exposes guided MCP server and cron job presets through `/api/extensions/templates` and guarded writes through `/api/extensions/configure`. It writes only to app-configured registry paths, validates entries before writing, requires confirmation, and refreshes the running web registry and cron runner. The lower-level `extension.configure` tool remains available for CLI and JSON automation.

Streaming generation is now covered from provider SSE parsing through orchestrator events and the web `/api/generate/stream` endpoint. The UI updates the assistant bubble while content arrives and falls back to `/api/generate` if streaming is unavailable before content starts.

Local knowledge import now chunks pasted documents into `knowledge` memory records with document metadata. It is exposed through `knowledge.ingest`, `/api/knowledge`, and the Advanced Knowledge panel, with confirmation required before local records are written. Guarded forget controls now remove a single memory record or all chunks for one imported document id only when confirmation is supplied.

Memory maintenance now reports active, pending, and expired temporal records separately. Expired-memory pruning is a separate guarded action through `memory.prune_expired`, `/api/memory/prune-expired`, the Advanced Memory panel, and optional write-local cron jobs; future `valid_from` records are preserved.

Runtime profile discovery now exposes `/api/config/profiles` and the Advanced Profiles panel. It scans runnable public profiles, includes the active config even when it is outside `configs/`, and reports setup readiness, backend, expert count, model names, hardware fit, local recommendation, and side-effect-labelled launch hints without switching profiles or starting model processes. Hardware fit is computed from the detected machine profile, configurable model candidate manifests, and conservative model-name fallbacks for unknown models. The read-only recommendation is also available through CLI `--recommend-profile` and `/api/config/recommendation`; it scores setup readiness, hardware fit, general-purpose coverage, routing capability, and active/default tie-breaks, then returns rationale plus next actions. Guarded profile preparation is available through CLI `--prepare-profile`, CLI `--prepare-recommended-profile`, web `/api/config/prepare-profile`, and Advanced Profiles; it reuses the setup runner for the selected profile and requires confirmation before install/download side effects. Guarded startup runbook orchestration is available through CLI `--startup`, web `/api/startup`, web `/api/startup/run`, and Advanced Startup; it combines setup inspection, optional preparation, optional model starts, and System Doctor evidence while requiring confirmation for every side effect. Guarded profile activation is available through CLI `--activate-profile`, CLI `--activate-recommended-profile`, web `/api/config/activate-profile`, Advanced Profiles, and the allowlisted `profile.activate` tool. It validates the target profile, requires confirmation, writes only `default_moe_config`, leaves the running process unchanged, and returns a restart command. The same active-profile hardware fit is included in System Doctor readiness; `too_large` is a required failure while `stretch` and `unknown` become operator warnings. System Doctor can also render a metadata-only Markdown handoff report through CLI and `/api/doctor/report.md`. Each displayed launch hint has a guarded clipboard control so operators can copy the exact command while keeping execution explicit.

Model log diagnostics now expose bounded, sanitized model server log tails through CLI `--models-logs`, web `/api/models/logs`, and the Advanced Runtime Model Logs panel. The reader only opens runtime-plan-generated log paths and redacts secret-looking values before returning text.

Local data backup now exports chat sessions and memory records into a schema-versioned JSON bundle through `data.export`, `/api/data/export`, and the Advanced Local Data panel. Restore through `data.import` or `/api/data/import` supports `merge` and `replace` modes and requires confirmation before writing local stores.

Local audit trail logging now records sensitive host-side actions in `<runtime.work_dir>/audit.jsonl` and exposes recent events through `/api/audit` plus the Advanced Audit Trail panel. Guarded retention pruning through `/api/audit/prune` and the same panel keeps the latest configured number of events, requires confirmation, and records its own `audit.prune` event. Audit events intentionally store metadata only, not prompt text, chat transcripts, memory text, environment variables, or model log bodies.

Generation run logging now records successful chat generations in `<runtime.work_dir>/runs.jsonl` and exposes recent records plus aggregate metadata-only summaries through CLI `--runs`, web `/api/runs`, and the Advanced Run Log panel. Guarded retention pruning is available through CLI `--runs-prune --runs-confirm`, web `/api/runs/prune`, and the same panel. Run records intentionally store prompt hash, prompt character count, route/model ids, latency, token usage, throughput, context telemetry, memory ids, error counts, and disagreement status, but not prompt text or answer text. Summaries add average and p95 latency, token totals, top models/experts, context pressure, memory usage, error totals, and operator recommendations without expanding the privacy surface.

The hardware recommendation is now general-purpose MoE with one strong resident general expert, one small resident fallback/compaction expert, and cold-loaded specialists only when evals justify them.

MCP execution is intentionally narrow. The filesystem MCP server advertises write/edit tools, so the example enabled profile is guarded by `allow_process_execution=true`, per-call `confirm_process_execution=true`, per-call `confirm_tool_call=true`, and a server-level `allowed_tools` list that keeps the example profile read-oriented.
