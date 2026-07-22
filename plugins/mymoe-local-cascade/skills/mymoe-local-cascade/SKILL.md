---
name: mymoe-local-cascade
description: Inspect local AI configuration, plan bounded delegation, run classification, extraction, or summarization on configured local models, and inspect metadata-only receipts. Use when Codex should reduce frontier-model work by handing a small, explicit, verifiable task to myMoE Local Cascade.
---

# myMoE Local Cascade

Use the frontier model for orchestration and judgment, then delegate only bounded work that a local model can complete and verify. The reusable local-cascade core remains usable without Codex; this skill only teaches Codex how to call its MCP adapter.

## Workflow

1. Install the `mymoe-local-cascade-mcp` entrypoint. The bundled launcher is currently a POSIX/macOS alpha and requires `python3`; Windows clients must explicitly configure the installed console entrypoint. If Codex has only a cached plugin copy, set `MYMOE_PROJECT_ROOT` to an absolute myMoE checkout with locked dependencies; the launcher uses the locked `local-cascade` extra in offline mode and fails closed instead of downloading. Also configure absolute paths in `MYMOE_LOCAL_CASCADE_CONFIG` and `MYMOE_LOCAL_CASCADE_MOE_CONFIG`. Never print their values. Start from the two shipped LocalCascade examples, replace the model placeholders/endpoints, and keep an explicit terminal-finish-reason allowlist per expert.
2. Call `machine_inspect` before choosing local execution. It reports the current hardware snapshot and strict configuration binding; model readiness is not probed.
3. Keep the task self-contained and choose an explicit `task_kind`: `classification`, `extraction`, or `summarization`.
4. Call `delegate_plan` with an efficiency profile and a small step limit. Require `status=ready`, a plan identifier, a plan digest, and acceptable reason codes before continuing.
5. Read profile policy literally: `economy` tries only the eligible tier with the lowest configured `cost_rank`; `balanced` follows configured `cost_rank` order up to `max_steps`; `quality` tries only the eligible tier with the highest configured `cost_rank`. These ranks are operator declarations, not measured cost or quality.
6. Treat model assets as `not_evaluated`. This alpha never installs or downloads assets; any installation must be a separate, explicit action outside the plugin.
7. Call `delegate_run` with the exact task, plan identifier, and plan digest. Do not alter the task between planning and execution. Set `max_output_chars` at least as high as the configured verifier maximum; the adapter fails before inference instead of truncating a verified result.
8. Use `receipt_inspect` when provenance or debugging is needed. Receipts contain metadata and hashes, never raw task text or model output.

## Delegation boundary

- Delegate only work whose expected output can be checked cheaply.
- Keep high-impact decisions, ambiguous architecture choices, final integration, and user-facing commitments with Codex.
- Do not use this alpha to execute shell, filesystem, browser, desktop, non-loopback-network, or installation actions. The adapter uses the local network stack and permits only a numeric loopback first hop; it does not prove what the local server does downstream.
- Request concise answers and reason codes. Never request or expose hidden reasoning or chain-of-thought.
- Do not pass secrets, credentials, environment dumps, or unnecessary repository context.
- Planning retains the exact task only in the MCP process memory, with bounded oldest-first eviction. A restart clears it. Do not plan material that should not be held in that process.
- One MCP server accepts only one local cascade run at a time. A completed plan remains reusable, so caller retries can repeat local compute.

## Token expectations

Local delegation can reduce later frontier input and output tokens, especially for repeated small subtasks. It cannot recover tokens already spent on the initial request, skill loading, machine inspection, or frontier planning. Measure end-to-end usage before claiming savings.
