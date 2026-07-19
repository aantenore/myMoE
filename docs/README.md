# myMoE Documentation

This is the documentation map for myMoE. Start with the path that matches what you are trying to do; the deeper pages explain the implementation and link to the relevant source files.

## Start Here

| If you want to... | Read... |
| --- | --- |
| Understand the complete system | [How myMoE works](how-it-works/README.md) |
| Install and start local models | [Installation](installation.md) |
| Use the web app, CLI, or HTTP API | [UI and CLI](ui.md) |
| Understand routing decisions | [Routing](router.md) |
| Control where inference may execute | [Execution Scope Guard](execution-scopes.md) |
| Understand chat context, memory, and persistence | [Context and Memory Architecture](context-architecture.md) |
| Run a tool-calling task safely | [Agent Runtime](agent-runtime.md) |
| Use local Codex first and escalate only after verification | [Hybrid Assistant Bridge](hybrid-assistant-bridge.md) |
| Learn from verified local/premium outcomes without changing live routes | [Verified Outcome Routing Lab](verified-outcome-routing.md) |
| Inspect the independent candidate evidence contract | [Independent Candidate Attestation Predicate v1](spec/independent-candidate-attestation/v1/README.md) |
| Check readiness or troubleshoot model startup | [Installation: Doctor](installation.md#doctor) and [Agent Runtime: Local Model Requirement](agent-runtime.md#local-model-requirement) |
| Validate a change | [CI](ci.md) and [Evaluation](evaluation.md) |

## System Design

- [How myMoE works](how-it-works/README.md) — end-to-end request, startup, routing, agent, and data-lifecycle diagrams.
- [Architecture](architecture.md) — system-level MoE decision, runtime modes, contracts, limitations, and release gates.
- [Routing](router.md) — exact score composition, semantic examples, distilled artifact, top-k, aggregation, and fallbacks.
- [Execution Scope Guard](execution-scopes.md) — scope/transport policy, fail-closed eligibility, fallback widening, and Mesh trust assumptions.
- [Context and Memory Architecture](context-architecture.md) — context budgets, session summaries, memory retrieval, compaction, and run telemetry.
- [Agent Runtime](agent-runtime.md) — strict tools, risk policy, approvals, MCP, plugins, cron, and diagnostic surfaces.
- [Hybrid Assistant Bridge](hybrid-assistant-bridge.md) — task-level local/premium policy, verification receipts, minimal escalation capsules, and CLI launch boundary.
- [Verified Outcome Routing Lab](verified-outcome-routing.md) — structural task signals, metadata-only outcomes, versioned scorecards, shadow replay, and preregistered paired promotion qualification without runtime activation.
- [Independent Candidate Attestation Predicate v1](spec/independent-candidate-attestation/v1/README.md) — stable provenance and integrity contract for candidate evidence.

## Run and Operate

- [Installation](installation.md) — Apple Silicon MLX, Windows/Linux Ollama, llama.cpp, model profiles, startup, and Doctor.
- [UI and CLI](ui.md) — chat behavior, streaming, session management, memory, local data, Advanced panels, API examples, and screenshots.
- [Gemma 4 E4B Runtime](gemma-e4b-runtime.md) — pinned dependency compatibility and measured result for that optional profile.
- [CI](ci.md) — local and GitHub Actions verification.

## Models and Evidence

- [Model Selection](model-selection.md) — hardware assumptions and model-role decisions.
- [Performance Benchmarking](performance-benchmarking.md) — benchmark method and candidate set.
- [Tested Performance](tested-performance.md) — measured results for the tested machine and profile.
- [Evaluation](evaluation.md) — routing, quality, integrity, and release-gate contracts.
- [Distillation Plan](distillation-plan.md) — route-label and answer-distillation stages.

Files under [`../outputs/`](../outputs/) are generated or historical evidence. They are not normative runtime documentation. When a report conflicts with current configuration or source code, the current configuration and source code win.

## Current Language Claims

Three different concepts are easy to confuse:

1. **Response-language hints** are the values configured in [`../configs/app.json`](../configs/app.json): `it`, `en`, `fr`, `de`, `es`, `pt`, `nl`, `pl`, `ar`, `hi`, `ja`, `ko`, and `zh`, plus `auto`.
2. **Routing examples** are multilingual phrases embedded in a particular MoE profile. They can cover a different set and can be replaced with the profile.
3. **Evaluated router coverage** is what a disjoint holdout actually measures. The current 52-case holdout contains four cases each for `ar`, `de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `nl`, `sv`, `tr`, and `zh`.

None of these lists is a universal language-quality guarantee. Answer quality still depends on the selected local model, and a new language should receive both configured routing examples and independent evaluation cases before support is claimed.

## Documentation Rules

- Runtime diagrams describe implemented behavior, not planned behavior.
- The online generation path ends when a response is returned and persisted; release quality gates run offline.
- Normal generation retrieves memory but does not create durable memory records automatically.
- A chat exchange is persisted only after a successful final response.
- `compaction_needed` is an advisory signal; durable model-generated compaction is an explicit action.
- Examples use the locked Python 3.12 environment because the project requires Python 3.10 or newer.

[Back to the project README](../README.md)
