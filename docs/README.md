# myMoE Documentation

This is the documentation map for myMoE. Start with the path that matches what you are trying to do; the deeper pages explain the implementation and link to the relevant source files.

## Start Here

| If you want to... | Read... |
| --- | --- |
| Let the cheapest suitable offline model role try a bounded task, verify it, and escalate only when needed | [LocalCascade](local-cascade.md) |
| Fingerprint the bindings declared for a local cell without starting it | [Bound Cell Attestor](cell-runtime-binding.md) |
| Find which fully evidenced local model/runtime/harness cell is eligible now | [Adaptive Cell Advisor](adaptive-cell-advisor.md) |
| Recheck that exact cell and current resources without running it | [Adaptive Cell Execution Gate](cell-execution-gate.md) |
| Stop participating local agents from counting the same observed free memory twice | [Cooperative Resource Lease](cooperative-resource-lease.md) |
| Use that exact already-running numeric-loopback cell for one guarded inference attempt | [Bound Cell Run v1](bound-cell-run.md) |
| Use Cline as a private local coding agent | [Local Coding Fabric](local-coding-fabric.md) |
| Exercise a local web app with a local model | [Browser Capability Cell](browser-capability-cell.md) |
| Read one selected desktop window with a local model | [Desktop Semantic Cell](desktop-semantic-cell.md) |
| Understand the complete system | [How myMoE works](how-it-works/README.md) |
| Install and start local models | [Installation](installation.md) |
| Use the web app, CLI, or HTTP API | [UI and CLI](ui.md) |
| Understand routing decisions | [Routing](router.md) |
| Control where inference may execute | [Execution Scope Guard](execution-scopes.md) |
| Understand chat context, memory, and persistence | [Context and Memory Architecture](context-architecture.md) |
| Run a tool-calling task safely | [Agent Runtime](agent-runtime.md) |
| Use local Codex first and escalate only after verification | [Hybrid Assistant Bridge](hybrid-assistant-bridge.md) |
| Check whether a local model can really use Codex workspace tools | [Hybrid Assistant Bridge: Local Provider Compatibility](hybrid-assistant-bridge.md#local-provider-compatibility-probe) |
| Learn from verified local/premium outcomes and optionally run a signed, narrow canary | [Verified Outcome Routing and Signed Canary Authority](verified-outcome-routing.md) |
| Inspect the independent candidate evidence contract | [Independent Candidate Attestation Predicate v1](spec/independent-candidate-attestation/v1/README.md) |
| Check readiness or troubleshoot model startup | [Installation: Doctor](installation.md#doctor) and [Agent Runtime: Local Model Requirement](agent-runtime.md#local-model-requirement) |
| Validate a change | [CI](ci.md) and [Evaluation](evaluation.md) |

## System Design

- [LocalCascade](local-cascade.md) — replaceable local roles, deterministic verifier-driven escalation, separate evidence accounting, and a paired-run measurement contract.
- [How myMoE works](how-it-works/README.md) — end-to-end request, startup, routing, agent, and data-lifecycle diagrams.
- [Bound Cell Attestor](cell-runtime-binding.md) — bounded fingerprints of declared local runtime/model/harness bindings, short-lived receipts, and explicit non-authority.
- [Adaptive Cell Advisor](adaptive-cell-advisor.md) — whole-cell passports, live resource admission, safe abstention, mini-app/API, deterministic contract benchmark, and market boundary.
- [Adaptive Cell Execution Gate](cell-execution-gate.md) — receipt-bound fresh admission, exact drift checks, strict dry-run policy, and a non-authorizing result.
- [Cooperative Resource Lease](cooperative-resource-lease.md) — atomic same-user/same-host accounting, conservative pool claims, delivery fencing, crash quarantine, strict receipts, and the boundary between cooperation and real RAM/VRAM reservation.
- [Bound Cell Run v1](bound-cell-run.md) — implemented alpha with pre-inspection, snapshot-bound atomic cooperative admission, two `GET /models` probes, one delivery-fenced compute-only completion `POST`, and a post-inspection; no tools, lifecycle, retry, process/residency attestation, or arbitrary semantic-correctness claim.
- [Local Coding Fabric](local-coding-fabric.md) — exact Cline connection, gateway contract, 24 GiB resource modes, network boundaries, and desktop sidecar roadmap.
- [Browser Capability Cell](browser-capability-cell.md) — persistent local-only browser tools, deterministic canary, exact approvals, schema attestation, and desktop adapter roadmap.
- [Desktop Semantic Cell](desktop-semantic-cell.md) — one read-only semantic window tool, exact target binding, bounded redaction, Cua Driver admission, threat model, and platform limits.
- [Architecture](architecture.md) — system-level MoE decision, runtime modes, contracts, limitations, and release gates.
- [Routing](router.md) — exact score composition, semantic examples, distilled artifact, top-k, aggregation, and fallbacks.
- [Execution Scope Guard](execution-scopes.md) — scope/transport policy, fail-closed eligibility, fallback widening, and Mesh trust assumptions.
- [Context and Memory Architecture](context-architecture.md) — context budgets, session summaries, memory retrieval, compaction, and run telemetry.
- [Agent Runtime](agent-runtime.md) — strict tools, risk policy, approvals, MCP, plugins, cron, and diagnostic surfaces.
- [Hybrid Assistant Bridge](hybrid-assistant-bridge.md) — task-level local/premium policy, verification receipts, minimal escalation capsules, and CLI launch boundary.
- [Verified Outcome Routing and Signed Canary Authority](verified-outcome-routing.md) — an offline evidence lab plus a separate, disabled-by-default runtime boundary for an operator-signed, less-premium canary.
- [Independent Candidate Attestation Predicate v1](spec/independent-candidate-attestation/v1/README.md) — stable provenance and integrity contract for candidate evidence.

## Run and Operate

- [LocalCascade contract benchmark](local-cascade.md#deterministic-contract-benchmark) — exercise the offline cascade with frozen injected attempts and no model download or invocation.
- [Installation](installation.md) — Apple Silicon MLX, Windows/Linux Ollama, llama.cpp, model profiles, startup, and Doctor.
- [Local Coding Fabric](local-coding-fabric.md#five-minute-cline-setup) — connect Cline to the running local gateway.
- [Cooperative Resource Lease](cooperative-resource-lease.md#exact-state-machine) — understand acquisition, delivery fencing, release, sticky unknown outcomes, and operational limits.
- [Bound Cell Run v1](bound-cell-run.md#command-workflow) — explicitly confirm one exact admitted inference attempt and keep its metadata-only evidence envelope separate from the answer; confirmation authorizes only that invocation.
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

- Runtime diagrams describe only implemented behavior.
- The online generation path ends when a response is returned and persisted; release quality gates run offline.
- Normal generation retrieves memory but does not create durable memory records automatically.
- Cline owns its file, terminal, browser, MCP, approval, and conversation state; the myMoE gateway supplies inference and records operational metadata only.
- Cooperative Resource Lease serializes only processes using the same database and sentinel root. It records conservative claims but does not reserve RAM/VRAM through the operating system, coordinate other users or hosts, or manage model lifecycle.
- Bound Cell Run requires explicit `--confirm` one-shot authority, permits one compute-only inference attempt, and never turns that confirmation into future authority. Its full endpoint sequence is atomic cooperative admission, one `GET /models`, a durable delivery fence, one completion `POST`, settlement, then a second `GET /models`, with sampled static evidence before and after; it does not manage model lifecycle, use tools, attest process/residency identity, or verify arbitrary answer semantics. The endpoint must use an explicit numeric loopback IP, never `localhost`.
- A chat exchange is persisted only after a successful final response.
- `compaction_needed` is an advisory signal; durable model-generated compaction is an explicit action.
- Examples use the locked Python 3.12 environment because the project requires Python 3.10 or newer.

[Back to the project README](../README.md)
