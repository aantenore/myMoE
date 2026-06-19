# Plan

Build a local-first MoE in stages, validating each stage before adding complexity. The runtime path stays local; stronger cloud/agent teachers are allowed only for dataset creation and distillation, not production inference.

## Scope

- In: system-level MoE router, local expert endpoints, evaluation harness, distillation plan.
- Out: training a full sparse transformer MoE from scratch, cloud runtime dependency, opaque provider-specific core logic.

## Action Items

- [x] Reset the old project and remove previous model/tool artifacts.
- [x] Create a dependency-free MoE orchestrator prototype.
- [x] Add deterministic fixture experts for offline routing tests.
- [x] Add local OpenAI-compatible provider boundary for llama.cpp, Ollama, LM Studio or vLLM.
- [x] Add an eval set and smoke evaluation harness.
- [x] Run routing experiment and fix config-level failure.
- [x] Plug real local GGUF specialist profiles into the live config set.
- [x] Add automatic model download planning for MLX, Ollama, and GGUF profiles.
- [x] Add local fallback-backed context compaction provider.
- [x] Add guarded runtime self-configuration from CLI and UI.
- [x] Add guarded model process management from CLI and UI.
- [x] Add System Doctor readiness report from CLI, API, and UI.
- [x] Add privacy-safe support bundle export from CLI, API, and UI.
- [x] Add sanitized performance decision report from CLI, API, and UI.
- [x] Add progressive streamed chat responses with non-streaming fallback.
- [x] Add local knowledge import into context retrieval.
- [x] Add guarded local memory and knowledge deletion controls.
- [x] Add read-only memory maintenance reports and guarded expired-memory pruning.
- [x] Add confirmed local data backup and restore for chats plus memory.
- [x] Add local audit trail for sensitive host-side actions.
- [x] Add guarded audit trail retention pruning from API and UI.
- [x] Add Plugin Studio and plugin-local skill discovery.
- [x] Add manual extension registry audit from CLI, API, and UI.
- [x] Add guarded extension self-configuration for MCP server and cron job entries.
- [x] Plug one real local GGUF expert into `configs/moe.live.qwen25-coder.json`.
- [x] Benchmark single expert vs MoE top-1 routing.
- [x] Add top-2 routing plus deterministic disagreement reporting.
- [x] Generate route-label dataset with curated teacher labels.
- [x] Train and enable a distilled local router artifact.
- [x] Expand route evals and distilled labels to 50+ multilingual cases.

## Milestones

### M0: Deterministic Routing Fixture

Status: complete.

Goal: verify router/config/eval harness.

### M1: One Real Expert

Status: complete with Qwen2.5-Coder-1.5B Q4_K_M smoke model.

Run one local model server as the `coder`, `single`, or `general` expert. Synthetic experts remain only inside deterministic fixture tests.

### M2: Three Real Experts

Run local experts on separate ports:

- `primary-general`: Qwen3 30B-A3B MLX 4-bit.
- `fallback-compaction`: Gemma 4 E4B MLX 4-bit.
- `coding-agentic`: optional Gemma 4 12B Agentic GGUF v2 or Qwen3-Coder profile.

### M3: Distilled Router

Create teacher-labelled route data and replace or augment keyword rules with a learned local classifier.

Status: complete for the current centroid classifier artifact. The live general router now has 52 curated multilingual route labels and the extended fixture eval has 56 cases. Next improvements should add domain-specific slices only when a new tool, model, or specialist route is introduced.

### M4: Distilled Student

Only if latency is unacceptable, distill frequent task classes into one smaller local model.
