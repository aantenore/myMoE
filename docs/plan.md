# Plan

Build a local-first MoE in stages, validating each stage before adding complexity. The runtime path stays local; stronger cloud/agent teachers are allowed only for dataset creation and distillation, not production inference.

## Scope

- In: system-level MoE router, local expert endpoints, evaluation harness, distillation plan.
- Out: training a full sparse transformer MoE from scratch, cloud runtime dependency, opaque provider-specific core logic.

## Action Items

- [x] Reset the old project and remove previous model/tool artifacts.
- [x] Create a dependency-free MoE orchestrator prototype.
- [x] Add mock experts for deterministic offline routing tests.
- [x] Add local OpenAI-compatible provider boundary for llama.cpp, Ollama, LM Studio or vLLM.
- [x] Add an eval set and smoke evaluation harness.
- [x] Run routing experiment and fix config-level failure.
- [ ] Plug one real local GGUF expert into `configs/moe.local.example.json`.
- [x] Plug one real local GGUF expert into `configs/moe.live.qwen25-coder.json`.
- [x] Benchmark single expert vs MoE top-1 routing.
- [ ] Add top-2 routing plus deterministic disagreement reporting.
- [ ] Generate route-label dataset with a strong teacher.
- [ ] Train or implement a distilled local router.

## Milestones

### M0: Mock MoE

Status: complete.

Goal: verify router/config/eval harness.

### M1: One Real Expert

Status: complete with Qwen2.5-Coder-1.5B Q4_K_M smoke model.

Run one local GGUF server as the `coder` or `single` expert. Keep mock experts for the rest when testing hybrid routing.

### M2: Three Real Experts

Run local experts on separate ports:

- `coder`: coding-focused GGUF.
- `architect`: MoE or reasoning-oriented GGUF.
- `general`: smaller dense fallback.

### M3: Distilled Router

Create teacher-labelled route data and replace keyword rules with a learned local classifier.

### M4: Distilled Student

Only if latency is unacceptable, distill frequent task classes into one smaller local model.
