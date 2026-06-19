# Architecture

## Design Decision

Use a system-level MoE first, not a trained monolithic sparse transformer.

Reason: local hardware can run quantized local experts, but training or fine-tuning a true sparse MoE is much more expensive than routing among already-good local experts. This design keeps the runtime local while leaving room for later distillation.

The target product is general-purpose. Coding is only one route, not the default identity of the app.

## Components

```mermaid
flowchart LR
  U["User prompt"] --> O["MoE Orchestrator"]
  O --> R["Configurable Router"]
  R --> E1["Primary General Expert"]
  R --> E2["Fast Summary / Fallback Expert"]
  R --> E3["Optional Specialist Expert"]
  E1 --> A["Aggregator / Synthesizer"]
  E2 --> A
  E3 --> A
  A --> Q["Quality Gate"]
  Q --> U
```

## Core Contracts

- `MoEConfig`: immutable parsed configuration.
- `ExpertConfig`: provider id, endpoint, model id, generation params, weight.
- `RoutingRule`: configured keyword/weight mapping to expert ids.
- `Router`: pure deterministic scorer with rules and optional local semantic examples, no provider names in code.
- `Provider`: local inference boundary. Normal use is OpenAI-compatible HTTP against local model servers; synthetic providers are confined to deterministic test fixtures.
- `Orchestrator`: applies route, fallback, timeout, correlation id propagation.
- `Evaluator`: deterministic routing and behavior checks.
- `System Doctor`: read-only control-plane aggregator for setup readiness, endpoint health, model process reachability, extension audit, and cron state.
- `Support Bundle`: privacy-safe diagnostic export for issue reports and handoffs.

## Runtime Modes

### Mode 0: Deterministic Test Fixture

No model required. Validates config, router behavior, evaluator, provider contracts, and CLI in tests. This is not a user-facing application mode.

### Mode 1: Top-1 Resident General Expert

Route to one strong general local endpoint. This is the cheapest real baseline.

### Mode 2: MoE With One Heavy Resident Expert

Keep one heavy general model resident plus one small fast model for compaction/fallback. Cold-load specialists when their eval slice justifies it.

### Mode 3: Top-2 Expert Comparison

Call two experts in parallel and expose a deterministic disagreement report. More expensive, but useful for high-risk, ambiguous, or multimodal tasks. The current report compares lexical overlap, length delta, and expert-specific terms; it does not call another model to judge the answers.

### Mode 4: Distilled Router

Replace hand-written routing rules with a trained small classifier. The experts remain local.

### Mode 5: Distilled Student

If system-level MoE is too slow, distill common expert behavior into one smaller local model.

## Comparable Tool Pattern

The closest local-first assistants usually share the same control-plane shape:

- Chat front end with switchable local or remote model providers.
- Knowledge/RAG layer for files, workspaces, or synced sources.
- Tool calling layer where the model proposes a tool call, the host executes it, then the result is returned to the model.
- Agent presets or workspaces that bind a model, instructions, tools, memory, and retrieval settings.
- Optional automations for scheduled prompts or maintenance jobs.

myMoE follows that pattern, but makes routing and execution policy first-class configuration. Open WebUI emphasizes a broad chat workspace with tools, RAG, memory, automations, and multiple providers. AnythingLLM exposes agents as LLM sessions with custom tools, MCP, and agent flows. LM Studio focuses on running local models behind local REST/OpenAI-compatible endpoints. Ollama exposes local model serving, context length controls, and tool calling primitives. myMoE sits one layer above those runtimes: it can use local OpenAI-compatible servers, but its core value is deciding which local expert, context, memory, and tool policy to use for each request.

## Multilingual Behavior

The app can work across languages when the selected local model supports them, but this should be treated as an evaluated capability rather than a blanket guarantee.

The runtime uses a language-preserving policy:

- Keep UI and documentation in English for product consistency.
- Detect or infer the user's language from the prompt.
- Ask the selected model to answer in the user's language unless the user requests otherwise.
- Keep routing examples multilingual so task classification does not depend only on English keywords.
- Add eval cases per language before claiming support for that language.
- Prefer local multilingual embedding or classifier backends later if rule/character-ngram routing becomes too brittle.

This is why the primary general expert matters more than a coding specialist. A coder-only model may be excellent for Python and terminal tasks, but a general-purpose assistant needs stronger multilingual instruction following, summarization, planning, tool-use comprehension, and conversational behavior.

## Routing Cost Policy

Do not use the heaviest resident model as the default request classifier. Classification runs before every request, so it should be cheap, deterministic, debuggable, and trainable. The heavy model should be reserved for answer generation, synthesis, or ambiguous fallback.

The practical policy is:

- Use configured rules, semantic examples, and a distilled local classifier for normal routing.
- Use the small fallback model for compaction, lightweight classification experiments, and cheap retries.
- Escalate to the heavy general model only when the router has low confidence or when the request itself requires deep reasoning.
- Use stronger teacher models offline to label routing datasets, then distill those labels into local artifacts.

## Failure Modes

- Router examples overfit narrow phrasing and miss semantic intent outside the evaluated languages.
- Multiple experts increase latency linearly if called sequentially.
- Synthesis can hide a bad expert answer instead of exposing disagreement.
- Local context limits differ per model; routing must know each expert context budget.
- Model-specific chat templates can break answer quality if endpoints are not normalized.

## Validation Gates

1. Config loads and validates.
2. Router selects expected experts on known prompts.
3. Provider boundary preserves `correlation_id`.
4. Local endpoint smoke test returns valid text under timeout.
5. MoE beats single-model baseline on a small rubric before adding complexity.
