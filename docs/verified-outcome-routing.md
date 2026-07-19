# Verified Outcome Routing Lab

The Verified Outcome Routing Lab is a shadow-only feedback layer for the
Hybrid Assistant Bridge. It answers a narrower question than semantic or
complexity routing:

> For this capability and difficulty band, which policy would have delivered a
> mechanically verified result with the best acceptable quality, cost,
> latency, privacy, and premium-use profile?

Version 1 never changes a live route. It records content-free outcomes, builds
a versioned scorecard, and replays alternative decisions offline. Activation is
deliberately deferred until a disjoint evaluation demonstrates that a proposed
policy improves the chosen objective without bypassing an existing authority,
budget, capability, or execution-scope guard.

## Why This Is a Separate Layer

The local `RuleRouter` already solves expert affinity with deterministic rules,
character n-gram examples, and a distilled local artifact. The Execution Scope
Guard decides where a request is allowed to run. The Assistant Bridge owns
local execution, independent verification, bounded premium escalation, and
content-free route receipts.

The lab reuses all three boundaries:

```mermaid
flowchart LR
  T["Task envelope metadata"] --> S["Structural signal provider"]
  T --> G["Existing hard guards"]
  G --> R["Existing RouteDecisionReceipt"]
  R --> E["Local or premium execution"]
  E --> V["Deterministic or independent verification"]
  S --> O["VerifiedOutcomeRecord"]
  R --> O
  V --> O
  O --> A[("Append-only outcome log")]
  A --> C["Versioned route scorecard"]
  C --> P["Shadow policy replay"]
  P --> Q{"Disjoint quality gate"}
  Q -->|"insufficient or regressed"| H["Keep current live policy"]
  Q -->|"future explicit activation"| F["Versioned, rollbackable policy"]
```

This avoids building another provider gateway or another four-tier prompt
classifier. Existing projects already cover those areas: [vLLM Semantic
Router](https://github.com/vllm-project/semantic-router), [LiteLLM Auto
Routing](https://docs.litellm.ai/docs/proxy/auto_routing), and
[RouteLLM](https://github.com/lm-sys/RouteLLM). myMoE instead closes the
decision-to-verification loop while retaining its local-first authority model.

## Content-Free Contracts

### `TaskSignals`

The built-in provider uses only fields already present in the route receipt:
capability identifiers, tool count, risk class, objective length, constraint
count, and an optional context-token estimate. It never stores or classifies
the objective text. The provider emits:

- a request fingerprint;
- declared capabilities;
- `simple`, `medium`, `complex`, or `very_complex`;
- confidence and an explicit abstention flag;
- structural counters and provider identity.

The provider interface is replaceable. A future vLLM Semantic Router or
LiteLLM adapter can emit the same contract, but it cannot expand the set of
routes admitted by myMoE's hard guards.

### `VerifiedOutcomeRecord`

Each append-only record binds:

- the route receipt and task fingerprints;
- configuration and provider-runtime digests;
- planned route and final provider identifiers;
- capability/difficulty signals and confidence;
- verification status, evidence strength, evidence digest, and failure class;
- latency, token counts, premium calls, remote payload size, and optional cost.

Prompt text, response text, constraints, diffs, verifier output, and reasoning
are not fields in this schema. Cost remains absent unless the caller supplies a
versioned pricing contract or a measured amount.

### `RouteScorecard`

The builder aggregates compatible records by configuration digest, route plan,
the exact canonical capability set, and difficulty. Marginal evidence for
`analysis` and `code` cannot be combined into evidence for an
`analysis + code` request. Each cell contains sample counts, verified success,
p95 latency, mean tokens, premium calls, remote payload, and cost when complete.
Abstained signals are always excluded, and the artifact records the configured
minimum confidence used to exclude weak signals. The artifact also binds its
source digest and freshness window. Mixed configuration cohorts, stale
artifacts, non-finite metrics, and insufficient evidence fail closed.

### `ShadowRouteDecision`

The shadow selector receives the current receipt, its hard-eligible routes,
signals, a scorecard, and a replaceable profile policy. It reports the current
route, recommendation, candidate exclusions, normalized utility components,
policy digest, and scorecard digest. `applied` is always `false` in contract
version 1.

## Run the Shadow Loop

Start from a JSON export of `BridgeRunResult.metadata_payload()`. The export may
contain route, verifier, command, and capsule metadata, but must not contain the
user-facing `result.content` object.

Derive structural signals without loading the task text:

```bash
PYTHONPATH=src python3 experiments/derive_route_signals.py \
  --bridge-metadata work/bridge-run-metadata.json \
  --out work/task-signals.json
```

Append the verified result. Omit `--estimated-cost-usd` unless the value comes
from a real, versioned pricing or accounting contract:

```bash
PYTHONPATH=src python3 experiments/record_verified_outcome.py \
  --bridge-metadata work/bridge-run-metadata.json \
  --signals work/task-signals.json \
  --store work/verified-routing-outcomes.jsonl
```

Build a scorecard from deterministic and independently attested evidence:

```bash
PYTHONPATH=src python3 experiments/build_route_scorecard.py \
  --records work/verified-routing-outcomes.jsonl \
  --minimum-evidence-strength independent \
  --minimum-confidence 0.70 \
  --ttl-seconds 2592000 \
  --out work/verified-routing-scorecard.json
```

Ask for a shadow recommendation. This command cannot execute a provider or
alter the receipt's live route:

```bash
PYTHONPATH=src python3 experiments/recommend_verified_route.py \
  --bridge-metadata work/bridge-run-metadata.json \
  --signals work/task-signals.json \
  --scorecard work/verified-routing-scorecard.json \
  --policy configs/verified-routing-policy.example.json \
  --out work/verified-routing-decision.json
```

Run the deterministic contract simulation used by CI:

```bash
make eval-verified-routing
```

Its committed output is
[`outputs/verified-routing-shadow-eval.json`](../outputs/verified-routing-shadow-eval.json).
The example policy normalizations are saturation scales, not provider prices;
replace them with values appropriate to the measured environment.

## Evidence Hierarchy

Scorecards can set a minimum evidence strength. The supported order is:

1. deterministic tests, contracts, and verifiers;
2. independently attested evidence;
3. explicit user evaluation;
4. a versioned judge rubric;
5. implicit interaction feedback.

The shipped example accepts only the first two classes. An inconclusive record
is retained for diagnostics but is not counted as verified success or failure.

## Policy Profiles

The five existing user profiles remain distinct objectives, not model names:

| Profile | Shadow objective | Non-negotiable boundary |
| --- | --- | --- |
| `economy` | Minimize expected cost and premium use after a quality floor. | Never exceed the receipt budget or eligible routes. |
| `balanced` | Trade verified success against latency, cost, egress, and premium use. | Existing scope and consent still win. |
| `quality` | Maximize verified success, then latency and cost. | Capability and authority gaps still block. |
| `privacy` | Minimize remote payload and premium use. | Remote remains unavailable without the existing explicit opt-in. |
| `offline` | Compare local plans only. | Remote candidates are always excluded. |

Weights, evidence floors, confidence thresholds, freshness, and minimum sample
counts live in JSON configuration rather than provider-specific code.

## Evaluation Contract

The deterministic lab compares three policies on the same cases:

- `local_only`;
- the current profile baseline;
- the verified shadow recommendation.

It reports verified success, false-local rate, unnecessary-premium rate,
escalation precision/recall, premium calls and tokens, optional cost, p95
latency, remote egress, Brier score, and expected calibration error. Results are
stratified by capability, difficulty, language, and context band when those
dimensions are available.

Synthetic fixtures validate formulas and invariants; they are not product
performance evidence. A live policy must be trained and evaluated on disjoint,
versioned records from the target environment.

## Safety Invariants

- Shadow analysis cannot add a route absent from the original receipt.
- `privacy` and `offline` cannot be softened by utility weights.
- Missing, stale, low-confidence, or statistically thin evidence returns the
  current baseline route and an abstention reason.
- Scorecard generation never learns from prompt or response bodies.
- No live weight mutation, exploration, or automatic policy activation exists
  in version 1.

The outcome loop is intentionally conservative: optimization starts only after
authority and verifiability are already established.
