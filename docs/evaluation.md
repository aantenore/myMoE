# Evaluation

## Baselines

Always compare at least:

1. single local general model,
2. MoE top-1 routing,
3. MoE top-2 routing with comparison/concatenation and disagreement reporting.

## Metrics

| Dimension | Measurement | Gate |
| --- | --- | --- |
| Training-fit routing accuracy | expected expert vs selected expert | diagnostic only; never a generalization claim |
| Holdout routing accuracy | expected expert vs selected expert | >= 0.90 for the committed live routing gate |
| Latency | wall-clock seconds and tokens/sec | must be acceptable interactively |
| Completeness | rubric per task family | no regression vs single model |
| Failure transparency | errors preserve correlation id and expert id | required |
| Compare transparency | top-k compare responses expose deterministic disagreement metadata | required for `aggregation = compare` |
| Local footprint | RAM/VRAM/disk | must fit target machine |

## Current Eval Sets

The base eval set is intentionally small and deterministic. It covers:

- coding,
- architecture,
- general language,
- mixed coding/architecture prompts.

The extended eval set adds broader router pressure:

- coding provider and test requests,
- architecture and gateway decisions,
- general writing and summarization,
- mixed prompts where keywords intentionally collide.

The live routing data has separate roles:

- `eval_set_live_general.jsonl` is the curated source used to create the 52
  training labels. Its score measures training fit only.
- `eval_set_live_general_holdout_v5.jsonl` is a frozen, independently authored
  52-case holdout with 26 cases per expert, 13 per complexity level, and all 13
  evaluated routing languages: Arabic, German, English, Spanish, French, Hindi,
  Italian, Japanese, Korean, Dutch, Swedish, Turkish, and Chinese. This set is
  separate from the app-level response-language hints.
- The quality gate requires zero shared ids and zero shared normalized prompt
  hashes, then binds config, training labels, artifact, holdout, and report with
  SHA-256 provenance.

The committed holdout is a regression suite once observed. Rotate or add a new
unseen set before making a fresh generalization claim.

## Current Results

- Deterministic routing fixture eval: `8/8` after config adjustment.
- Extended deterministic routing eval: `56/56` across coding, architecture, general writing, and mixed prompts.
- Live small-model benchmark: `196.47 tok/s` generation on Qwen2.5-Coder-1.5B Q4_K_M.
- Live single-expert eval: 6 real calls, average latency `0.585 s`, average server-reported generation speed `194.93 tok/s`.
- Training-fit routing diagnostic: `52/52`; this is expected because those
  cases generated the labels and must not be presented as generalization.
- Leakage-free live routing holdout: `52/52` (`100%`), 95% Wilson interval
  `93.1%-100%`, with zero id or prompt-hash overlap. Evidence lives in
  `outputs/live-general-routing-holdout.json`.
- Current live answer-quality benchmark: 72 completed executions with zero
  response failures or truncation. Single-general and routed top-1 both
  achieved `1.0` task success, `1.0` quality pass rate, and `1.0` quality
  score. Mean latency was `2.5915 s` vs `2.4913 s`; the six fast-routed pairs
  achieved median latency ratio `0.4231` while routing 25% of cases away from
  the general model.
- Top-2 remains diagnostic: it reported complete comparisons and disagreement,
  but achieved `0.75` task success, `0.9437` quality score, and `3.0044 s` mean
  latency. It is not the value route and its result is not masked by top-1.
- Current host evidence passed with `100%` sample coverage, `86.1555%` peak RAM
  use, and `249,749,832` bytes swap growth.
- The release quality gate now passes with `release_ready = true`. The previous
  72-record evidence remains preserved as
  `outputs/quality-benchmark-2026-07-10.json` for historical comparison.

Interpretation: the harness works, the current router clears the committed
multilingual routing gate, and routed top-1 demonstrates benchmark-bounded
value without quality regression in the current live run. The bidirectional
runtime fallback preserves availability when a fast route misses or its
endpoint is offline. The deterministic rubric does not establish broad
semantic superiority, so future model/profile changes still require fresh live
evidence.

## Verified Outcome Routing and Signed Canary Eval

The shadow lab has a separate evaluation contract because it predicts whether
premium escalation is needed rather than which local expert best matches a
prompt. On paired counterfactual cases:

- `need_premium = not local_verified and premium_verified`;
- a local decision when premium was needed is a false local;
- a premium decision when local was verified is unnecessary premium use;
- cases where neither route verifies are reported as unrecoverable and do not
  inflate escalation precision or recall.

The report also tracks verified success, tokens, optional measured cost, p95
latency, remote payload size, Brier score, expected calibration error, and
strata for capability, difficulty, language, and context band. CI expands the
compact fixture into 64 pairwise-covered deterministic cases and compares
`local_only`, `current_baseline`, and `verified_shadow`.

The committed
[`verified-routing-shadow-eval.json`](../outputs/verified-routing-shadow-eval.json)
is a formula and invariant test only. It declares
`claim_scope = synthetic_deterministic_only` and `empirical_claim = false`; its
numbers must never be presented as measured product savings. Activation would
require disjoint real outcome cohorts from the target configuration plus a
separate regression gate.

That regression gate now exists as an offline structural-eligibility contract. It
requires a content-addressed plan frozen before execution, exact scorecard
reconstruction from the supplied training set, zero training/holdout overlap,
complete baseline/candidate arms for every planned task, per-cell sample floors,
Wilson lower bounds, zero candidate-only failures, monotone premium/egress/cost
usage, and bounded latency. A passing report may emit a short-lived canary
manifest. The disabled-by-default runtime can consume it only with a separate
operator-signed authorization bound to the stable Bridge configuration, route
policy, scorecard, pinned public key, assignment-bucket threshold, and contained
time window. No empirical manifest or authorization is checked in because real
paired Assistant Bridge evidence is not yet available.

The Signed Route Canary Authority has a deterministic contract suite, not a
product-savings benchmark. It verifies that:

- the manifest alone has no authority and an invalid, wrong-key, expired, or
  mismatched Ed25519 DSSE authorization preserves the baseline;
- configuration, manifest, policy, scorecard, signal-provider, runtime-plan,
  and route-receipt digests stay bound end to end;
- HMAC assignment is stable for the same secret and task fingerprint, the
  runtime rejects a secret whose SHA-256 does not match the manifest, and no
  more than 500 of 10,000 deterministic hash buckets can be eligible. This is a
  sampling threshold, not a guarantee that at most 5% of observed requests will
  be routed through the canary;
- only an exact qualified cell and a monotone transition toward less premium
  use can apply; blocked, outside-bucket, wider, stale, or abstained decisions
  retain the baseline;
- the locked content-bound local chronology rejects local clock rollback,
  activation rollback, and signed equivocation while its prior state remains
  intact; it makes no tamper-resistance or external timestamp claim;
- canary lineage is preserved in route receipts and verified outcome records;
- `verified_routing.enabled=false` bypasses canary evaluation entirely.

These tests establish structural safety only. A real canary still requires the
disjoint empirical qualification above, an operator decision, monitoring of the
same quality/latency/premium/egress/cost dimensions, and expiry or kill-switch
rollback. Neither synthetic shadow metrics nor signature tests justify a claim
of live savings.

`route_canary` is an optional alpha extension to the existing
`RouteDecisionReceipt` 2.0 and `VerifiedOutcomeRecord` 1.0 JSON shapes. Current
readers accept legacy payloads that omit it. Older strict readers reject the new
field, so producers and consumers that exchange canary-bearing payloads must be
upgraded together; this is not a forward-compatible wire extension for older
binaries.

The benchmark contract lives in `configs/quality-benchmark.json`,
`experiments/quality_benchmark_cases.jsonl`, and
`experiments/run_quality_benchmark.py`. Routed top-1 is the value variant and
must not regress against the single-general baseline; top-2 is diagnostic and
must report complete comparison/disagreement evidence without masking a top-1
regression. The release profile requires a complete provenance-bound artifact
from the live Qwen3 4B and Qwen3 1.7B endpoints. The offline CI overlay may
report unavailable endpoints, but it cannot declare release readiness.

The live contract runs three repetitions. With the current eight-case dataset,
the two fast-routed cases therefore contribute six paired latency observations.
The benchmark keeps the default profile's configurable concise system prompts
and caps generation at 768 tokens. This represents the interactive product
contract; operators can use a quality-first profile for intentionally longer
answers.
Both the single-general baseline and routed top-1 must also stay at or below a
30-second mean latency; a relative speedup cannot make an unusably slow pair
release-ready.
Release evidence also requires host RAM and swap counters: swap growth must stay
at or below 4 GiB and observed peak RAM use at or below 95%. These host-wide
limits require valid before/after and peak counters plus at least 90% periodic
sample coverage. The bounded coverage tolerance handles transient host-command
timeouts under accelerator load without accepting missing memory evidence.
These limits intentionally tolerate ordinary desktop noise while rejecting the
catastrophic pressure seen with the earlier large-model topology. Missing RAM or
swap counters fail the release profile; an offline CI run may only skip an
absent or explicitly blocked live artifact.

Provenance binds the benchmark code, wrapper, lockfile, source configuration,
dataset, router artifact, benchmark-process Python/package versions, and the
locally cached Hugging Face snapshot revisions. The standard OpenAI-compatible
`/models` endpoint exposes neither the served snapshot revision nor the
inference-server package build. Both cache-to-server matching and server-package
identity are therefore recorded as explicitly unverified rather than presented
as attested.

## Quality Gate

Before adding another model download, the project must pass:

```bash
./scripts/run_all_checks.sh
```

The automated gate checks:

- compileability for source, tests, scripts, and experiments,
- unit tests for config, router, provider contracts, runtime, evaluator, CLI, and orchestrator,
- base and extended routing eval,
- a disjoint live routing holdout regenerated on every run,
- the 64-case Verified Outcome Routing shadow simulation,
- deterministic Signed Route Canary authority, lineage, hash assignment, chronology,
  fail-closed, and kill-switch tests,
- artifact and report provenance freshness,
- zero train/holdout id or normalized prompt-hash overlap,
- minimum holdout routing accuracy of `0.90` across at least `50` cases,
- minimum extended routing accuracy of `0.90`,
- minimum extended routing eval size of `50`,
- required project files,
- no leftover local live eval listener on `127.0.0.1:8101` or
  `127.0.0.1:8102`.
