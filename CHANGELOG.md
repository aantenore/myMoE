# Changelog

All notable changes to myMoE are documented in this file.

## [Unreleased]

## [0.4.0-alpha.1] - 2026-07-19

### Added

- An installable `mymoe-paired` CLI for provider-free journal status and
  fail-closed execution of exact frozen cases, plus a private directory
  exchange for a separately operated signed independent verifier. Execution
  requires public trust, a preinitialized evidence store, and embedded pricing;
  myMoE never owns the signing key.
- Verified Hybrid Execution contracts for content-addressed AB/BA runs that
  bind the evidence plan, normalized item, exact routes, one frozen source
  snapshot, Bridge/runner lineage, and a versioned pricing contract.
- Preregistered runner-source, signed-attestation-policy, and semantic
  execution-harness identities derived from the inspected configuration rather
  than accepted as caller-supplied hashes.
- A metadata-only append-only paired-run journal with durable pre-invocation
  claims, ordered checkpoints, safe partial resume, concurrent-claim exclusion,
  and an explicit indeterminate state that forbids automatic retry after an
  ambiguous provider call. Stable hashes and provider/runtime metadata remain
  sensitive and are not publication-safe; POSIX permissions are checked while
  Windows ACL privacy remains operator-managed.
- An evaluation-only Assistant Bridge route constraint that can retain the
  guarded baseline or reduce premium use, rechecks every existing hard guard,
  executes only in a disposable workspace, and has no source-apply authority.
- Exact per-command token cost evidence derived from provider/model pricing
  contracts instead of accepting an unbound caller total.
- Evaluation-only DSSE attestations that preserve independently signed pass and
  fail results without widening the pass-only contract used for source apply.
- Content-addressed receipts that let qualification and signing reconstruct
  outcomes from the original signals, result metadata, candidate artifacts,
  pricing lineage, and signed verifier envelopes instead of trusting JSONL
  summary rows.
- A preregistered paired-evidence gate that rebuilds scorecard lineage, rejects
  training/holdout overlap, applies intention-to-treat coverage, and evaluates
  exact profile/capability/difficulty/runtime cells.
- Content-addressed promotion reports plus short-lived, structural-eligibility canary
  manifests for transitions that monotonically reduce premium use.
- Atomic no-clobber evidence writes, strict gate/plan contracts, Wilson success
  bounds, cost completeness, latency, egress, premium-use, and hard-invariant
  guardrails.

### Changed

- Refreshed the live 72-execution Qwen3 quality artifact. Routed top-1 retained
  full task success and deterministic quality while the release quality gate
  became ready.

### Known limitations

- Signed canary manifests are consumed only when the disabled-by-default
  runtime, pinned operator key, short-lived authorization, lineage, chronology,
  and deterministic assignment all agree; every failure retains the baseline.
- Real paired Assistant Bridge evidence has not yet been collected, so the
  repository ships no empirical canary manifest and makes no measured savings
  claim. Independent evidence and complete provider usage remain prerequisites
  for eligibility; a deterministic-only or cost-incomplete run is diagnostic.
- DSSE signatures establish the integrity and provenance of recorded
  evaluations; they do not establish that benchmark inputs are representative
  or that a verifier's checks measure the right real-world quality.

## [0.3.0-alpha.1] - 2026-07-19

### Added

- A shadow-only Verified Outcome Routing Lab that connects content-free route
  receipts to final verification and operational metrics.
- Replaceable structural task-signal contracts with explicit difficulty,
  confidence, out-of-distribution detection, and abstention.
- Immutable, content-addressed `VerifiedOutcomeRecord` JSONL storage plus
  scripts to derive signals and record Assistant Bridge outcomes.
- Versioned route scorecards grouped by configuration, plan, capability, and
  difficulty, with evidence-strength floors, expiry, source digests, verified
  success, p95 latency, tokens, premium use, egress, and optional measured cost.
- Configurable `economy`, `balanced`, `quality`, `privacy`, and `offline`
  shadow policies with hard-eligibility preservation, Pareto filtering, utility
  scoring, and conservative abstention.
- A 64-case pairwise-covered deterministic simulation for escalation metrics,
  calibration, costs, latency, tokens, egress, and per-dimension strata.

### Changed

- Execution-scope rechecks now preserve the original attestation authority
  instead of replacing its provenance with a generic guard label.
- The canonical CI gate now regenerates the Verified Outcome Routing shadow
  simulation on every supported operating system and Python version.

### Known limitations

- Contract version 1 can recommend but never apply a route. There is no online
  learning, automatic policy activation, or exploration.
- The committed 64-case report is synthetic contract evidence, not a measured
  savings or quality claim.
- Cost remains unknown unless a caller supplies a real versioned pricing or
  accounting value; cost-weighted policies abstain on incomplete cells.

## [0.2.0-alpha.1] - 2026-07-19

### Added

- A provider-neutral Execution Scope Guard for routing, fallback, streaming,
  parallel generation, and agent execution.
- Explicit `device_only`, `private_mesh`, `public_mesh`, and `paid_remote`
  policy vocabulary plus `direct_local`, `mesh_llm`, and `gateway` transports.
- Fail-closed `scope_blocked` behavior and immediate pre-invocation rechecks.
- Explicit execution policy and expert declarations in shipped local profiles.
- Structured `scope_blocked` responses across CLI, Web, streaming, chat
  compaction, tools, and generation smoke reports.

### Changed

- Product positioning now describes myMoE as a local-first orchestration
  runtime rather than a prototype.
- Model HTTP calls now reject cross-origin redirects, validate the final URL,
  preserve POST bodies only for 307/308 redirects, and ignore ambient proxies
  for loopback endpoints.
- Compaction, agent calls, health checks, model management, and benchmark
  readiness probes now enforce the same execution-scope policy before network
  access.
- Model management re-attests before each probe and again immediately before a
  local process start, and never launches a local server command for an
  attested Mesh or gateway transport.

### Known limitations

- The built-in attestor authorizes only direct-local loopback experts. Loopback
  proves the first network hop, not model placement.
- Mesh-LLM v0.73.1 `/api/status` is not request-bound and has no
  `schema_version`; the Mesh adapter therefore remains disabled and fail-closed.
- `paid_remote` is reserved for the separately approval- and budget-gated
  Assistant Bridge and is not enabled for normal chat routing.
