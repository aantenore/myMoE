# Changelog

All notable changes to myMoE are documented in this file.

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
