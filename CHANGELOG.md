# Changelog

All notable changes to myMoE are documented in this file.

## [Unreleased]

### Added

- `mymoe advisor`, an offline, read-only Adaptive Cell Advisor that evaluates a
  task demand against live local resources and exact configured cell passports,
  then recommends the best eligible configured cell with current verified
  evidence or abstains with machine-readable reason codes.
- Versioned catalog and passport contracts that keep cell declarations,
  observed component identities, resource estimates, and workload-specific
  measurements separate and content-addressed.
- Deterministic hard eligibility filters followed by a profile-weighted Pareto
  selection across verified quality, p95 latency, and peak memory evidence.
- Metadata-only human and JSON receipts binding the exact task fingerprint,
  optional caller-supplied intent family, workload demand, evaluation contract,
  catalog, live resource snapshot, candidate assessments, and final advice.
- `--task-file` and `--task-stdin` alternatives to command-line task text, plus
  a full content-addressed receipt envelope and atomic no-clobber `--out`
  publication.
- A vendor-neutral, deliberately unqualified example catalog that demonstrates
  safe abstention instead of presenting invented model or benchmark evidence.
- `mymoe advisor-init`, which materializes a self-contained zero-claim starter
  without overwriting existing paths, and an installed **Find the right local
  setup** mini-app launched with `mymoe-web --app-config <dir>/app.json`.
- A strict, bounded, no-store `GET /api/advisor/config` and `POST /api/advisor`
  boundary that keeps workload, capabilities, tool surfaces, and risk under
  local policy while returning three non-technical states plus the complete
  technical receipt.
- A deterministic synthetic contract benchmark, integrated into the canonical
  CI plan, covering profile-dependent valid selection, stale/resource-pressure
  abstention, and separate exact lineage for paraphrases with a shared
  caller-declared intent family. It makes no empirical model-quality,
  performance, or semantic-normalization claim.

### Security

- The advisor makes zero model calls, uses no network, and does not download,
  start, stop, or reconfigure a model. Every v1 receipt is non-applying and
  non-authorizing.
- Catalog, evaluation-contract, and referenced evidence inputs are size-bounded;
  referenced evidence must be a regular non-symlink file below the catalog root
  and must match its declared SHA-256 digest.
- Catalog and app configuration JSON reject duplicate keys at every depth; app
  configuration reads are bounded and use the same no-follow regular-file
  boundary.
- The loopback control plane rejects non-loopback clients, malformed or
  wrong-port Host headers, and browser origins that do not exactly match the
  request authority on every supported HTTP method, preventing DNS-rebinding
  and cross-port local-origin access to Advisor, chat, memory, and operational
  endpoints. The separately authenticated `/v1/*` model gateway preserves its
  documented loopback-origin policy for local editor and web clients. Generic
  control-plane JSON is now bounded, depth-limited, and duplicate-key safe as
  well.
- Receipt publication pins its destination directory, uses directory-relative
  no-replace operations on POSIX, and cannot redirect cleanup through a swapped
  parent path. Read-only host probes discard output from failed processes.
- The macOS coding canary now selects its read-denial probe from any explicit
  sandbox-denied root, so a checkout outside the user's home remains
  qualifiable without weakening the host-file boundary.
- Advisor task files use the same bounded non-link loader. Receipt publication
  requires an existing real parent, rejects input aliases and existing targets,
  and forces mode `0600` on POSIX. Windows inherits the destination directory
  ACL; no owner-only DACL is claimed.
- The mini-app accepts only strict fixed-length UTF-8 JSON with allowlisted
  fields, returns `Cache-Control: no-store`, does not persist task or receipt,
  and cannot widen the locally configured workload or risk policy.
- Unknown, stale, future-dated, mismatched, insufficient, or inapplicable
  identity, evaluation, platform, and resource evidence fails closed before
  ranking.

### Known limitations

- Windows swap use is deliberately unknown and therefore causes abstention.
  Linux and Windows GPU identity and accelerator-memory discovery are not
  implemented, so accelerator cells abstain there when that evidence is needed.
- Source path, byte, and digest verification does not authenticate who produced
  the evidence or prove that its claims are true; operators must trust the local
  evidence producer.
- A live resource snapshot can become stale immediately after collection. The
  advice does not reserve memory and cannot authorize a later execution.
- v1 does not implement ACP delivery, desktop or browser actions, model
  concurrency or lifecycle changes, response caching, or an LLM-based semantic
  intent normalizer.
- Task wording is fingerprinted but not interpreted. Workload, capabilities,
  tool surfaces, risk class, and goal are explicit caller or local-policy
  declarations; a matching intent-family digest never authorizes response
  reuse.

## [0.8.0-alpha.1] - 2026-07-21

### Added

- An opt-in Desktop Semantic Cell that lets the built-in local-model agent read
  the bounded accessibility state of one operator-selected application process
  and window through a single `desktop.observe` contract.
- A provider-neutral `attest/start/observe/close` lifecycle with Cua Driver
  `0.10.0` as the first pinned local MCP stdio adapter.
- An owned-daemon boundary that launches Cua in `bounded` mode on a private
  POSIX socket, applies exact session and target-argument policies, verifies
  daemon process and policy state, and revokes and stops the daemon on close.
- Dedicated app and provider configuration examples plus a deterministic
  live canary for the read-only desktop contract.
- A `desktop-init` binder that hashes the installed native provider and current
  target process, validates the exact platform-native catalog and observe
  schema, writes that platform's admitted schema digest, disables provider
  telemetry, erases its telemetry identifier, and creates an installable
  workspace exclusively. Files request mode `0600` on POSIX; Windows retains
  the destination directory's inherited ACL rather than claiming an owner-only
  DACL.
- A Desktop Semantic Cell guide covering the plain-language use case,
  originality boundary, provider admission, threat model, platform limits, and
  future resource-aware semantic-versus-vision routing.
- A deterministic 512-node payload benchmark that gates redaction, forbidden
  addressing removal, bounded output, and reductions in model-visible tools and
  serialized observation size.
- A Linux, macOS, and Windows CI contract job that installs the exact optional
  provider wheel, checks its locked version, complete platform catalog (53, 49,
  and 50 tools respectively), catalog-name digest, and exact semantic schema,
  and reports the observed native executable digest without GUI access,
  telemetry, or update checks.
- A repository security policy describing supported alpha versions, private
  vulnerability reporting, and the project's main trust boundaries.

### Security

- The model sees only `desktop.observe`; upstream application and window
  discovery, screenshots, coordinates, mouse and keyboard actions, clipboard,
  files, shell, process control, and every other provider tool remain hidden.
- Every observation is bound to one configured target and live provider state.
  Exact approval also binds the target and configuration digest. Application
  restart during a call, target drift, provider, daemon, policy, or schema
  mismatch, empty, fully invalid after normalization, or degraded output,
  unexpected image content, oversized output, and invalid structure fail closed.
- Screenshot capture is disabled in the upstream request with
  `include_screenshot=false`. Semantic output is bounded, normalized, redacted,
  stripped of secure values, and labelled as untrusted content before it reaches
  the model.
- The serialized-result budget drops only trailing nodes, retaining a useful
  semantic prefix inside the normal agent-loop limit instead of replacing the
  entire accessibility tree with a truncation marker.

### Known limitations

- This alpha is read-only and accessibility-tree-only. It does not prove visual
  layout, colors, canvas content, images, animation, or pixel correctness, and
  it grants no authority to perform desktop actions.
- macOS TCC permissions and Linux toolkit and Wayland behavior vary by host and
  target application. An application can expose an incomplete or misleading
  accessibility tree; provider completeness remains explicitly unknown. The
  POSIX runtime is live-qualified on macOS; Linux requires a local bound-window
  canary.
- Only the provider/schema wheel contract is tested on Windows. The owned-daemon
  runtime fails closed there until private named-pipe ownership and teardown are
  qualified.
- Cua Driver is a pre-1.0 trusted dependency. The private socket and bounded
  policies narrow authority but do not sandbox a compromised admitted provider
  process. A host-level egress boundary is still required for an independently
  enforced air-gap claim.

## [0.7.0-alpha.1] - 2026-07-21

### Added

- An opt-in Browser Capability Cell that lets the built-in local-model agent
  navigate, observe, type, and click inside one explicitly approved local web
  origin through four myMoE-owned contracts rather than raw Playwright tools.
- A deterministic five-step browser canary covering navigation, accessibility
  observation, typing, clicking, and denial of a second unapproved loopback
  service without requiring a model.
- `mymoe browser-init`, which materializes a self-contained app, MCP, model, and
  context-policy workspace at its chosen location from the installed wheel
  without overwriting existing files.
- `mymoe browser-prefetch`, which fills a fresh npm cache without executing the
  provider package or lifecycle scripts, records the resolved dependency-lock
  digest, and verifies the pinned top-level archive before first execution.
- A persistent, UTF-8 MCP stdio session with bounded messages, deterministic
  process-tree teardown, POSIX process groups, and a kill-on-close Windows Job
  Object launcher.
- Cross-platform CI jobs that admit the exact browser dependency and run the
  real canary with offline provider resolution on Linux, macOS, and Windows
  with Python 3.10 and 3.12.

### Security

- Normal browser HTTP(S) traffic is forced through a parent-owned proxy that
  forwards only the approved scheme, loopback host, and port. Chromium's implicit
  loopback bypass is disabled, and redirects are rechecked from every returned
  observation.
- Click and type approvals bind browser session, exact origin, full snapshot
  hash, revision, reference, and accessible label. A fresh pre-action snapshot
  must match before the action is sent upstream.
- Browser-capability MCP servers are exclusive to the guarded runner; generic
  raw-MCP discovery and call paths reject them.
- The provider launch is an exact offline argument profile with an empty
  provider environment. myMoE recomputes the cached package archive SHA-512,
  hashes Node plus the configured and effective launch arguments, and verifies live
  upstream schema digests before exposing tools. The profile selects Google
  Chrome explicitly instead of inheriting an upstream default. Windows executes `node.exe`
  plus `npx-cli.js` directly instead of crossing a batch-file argument parser.
- Provider errors, state drift, invalid bindings, oversized MCP responses, and
  dead processes invalidate the entire browser lifecycle and fail closed.
- Offline archive verification has a separate configurable 10-180 second
  startup bound, and the canary never retries a failed runtime attestation.

### Known limitations

- This alpha qualifies accessible interactions with one local HTTP(S) origin;
  it does not qualify the public web, login sessions, downloads, visual layout,
  WebSocket upgrades, desktop control, or a model's ability to complete a real
  browser task.
- The proxy constrains normal browser HTTP(S) traffic, not a compromised Node
  dependency, non-HTTP browser networking, server-side egress by the local app,
  or the host network. The Node/npm toolchain and remaining dependency tree are
  trusted computing-base components rather than OS-sandboxed code.
- The pre-action accessibility snapshot narrows stale-state risk but cannot make
  a JavaScript-driven action atomic or prove visual and semantic intent. Use a
  disposable environment for untrusted applications.

## [0.6.0-alpha.1] - 2026-07-20

### Added

- `mymoe coding-canary`, a metadata-only macOS qualification for one exact
  Cline CLI 3.0.46, myMoE gateway, pinned local model, runtime configuration,
  hardware fingerprint, disposable single-file edit, and pristine test cell.
- A separate `coding-canary` package extra and a bounded missing-extra error,
  keeping the base distribution dependency-free and avoiding import tracebacks.
- A Cline `PreToolUse` policy gate, independent NDJSON tool-input validation,
  parent-owned authenticated inference broker, exact model and route pinning,
  stable workspace attestation, and a separately isolated pristine verifier.
- A canonical digest of the complete effective myMoE runtime configuration,
  including routing, timeouts, provider parameters, execution declarations,
  and rules. The loopback `/api/config` endpoint exposes only the digest for
  declared-versus-live binding.
- Deterministic tests for hook ordering, Cline event lifecycles, exact tool
  inputs, late hardlinks, wrong executable digests, gateway drift, proxy races,
  metadata privacy, and fail-closed result classification.
- Exact handling for Cline 3.0.46's fixed AI SDK warning banner while every
  other non-NDJSON stdout line remains fail-closed.

### Security

- The canary hashes and rejects an untrusted Cline executable before running
  its bounded version probe, and rechecks its identity after the agent run.
- Its generated `sandbox-exec` policy denies host-file writes by default,
  re-allows only isolated state, temporary data, and the exact source edit,
  blocks general network and network bind, and is exercised by a live policy
  probe before the agent starts.
- Proxy handler threads are joined before evidence is frozen, candidate bytes
  must still equal the exact attested fix before verification, and reports omit
  prompts, tool inputs, outputs, credentials, and raw filesystem paths.

### Known limitations

- The canary is diagnostic-only, macOS-only, and pinned to one Cline version
  and one narrow edit-and-test contract. It never authorizes routing and does
  not qualify real repositories, unrestricted terminal use, browser or desktop
  control, MCP, Git publication, or general autonomy.
- The measured Apple M5 Pro / 24 GiB Qwen3 Coder 30B-A3B 4-bit cell returned
  `incompatible`: the isolation and binding gates passed, but the pre-tool gate
  rejected an editor request outside the exact fixture contract.

## [0.5.0-alpha.1] - 2026-07-20

### Added

- A loopback OpenAI-compatible gateway for editor-agent harnesses, exposing
  `/v1/models` and regular or streaming `/v1/chat/completions` with routed
  `mymoe` and pinned `mymoe/<expert-id>` aliases, fresh execution-scope checks,
  bounded proxying, and metadata-only audit events.
- A Cline Local Coding Fabric guide with exact VS Code setup, a read-only
  compatibility canary, 24 GiB model guidance, explicit air-gapped versus
  browser-connected semantics, MCP trust boundaries, and an
  accessibility-first desktop sidecar roadmap.
- A successful isolated Cline 4.0.10 read-only canary through myMoE and local
  Qwen3-4B, with correct file-derived output, metadata-only gateway evidence,
  and a byte-identical workspace after execution.
- A compact Local Coding Fabric setup link in the existing web UI welcome view.
- Managed model servers now start with Hugging Face and Transformers offline
  mode enforced, so model downloads remain an explicit bootstrap step rather
  than an implicit runtime network action.
- Packaged fallback configuration and context policy assets, allowing the
  installed `mymoe-web` console script to start from an empty directory without
  depending on a source checkout.

### Security

- The shared UI/control-plane listener is loopback-only, gateway streams and
  JSON structure are bounded, diagnostic reports omit secret-environment names,
  and managed model processes disable implicit model-network access.

## [0.4.0-alpha.1] - 2026-07-20

### Added

- `mymoe assistant-probe`, a bounded diagnostic that checks whether the
  configured local model can recover a random marker through the Codex
  workspace tool protocol without retaining prompt, response, or marker content
  in its metadata report.
- A metadata-only real-host compatibility snapshot for the shipped
  `qwen3:4b`/Ollama provider, including public command/runtime identity, a
  stable declared-config digest, and the content-addressed Ollama model digest.
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

- POSIX directory-sidecar consumers now wait only for the adapter's exact
  link-before-unlink publication state. Persistent hard links remain rejected,
  and response content is never read before the final path has one link.
- The shipped local Assistant Bridge provider now advertises read-only
  `analysis` and no tools because filesystem/tool compatibility was not
  demonstrated by the bounded live probe. Its sandbox and workspace ceilings
  are also read-only. Operators can opt into broader capabilities only after
  evaluating their exact model/runtime combination.
- Direct Codex command-plan construction now rejects capability, tool, risk,
  network, and workspace requests outside the provider declaration. Probe
  reports reuse the hardened no-link atomic writer and distinguish incompatible
  responses from operationally indeterminate runs.
- Codex execution now requires version 0.138 or newer and requests strict named
  permission profiles instead of the legacy sandbox flag. The requested
  profiles expose only the minimal runtime plus the selected workspace, keep
  shell network disabled, and bind native cached-web authority separately into
  receipts and command digests. Receipts explicitly mark the effective profile
  as unattested because managed requirements and the beta Codex permission
  runtime remain part of the trusted computing base; unknown configuration
  fails closed without a myMoE legacy fallback.
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
