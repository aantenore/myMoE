# Changelog

All notable changes to myMoE are documented in this file.

## [0.2.0-alpha.1] - 2026-07-19

### Added

- A provider-neutral Execution Scope Guard for routing, fallback, streaming,
  parallel generation, and agent execution.
- Explicit `device_only`, `private_mesh`, `public_mesh`, and `paid_remote`
  policy vocabulary plus `direct_local`, `mesh_llm`, and `gateway` transports.
- Fail-closed `scope_blocked` behavior and immediate pre-invocation rechecks.
- Explicit execution policy and expert declarations in shipped local profiles.

### Changed

- Product positioning now describes myMoE as a local-first orchestration
  runtime rather than a prototype.

### Known limitations

- The built-in attestor authorizes only direct-local loopback experts. Loopback
  proves the first network hop, not model placement.
- Mesh-LLM v0.73.1 `/api/status` is not request-bound and has no
  `schema_version`; the Mesh adapter therefore remains disabled and fail-closed.
- `paid_remote` is reserved for the separately approval- and budget-gated
  Assistant Bridge and is not enabled for normal chat routing.
