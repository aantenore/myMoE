# Process-bound Runtime Supervisor

**It answers one practical question: is this local model endpoint still the
same server process myMoE started for this exact model?**

In plain terms, a loopback URL alone is not enough: another local program could
already own the port or replace the server after startup. The process-bound
runtime supervisor starts one direct `llama-server`, binds its process identity
to one numeric-loopback listener, checks the advertised model, and refuses to
report readiness when those observations disagree.

This is a POSIX-only v1 alpha. It is a runtime-lifecycle boundary, not a coding
agent, editor integration, MCP host, model router, downloader, sandbox, or
cryptographic attestation system.

## What changes for an operator

Without this boundary, an application can know that `127.0.0.1:PORT` responds
but cannot show that the responder is the process it launched. With the v1
supervisor, readiness requires all of these observations to agree:

- the endpoint was vacant before launch, so the supervisor never attaches to an
  existing service;
- one exact, previously anchored `llama-server` executable was launched
  directly, without a shell;
- the root PID, creation time, executable digest, and root-only process shape
  still match;
- the numeric-loopback listener belongs only to that root PID;
- bounded `GET /health`, `GET /props`, and `GET /v1/models` probes identify the
  configured model;
- the same process and listener evidence is observed again after the probes.

The complete static binding is re-hashed immediately before spawn and again
after initial readiness; process, listener, and application evidence is then
collected again before `ready` can be returned. Foreground continuity checks
sample process/application state frequently, compare cheap executable/GGUF
inode, size, mode, modification, and change-time fingerprints, and repeat the
full static re-hash at a bounded 300-second cadence or immediately after
fingerprint drift. Every such long re-hash is followed by another dynamic
inspection. This avoids reading a multi-gigabyte GGUF every few seconds while
still failing closed on ordinary file or runtime changes.

If cleanup cannot prove that the owned process group is empty and the endpoint
is vacant, the result becomes sticky `unknown_blocking`. A new launch through
the installed CLI is refused until an operator resolves that ambiguity outside
this contract.

The installed CLI always uses one canonical private coordination domain; it
does not expose a state-directory override. Embedders that construct the Python
service directly must share the same lease store to retain this guarantee; a
separate database is a separate coordination domain. The alpha has no automatic
or CLI reconciliation command. An unknown row keeps its exact host and port
reserved but does not consume the normal live-lease count; other reviewed
endpoints remain usable. Do not delete the row or reuse that port merely because
it looks idle. Use a different bound endpoint, or retain the evidence for a
future separately reviewed reconciliation workflow.

## V1 lifecycle

```mermaid
flowchart LR
  B["Exact static binding: binary + GGUF + launch plan"] --> L["Prepare owner-bound lease"]
  L --> V["Prove numeric-loopback port is vacant"]
  V --> S["Spawn one direct llama-server"]
  S --> O1["Observe root process + listener"]
  O1 --> G["Bounded GET identity probes"]
  G --> O2["Re-observe root process + listener"]
  O2 --> R["Ready metadata receipt"]
  R --> T["Terminate owned process group"]
  T --> C["Prove group empty + port vacancy"]
  C --> X["Stopped, or sticky unknown_blocking"]
```

The static and dynamic checks have different jobs. The
[Bound Cell Attestor](cell-runtime-binding.md) anchors the exact runtime and
GGUF identities before launch. The supervisor then binds that static evidence
to the process and listener it owns. A model identifier returned over HTTP is
supporting application evidence; it does not replace the separately anchored
GGUF digest.

The metadata lease ledger coordinates one owner and one endpoint authority. It
stores digests, states, reason codes, process identity, and listener evidence.
The raw lease capability is never serialized. The ledger does not start,
inspect, stop, or authorize a process by itself.

## State and failure contract

The durable states are:

| State | Meaning |
| --- | --- |
| `prepared` | Exact static binding and owner are recorded; no runtime is ready. |
| `starting` | The owner is attempting the one direct launch. |
| `ready` | Process, listener, and application evidence agreed at the sampled readiness boundary. |
| `stopping` | Teardown is in progress; no new inference authority is implied. |
| `stopped` | Owned-process exit and endpoint vacancy were both observed. |
| `revoked` | The lease was invalidated before a safe ready state could be retained. |
| `unknown_blocking` | Ownership or cleanup is ambiguous; ordinary transitions cannot clear it. |

There is no adoption of an existing process and no automatic restart. PID
reuse, endpoint substitution, executable drift, an unexpected descendant,
advertised-model drift, or an unverified cleanup fails closed. A `ready`
receipt is sampled metadata, not permanent authority and not permission to run
inference.

The final receipt states that the supervisor control plane made no remote
request and that the offline launch profile was used. Runtime egress remains
`not_observed`: the supervisor neither sandboxes nor attests the server's full
network behavior.

The example policy is
[`runtime-supervisor-policy.example.json`](../configs/runtime-supervisor-policy.example.json).
Its fixed booleans make the authority boundary explicit: metadata only, no
process mutations through the ledger, no persisted raw token, no adoption, and
no automatic restart. The direct launcher is the separate component that owns
the process lifecycle.

## Deliberately excluded

V1 does not provide:

- attach mode, daemon discovery, process adoption, or recovery by reconnecting;
- a multi-model router, model autoloading, remote model download, or delete API;
- `POST` control probes or an additional supervisor inference endpoint; the
  owned `llama-server` still exposes its native local inference API;
- agent, editor, browser, tool, MCP, UI, or workspace authority;
- hostname endpoints, non-loopback binds, proxy use, or redirects;
- automatic restart or fallback to another runtime;
- descendant-process containment; v1 admits a root-only process and rejects
  observed descendants;
- Windows lifecycle support.

The owner must remain alive in the foreground for the owned lifecycle. Each
additional model requires its own direct server, endpoint, binding, and lease;
v1 does not combine them behind a runtime router.

## Where it fits instead of replacing everything else

- [`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
  remains the inference engine. Its broader server/router surface can solve
  different problems; this profile deliberately selects one direct model and
  disables optional control surfaces. The upstream
  [security policy](https://github.com/ggml-org/llama.cpp/security) also warns
  against treating its router as a boundary in an untrusted environment.
- `systemd`, `launchd`, and container runtimes are better process/service or
  isolation boundaries. They do not by themselves bind a reviewed GGUF,
  launch plan, advertised model, and sampled listener owner into one receipt.
  Use them around this supervisor when stronger isolation or restart policy is
  needed; v1 intentionally provides neither.
- Coding clients such as
  [Goose](https://github.com/aaif-goose/goose/blob/main/documentation/docs/getting-started/providers.md)
  or [Continue](https://docs.continue.dev/customize/model-providers/top-level/openai)
  can consume a local OpenAI-compatible endpoint. The supervisor is not a
  coding client and does not improve model quality; it only makes the local
  runtime lifecycle less ambiguous.

## Threat model and trusted computing base

The supervisor is designed to catch accidental port collisions, stale PIDs,
PID reuse, a listener owned by another local process, runtime executable drift,
unexpected descendants, advertised-model changes, and incomplete cleanup. It
does this with OS and application observations around bounded readiness and
teardown boundaries.

It does **not** isolate mutually hostile processes running as the same OS user.
It does not prevent a compromised runtime from lying in its JSON responses,
proxying elsewhere, loading undeclared dynamic code, or changing between
samples. SHA-256 digests here are content-integrity anchors, not signatures,
remote attestation, a hardware root of trust, or authenticated provenance.
V1 also does not pin the GGUF through an inherited immutable file descriptor;
a hostile same-user process that swaps and restores bytes entirely between
samples is outside this boundary. Use OS isolation and immutable deployment
storage when that actor is in scope.

The trusted computing base includes:

- the same-user POSIX kernel process and socket information;
- the owner Python process, `psutil`, filesystem reads, and clock observations;
- on macOS, the root-owned system `/usr/sbin/lsof` fallback used when global
  `psutil` listener enumeration is denied, plus a non-reuse bind vacancy probe;
- the direct `llama-server` executable and its runtime behavior;
- the separately reviewed static binding producer and catalog;
- the local metadata lease store and the operator who resolves sticky unknown
  state.

Use an OS user, container, or virtual-machine boundary when the local runtime
must be isolated from hostile code. The process-bound supervisor does not
replace those controls.

## Deterministic contract benchmark

Run the model-free benchmark from the repository root:

```bash
uv run python experiments/benchmark_process_bound_runtime.py --check
```

The benchmark uses deterministic fakes and actively blocks real socket
creation, process spawning, subprocess calls, and URL fetches. It checks the
happy path, occupied port, PID reuse, port substitution, unexpected descendant,
binding drift, verified cleanup, and sticky cleanup ambiguity.

A passing result is evidence that the injected contract scenarios do not
produce a false `ready`. It is **not production evidence** for OS containment,
real `llama-server` compatibility, model identity or quality, security,
latency, memory use, throughput, or performance on a particular machine.

The production adapter has separately completed one real macOS arm64
compatibility run. Its sanitized, path-free result is published as
[`process-bound-runtime-live-canary.json`](../outputs/process-bound-runtime-live-canary.json).
The run loaded an already-present GGUF, reached verified readiness, repeated
the inspection, made zero inference calls, and proved teardown. It is one
compatibility data point, not a cross-platform certification or a security
proof. Run `mymoe-runtime check` with the exact binding on every machine and
runtime build you intend to rely on.
