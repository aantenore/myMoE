# Hybrid Assistant Bridge

Status: implemented and adversarially tested local-first execution slice.

## Product decision

myMoE should not become another model gateway. Mature projects already cover that
layer:

- [LiteLLM](https://docs.litellm.ai/) normalizes providers and supplies retries,
  fallbacks, budgets, and gateway controls.
- [RouteLLM](https://github.com/lm-sys/RouteLLM) trains and evaluates weak-versus-
  strong model routers.
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) performs
  system-level semantic model, tool, and cache routing.
- [Codex OSS mode](https://learn.chatgpt.com/docs/config-file/config-advanced#oss-mode-local-providers)
  already runs against Ollama or LM Studio with `--oss --local-provider`; Codex
  also supports custom model providers.

The missing application-level decision is different: **did the local assistant
produce enough verified evidence to finish this task, and if it did not, what is
the smallest safe handoff to a premium assistant?**

The Hybrid Assistant Bridge therefore adds proof-carrying, failure-driven
escalation above provider/model routers. Existing gateways remain replaceable
adapter targets; this slice adds no gateway dependency.

## Boundary

```text
task + profile + capabilities + budget
                  |
                  v
        deterministic route policy
                  |
       +----------+-----------+
       |                      |
       v                      v
 local Codex OSS       premium Codex
       |
       v
 mechanical verification
       |
   pass|fail
       |  +---------------------------+
       |                              |
       v                              v
   finish locally          minimal escalation capsule
                                      |
                                      v
                               premium Codex
```

The bridge owns task-level policy and evidence. Codex, LiteLLM, RouteLLM, vLLM,
Ollama, and LM Studio remain execution adapters. The bridge must never infer
authorization from a model response.

## Contracts

- `AssistantTaskEnvelope`: objective, chosen profile, constraints, capability
  demand, risk class, and bounded premium-call budget.
- `CapabilityDemand`: required capabilities and tools. Provider support comes
  from configuration, not hardcoded model names.
- `WorkspaceAttestation`: the exact workspace snapshot fingerprint plus Git
  `HEAD`, index and status digests, the complete in-scope file-manifest digest,
  file count, and byte count. Diff evidence is collected separately when an
  escalation needs it; snapshot telemetry never relabels manifest data as a
  staged, unstaged, or untracked diff.
- `RouteDecisionReceipt`: deterministic metadata-only proof of the chosen route,
  policy rules, effective model/adapter/sandbox, workspace fingerprint, budget,
  and configuration digest.
- `VerificationEvidence`: mechanical evidence bound to the exact task,
  workspace state, verifier contract, and artifact digest. It records no
  assistant output or hidden reasoning. Run telemetry separates `prior`
  evidence used to explain escalation from `final` evidence produced against
  the completed candidate; only the final phase can satisfy completion.
- `EscalationCapsule`: a bounded, redacted task objective plus constraints,
  failure codes, evidence references, and an optional bounded diff. It excludes
  the original conversation and local execution transcript. Before publication,
  a final recursive pass covers every non-generated string, capability/tool
  identifier, and mapping key; validated generated hashes and ids remain opaque
  lineage metadata.

Routes are `local`, `premium`, `local_then_verify`, and `blocked`. Profiles are
`economy`, `balanced`, `quality`, `privacy`, and `offline`.

| Profile | Initial behavior | Remote boundary |
| --- | --- | --- |
| `economy` | Run locally when capable; use premium only for a known local capability, tool, or risk gap. | Allowed within the bounded task budget. |
| `balanced` | Run locally, verify, then escalate only on failure. | Allowed within the bounded task budget. |
| `quality` | Start with the configured premium adapter. | Allowed within the bounded task budget. |
| `privacy` | Run locally; failure-driven escalation is enabled only by an explicit task opt-in. | Hard-blocked without `allow_remote=true`. |
| `offline` | Run locally. | Always blocked. |

## Policy invariants

1. `offline` never invokes a remote provider.
2. `privacy` never invokes a remote provider unless the task explicitly opts in.
3. Premium calls cannot exceed the lower of profile and task budgets. The
   current ledger consumes the budget after bridge-level preflight and directly
   before the hardened runtime call. A two-phase reservation is still required
   to release that authority when the runtime rejects the executable, working
   directory, or process launch after this last bridge preflight.
4. Missing capabilities, required tools, or unsupported risk classes fail
   closed or escalate only when policy and budget both permit it.
5. A successful local verification stops the flow before any premium invocation.
6. A failed local verification sends only an `EscalationCapsule`, never the full
   local transcript. Premium receives a capsule-only temporary workspace unless
   a write task explicitly sets both remote and remote-workspace authority.
7. Route rationale is structured policy data, not chain-of-thought.
8. Command execution uses an argument vector and stdin, never a shell-built
   command string.
9. Missing launchers, OS launch errors, output overflows, invalid configuration,
   stale evidence, and malformed contracts fail closed.
10. Planning, audit, and run-log telemetry contain hashes, counts, status codes,
    and timings only. Successful execution returns the final answer separately
    in `result.content`; it is never written to those metadata stores.
11. Read-only work gets a read-only sandbox. Only an exact confirmation bound to
    the inspected receipt, command plan, external evidence, diff/capsule options,
    and initial command semantics can grant `write_local`; external, destructive, and
    privileged effects are not supported by this bridge.
12. Local Codex runs with isolated state, ignored user configuration/rules, a
    temporary home, a sanitized environment, no native web tool, and agent-tool
    network disabled.

Premium authentication is copied into that temporary home through an exclusive,
no-follow file descriptor, restricted on the descriptor itself, synced, read
back under a bounded attestation, and verified again immediately before the
premium budget reservation. This is same-user change detection, not hard
containment: a process with the same account authority can still race after the
last check, so stronger isolation requires a separate OS security boundary.

## What cost it can and cannot avoid

A preflight launcher can keep the complete initial task local and contact a paid
assistant only after verification fails. A component invoked from inside an
already-running premium Codex session cannot undo the paid input already sent to
that session. MCP delegation can reduce subsequent premium work, but it cannot
make that initial usage zero.

The first slice is a CLI launcher because it can enforce that boundary before a
premium session exists. An MCP adapter is a later integration surface; it should
use an established SDK and expose the same contracts rather than inventing a
private wire protocol.

## CLI usage

Planning is the default and does not launch Codex. The output contains a
metadata-only `RouteDecisionReceipt`, a redacted argv shape, and a
`confirmation_id`. The task remains stdin data and is represented only by a
digest and character count. Planning resolves the configured executable through
the sanitized `PATH` and binds its stat/content identity, but never invokes it;
provider and verifier version-probe configuration is intentionally unsupported.

```bash
.venv/bin/mymoe \
  --assistant-task-file tests/fixtures/assistant-bridge.task.json \
  --assistant-capsule-out work/runtime/escalation-capsule.json \
  --json
```

Execute only after inspecting the plan, using the exact `confirmation_id` it
returned and otherwise identical task, configuration, adapter, workspace state,
external evidence, diff policy, and capsule target:

```bash
.venv/bin/mymoe \
  --assistant-task-file tests/fixtures/assistant-bridge.task.json \
  --assistant-bridge-execute \
  --assistant-confirm-receipt 'confirm-<64 lowercase hex characters>' \
  --assistant-capsule-out work/runtime/escalation-capsule.json \
  --json
```

The app-level `permissions.assistant_bridge_execution_policy` supports
`disabled`, `local_only`, and `hybrid_receipt_confirmation`. The shipped app
config opts in explicitly; an omitted setting defaults to `disabled`. Both
denial policies are checked without invoking provider or verifier binaries.

`--assistant-capsule-out` writes bounded but still task-bearing data. Review it
before sharing. `--assistant-include-diff` is also explicit because even a
bounded, redacted diff can contain proprietary code. Diff collection reflects
the current worktree diff, including pre-existing edits, so use it from a known
baseline. Capsule persistence uses an unpredictable exclusive peer file,
rejects link/reparse targets and unsafe parents, synchronizes content, and then
performs an atomic same-directory replacement (write-through replacement on
Windows).

Provider commands, explicit models, capabilities, tools, risk ceilings,
timeouts, profiles, trusted verifier contracts, durable state, and capsule
limits live in
[`configs/assistant-bridge.json`](../configs/assistant-bridge.json). The local
adapter uses Codex OSS mode and defaults to Ollama; switch one run to LM Studio
with `--assistant-local-provider lmstudio`. The premium adapter runs normal
Codex. Both executable paths are replaceable in configuration for testing or a
managed installation.

The currently implemented launcher adapter is `codex_cli`. It deliberately
ignores ambient Codex configuration and does not load `codex_profile`, MCP, or
plugin state. Authority-affecting `extra_args` are rejected rather than merged.
A trusted executable plus launcher prefix can integrate an existing gateway;
that adapter remains part of the trusted computing base, while the task,
receipt, evidence, budget, and capsule contracts remain unchanged.

The shipped verifier rejects known failure language, runs configured mechanical
commands for code/tool/write tasks, and accepts strict external evidence.
External evidence must match a configured verifier id and spec digest and must
bind the current task and workspace fingerprints. The fixture files demonstrate
the schema; their example workspace fingerprint is intentionally not valid for
an arbitrary live checkout.

The Git hygiene verifier is a fixed builtin, not an ambient command. It creates
a synthetic repository whose `HEAD` is the attested source baseline, applies the
already-verified candidate changes only to that disposable worktree, marks new
files intent-to-add, and runs a trusted `git diff --check` with external diff and
text-conversion hooks disabled. A trusted synthetic `.git/info/attributes`
policy has higher precedence than candidate `.gitattributes`, so candidate
files cannot opt out of text classification or the fixed whitespace rules.
This metadata does not alter the attested source baseline, candidate manifest,
or candidate attributes, and candidate `.git` metadata is never copied. This is
a portable baseline integrity check, not proof of business correctness.

Arbitrary command verifiers are treated as untrusted candidate code. They run
only through an attested OS-owned hard-sandbox backend: fixed
`/usr/bin/sandbox-exec` with a deny-default Seatbelt profile on macOS, or fixed
`/usr/bin/bwrap` with namespaces, no network namespace, dropped capabilities,
read-only runtime/system binds, a tmpfs `/tmp`, and a writable disposable
workspace on Linux. If the backend is missing, altered, or unsupported, the
route is blocked before a provider, verifier, premium authorization, or budget
reservation can launch; there is no direct-host fallback. Windows currently has
no command-verifier backend and therefore fails closed for routes that select
one. An unsupported plan reports no active workspace, runtime, system-root, or
temporary-storage guarantees; policy intent is not presented as a capability.
The trusted Git builtin remains a separate fixed boundary.

Python project verifiers can declare safe relative `workspace_python_paths`.
The bridge translates only an explicit `{python} -m <module>` contract into a
fixed isolated `runpy` adapter: candidate paths precede editable-install paths,
while third-party dependencies remain in an attested read-only Python runtime.
Host `PYTHONPATH` is never accepted. Projects should replace or extend
`command_verifiers` with their native tests, linters, schema validators, or
evaluation gates. Natural-language output checks validate the response
contract; they do not prove that every factual statement is true.

## Isolation boundary

`offline` is an enforceable bridge policy, not a claim that an arbitrary binary
is trustworthy. The configured local executable and the installed Codex sandbox
remain part of the trusted computing base. The bridge removes proxy variables
and secret-bearing environment state, uses an empty temporary `CODEX_HOME`,
ignores ambient config/rules, omits `--search`, and requests network-disabled
tool sandboxing. Host firewall or container isolation is still appropriate when
the launcher itself is outside your trust boundary.

Verifier containment protects the host from the verifier process; it is not a
multi-tenant boundary against a concurrent actor with the same host account.
The selected sandbox executable, kernel containment implementation, trusted Git,
declared read-only runtime roots, and their installed dependencies remain in the
trusted computing base. Runtime roots and launcher identities are re-attested
when the confirmed plan is rebuilt, but same-user mutation races and kernel or
sandbox vulnerabilities are outside this bridge's guarantee. Linux operation
also depends on the host permitting bubblewrap namespaces; CI installs and runs
the real backend. macOS and Linux live probes verify that network access and
reads/writes outside declared roots are denied. Per-verifier temporary homes are
identity-checked, removed after every run, and cleanup failures fail closed.

## Acceptance evidence

- privacy and offline profiles cannot silently escape to remote execution;
- capability, tool, risk, and budget gaps produce deterministic routes;
- a passing verifier avoids premium execution;
- a nonempty failure message cannot satisfy completion;
- a failing verifier produces a bounded redacted capsule;
- task text that resembles shell syntax remains stdin data, not an executable
  argument;
- a missing Codex binary fails closed;
- task, receipt, prompt, evidence, runtime override, and confirmation hashes are
  collision-resistant bindings;
- Git index/status plus the complete in-scope file manifest participate in
  lineage, while exact staged/unstaged/untracked content is separate diff
  evidence;
- prior escalation evidence and final candidate evidence are reported as
  distinct phases, and only final evidence gates completion;
- final output is user-visible while audit/run logs remain metadata-only;
- full repository checks remain green.
