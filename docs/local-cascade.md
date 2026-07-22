# LocalCascade

LocalCascade lets the model role with the lowest configured `cost_rank` try a
bounded task before roles with higher configured ranks. Every candidate must
pass the same deterministic verifier and its reported token ceilings. If it
fails or abstains, LocalCascade sends only reason codes to the next tier, not
the failed answer. If no tier passes, it returns no content and records an
exhausted receipt. `cost_rank` is operator-assigned ordering; it is not measured
price, latency, energy use, quality, or model capability.

In practical terms, a laptop can map a lower-ranked role to bounded extraction,
classification, or summarization and advance only when the contract rejects its
result. Model lifecycle remains outside LocalCascade. The roles are names in
configuration, not hardcoded models. Replacing a runtime, quantization, or model
does not require changing the cascade contract.

## What is implemented

LocalCascade v1.1 has four strict boundaries:

1. `LocalCascadeTaskV1` describes one bounded classification, extraction, or
   summarization task and its output format.
2. `LocalCascadeConfigV1` orders replaceable roles by `cost_rank` and defines
   one deterministic verifier.
3. `LocalCascadeAttemptPort` is the injectable boundary that a runtime adapter
   implements for exactly one attempt carrying
   `requested_execution_scope=offline_local`.
4. `LocalCascadeReceiptV1` records task/config digests, tier lineage, verifier
   decisions, timings, and source-labelled token observations. Accepted content
   is returned separately and never embedded in the receipt.

Execution is sequential by increasing configured `cost_rank`. A passing attempt
stops the cascade. A failed output is hashed for lineage and discarded; the
next request contains only stable verifier reason codes. Reported actual or
estimated input/output counts above the selected tier ceiling also force
advancement. Unknown counts remain unknown instead of being invented.

Configuration accepts only `requested_execution_scope=offline_local`, rejects
permission for external network, tools, or workspace writes, and allows exactly
one active attempt. The receipt explicitly records
`execution_scope_attestation=adapter_declared_unverified`: these are requested
constraints, not proof that an adapter or downstream runtime obeyed them.

The core contains no installer or model lifecycle implementation and does not
choose an adapter. Operators explicitly bind each configured role to a runtime
outside this contract. The core cannot sandbox that adapter, enforce network
isolation, or attest the process and model that ultimately produced a result.

For JSON-object output, parsing rejects duplicate keys, non-finite constants,
numeric overflow, excessive nesting, non-object roots, unknown fields when
disabled, and declared field type/value mismatches.

### Run identity and digest privacy

Every invocation receives a random `run_id`, so repeated equivalent runs remain
distinct events. Separately, `evidence_sha256` excludes run identity and
volatile timing and is deterministic for equivalent semantic evidence. This
supports comparison without pretending two invocations are one run.

Task, request, output, configuration, and evidence digests are deterministic
and unsalted. They omit raw content, but identical values remain correlatable
and low-entropy values may be guessable by hashing candidates. These digests are
integrity and lineage aids, not encryption, anonymity, signatures, or trusted
provenance.

## Configuration-first roles

Start from
[`configs/local-cascade.example.json`](../configs/local-cascade.example.json).
It declares three replaceable roles:

| Role | Intended use | Residency assumption |
| --- | --- | --- |
| `utility` | Narrow, low-complexity bounded work | Lowest operator-assigned `cost_rank`; no measured-cost claim |
| `resident-generalist` | Work that needs a stronger general local model | Commonly kept available when resources allow |
| `cold-specialist` | The hardest supported bounded work | Loaded only by an external lifecycle owner |

`model_ref` values such as `local_cascade_utility` are logical references and
must exactly match an expert `id` in the separately supplied myMoE
configuration. They do not identify a provider, prescribe a model, authorize a
download, or prove that the referenced cell is available. The operator may map
the same stable expert ids to different local providers or models per machine
while preserving the task, verifier, and receipt contracts.

The optional plugin has no implicit configuration discovery. Set both paths
explicitly before launching it:

```bash
export MYMOE_LOCAL_CASCADE_CONFIG="$PWD/configs/local-cascade.example.json"
export MYMOE_LOCAL_CASCADE_MOE_CONFIG="$PWD/configs/your-local-moe.json"
```

The first file is this strict cascade contract. The second is the existing
myMoE configuration whose expert ids include `local_cascade_utility`,
`local_cascade_resident`, and `local_cascade_specialist`, with provider and model
values selected by the operator. Missing paths, an unknown expert id, or a
malformed configuration fail closed. The cascade contract requests offline
execution but cannot enforce runtime isolation. Inspection and planning contain
no install, download, start, stop, load, unload, or swap operation.

The plugin connects to an explicitly configured loopback endpoint. Loopback
establishes only that first network hop as local; it does not attest whether the
receiver proxies elsewhere, which downstream runtime runs, which model answers,
or whether later network activity occurs. LocalCascade therefore makes no
evidence-qualified model, route, or isolation claim.

The checked example is deliberately restrictive:

- `requested_execution_scope` is `offline_local`;
- network, tools, and writes are false;
- `parallel_attempts` is one;
- each result must contain `decision=` and `evidence=` and must not contain the
  forbidden marker;
- input and output ceilings are explicit per role.

These are contract ceilings, not hardware recommendations or model quality
claims.

## Reduction happens in layers

LocalCascade is one layer of a larger token-efficiency path. The useful order
is:

1. **Select context and reuse stable prefixes first.** Do not send irrelevant
   files, old turns, or repeated instructions merely to compress them later.
2. **Filter command output by command semantics.** An RTK-like filter can keep
   failures, summaries, and changed rows while retaining full raw evidence out
   of band when required.
3. **Use compact structured internal handoffs.** A Caveman-like internal format
   can reduce repeated prose between agents, while user-facing output stays
   clear and complete.
4. **Delegate bounded attempts through configured roles.** LocalCascade requests
   offline-local execution and applies deterministic acceptance, while the
   adapter remains an unverified boundary.
5. **Advance from explicit checks.** A higher-ranked role, or an external
   premium boundary owned by the caller, is considered only after abstention or
   a contract failure.

These layers operate on different surfaces and their percentages are
**non-additive**. Cutting shell bytes by one percentage and prompt tokens by
another does not justify summing those percentages. A local call also does not
automatically reduce tokens already consumed by a host assistant before the
delegation decision.

## Accounting without a misleading total

The receipt and benchmark keep the following categories separate:

| Observation | Meaning | May be combined with |
| --- | --- | --- |
| Actual local input/output tokens | Count reported by the local runtime | Actual counts from the same metering contract and direction |
| Estimated local input/output tokens | Count produced by an identified estimator | Other estimates only when the estimator contract matches |
| Unknown input/output attempts | The runtime supplied no usable count | Nothing; it remains an unknown count of attempts |
| Context-selection reduction | Exact retained and removed UTF-8 bytes in the fixture | Byte measurements from the same surface |
| Command-output reduction | Exact retained and removed UTF-8 bytes in the fixture | Byte measurements from the same surface |
| Premium counterfactual | Simulated calls under a declared policy | Only a paired live run can turn this into measured evidence |

Input and output directions remain visible. Actual, estimated, and unknown
observations are never collapsed into a headline token total. Context and tool
output use bytes in the contract fixture because treating bytes as model tokens
would be false precision.

## Deterministic contract benchmark

Run the checked fixture from the repository root:

```bash
uv run python experiments/benchmark_local_cascade.py --check
```

The benchmark:

- loads the public role configuration through the strict v1.1 parser;
- runs four frozen tasks through an injected attempt port;
- covers immediate acceptance, verifier-driven escalation, abstention, and
  exhaustion;
- records actual, estimated, and unknown local token observations separately;
- measures context selection and command-output filtering in UTF-8 bytes;
- reports local attempts and verifier pass/fail decisions;
- proves that reported input and output token counts above configured ceilings
  are rejected;
- checks strict JSON rejection for duplicate, non-finite, overflowing, and
  excessively nested fixtures;
- demonstrates unique per-run ids alongside stable semantic
  `evidence_sha256`;
- labels baseline and avoided premium calls as a simulation; and
- compares the regenerated JSON byte-for-byte with
  [`outputs/local-cascade-contract-benchmark.json`](../outputs/local-cascade-contract-benchmark.json).

The injected fixture performs zero downloads, model invocations, network calls,
tool calls, workspace writes, and premium calls. Therefore it is **not** a live
model quality, cost, latency, energy, isolation, or token-savings benchmark and
does not evidence-qualify a model or route.

## Extending it to real paired evidence

The experiment publishes a `paired_runner_extension` contract for later live
measurement. A compliant paired runner should inject two separately metered
boundaries without changing the evaluation:

- a real local attempt adapter for the configured role cells; and
- a separately metered comparison runner selected by the operator.

Both paths must receive the identical frozen task set, source context, tool
authority, verifier contract, and pass criteria. Record local and comparison
input/output tokens in separate fields, preserve unknown observations, and
compare outcomes only after the same verifier accepts them. Also record
end-to-end latency and failures; a path that fails the acceptance contract is
not equivalent work.

An overall reduction claim becomes defensible only after repeated paired runs
meet those controls. Results remain specific to the measured hardware, runtime,
models, quantization, harness, task set, and date.

## Current limits

- V1.1 supports classification, extraction, and summarization with text or a
  strict top-level JSON-object verifier; it is not a general coding-agent loop.
- Deterministic structural checks can reject malformed output but do not prove
  truth, deep semantic correctness, or safety.
- A logical role reference is not runtime or artifact attestation.
- `requested_execution_scope` and loopback first-hop transport do not prove
  downstream isolation; the receipt says the adapter declaration is unverified.
- Sequential execution avoids overlapping model attempts; an external lifecycle
  owner must still manage residency and resources.
- Local acceptance does not erase prompt or completion tokens already consumed
  by a surrounding host assistant.
- The core has no premium-provider dependency. Any premium escalation remains a
  separate, explicit caller policy and accounting boundary.
- Unsalted deterministic digests can correlate repeats and enable guessing of
  low-entropy content even though raw task and output bodies are omitted.

[Back to the documentation map](README.md)
