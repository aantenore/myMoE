# Execution Scope Guard

The Execution Scope Guard is the fail-closed boundary between routing a request
and invoking an expert. It applies to web chat, persistent and stateless CLI
chat, streaming, comparison modes, fallbacks, and the separate agent runtime.
It answers a different question from semantic routing: **where is this request
allowed to execute?**

The default policy is `device_only`. A route cannot silently widen its scope,
and a request with no eligible expert fails with the stable reason code
`scope_blocked` before an ineligible provider is invoked.

## Scope Vocabulary

Scopes are ordered by increasing egress and trust surface:

| Scope | Meaning | Default status |
| --- | --- | --- |
| `device_only` | Inference stays within the current device boundary. | Allowed by the shipped profiles. |
| `private_mesh` | Inference may use explicitly trusted peers in a private mesh. | Blocked without an external, fresh attestor. |
| `public_mesh` | Inference may use peers outside a private trust domain. | Blocked by the shipped profiles. |
| `paid_remote` | A paid remote model is involved. | Reserved for the approval- and budget-gated Assistant Bridge, not normal MoE chat. |

The scope name is a policy label, not evidence by itself. An expert declaration
states the intended boundary; an attestor must establish the effective boundary
for transports that can route beyond the device.

## Transport Vocabulary

| Transport | Meaning | Evidence rule |
| --- | --- | --- |
| `direct_local` | Direct call to a model server on the same device. | Requires a loopback HTTP endpoint. |
| `mesh_llm` | Call through a Mesh-LLM control or data plane. | Always requires an external attestor, even when the first hop is loopback. |
| `gateway` | Call through another gateway or provider boundary. | Always requires an external attestor. |

A loopback URL proves only that the first network hop terminates on the current
device. It does **not** prove where inference runs. A proxy, gateway, or mesh
bound to `127.0.0.1` can still forward the request elsewhere. Such an endpoint
must declare its real transport; misdeclaring it as `direct_local` violates the
operator trust assumption and invalidates the `device_only` claim.

## Configuration Contract

The policy belongs to the MoE profile, so models and their execution boundary
can be replaced together without hardcoded provider logic:

```json
{
  "execution": {
    "max_scope": "device_only",
    "allowed_scopes": ["device_only"],
    "allow_scope_widening": false
  },
  "experts": [
    {
      "id": "general",
      "provider": "openai_compatible",
      "base_url": "http://127.0.0.1:8101/v1",
      "model": "replaceable-local-model",
      "role": "general",
      "execution": {
        "scope": "device_only",
        "transport": "direct_local"
      }
    }
  ]
}
```

`max_scope` is the policy ceiling. `allowed_scopes` is the explicit allowlist
within that ceiling. `allow_scope_widening=false` prevents a fallback from
moving a request from a narrower selected scope to a broader one. Enabling
widening does not bypass the allowlist or attestation requirements.

Declarations are per expert because two OpenAI-compatible endpoints can have
different placement and trust properties. Provider protocol, endpoint address,
transport, and execution scope are independent configuration dimensions.

## Enforcement Lifecycle

1. Configuration parsing rejects unknown scope and transport values.
2. The guard resolves fresh evidence for every candidate before routing.
3. The router scores only eligible experts and filters ineligible fallbacks.
4. Unless explicitly enabled, fallback scope cannot exceed the scope selected
   for the original route.
5. The orchestrator rechecks the chosen expert immediately before every normal,
   streaming, parallel, or fallback provider invocation.
6. Missing, stale, contradictory, or unavailable evidence produces
   `scope_blocked`; the runtime does not silently try a broader target.

The second check narrows the time-of-check/time-of-use window. It is not a
cryptographic proof of model placement; the strength of the result remains
bounded by the configured transport and attestor.

## Mesh-LLM Boundary

[`configs/moe.mesh-private.example.json`](../configs/moe.mesh-private.example.json)
shows the intended configuration seam, but it is deliberately not runnable
with the built-in attestor.

[Mesh-LLM v0.73.1](https://github.com/Mesh-LLM/mesh-llm/releases/tag/v0.73.1)
exposes `/api/status`, but that response is an operational snapshot rather than
a request-bound execution receipt. Its
[status payload](https://github.com/Mesh-LLM/mesh-llm/blob/v0.73.1/crates/mesh-llm-host-runtime/src/api/status.rs)
does not carry a `schema_version`, and the snapshot does not prove which peer
will serve the next inference request. It therefore cannot, by itself,
authorize a `private_mesh` claim.

The Mesh adapter remains disabled and fail-closed until a pinned adapter can
validate all of the following:

- a versioned, compatible evidence schema;
- fresh mesh and node identity plus ownership/trust state;
- the effective peer admission and egress policy;
- model and release identity relevant to the selected expert;
- evidence bound to the exact request, or an equivalent placement receipt;
- a short validity window and a fresh check immediately before invocation.

Until that contract exists, changing only `max_scope` or declaring
`private_mesh` is not sufficient. The default attestor accepts only
`direct_local` experts on loopback endpoints and blocks `mesh_llm` and
`gateway` transports.

## Premium Boundary

Normal MoE generation does not gain premium access by declaring
`paid_remote`. Premium Codex execution remains exclusively under the Hybrid
Assistant Bridge, where capability policy, explicit privacy consent, bound
verification, redacted capsules, and durable budgets are evaluated together.
The execution-scope vocabulary does not replace those controls.

## Trust and Logging

The built-in local attestor trusts the operator to describe the real transport
and treats loopback as a direct-local network boundary, not as compute
attestation. External attestors are provider-neutral extension points and must
fail closed when evidence is unavailable or inconsistent.

Scope decisions should be observable through reason codes, expert identifiers,
scope/transport labels, counts, timestamps, and digests. They must not add
prompt, answer, credential, or raw attestation bodies to metadata-only run or
audit logs.
