# Bound Cell Attestor

**In plain terms: it fingerprints the files and configuration bindings you
declare for a local cell, without starting it.** A `verified` result means that
the configuration was validated and fingerprinted and that the observed model,
runtime, harness, and tool-contract identities matched separately reviewed
catalog anchors during one bounded local observation. It does not discover or
prove every file a future process will load, and it does not mean that the
producer is trusted or cryptographically authenticated.

`mymoe cell-bind inspect` turns one declared local model/runtime/harness cell
into two linked records:

- a content-addressed **binding manifest** describing the declared local
  bindings and their observed fingerprints;
- a short-lived **inspection receipt** saying either `verified` or `abstained`.

The inspection is offline and read-only. It does not start, stop, call, or
download a model; access the network; reserve RAM or VRAM; mutate runtime state;
or grant execution authority. Optional `--out` publication is the only write
and must target a new file outside the request, catalog, runtime configuration,
runtime, and model-artifact roots.

## Quick start

From a fresh clone, exercise the complete public contract without downloading
or starting a model:

```bash
uv run python experiments/benchmark_runtime_binding.py
```

The benchmark creates temporary sample artifacts and covers verified,
abstained, drift, bounded-read, and non-authorizing cases. To inspect your own
cell, create a `CellBindingInspectRequest` JSON document, then run:

```bash
mymoe cell-bind inspect --request ./cell-binding-request.json
```

[`configs/cell-binding-request.example.json`](../configs/cell-binding-request.example.json)
is a copyable request template, not a ready-made local cell. Its paths and
catalog identities are placeholders for files you own; the inspector never
downloads or invents them.

Use JSON for automation and optionally publish the same complete bundle to one
new file:

```bash
mymoe cell-bind inspect \
  --request ./cell-binding-request.json \
  --json \
  --out ./cell-binding-inspection.json
```

The file named by `--out` contains both `binding_manifest` and
`inspection_receipt`. Publication is atomic and no-clobber: the parent directory
must already exist, an existing regular file, link, or special file is rejected,
and POSIX output permissions are `0600`. The destination must be outside every
input and observed artifact root so publication cannot invalidate its own
snapshot. The CLI resolves request and output paths once, carries private
physical identities for the inspected runtime and model roots, and rechecks
those identities in the output writer. A failed post-publication parent check
removes only the just-staged inode through the already pinned directory handle.

## Request contract

The request uses schema version `1.0` and contract
`CellBindingInspectRequest`. It declares:

| Field | Meaning |
| --- | --- |
| `cell_id`, `expert_id` | The exact configured cell and expert to bind. |
| `adapter_id` | The expected adapter contract; v1 accepts `managed_direct_local_openai_v1`. |
| `catalog_path` | Adaptive Cell catalog, relative to the request directory. |
| `runtime_config_path` | myMoE runtime config, relative to the request directory. |
| `runtime_root` | Real directory containing the runtime executable, driver, and harness. |
| `model_artifact_root` | Real directory below which the configured local model must resolve. |
| `runtime_components` | Exactly one relative path for each role: `runtime_executable`, `driver`, and `harness`. |
| `observation_ttl_seconds` | Receipt lifetime from 1 to 120 seconds. |
| `hash_limits` | Explicit bounds: `max_files` (also the traversal-entry ceiling), `max_total_bytes`, `max_depth`, and `max_file_bytes`. |

This complete shape uses conservative explicit bounds. Replace the identifiers
and relative paths with the files in your own cell; the selected expert in
`configs/moe.local.json` would, for example, name `models/coder.gguf` as its
local model and `runtime/bin/llama-server` as its `runtime_executable`:

```json
{
  "schema_version": "1.0",
  "contract": "CellBindingInspectRequest",
  "cell_id": "coder-local",
  "expert_id": "coder",
  "adapter_id": "managed_direct_local_openai_v1",
  "catalog_path": "configs/adaptive-cells.json",
  "runtime_config_path": "configs/moe.local.json",
  "runtime_root": "runtime",
  "model_artifact_root": "models",
  "runtime_components": [
    { "role": "driver", "path": "lib/driver.py" },
    { "role": "harness", "path": "lib/harness.py" },
    { "role": "runtime_executable", "path": "bin/llama-server" }
  ],
  "observation_ttl_seconds": 60,
  "hash_limits": {
    "max_files": 4096,
    "max_total_bytes": 68719476736,
    "max_depth": 16,
    "max_file_bytes": 34359738368
  }
}
```

All four top-level paths are relative POSIX paths resolved from the directory
containing the request. Runtime component paths are relative to `runtime_root`.
The expert's configured model is also a relative path and must remain below
`model_artifact_root`. Absolute paths, `..`, links, special files, duplicate or
physically aliased runtime component locations (including case aliases and
hard links), ambiguous trees, and limits outside the accepted contract fail
closed. Both artifact roots must themselves be directories; for a GGUF model,
`model_artifact_root` names its containing directory rather than the file.

The explicit hash limits are part of the request lineage. They bound traversal
and hashing; they are not hints that the inspector may silently relax.
`max_files` conservatively counts every directory entry enumerated, including
subdirectories, so wide trees are bounded before their names are fully
materialized or walked. V1 also enforces hard ceilings of 100,000 entries, 2
TiB total bytes, depth 64, and 2 TiB per file. A request may choose lower bounds,
as the example does, but cannot raise those ceilings.

### V1 applicability boundary

V1 deliberately accepts only a cell that is already declared offline-capable,
`compute_only`, and without tool surfaces. Its runtime configuration must use a
non-widening `device_only` policy and exactly one selected OpenAI-compatible
expert with `device_only` / `direct_local` execution. The expert must declare a
local `llama_cpp`, `mlx_lm`, or `mlx_vlm` backend, `runtime_model_source: local`,
an exact request-relative `runtime_executable` equal to
`runtime_root/runtime_component.path`, and an
uncredentialed loopback base URL with an explicit port. For example,
`http://127.0.0.1:8101/v1` can bind the `--host 127.0.0.1 --port 8101` launch
plan; an executable or endpoint that differs from the generated plan is
rejected.

The bound launch plan uses the inspection request directory as its explicit
logical working directory. Therefore an executable declared as
`runtime/bin/llama-server` and a model declared as `models/coder.gguf` resolve
to the same request-root files the inspector hashes. This is a plan identity,
not an instruction to the generic model manager and not execution authority; a
future consumer must preserve that working-directory contract.

These restrictions are part of the adapter contract, not a claim that every
local inference runtime has this shape. A new adapter should define a separate
versioned contract rather than weakening these checks.

### Platform boundary for model artifacts

V1 deliberately has a narrower artifact-shape boundary on Windows. Single
regular files can be opened and hashed there through no-follow Windows handles,
so a local `llama_cpp` expert backed by one `.gguf` file is supported. Recursive
model-directory traversal is currently enabled only on POSIX hosts that provide
the required descriptor-relative, no-follow primitives. Because `mlx_lm` and
`mlx_vlm` models are directories, their binding inspection is therefore not
available on Windows in v1 and fails closed; this is an inspector limitation,
not a claim that MLX inference itself runs on Windows.

| Host and model shape | V1 inspection |
| --- | --- |
| Windows + one local GGUF file | Supported |
| Windows + MLX model directory | Rejected fail-closed |
| Supported POSIX host + one local GGUF file | Supported |
| Supported POSIX host + MLX model directory | Supported |

The inspector never replaces the unavailable Windows directory guarantee with
an ordinary recursive path walk. Adding Windows directory support requires a
separate no-follow, handle-relative implementation with equivalent mutation and
reparse-point checks.

## Result and exit contract

| Exit | Meaning |
| ---: | --- |
| `0` | `inspection_receipt.status` is `verified`. |
| `1` | The request was valid, but the inspector safely `abstained`. |
| `2` | Invocation, request, contract, input, or publication was invalid, or the operation failed safely. |

Human-readable text is the default. `--json` emits the complete stable bundle
to standard output. The JSON shape is:

```json
{
  "schema_version": "1.0",
  "contract": "BoundCellInspector",
  "request_sha256": "...",
  "binding_manifest": { "digest": "..." },
  "inspection_receipt": { "status": "verified", "digest": "..." },
  "digest": "..."
}
```

The actual manifest also fingerprints the cell declaration, runtime
configuration, expert configuration, adapter contract, platform, launch plan,
endpoint authority, model reference, selected producer modules, and every
declared component that was observed.
The model artifact tree must be completely hashed inside the declared limits
for any receipt to be emitted. An incomplete observation fails closed with exit
`2`; it does not become a partial receipt. Once observation is complete, a
missing or mismatched catalog identity produces `abstained` with stable reason
codes.

The manifest exposes four observed identity digests: `model_identity_sha256`,
`runtime_identity_sha256`, `harness_identity_sha256`, and
`tool_contract_identity_sha256`. This makes first use honest and practical:
an unqualified catalog produces `abstained` plus the observed identities; after
an operator independently reviews those exact files, the accepted digests can
be recorded as the declaration's `expected_*` fields and the inspection can be
repeated. The command never edits or promotes the catalog itself.

Model evidence does not include the operator's relative filenames in plaintext.
Each public component uses a stable SHA-256 identity of its tree-relative path
as its pseudonymous name, together with byte count and content digest. The
single-file shape carries only the contract marker `.gguf`;
`model_artifact_kind` distinguishes it from a directory, and
`model_artifact_manifest_sha256` is recomputed from those public components.
Renaming an MLX config or shard therefore changes the model identity even when
its bytes and suffix stay the same. These deterministic unsalted path digests
are pseudonyms, not secrecy: common filenames can still be guessed offline.

The four identity families are `model`, `runtime`, `harness`, and
`tool_contract`. Each may produce `<family>_identity_unknown` when the catalog
does not declare an expected digest, or `<family>_identity_mismatch` when the
declared digest differs from the observed bytes. `unknown` is not treated as a
match, even when every local file was read successfully.

The receipt's `observed_component_count` counts declared artifacts that were
successfully read and hashed and therefore equals `component_count` in v1. It is
not a count of catalog identity matches; `status` and `reason_codes` carry that
decision.

Errors are JSON on standard error, including when `--json` was not requested:

```json
{
  "error": "cell_binding_error",
  "code": "invocation_invalid",
  "message": "Invalid cell binding inspection invocation."
}
```

Errors never echo the request document, a supplied path, or an underlying
exception. This makes the failure surface safe for logs while the exit code and
stable code remain machine-readable.

## What `verified` guarantees

Within the captured snapshot, `verified` means:

1. the request, catalog, runtime config, selected cell, and expert formed one
   valid declared contract, and the configuration was validated and
   fingerprinted;
2. every declared runtime component was a bounded regular non-link file and its
   observed bytes were hashed;
3. the configured local model artifact tree was completely observed within the
   declared roots and limits;
4. the observed model, runtime, harness, and tool-contract identities matched
   the separately reviewed expected digests in the selected catalog;
5. the manifest and receipt self-digests matched their canonical content; and
6. no model, process lifecycle, network, tool, or execution-authority operation
   was used.

The receipt is intentionally short-lived. A consumer must check its expiry and
exact manifest binding rather than treating an old JSON document as current
state.

## Limits and non-guarantees

- A self-digest checks canonical internal consistency and can reveal accidental
  corruption when it is not recomputed. An adversary can rewrite the content
  and recompute the embedded digest. Detecting deliberate rewriting or later
  drift requires a separately trusted anchor; a self-digest is **not a
  signature or authenticated provenance** and does not establish a trusted
  producer.
- Model path identities are deterministic unsalted hashes. Raw relative paths
  are absent, but common filenames remain dictionary-guessable; the receipt
  must not be treated as a confidential filename-hiding format.
- The observation is a snapshot and can become stale immediately after the
  files are hashed. The receipt TTL bounds a claim; it does not lock files or
  reserve resources.
- Only declared static files and configuration-derived bindings are covered.
  The inspector does not discover or prove that the declared driver or harness
  is what a future process will actually load.
  Dynamic libraries, runtime-loaded plugins, environment variables, mutable
  caches, device drivers, kernel state, and remote dependencies are not proven.
- On Windows, v1 hashes only a single regular model artifact file. Recursive
  model directories, including MLX model layouts, fail closed as described in
  the platform matrix above.
- File presence and digest equality do not prove that a process is resident,
  healthy, compatible with the current driver stack, or able to satisfy a task.
- `verified` does not authorize execution and cannot be passed as permission to
  a launcher. There is deliberately no `apply`, `start`, or `execute` command.
- Optional `--out` publication must remain outside every request, catalog,
  configuration, runtime, and model-artifact input or root; the inspection
  itself remains read-only. Root membership is checked against the physical
  identities captured during inspection, not only against their later
  pathnames.
- The inspector does not provide a general software supply-chain attestation,
  reproducible-build claim, sandbox, or malware analysis.

## Threat model

The v1 boundary is designed to fail closed for malformed JSON, path escape,
links and special files, file replacement during observation, unbounded trees,
incomplete component identity, model artifacts outside the declared root,
digest mismatch, output destinations inside an input or observed artifact root,
unsafe output aliases, root-rename races between inspection and publication,
and output replacement races.

It does not defend against a privileged attacker who can alter the inspector,
the Python runtime, or the operating system while the command runs. It also
does not make a locally produced receipt trustworthy to another machine or
organization. For that, a separate signer, protected key, verifier identity,
and trust policy would be required.

## Relationship to OCI and SLSA

This feature borrows two useful ideas but claims compatibility with neither:

| Concept | What it addresses | How this attestor differs |
| --- | --- | --- |
| [OCI Image Manifest](https://specs.opencontainers.org/image-spec/manifest/) | Content-addressed configuration and layers for a distributable container image. | The attestor fingerprints operator-declared runtime, driver, harness, model, and configuration bindings; it neither builds nor distributes an OCI image. |
| [SLSA provenance](https://slsa.dev/spec/v1.2/provenance) and [artifact verification](https://slsa.dev/spec/v1.2/verifying-artifacts) | Verifiable information about where and how artifacts were produced, checked against expectations and roots of trust. | The attestor fingerprints declared current local bytes and records self-consistency digests. It has no signed provenance, builder identity, SLSA level, or root-of-trust verification. |

An OCI digest can identify packaged image content, and SLSA provenance can
describe how an artifact was produced. This binding manifest addresses a
different local need: fingerprinting the loose files, model tree, harness, and
configuration explicitly declared for one cell so a separately anchored
consumer can detect drift. It does not prove what a future process will load.
Authenticated provenance could be layered around it later without changing its
non-authorizing role.

## Related boundaries

- [Adaptive Cell Advisor](adaptive-cell-advisor.md) chooses among configured,
  evidenced cells without running them.
- [Adaptive Cell Execution Gate](cell-execution-gate.md) rechecks an Advisor
  receipt and live resource admission without executing the cell.
- [Architecture](architecture.md) explains how the control plane and local
  inference runtimes fit together.

[Back to the documentation map](README.md)
