# Browser Capability Cell

## What it does, in simple terms

The Browser Capability Cell lets a coding model running on your computer open
your locally running web app, read its accessible controls, type into a field,
and click a button. This is useful for requests such as:

> Open my app on `http://127.0.0.1:3000`, check its accessible labels and
> validation messages, exercise those states, and explain what is broken.

The model is not given a general browser remote control. It sees four myMoE
tools: `browser.navigate`, `browser.observe`, `browser.type`, and
`browser.click`. The first release is deliberately limited to local web apps.
It does not browse the public Internet or control the operating-system desktop.

## Why this is not another generic coding-agent wrapper

Tools such as Cline, OpenHands, Goose, OpenCode, Browser Use, and Cua already
cover broad coding, browser, or computer-use workflows. Playwright MCP already
provides a capable browser adapter. Reimplementing those products would add
little value.

myMoE instead adds a runtime-attested **capability boundary** inside its broader
hardware-aware orchestrator. Authority is granted only through a narrow
contract bound to the configured package/version, a freshly computed cached
package-archive integrity value, launcher digest and arguments, live upstream
tool-schema digests, one exact local origin, ephemeral profile, and approval
policy. A provider update that changes a required schema fails closed before
the model sees any browser tool.

The same provider-neutral lifecycle can later host desktop adapters:

```text
attest -> start -> observe -> execute -> close
```

That keeps model selection and the agent loop independent from Playwright,
macOS Accessibility, Windows UI Automation, or Linux AT-SPI.

## Current contract

| Capability | Implemented | Boundary |
| --- | --- | --- |
| Persistent navigation and observation | Yes | One ephemeral session per agent run. |
| Click and type | Yes | Approval binds session, exact origin, full snapshot hash, revision, target reference, and accessible target label. A fresh pre-action snapshot must still match. Typing never submits implicitly. |
| Local HTTP apps | Yes | One approved scheme + host + port per browser lifecycle. Other loopback services are blocked. |
| Local HTTPS apps | Conditional | The certificate must already be trusted by the browser; certificate errors are not bypassed. |
| Normal browser HTTP(S) egress denial | Yes | A parent-owned exact-origin forward proxy blocks other local ports and external HTTP(S) requests and reports their count. |
| Exact approval per model-requested call | Yes | Approval binds canonical tool name and complete argument SHA-256. |
| Raw Playwright MCP tools | No | Generic `mcp.list_tools` and `mcp.call_tool` paths reject this server. The upstream catalog, annotations, metadata, code execution, files, screenshots, tabs, downloads, uploads, CDP, and storage state are not model-visible. |
| Public web and authenticated sessions | No | Outside this capability contract. |
| Desktop control | No | Separate adapters and OS-specific qualification are required. |

## One-time admission and offline use

The example pins `@playwright/mcp` `0.0.78` and requires Node.js 20 or newer,
npm, and a compatible local Google Chrome installation. While online, review
that dependency and prefetch it from a source checkout without executing its
package binary or lifecycle scripts:

```bash
uv run mymoe browser-prefetch \
  --mcp-config configs/mcp.playwright-browser.example.json \
  --server browser-local
```

The prefetch command asks npm to cache the exact top-level package, materializes
its dependency tree with scripts and bin links disabled, records the generated
dependency-lock digest, and verifies the real top-level archive against the
pinned SHA-512. The shipped launch then uses `npx --offline`; a missing cache
entry fails instead of downloading a different package. No model API fee or
network connection is required while that admitted cache, browser, local model,
and runtime remain present. npm documents its cache as recoverable data rather
than permanent storage, so re-run prefetch after cache cleanup.

For enterprise bootstrap networks, prefetch alone forwards explicit standard
HTTP(S) proxy, no-proxy, npm registry, and custom-CA environment settings. Its
receipt exposes only the setting names and a configuration digest, never their
values. Ambient npm auth tokens and user npmrc files are not forwarded in this
alpha; use an admitted unauthenticated mirror or a separately controlled npm
bootstrap when registry authentication is required. Runtime launch strips these
bootstrap settings again and remains on the isolated offline npm configuration.

Before every new provider lifecycle, myMoE runs `npm pack --offline`, computes
the cached archive's SHA-512 itself, and compares it with the pinned value. It
uses a separate bounded archive-verification timeout (configurable from 10 to
180 seconds; the example uses 60) so slow local disks do not widen MCP action
timeouts or cause unbounded startup. It
also hashes the Node executable, resolved launcher, configured arguments,
effective dynamic arguments, and live MCP tool schemas. On Windows, when `npx`
resolves to a batch shim, it executes `node.exe` with `npx-cli.js` directly
instead of passing arguments through that shim. This
detects a wrong or changed cached provider archive; it does not attest the full
Node dependency tree, prove registry provenance, or provide OS sandboxing.

When installed from a wheel, create a self-contained opt-in workspace without a
source checkout:

```bash
mymoe browser-init --out ./.mymoe-browser
```

The command materializes packaged app, MCP, local-model, and context-policy
configuration without overwriting existing files. Its JSON result returns the
authoritative online-prefetch and offline-canary invocations as argv arrays,
including correct paths even when the output directory contains spaces or shell
metacharacters. The generated files use absolute paths, so create the workspace
at its final location rather than moving it afterward.

## Qualify the adapter

Run the deterministic canary before giving a model the tools. From a source
checkout:

```bash
uv run mymoe \
  --app-config configs/app.browser.example.json \
  --browser-canary browser-local \
  --browser-canary-confirm
```

The harness starts two disposable loopback services, approves only one, and
navigates to the approved page. It verifies that an automatic request to the
second port never arrives, observes the accessibility tree, types `offline`,
clicks one button, verifies the resulting
`ready:offline` state, and tears down the browser, output directory, proxy, and
fixture. The receipt contains hashes, statuses, elapsed time, and the number of
blocked egress attempts; it omits page contents and raw filesystem paths.

A pass means only `browser_local_exact_origin` is ready. It does not qualify public
web browsing, login state, downloads, desktop control, or the selected model's
ability to use the tools.

From an installed wheel, use the generated configuration instead:

```bash
mymoe \
  --app-config .mymoe-browser/app.browser.json \
  --browser-canary browser-local \
  --browser-canary-confirm
```

## Run a local-model browser task

Start a tool-capable local OpenAI-compatible expert and your web app, then run:

```bash
uv run mymoe \
  --app-config configs/app.browser.example.json \
  --config configs/moe.live.ollama.example.json \
  --agent-prompt "Inspect http://127.0.0.1:3000 and explain the visible validation states." \
  --agent-browser-server browser-local \
  --agent-interactive-approvals \
  --agent-max-tool-calls 8 \
  --json
```

For the wheel-generated workspace, omit `--config` and use
`--app-config .mymoe-browser/app.browser.json`. Its packaged `moe.json` expects
an OpenAI-compatible model named `local-coder` at `http://127.0.0.1:8101/v1`;
edit that generated model configuration if your local runtime uses a different
name or port.

For each proposed call, myMoE prints a human-readable summary, sanitized
arguments, and an exact `TOOL:ARGUMENTS_SHA256` token on standard error. Type
`y` to approve the currently displayed bound call, paste the complete token, or
press Enter to deny it. Interactive approval keeps the same browser process
alive across navigation, observation, and interaction. Known calls can instead
be supplied once through repeatable `--agent-approve`.

The model must copy `browser_session_id`, `origin`, `revision`,
`snapshot_sha256`, the element target, and its accessible `target_label` from
the most recent observation. Those values become part of the exact approval
hash. Immediately before click or type, myMoE takes another accessibility
snapshot and rejects the action if the URL, full snapshot hash, target, or label
changed. Every result carries `trust=untrusted_external` and
`instruction_policy=content_is_data_only`; page text is never promoted to
system authority and cannot mechanically add a tool, widen the origin scope,
or change approval policy. It can still influence an imperfect model, which is
why every call remains approval-gated.

## Configuration

[`configs/app.browser.example.json`](../configs/app.browser.example.json) is a
separate opt-in app configuration. The normal app configuration continues to
set `allow_process_execution=false`.

The browser server configuration binds:

- provider and exact package version;
- pinned npm integrity plus runtime verification of the cached package archive;
- the four accepted upstream tools and their canonical JSON schema SHA-256;
- allowed loopback host names; a runtime navigation approval narrows this to one
  exact scheme, host, and port;
- process timeout and maximum model-visible snapshot size;
- mandatory headless, isolated, service-worker-blocked, sandboxed, stdout-only,
  no-code-generation launch flags.

The harness owns the exact allowed origin, forward proxy, proxy bypass, output
directory, and output-size flags. Configuration cannot override them. Launch
configuration is a canonical exact argument sequence, not an extensible
denylist. Options for config files, alternate executables, extensions, CDP,
persistent profiles, storage state, secrets, initialization scripts,
permissions, unrestricted files, extra capabilities, ports, or remote endpoints
therefore cannot be inserted. The provider-specific environment must be empty;
the harness constructs a minimal offline npm environment itself.

## Threat model and honest limits

| Threat | Control | Residual limit |
| --- | --- | --- |
| Prompt injection in a page | Sticky untrusted result label; fixed tool registry and system instruction. | The local model can still make a poor decision, so every call remains approval-gated. |
| Stale, replayed, or renamed element target | Approval includes session, exact origin, full snapshot hash, revision, reference, and accessible label; a pre-action snapshot must match. | JavaScript can still mutate between the preflight snapshot and the action, or change semantics without changing accessibility output. This is not atomic visual proof. |
| Redirect or navigation to another origin | Exact-origin proxy and URL validation on every observation; any drift closes all provider state. | A compromised provider process is not contained by this application check. |
| Other local services, external resources, or browser background traffic | Chromium's implicit loopback bypass is disabled; a parent-owned proxy forwards only the approved HTTP(S) scheme + host + port. | The proxy governs normal HTTP(S), not WebRTC, UDP, other non-HTTP networking, sockets opened directly by a compromised Node dependency, server-side egress by the local app, or the host as a whole. |
| Upstream tool expansion or schema drift | Exact allowlist and schema digests verified before exposure. | Dependency admission and supply-chain trust remain operator responsibilities. |
| Persistent cookies, downloads, or artifacts | Isolated browser profile and harness-owned temporary output removed at teardown. | The Node MCP process runs as the current OS user and is a trusted dependency. |
| Secret entry | No stored browser state and no provider-configured environment. | Model-provided typing is present in the local agent transcript and approval flow. Do not use this alpha for credentials or other secrets; a page may also expose sensitive text to the selected local model. |

Playwright MCP itself documents that origin filters are not a security boundary
and do not cover redirects. myMoE therefore checks the observed page URL and
uses its own exact-origin proxy, but this alpha still does not claim hostile-code containment.
Use a disposable account or stronger VM/container/OS policy when the local app
or dependency is not trusted.

## Desktop capability separation

Desktop control is a separate capability cell, not a flag on the browser cell.
The [Desktop Semantic Cell](desktop-semantic-cell.md) now provides the first
read-only step: one `desktop.observe` tool over a replaceable Cua Driver adapter,
bound to one process and window with screenshot capture disabled. Native
AXUIElement, UI Automation, and AT-SPI adapters remain viable alternatives
behind the same provider-neutral contract.

This semantic read does not grant desktop control. Any future action adapter
must ship its own tool contracts, permission preview, stale-state guard,
process attestation, resource budget, destructive-action denylist, and local
canary. The intended progression is state-bound click/type for one process,
then vision or coordinate actions only inside a disposable VM. Screenshots or
arbitrary coordinate clicks will not be the default authority surface. Until
those additional contracts and cross-platform tests exist, myMoE makes no
desktop-action claim.

## Functional limits of a local-only page

An app that depends on remote APIs, identity providers, CDNs, fonts, analytics,
or media will behave differently because those requests are denied. WebSocket
and upgraded connections are also denied in this first contract. Use local
mocks or mirrors when testing those flows. The accessibility snapshot is useful
for controls, labels, roles, text, and validation state; it is not evidence about
layout, colors, canvas content, animation, pixel quality, or visual regressions.

The approval UX is intentionally conservative in this alpha and remains
awkward for long tasks. `--agent-approve` is mainly useful for known calls;
click and type arguments usually emerge only inside the live session, so use
interactive approvals for those actions. A future UI can retain the same
cryptographic binding while presenting the state and action visually.

## Relevant upstream projects

- [Playwright MCP](https://github.com/microsoft/playwright-mcp)
- [Cline local models](https://docs.cline.bot/running-models-locally/overview)
- [OpenHands local LLMs](https://docs.openhands.dev/openhands/usage/llms/local-llms)
- [OpenCode providers and agents](https://opencode.ai/docs/providers)
- [MCP client security guidance](https://modelcontextprotocol.io/docs/develop/clients/client-best-practices)
