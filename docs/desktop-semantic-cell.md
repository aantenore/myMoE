# Desktop Semantic Cell

## What it does, in simple terms

The Desktop Semantic Cell lets a local model read the meaningful controls and
text in one desktop application window selected by the operator. For example,
it can inspect the labels and validation messages in one editor window and
explain what is visible without taking a screenshot or sending the content to a
cloud model.

This alpha exposes exactly one model-visible tool: `desktop.observe`. The tool
is read-only. It cannot click, type, press keys, use coordinates, read the
clipboard, enumerate applications or windows, run a shell, or inspect a second
target.

```text
local model
    |
    | desktop.observe
    v
myMoE capability firewall
    |
    | bounded get_window_state(include_screenshot=false)
    v
pinned MCP proxy -> owned bounded daemon -> one operator-bound process + window
```

After the model runtime and adapter are installed, this path can run without a
paid model or a cloud request. Host networking is still an operating-system
boundary: use an egress policy if an independently enforced air gap is required.

## Why this is not a new computer-control framework

[Cua Driver](https://github.com/trycua/cua) already implements native desktop
inspection across operating systems. Its
[MCP server exposes 49 tools](https://cua.ai/docs/reference/cua-driver/mcp-tools),
including application discovery, screenshots, coordinate input, process
control, clipboard access, and other authority that this alpha does not need.
Reimplementing those adapters would duplicate useful upstream work; exposing
the complete catalog to a model would grant far too much authority.

myMoE instead contributes a provider-neutral **capability firewall**:

- the operator selects one exact application process and window before the
  agent starts;
- the model sees one stable myMoE contract rather than an upstream tool catalog;
- myMoE starts one dedicated daemon on a private POSIX socket, admits only the
  pinned executable, and checks its PID, socket owner, bounded mode, and policy
  digests before accepting an observation;
- daemon and session policies allow only `get_window_state` with the exact PID,
  window, screenshot flag, node bound, and depth bound configured by the
  operator;
- the provider version, executable, launch profile, tool schema, target, and
  live state are checked before and after an observation is accepted;
- `include_screenshot=false` disables screenshot capture, and image-bearing
  provider responses are rejected;
- accessibility content is bounded, normalized, redacted, and labelled as
  untrusted data before it reaches the model;
- an application restart, empty or explicitly degraded tree, target mismatch,
  schema drift, oversized result, or provider failure closes the cell instead
  of silently widening access;
- closing the cell revokes live grants, stops the owned daemon, and removes its
  private policy and socket directory.

The first adapter is pinned to Cua Driver `0.10.0` over local MCP stdio. The
provider boundary remains replaceable so native
[AXUIElement](https://developer.apple.com/documentation/applicationservices/axuielement_h),
[Windows UI Automation](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-providersoverview),
or [AT-SPI](https://gnome.pages.gitlab.gnome.org/at-spi2-core/devel-docs/architecture.html)
adapters can be qualified later without changing the model-visible tool.

The distinction from adjacent open-source work is deliberate:

| Project | Primary purpose | Relationship to this cell |
| --- | --- | --- |
| [Cua Driver](https://github.com/trycua/cua) | Broad cross-platform computer-use provider | Reused behind a narrow adapter; its raw catalog is not exposed. |
| [UI-TARS Desktop](https://github.com/bytedance/UI-TARS-desktop) | Vision-based desktop agent using screenshots and coordinate actions | Useful reference for a future visual cell, not the read-only semantic authority surface. |
| [Agent-S](https://github.com/simular-ai/Agent-S) | General computer-use agent framework with screenshot-driven control | Complementary agent research; myMoE is adding a policy and attestation boundary, not another general agent loop. |
| [OSWorld](https://github.com/xlang-ai/OSWorld) | Reproducible benchmark for real computer tasks | Candidate future VM-level evaluation, not a host permission mechanism. |

## Current contract

| Capability | Alpha status | Boundary |
| --- | --- | --- |
| Observe one selected application window | Implemented on POSIX; live-qualified on macOS | Target identity is configured by the operator and echoed through a fixed approval binding; it is not selectable by the model. Linux requires a local bound-window canary; Windows currently receives provider-contract checks only and fails closed at runtime. |
| Semantic accessibility tree | Implemented | Output is normalized, bounded, redacted, and marked as untrusted content. |
| Screenshot capture | Disabled | The adapter is called with `include_screenshot=false`; any image result fails closed. |
| Application or window discovery | Not exposed | Upstream discovery tools are outside the allowlist. |
| Click, type, keys, hotkeys, and coordinates | Not exposed | This release has no model-visible action contract. |
| Clipboard, files, shell, and process control | Not exposed | These capabilities cannot be requested through `desktop.observe`. |
| Password and secure fields | Values withheld intentionally | Secure-field values and provider-only addressing data are removed; ordinary visible application text can still be sensitive. |
| Public network or remote desktop | Not exposed | The provider transport is local stdio and the selected model is local. |

An observation is evidence about the accessibility representation returned for
one target at one moment. It is not visual proof, an authorization to act, or a
claim that the application is safe.

## Qualify the adapter

The source configuration keeps desktop support opt-in:

- [`configs/app.desktop.example.json`](../configs/app.desktop.example.json)
  enables only the dedicated desktop provider configuration;
- [`configs/mcp.cua-desktop.example.json`](../configs/mcp.cua-desktop.example.json)
  pins Cua Driver `0.10.0`, the accepted upstream schema, one target binding,
  output limits, and the exact local launch profile. Its zero executable
  digests deliberately make it a non-runnable binding example.

Install the optional pinned provider and use Cua's human-facing
[`list_windows`](https://cua.ai/docs/reference/cua-driver/mcp-tools#list_windows)
tool to choose one PID and window id. Discovery is an operator setup step; it
is never placed in the model's tool registry. For the actual cell, myMoE starts
and owns a fresh embedded daemon rather than trusting the temporary discovery
daemon or a pre-existing machine-wide daemon.

### Operator-only target discovery

After `uv sync --locked --extra desktop`, run this POSIX shell block from the
source checkout. It resolves the exact bundled Cua Driver binary, creates a
private socket directory, starts a standalone operator-only daemon, waits until
it is ready, prints the on-screen windows locally, and then stops and removes
the daemon. The environment disables telemetry and update checks for every
discovery command. Do not connect this socket to myMoE, MCP, or a model, and do
not paste window titles into prompts or issue reports.

```bash
CUA_DRIVER="$(uv run --no-sync python -c \
  'from cua_driver import get_binary_path; print(get_binary_path())')"
DISCOVERY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mymoe-cua-discovery.XXXXXX")"
chmod 700 "$DISCOVERY_DIR"
DISCOVERY_SOCKET="$DISCOVERY_DIR/daemon.sock"
DISCOVERY_PID=""

cleanup_desktop_discovery() {
  env CUA_DRIVER_RS_TELEMETRY_ENABLED=false \
      CUA_DRIVER_RS_UPDATE_CHECK=false \
      "$CUA_DRIVER" stop --socket "$DISCOVERY_SOCKET" >/dev/null 2>&1 || true
  if [ -n "$DISCOVERY_PID" ]; then
    wait "$DISCOVERY_PID" 2>/dev/null || true
  fi
  rm -rf "$DISCOVERY_DIR"
}
trap cleanup_desktop_discovery EXIT INT TERM

env CUA_DRIVER_RS_TELEMETRY_ENABLED=false \
    CUA_DRIVER_RS_UPDATE_CHECK=false \
    "$CUA_DRIVER" serve \
      --socket "$DISCOVERY_SOCKET" \
      --permission-mode standard \
      --no-overlay \
      >"$DISCOVERY_DIR/daemon.log" 2>&1 &
DISCOVERY_PID=$!

until env CUA_DRIVER_RS_TELEMETRY_ENABLED=false \
          CUA_DRIVER_RS_UPDATE_CHECK=false \
          "$CUA_DRIVER" status --socket "$DISCOVERY_SOCKET" >/dev/null 2>&1; do
  kill -0 "$DISCOVERY_PID" 2>/dev/null || {
    echo "Cua discovery daemon exited before becoming ready." >&2
    exit 1
  }
  sleep 0.05
done

env CUA_DRIVER_RS_TELEMETRY_ENABLED=false \
    CUA_DRIVER_RS_UPDATE_CHECK=false \
    "$CUA_DRIVER" call list_windows \
      '{"on_screen_only":true}' \
      --socket "$DISCOVERY_SOCKET"

cleanup_desktop_discovery
trap - EXIT INT TERM
```

On macOS, run Cua's documented `permissions grant` flow first if the admitted
standalone or host identity does not already have the required TCC grants. The
block was validated with Cua Driver `0.10.0`; `list_windows` is a tool passed to
`cua-driver call`, not a top-level `cua-driver list_windows` command. Copy only
the chosen numeric `pid` and `window_id` into `desktop-init`.

Create the runnable binding and run its live canary:

```bash
uv sync --locked --extra desktop

uv run mymoe desktop-init \
  --out ./.mymoe-desktop \
  --target-id editor \
  --target-pid 12345 \
  --window-id 67890

uv run mymoe \
  --app-config .mymoe-desktop/app.desktop.json \
  --desktop-canary desktop-local \
  --desktop-canary-confirm
```

`desktop-init` inspects the current process start time, name, and executable;
hashes both target and native provider executable; persistently disables Cua
telemetry; erases its telemetry installation identifier; and writes files with
exclusive creation without overwriting an existing workspace. On POSIX, the
files request mode `0600`; on Windows, they inherit the destination directory's
ACL and this command does not claim to establish an owner-only DACL. It returns
the exact canary and agent argv arrays as JSON. Re-run it into a new directory
after an application or provider update.

The live canary proves provider and owned-daemon attestation, a real bounded
semantic read, and the model-visible surface. It refuses empty or explicitly
degraded accessibility output, including a non-empty provider response whose
nodes are all removed as invalid during normalization. The release test suite
separately verifies that screenshots, coordinates, actions, discovery, secure
values, PID reuse, target changes, schema drift, and oversized trees fail
closed. A pass qualifies only the bound read-only contract; it does not qualify
Cua Driver generally, the selected local model, a different host, or desktop
actions.

CI separately installs the exact optional provider wheel on Linux, macOS, and
Windows, executes no GUI action, and verifies the locked provider version,
49-tool catalog, and canonical `get_window_state` schema digest with telemetry
and update checks disabled. It reports the observed native executable digest;
it does not compare that platform-specific digest with a separately admitted
binary digest. Hosted CI cannot qualify an interactive desktop session; that
remains the purpose of the local bound-window canary.

The deterministic payload benchmark uses a 512-node synthetic accessibility
tree with long text, provider-only tokens, coordinates, password fields, and
secret sentinels. It emulates the provider's own `max_elements` cap before
myMoE applies its serialized-result budget. The current release artifact
reports 14 useful delivered nodes, zero forbidden keys or sentinel leaks, a
97.96% tool-surface reduction (49 to 1), and a 98.52% serialized-payload
reduction. This measures the capability firewall itself; it does not measure
end-to-end operating-system latency, accessibility completeness, or model task
success. See
[`outputs/desktop-semantic-benchmark.json`](../outputs/desktop-semantic-benchmark.json).

With the local model and exact target configuration ready, expose only the
desktop cell to the built-in agent:

```bash
uv run mymoe \
  --app-config .mymoe-desktop/app.desktop.json \
  --config configs/moe.live.ollama.example.json \
  --agent-prompt "Describe the visible controls and validation state in the selected window." \
  --agent-desktop-server desktop-local \
  --agent-interactive-approvals \
  --json
```

The tool schema contains a fixed target id and configuration digest so an exact
approval cannot be replayed for another window. The model must echo those
harness-supplied constants, but cannot choose or alter them. Do not place
credentials or sensitive target content in command-line prompts.

## Provider admission and offline operation

Cua Driver is a pre-1.0 dependency and must be treated as part of the trusted
computing base. The `desktop` extra pins the cross-platform wheel to `0.10.0`;
`desktop-init` resolves its bundled native executable, hashes it, requires that
exact version, and disables the upstream default telemetry before writing the
binding. Review Cua's
[official installation and daemon guide](https://cua.ai/docs/how-to-guides/driver/install)
because the daemon and operating-system permission identity remain
platform-specific. Use host-level egress enforcement when an independently
proven air gap matters.

Local stdio prevents a remote MCP hop, but it is not process isolation. The Cua
daemon runs with the current user's operating-system permissions. This cell
narrows daemon authority through an owned private socket plus bounded session
and argument policies, while executable, process, schema, and policy checks
detect drift. A compromised provider binary remains inside the trusted
computing base; use a disposable VM for untrusted applications or stronger
adversaries.

## Threat model and honest limits

| Risk | Control | Residual limit |
| --- | --- | --- |
| Model asks for broader desktop access | Only `desktop.observe` is registered. Its fixed target/configuration binding is approval-visible, while process, window, coordinates, command, and action values are not model-selectable. | A future action tool requires a separate contract and evaluation; this alpha cannot act. |
| Wrong app or window | The operator-bound target and live process identity are checked before and after every observation. Target drift closes the lifecycle. | Process and window identifiers are operating-system facts, not cryptographic identities. Rebind after an app restart. |
| Screenshot or visual data reaches the model | Capture is disabled upstream and image result blocks are rejected. | Accessibility text can itself contain private information; the operator must choose the target carefully. |
| Password or secret leakage | Secure-field and sensitive values are removed, output is bounded, and raw provider output is not passed through. | Applications can mislabel controls or render secrets as ordinary text. Redaction cannot infer every semantic secret. |
| Prompt injection in window content | Every node is tagged as untrusted data under a fixed tool and scope. | A local model can still follow malicious visible instructions in its prose answer. There are no actions to execute in this release. |
| Accessibility-tree explosion or incomplete output | Provider, node, depth, text, and serialized-result budgets structurally retain a useful prefix rather than replacing the whole result. Empty, fully invalid after normalization, and explicitly degraded trees fail. | Cua does not currently attest tree completeness, so `provider_completeness=unknown`; omitted controls may exist even below the configured cap. |
| Provider drift or compromise | Exact version, executable, owned daemon PID, private socket namespace, bounded policies, and upstream schema are checked; unexpected content fails closed. | The admitted provider binary and current OS account remain trusted computing-base components. |
| Offline claim is mistaken | Inference and MCP transport can be local; telemetry is disabled and no network capability is exposed to the model. | Only an external host policy can prove that every process has no egress. |

Platform behavior is not uniform:

- **macOS:** accessibility reads require the appropriate TCC Accessibility
  grant attached to the admitted host or standalone driver identity. Cua's
  startup gate may also require Screen Recording even though this cell sends
  `include_screenshot=false` and never requests a capture. See Cua's
  [permission limits](https://cua.ai/docs/reference/cua-driver/limits#permission-boundaries).
- **Windows:** only the provider wheel and schema contract are checked in CI;
  the runtime has not been live-qualified. This alpha's owned-daemon runtime
  currently fails closed because its private
  named-pipe ownership and teardown contract has not yet been qualified. UI
  Automation coverage and elevated-integrity boundaries remain future runtime
  qualification work.
- **Linux:** the POSIX runtime is implemented, but it has not been live-qualified
  by hosted CI and requires a local bound-window canary. AT-SPI coverage varies
  by toolkit. Cua documents separate X11 and
  compositor-specific Wayland support; standard Wayland deliberately limits
  arbitrary input. This alpha performs no input, but discovery and semantic
  tree completeness can still vary. See the current
  [platform matrix](https://cua.ai/docs/reference/cua-driver/platform-support).

Accessibility trees do not prove colors, layout, canvas content, image state,
animation, or pixel-level correctness. Custom-drawn applications can expose a
sparse or misleading tree. This is a semantic inspection cell, not a visual
testing system.

## Roadmap: semantic first, vision only when justified

The next useful step is not to expose Cua's remaining tools wholesale. A
future policy can estimate whether the semantic tree is complete enough for a
task and consider a local vision model only when semantics abstain. That route
must account for available RAM/VRAM, latency, and model residency, and visual or
coordinate control belongs in a separately attested disposable-VM cell.

This keeps the common path cheap and inspectable while reserving expensive
local VLM inference and stronger containment for surfaces that actually need
them. It is a roadmap, not a capability claim for this release.
