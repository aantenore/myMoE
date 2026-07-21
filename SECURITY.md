# Security Policy

myMoE is an alpha local workstation runtime. Security boundaries are part of
the product contract, but this project is not a hardened multi-user service or
a substitute for operating-system, container, or virtual-machine isolation.

## Supported versions

| Version | Security fixes |
| --- | --- |
| Latest alpha release | Best effort |
| `main` | Active development; may contain unreleased changes |
| Older alpha releases | Upgrade before requesting a fix |

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/aantenore/myMoE/security/advisories/new)
for issues that could expose data, execute unintended commands, bypass a scope
or approval boundary, widen provider authority, forge evidence, or select the
wrong application, process, window, workspace, model, or execution target.

Include the smallest safe reproduction, affected version or commit, operating
system, configuration shape with secrets removed, expected boundary, and
observed result. Use disposable test data. Do not include credentials, private
prompts, model outputs, accessibility-tree contents, signing keys, tokens, or
other user data.

Please avoid opening a public issue until a sensitive report has been assessed.
Ordinary bugs with no confidentiality, integrity, or authority impact can use
the public issue tracker.

## Important trust boundaries

- Local model servers, coding-agent harnesses, MCP providers, browsers, desktop
  adapters, and their dependency trees are separate trusted components.
- Loopback HTTP and local stdio describe transport placement; they do not
  sandbox a process or prove that it has no network access.
- Accessibility text, browser pages, files, tool results, and model output are
  untrusted content even when they originate on the same machine.
- Metadata-only receipts can still be linkable or sensitive. Review every
  support bundle or evaluation artifact before sharing it.
- A passing canary qualifies only the exact version, configuration, hardware,
  target, and bounded behavior named by that canary.

The relevant per-feature threats and residual limits are documented in
[Execution Scope Guard](docs/execution-scopes.md),
[Agent Runtime](docs/agent-runtime.md),
[Browser Capability Cell](docs/browser-capability-cell.md), and
[Desktop Semantic Cell](docs/desktop-semantic-cell.md).
