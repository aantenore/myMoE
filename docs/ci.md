# CI

myMoE's canonical quality gate is `scripts/run_ci_checks.py`. It is implemented in Python and runs subprocesses with explicit argv lists instead of shell-specific command strings, so it can be used from macOS, Linux, Windows, GitHub Actions, or another CI runner.

Run the full gate locally:

```bash
python3 scripts/run_ci_checks.py
```

Inspect the plan without executing it:

```bash
python3 scripts/run_ci_checks.py --dry-run --json
```

The gate currently performs these steps:

1. Compile `src`, `tests`, `experiments`, and `scripts`.
2. Run the full `unittest` suite.
3. Run the base synthetic routing smoke eval.
4. Run the extended synthetic routing smoke eval.
5. Run the project quality gate from `configs/quality-gate.json`.
6. Refresh the hardware profile artifact.
7. Run the packaging smoke test, which installs the project in a temporary virtual environment and verifies the `mymoe` and `mymoe-web` console scripts.

`make check` and `scripts/run_all_checks.sh` both delegate to the same Python runner.

## GitHub Actions Template

The current GitHub token available in this workspace has `repo` scope but not `workflow` scope, so commits that create or modify `.github/workflows/*` are rejected by GitHub. Until a token with `workflow` scope is available, create this workflow manually in GitHub or push it from a credential that can manage Actions workflows.

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  checks:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Run quality gate
        run: python scripts/run_ci_checks.py
```
