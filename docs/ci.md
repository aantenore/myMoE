# CI

myMoE's canonical quality gate is `scripts/run_ci_checks.py`. It is implemented in Python and runs subprocesses with explicit argv lists instead of shell-specific command strings, so it can be used from macOS, Linux, Windows, GitHub Actions, or another CI runner.

Run the full gate locally:

```bash
uv run --locked --python 3.12 python scripts/run_ci_checks.py
```

The runner fails fast below Python 3.10, matching `pyproject.toml`, instead of
running most checks and failing later during packaging.

Inspect the plan without executing it:

```bash
python3 scripts/run_ci_checks.py --dry-run --json
```

The gate currently performs these steps:

1. Compile `src`, `tests`, `experiments`, and `scripts`.
2. Run the full `unittest` suite.
3. Run the base synthetic routing smoke eval.
4. Run the extended synthetic routing smoke eval.
5. Regenerate the leakage-free 52-case live routing holdout report.
6. Run the project quality gate from `configs/quality-gate.json`, including
   train/holdout separation and provenance freshness.
7. Refresh the hardware profile artifact.
8. Run the packaging smoke test, which installs the project in a temporary virtual environment and verifies the `mymoe` and `mymoe-web` console scripts.

`make check` and `scripts/run_all_checks.sh` both delegate to the same Python runner.

## GitHub Actions Template

The repository does not currently have an active GitHub Actions workflow.
`docs/github-actions-ci.yml` is the ready-to-install template for Linux, macOS,
and Windows with Python 3.10 and 3.12. It uses the official uv setup action, a
read-only token, dependency caching, concurrency cancellation, and
`uv run --locked` so CI rejects stale dependency state.

Install it when using a GitHub credential authorized to manage workflows:

```bash
mkdir -p .github/workflows
cp docs/github-actions-ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Add locked cross-platform CI"
git push origin main
```

Action versions follow the current official uv GitHub Actions guide:
https://docs.astral.sh/uv/guides/integration/github/
