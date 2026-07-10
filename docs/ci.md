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
6. Run the offline CI profile from `configs/quality-gate-ci.json`, including
   train/holdout separation and provenance freshness. The live answer-quality
   benchmark is reported as non-release-eligible when local model endpoints are
   unavailable; only `configs/quality-gate.json` can declare release readiness.
   The live result path is deliberately not a generic `required_files` entry:
   the profile-aware benchmark check requires it for release and permits it to
   be absent only in offline CI.
7. Refresh the hardware profile artifact.
8. Run the packaging smoke test, which installs the project in a temporary virtual environment and verifies the `mymoe` and `mymoe-web` console scripts.

`make check` and `scripts/run_all_checks.sh` both delegate to the same Python runner.

## GitHub Actions

The active workflow is `.github/workflows/ci.yml`; `docs/github-actions-ci.yml`
is its installable reference copy. It runs Linux, macOS, and Windows with Python
3.10 and 3.12. It uses the official uv setup action, a read-only token,
dependency caching, concurrency cancellation, and `uv run --locked` so CI
rejects stale dependency state.

Keep the active copy synchronized after intentionally changing the template:

```bash
cp docs/github-actions-ci.yml .github/workflows/ci.yml
```

Action versions follow the current official uv GitHub Actions guide:
https://docs.astral.sh/uv/guides/integration/github/
