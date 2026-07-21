# CI

myMoE's canonical quality gate is `scripts/run_ci_checks.py`. It is implemented in Python and runs subprocesses with explicit argv lists instead of shell-specific command strings, so it can be used from macOS, Linux, Windows, GitHub Actions, or another CI runner.

Run the full gate locally:

```bash
uv run --locked --extra assistant-bridge --python 3.12 python scripts/run_ci_checks.py
```

The runner fails fast below Python 3.10, matching `pyproject.toml`, instead of
running most checks and failing later during packaging.

The `uv run` command installs the project and its Assistant Bridge dependencies.
The runner preserves the inherited environment for child checks except for
`PYTHONPATH`, which the hardened Assistant Bridge runtime rejects as a code
injection path. Child checks therefore import the installed project instead of
injecting `src` into Python's module search path.

Inspect the plan without executing it:

```bash
uv run --locked --extra assistant-bridge --python 3.12 python scripts/run_ci_checks.py --dry-run --json
```

The gate currently performs these steps:

1. Compile `src`, `tests`, `experiments`, and `scripts`.
2. Validate the installed Assistant Bridge optional-dependency versions,
   package metadata, and process-tree runtime capability.
3. Run the full `unittest` suite.
4. Run the base synthetic routing smoke eval.
5. Run the extended synthetic routing smoke eval.
6. Regenerate the leakage-free 52-case live routing holdout report.
7. Run the 64-case deterministic Verified Outcome Routing shadow simulation.
   The fixture validates metrics, pairwise axis coverage, and privacy/cost
   accounting; it is explicitly synthetic and makes no empirical quality claim.
8. Run the deterministic Desktop Semantic Cell payload and tool-surface
   benchmark. Its canonical artifact excludes host wall-clock timing so repeated
   runs over the same fixture are byte-for-byte reproducible.
9. Run the deterministic Adaptive Cell Advisor contract benchmark and write
   `outputs/adaptive-cell-advisor-contract.json`. Its synthetic fixture checks
   profile-dependent selection, stale and resource-pressure abstention, and
   separate exact paraphrase lineage; it does not measure model quality or real
   performance.
10. Run the deterministic Adaptive Cell Execution Gate contract benchmark and
   write `outputs/cell-execution-gate-contract.json`. Its synthetic fixture
   checks unchanged admission, exact task and catalog drift, receipt expiry,
   and current resource pressure. It never runs a model or authorizes execution.
11. Run the offline CI profile from `configs/quality-gate-ci.json`, including
   train/holdout separation and provenance freshness. The live answer-quality
   benchmark is reported as non-release-eligible when local model endpoints are
   unavailable; only `configs/quality-gate.json` can declare release readiness.
   The live result path is deliberately not a generic `required_files` entry:
   the profile-aware benchmark check requires it for release and permits it to
   be absent only in offline CI.
12. Refresh the hardware profile artifact.
13. Run the packaging smoke test, which installs the project in a temporary
   virtual environment and verifies the `mymoe`, `mymoe-paired`, and
   `mymoe-web` console scripts, packaged browser, desktop, and Advisor templates,
   and the installed workspace initializers. The Advisor smoke runs from an
   unrelated working directory and verifies its zero-claim abstention path.

`make check` and `scripts/run_all_checks.sh` both delegate to the same Python runner.

## GitHub Actions

The active workflow is `.github/workflows/ci.yml`; `docs/github-actions-ci.yml`
is its installable reference copy. It runs Linux, macOS, and Windows with Python
3.10 and 3.12. It uses the official uv setup action, a read-only token,
dependency caching, concurrency cancellation, and `uv run --locked` so CI
rejects stale dependency state. Linux jobs install Bubblewrap before the gate so
the verifier isolation contract runs against its real OS-backed backend; macOS
and Windows use their platform-specific capability paths.

A separate Desktop Semantic Cell contract matrix installs the exact `desktop`
extra on Linux, macOS, and Windows under both Python 3.10 and 3.12. It executes
no GUI action; it verifies the pinned native provider version, complete
platform-specific catalog by both count and sorted-name digest (53 tools on
Linux, 49 on macOS, and 50 on Windows), and the exact native
`get_window_state` schema.

The matrix runs once for each pull request and again after changes reach
`main`. Feature-branch pushes do not start a duplicate matrix alongside the
pull-request event.

Keep the active copy synchronized after intentionally changing the template:

```bash
cp docs/github-actions-ci.yml .github/workflows/ci.yml
```

Action versions follow the [official uv GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/).
