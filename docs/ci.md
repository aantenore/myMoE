# CI

GitHub refused workflow pushes from the current OAuth token because it does not include the `workflow` scope. Until that scope is available, use this workflow template manually from GitHub or push it with a token that can manage Actions workflows.

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  checks:
    runs-on: macos-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Run quality gate
        run: ./scripts/run_all_checks.sh
```
