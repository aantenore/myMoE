"""Runtime probe for the verifier's candidate-first Python adapter."""

from pathlib import Path
import sys

import local_moe
import psutil
import tests


workspace = Path(sys.argv[1]).resolve(strict=True)
local_module = Path(local_moe.__file__).resolve(strict=True)
psutil_module = Path(psutil.__file__).resolve(strict=True)
test_roots = tuple(Path(item).resolve(strict=True) for item in tests.__path__)

if not local_module.is_relative_to(workspace / "src"):
    raise SystemExit(91)
if not any(item.is_relative_to(workspace / "tests") for item in test_roots):
    raise SystemExit(92)
if not psutil_module.is_relative_to(Path(sys.prefix).resolve(strict=True)):
    raise SystemExit(93)
