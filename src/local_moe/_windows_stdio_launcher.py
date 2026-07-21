from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Delay child creation until the parent has assigned this process to a Job."""

    if os.name != "nt" or len(sys.argv) < 2:
        return 125
    if os.read(0, 1) != b"\0":
        return 125
    child = subprocess.Popen(
        sys.argv[1:],
        stdin=0,
        stdout=1,
        stderr=2,
        close_fds=True,
    )
    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
