from __future__ import annotations

import json

from local_moe.hardware import write_hardware_report


if __name__ == "__main__":
    profile = write_hardware_report("outputs/hardware-profile.json")
    print(json.dumps(profile.__dict__, indent=2))
