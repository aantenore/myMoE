from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from local_moe.app_config import load_app_config
from local_moe.bootstrap import build_runtime_plan
from local_moe.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Start configured local model servers.")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--config")
    parser.add_argument("--only-first", action="store_true", help="Start only the first configured model server.")
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    config = load_config(args.config or app_config.default_moe_config)
    plan = build_runtime_plan(config, app_config.runtime.preferred_backends)
    commands = plan.model_commands[:1] if args.only_first else plan.model_commands
    if not commands:
        raise SystemExit(f"No start commands for backend {plan.backend}.")

    work_dir = Path(app_config.runtime.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, command in enumerate(commands, start=1):
        log_path = work_dir / f"model-{index}.log"
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        processes.append((process, log))
        print(f"started pid={process.pid} log={log_path} command={' '.join(command)}")

    try:
        while True:
            live = [process for process, _ in processes if process.poll() is None]
            if not live:
                raise SystemExit("All model servers exited.")
            time.sleep(2)
    except KeyboardInterrupt:
        print()
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
            log.close()


if __name__ == "__main__":
    main()
