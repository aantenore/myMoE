from __future__ import annotations

import argparse

from local_moe.app_config import load_app_config
from local_moe.config import load_config
from local_moe.model_servers import (
    ModelServerManager,
    model_server_action_payload,
    wait_for_managed_processes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start configured local model servers.")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--config")
    parser.add_argument("--only-first", action="store_true", help="Start only the first configured model server.")
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    config = load_config(args.config or app_config.default_moe_config)
    manager = ModelServerManager.from_config(
        config,
        preferred_backends=app_config.runtime.preferred_backends,
        work_dir=app_config.runtime.work_dir,
    )
    action = manager.start(confirm=True, only_first=args.only_first)
    for result in model_server_action_payload(action)["results"]:
        print(
            " ".join(
                [
                    f"status={result['status']}",
                    f"expert={result['expert_id']}",
                    f"pid={result['pid']}",
                    f"log={result['log_path']}",
                    f"command={result['command_display']}",
                ]
            )
        )
    if not action.ok:
        raise SystemExit(action.message or action.status)
    if any(item.managed for item in action.results):
        wait_for_managed_processes(manager)


if __name__ == "__main__":
    main()
