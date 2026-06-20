from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .app_config import load_app_config
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .config import load_config
from .doctor import build_doctor_report, render_doctor_report_markdown
from .environment import build_environment_report, render_environment_report_markdown
from .evaluator import evaluate_router, load_eval_cases
from .extensions import create_plugin_scaffold, load_extension_registry, registry_payload
from .model_servers import ModelServerManager, model_server_action_payload, wait_for_managed_processes
from .orchestrator import LocalMoE
from .performance_report import build_performance_report, render_performance_report_markdown
from .scheduler import cron_status, cron_summary_payload, run_due_jobs
from .setup_status import inspect_setup_status, setup_status_payload
from .setup_runner import run_runtime_setup, setup_run_payload
from .smoke import DEFAULT_SMOKE_PROMPT, build_generation_smoke_report
from .support_bundle import build_support_bundle, support_bundle_filename
from .tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MoE orchestrator")
    parser.add_argument("--config")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--prompt")
    parser.add_argument("--eval")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--doctor-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--doctor-out")
    parser.add_argument("--about", action="store_true")
    parser.add_argument("--about-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--about-out")
    parser.add_argument("--performance-report", action="store_true")
    parser.add_argument("--performance-report-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--performance-report-out")
    parser.add_argument("--support-bundle", action="store_true")
    parser.add_argument("--support-bundle-out")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--prepare-runtime", action="store_true")
    parser.add_argument("--prepare-execute", action="store_true")
    parser.add_argument("--prepare-download-models", action="store_true")
    parser.add_argument("--prepare-confirm", action="store_true")
    parser.add_argument("--models-status", action="store_true")
    parser.add_argument("--models-logs", action="store_true")
    parser.add_argument("--models-log-expert")
    parser.add_argument("--models-log-lines", type=int, default=120)
    parser.add_argument("--start-models", action="store_true")
    parser.add_argument("--stop-models", action="store_true")
    parser.add_argument("--models-confirm", action="store_true")
    parser.add_argument("--models-only-first", action="store_true")
    parser.add_argument("--list-extensions", action="store_true")
    parser.add_argument("--create-plugin")
    parser.add_argument("--run-tool")
    parser.add_argument("--tool-input", default="{}")
    parser.add_argument("--cron-status", action="store_true")
    parser.add_argument("--run-cron", action="store_true")
    parser.add_argument("--cron-dry-run", action="store_true")
    parser.add_argument("--cron-confirm-writes", action="store_true")
    parser.add_argument("--smoke-generate", action="store_true")
    parser.add_argument("--smoke-prompt", default=DEFAULT_SMOKE_PROMPT)
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    config_path = args.config or app_config.default_moe_config
    config = load_config(config_path)

    if args.doctor or args.doctor_out:
        report = build_doctor_report(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=args.app_config,
        )
        rendered = render_doctor_report_markdown(report) if args.doctor_format == "markdown" else json.dumps(report, indent=2)
        if args.doctor_out:
            out = Path(args.doctor_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")
            print(json.dumps({"written": str(out)}, indent=2))
        else:
            print(rendered)
        return

    if args.about or args.about_out:
        payload = build_environment_report(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=args.app_config,
        )
        rendered = render_environment_report_markdown(payload) if args.about_format == "markdown" else json.dumps(payload, indent=2)
        if args.about_out:
            out = Path(args.about_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")
            print(json.dumps({"written": str(out)}, indent=2))
        else:
            print(rendered)
        return

    if args.performance_report or args.performance_report_out:
        payload = build_performance_report()
        if args.performance_report_format == "markdown":
            rendered = render_performance_report_markdown(payload)
        else:
            rendered = json.dumps(payload, indent=2)
        if args.performance_report_out:
            out = Path(args.performance_report_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")
            print(json.dumps({"written": str(out)}, indent=2))
        else:
            print(rendered)
        return

    if args.support_bundle or args.support_bundle_out:
        payload = build_support_bundle(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=args.app_config,
        )
        if args.support_bundle_out:
            out = Path(args.support_bundle_out)
            if out.is_dir():
                out = out / support_bundle_filename()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({"written": str(out)}, indent=2))
        else:
            print(json.dumps(payload, indent=2))
        return

    if args.setup:
        print(
            json.dumps(
                setup_status_payload(
                    inspect_setup_status(
                        config_path,
                        config,
                        app_config,
                        app_config_path=args.app_config,
                    )
                ),
                indent=2,
            )
        )
        return

    if args.bootstrap:
        print(json.dumps(runtime_plan_payload(build_runtime_plan(config, app_config.runtime.preferred_backends)), indent=2))
        return

    if args.prepare_runtime:
        result = run_runtime_setup(
            config_path=config_path,
            app_config_path=args.app_config,
            execute=args.prepare_execute,
            download_models=args.prepare_download_models,
            confirm=args.prepare_confirm,
        )
        print(json.dumps(setup_run_payload(result), indent=2))
        if not result.ok and result.status not in {"planned", "needs_setup"}:
            raise SystemExit(2)
        return

    if args.models_status or args.models_logs or args.start_models or args.stop_models:
        manager = ModelServerManager.from_config(
            config,
            preferred_backends=app_config.runtime.preferred_backends,
            work_dir=app_config.runtime.work_dir,
        )
        if args.start_models:
            action = manager.start(confirm=args.models_confirm, only_first=args.models_only_first)
            print(json.dumps(model_server_action_payload(action), indent=2))
            if not action.ok:
                raise SystemExit(2)
            if any(item.managed for item in action.results):
                wait_for_managed_processes(manager)
            return
        if args.stop_models:
            action = manager.stop(confirm=args.models_confirm)
            print(json.dumps(model_server_action_payload(action), indent=2))
            if not action.ok:
                raise SystemExit(2)
            return
        if args.models_logs:
            print(
                json.dumps(
                    manager.logs(
                        expert_id=args.models_log_expert,
                        max_lines=args.models_log_lines,
                    ),
                    indent=2,
                )
            )
            return
        print(json.dumps(manager.status(), indent=2))
        return

    if args.list_extensions:
        print(json.dumps(registry_payload(_registry(app_config)), indent=2))
        return

    if args.cron_status:
        registry = _registry(app_config)
        print(
            json.dumps(
                cron_status(
                    registry.cron_jobs,
                    state_path=_cron_state_path(app_config),
                ),
                indent=2,
            )
        )
        return

    if args.run_cron:
        registry = _registry(app_config)
        summary = run_due_jobs(
            registry.cron_jobs,
            state_path=_cron_state_path(app_config),
            dry_run=args.cron_dry_run,
            confirm_writes=args.cron_confirm_writes,
            registry=registry,
        )
        print(json.dumps(cron_summary_payload(summary), indent=2))
        return

    if args.smoke_generate:
        payload = build_generation_smoke_report(config, prompt=args.smoke_prompt)
        print(json.dumps(payload, indent=2))
        if payload["status"] != "pass":
            raise SystemExit(2)
        return

    if args.create_plugin:
        path = create_plugin_scaffold(args.create_plugin, root=app_config.extensions.plugins_dir)
        print(json.dumps({"created": str(path)}, indent=2))
        return

    if args.run_tool:
        try:
            tool_input = json.loads(args.tool_input)
            result = LocalToolRunner(
                _registry(app_config),
                app_config=app_config,
                moe_config=config,
            ).run(args.run_tool, tool_input)
        except (json.JSONDecodeError, ToolExecutionError) as exc:
            print(json.dumps({"error": "tool_error", "message": str(exc)}, indent=2), file=sys.stderr)
            raise SystemExit(2) from exc
        print(json.dumps(tool_result_payload(result), indent=2))
        return

    if args.eval:
        cases = load_eval_cases(args.eval)
        print(json.dumps(evaluate_router(config, cases), indent=2))
        return

    moe = LocalMoE(config)

    if args.interactive:
        _interactive(moe, json_output=args.json_output)
        return

    if not args.prompt:
        parser.error("--prompt or --interactive is required unless --eval is provided")

    response = moe.generate(args.prompt)
    _print_response(response, json_output=args.json_output)


def _interactive(moe: LocalMoE, *, json_output: bool) -> None:
    print("myMoE interactive shell. Type /exit to quit.", file=sys.stderr)
    while True:
        try:
            prompt = input("mymoe> ")
        except EOFError:
            print(file=sys.stderr)
            return

        if prompt.strip() in {"/exit", "/quit"}:
            return
        if not prompt.strip():
            continue

        response = moe.generate(prompt)
        _print_response(response, json_output=json_output)


def _print_response(response: object, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_response_payload(response), indent=2))
        return

    print(response.content)
    print()
    print(
        json.dumps(
            _response_metadata(response),
            indent=2,
        )
    )


def _response_payload(response: object) -> dict[str, object]:
    payload = _response_metadata(response)
    payload["content"] = response.content
    payload["results"] = [item.__dict__ for item in response.results]
    return payload


def _response_metadata(response: object) -> dict[str, object]:
    return {
        "correlation_id": response.correlation_id,
        "selected": [item.__dict__ for item in response.route.selected],
        "fallback_order": list(response.route.fallback_order),
        "errors": list(response.errors),
        "disagreement": asdict(response.disagreement) if response.disagreement else None,
    }


def _registry(app_config: object) -> object:
    return load_extension_registry(
        plugins_dir=app_config.extensions.plugins_dir,
        skills_dir=app_config.extensions.skills_dir,
        tools_config=app_config.extensions.tools_config,
        mcp_config=app_config.extensions.mcp_config,
        cron_config=app_config.extensions.cron_config,
    )


def _cron_state_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/cron-state.json"


if __name__ == "__main__":
    main()
