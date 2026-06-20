from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .app_config import load_app_config
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .chat_runtime import generate_chat_turn
from .chat_store import ChatSession, FileChatStore, chat_session_payload, chat_summary_payload
from .compaction import LocalCompactionProvider
from .config import load_config
from .config_profiles import recommend_config_profile
from .context import ConversationTurn
from .context_policy import load_context_policy
from .doctor import build_doctor_report, render_doctor_report_markdown
from .environment import build_environment_report, render_environment_report_markdown
from .evaluator import evaluate_router, load_eval_cases
from .extensions import create_plugin_scaffold, load_extension_registry, registry_payload
from .memory import FileMemoryStore
from .model_servers import ModelServerManager, model_server_action_payload, wait_for_managed_processes
from .orchestrator import LocalMoE
from .performance_report import build_performance_report, render_performance_report_markdown
from .providers import ProviderError
from .profile_activation import activate_config_profile, activate_recommended_config_profile
from .run_log import RunLogStore, run_log_payload, run_log_prune_payload
from .runtime_optimizer import (
    build_runtime_optimizer_report,
    render_runtime_optimizer_markdown,
)
from .scheduler import cron_status, cron_summary_payload, run_due_jobs
from .setup_status import inspect_setup_status, setup_status_payload
from .setup_runner import run_runtime_setup, setup_run_payload
from .smoke import DEFAULT_SMOKE_PROMPT, build_generation_smoke_report
from .startup import run_startup_readiness
from .support_bundle import build_support_bundle, support_bundle_filename
from .tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MoE orchestrator")
    parser.add_argument("--config")
    parser.add_argument("--app-config", default="configs/app.json")
    parser.add_argument("--prompt")
    parser.add_argument("--eval")
    parser.add_argument("--interactive", action="store_true")
    chat_session_group = parser.add_mutually_exclusive_group()
    chat_session_group.add_argument("--chat-session")
    chat_session_group.add_argument("--new-chat", action="store_true")
    parser.add_argument("--chat-title")
    parser.add_argument("--chat-query")
    parser.add_argument("--list-chats", action="store_true")
    parser.add_argument("--chats-limit", type=int, default=20)
    parser.add_argument("--export-chat")
    parser.add_argument("--compact-chat")
    parser.add_argument("--compact-expert")
    parser.add_argument("--delete-chat")
    parser.add_argument("--rename-chat")
    parser.add_argument("--chat-confirm", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--doctor-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--doctor-out")
    parser.add_argument("--about", action="store_true")
    parser.add_argument("--about-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--about-out")
    parser.add_argument("--runs", action="store_true")
    parser.add_argument("--runs-limit", type=int, default=100)
    parser.add_argument("--runs-prune", action="store_true")
    parser.add_argument("--runs-keep", type=int, default=1000)
    parser.add_argument("--runs-confirm", action="store_true")
    parser.add_argument("--performance-report", action="store_true")
    parser.add_argument("--performance-report-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--performance-report-out")
    parser.add_argument("--runtime-optimizer", action="store_true")
    parser.add_argument("--runtime-optimizer-format", choices=["json", "markdown"], default="json")
    parser.add_argument("--runtime-optimizer-out")
    parser.add_argument("--runtime-optimizer-runs-limit", type=int, default=100)
    parser.add_argument("--support-bundle", action="store_true")
    parser.add_argument("--support-bundle-out")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--recommend-profile", action="store_true")
    parser.add_argument("--activate-profile")
    parser.add_argument("--activate-recommended-profile", action="store_true")
    parser.add_argument("--profile-confirm", action="store_true")
    parser.add_argument("--prepare-runtime", action="store_true")
    prepare_profile_group = parser.add_mutually_exclusive_group()
    prepare_profile_group.add_argument("--prepare-profile")
    prepare_profile_group.add_argument("--prepare-recommended-profile", action="store_true")
    parser.add_argument("--prepare-execute", action="store_true")
    parser.add_argument("--prepare-download-models", action="store_true")
    parser.add_argument("--prepare-confirm", action="store_true")
    parser.add_argument("--startup", action="store_true")
    parser.add_argument("--startup-prepare", action="store_true")
    parser.add_argument("--startup-download-models", action="store_true")
    parser.add_argument("--startup-start-models", action="store_true")
    parser.add_argument("--startup-confirm", action="store_true")
    parser.add_argument("--startup-only-first", action="store_true")
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

    if args.runs or args.runs_prune:
        store = RunLogStore(_run_log_path(app_config))
        if args.runs_prune:
            if not args.runs_confirm:
                print(
                    json.dumps(
                        {
                            "error": "confirmation_required",
                            "message": "Run log pruning requires --runs-confirm.",
                        },
                        indent=2,
                    )
                )
                raise SystemExit(2)
            print(json.dumps(run_log_prune_payload(store.prune(keep=args.runs_keep)), indent=2))
            return
        report = store.read_report(limit=args.runs_limit)
        print(
            json.dumps(
                run_log_payload(
                    report.records,
                    path=store.path,
                    valid_count=report.valid_count,
                    skipped_count=report.skipped_count,
                    total_lines=report.total_lines,
                ),
                indent=2,
            )
        )
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

    if args.runtime_optimizer or args.runtime_optimizer_out:
        payload = build_runtime_optimizer_report(
            config_path=config_path,
            app_config=app_config,
            app_config_path=args.app_config,
            run_limit=args.runtime_optimizer_runs_limit,
        )
        if args.runtime_optimizer_format == "markdown":
            rendered = render_runtime_optimizer_markdown(payload)
        else:
            rendered = json.dumps(payload, indent=2)
        if args.runtime_optimizer_out:
            out = Path(args.runtime_optimizer_out)
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

    if args.list_chats:
        chat_store = _chat_store(app_config)
        if args.chat_query:
            summaries = chat_store.search_sessions(args.chat_query, limit=args.chats_limit)
        else:
            summaries = chat_store.list_sessions(limit=args.chats_limit)
        print(
            json.dumps(
                {
                    "count": len(summaries),
                    "sessions": [chat_summary_payload(item) for item in summaries],
                },
                indent=2,
            )
        )
        return

    if args.export_chat:
        try:
            print(_chat_store(app_config).export_markdown(args.export_chat), end="")
        except KeyError as exc:
            print(json.dumps({"error": "not_found", "message": "Chat session not found."}, indent=2), file=sys.stderr)
            raise SystemExit(2) from exc
        return

    if args.rename_chat:
        if not args.chat_title:
            print(json.dumps({"error": "bad_request", "message": "--chat-title is required with --rename-chat."}, indent=2), file=sys.stderr)
            raise SystemExit(2)
        try:
            session = _chat_store(app_config).rename_session(args.rename_chat, args.chat_title)
        except KeyError as exc:
            print(json.dumps({"error": "not_found", "message": "Chat session not found."}, indent=2), file=sys.stderr)
            raise SystemExit(2) from exc
        print(json.dumps(chat_session_payload(session), indent=2))
        return

    if args.compact_chat:
        if not args.chat_confirm:
            print(
                json.dumps(
                    {
                        "error": "confirmation_required",
                        "message": "Chat compaction writes a durable summary and requires --chat-confirm.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2)
        chat_store = _chat_store(app_config)
        session = chat_store.get_session(args.compact_chat)
        if session is None:
            print(json.dumps({"error": "not_found", "message": "Chat session not found."}, indent=2), file=sys.stderr)
            raise SystemExit(2)
        turns = _conversation_turns(session)
        if not turns:
            print(
                json.dumps(
                    {"error": "bad_request", "message": "Chat session has no turns to compact."},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            compacted = LocalCompactionProvider(config, expert_id=args.compact_expert).compact(
                turns=turns,
                existing_summary=session.summary,
            )
            updated = chat_store.update_summary(
                session.id,
                compacted.summary,
                meta={
                    "expert_id": compacted.expert_id,
                    "model": compacted.model,
                    "correlation_id": compacted.correlation_id,
                    "input_messages": len(turns),
                    "previous_summary_present": bool(session.summary.strip()),
                },
            )
        except (KeyError, ProviderError, ValueError) as exc:
            print(json.dumps({"error": "compact_error", "message": str(exc)}, indent=2), file=sys.stderr)
            raise SystemExit(2) from exc
        print(
            json.dumps(
                {
                    "session": chat_session_payload(updated),
                    "summary": updated.summary,
                    "summary_updated_at": updated.summary_updated_at,
                    "compaction": {
                        "expert_id": compacted.expert_id,
                        "model": compacted.model,
                        "correlation_id": compacted.correlation_id,
                        "input_messages": len(turns),
                        "previous_summary_present": bool(session.summary.strip()),
                    },
                },
                indent=2,
            )
        )
        return

    if args.delete_chat:
        if not args.chat_confirm:
            print(
                json.dumps(
                    {
                        "error": "confirmation_required",
                        "message": "Chat deletion requires --chat-confirm.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2)
        deleted = _chat_store(app_config).delete_session(args.delete_chat)
        if not deleted:
            print(json.dumps({"error": "not_found", "message": "Chat session not found."}, indent=2), file=sys.stderr)
            raise SystemExit(2)
        print(json.dumps({"deleted": True, "session_id": args.delete_chat}, indent=2))
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

    if args.recommend_profile:
        print(
            json.dumps(
                recommend_config_profile(
                    active_config_path=config_path,
                    app_config=app_config,
                    app_config_path=args.app_config,
                ),
                indent=2,
            )
        )
        return

    if args.activate_profile or args.activate_recommended_profile:
        if args.activate_recommended_profile:
            payload = activate_recommended_config_profile(
                active_config_path=config_path,
                app_config=app_config,
                app_config_path=args.app_config,
                confirm=args.profile_confirm,
            )
        else:
            payload = activate_config_profile(
                args.activate_profile,
                active_config_path=config_path,
                app_config=app_config,
                app_config_path=args.app_config,
                confirm=args.profile_confirm,
            )
        print(json.dumps(payload, indent=2))
        if payload["status"] != "ok":
            raise SystemExit(2)
        return

    if args.prepare_profile or args.prepare_recommended_profile:
        if args.prepare_recommended_profile:
            recommended = recommend_config_profile(
                active_config_path=config_path,
                app_config=app_config,
                app_config_path=args.app_config,
            )["recommendation"]
            profile_path = str(recommended.get("profile_path") or "")
            if not profile_path:
                print(json.dumps({"error": "profile_error", "message": "No recommended runtime profile is available."}, indent=2))
                raise SystemExit(2)
        else:
            profile_path = args.prepare_profile
        result = run_runtime_setup(
            config_path=profile_path,
            app_config_path=args.app_config,
            execute=args.prepare_execute,
            download_models=args.prepare_download_models,
            confirm=args.prepare_confirm,
        )
        payload = setup_run_payload(result)
        payload["profile_path"] = profile_path
        print(json.dumps(payload, indent=2))
        if not result.ok and result.status not in {"planned", "needs_setup", "confirmation_required"}:
            raise SystemExit(2)
        return

    if args.startup:
        payload = run_startup_readiness(
            config_path=config_path,
            app_config_path=args.app_config,
            prepare=args.startup_prepare,
            download_models=args.startup_download_models,
            start_models=args.startup_start_models,
            confirm=args.startup_confirm,
            only_first=args.startup_only_first,
        )
        print(json.dumps(payload, indent=2))
        if payload["status"] in {"confirmation_required", "error", "manual_required", "needs_setup", "needs_attention"}:
            raise SystemExit(2)
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
                app_config_path=args.app_config,
                active_config_path=config_path,
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
        _interactive(
            moe,
            app_config=app_config,
            chat_session_id=args.chat_session,
            new_chat=args.new_chat,
            chat_title=args.chat_title,
            json_output=args.json_output,
        )
        return

    if not args.prompt:
        parser.error("--prompt or --interactive is required unless --eval is provided")

    if args.chat_session or args.new_chat or args.chat_title:
        try:
            payload = _generate_persistent_prompt(
                moe,
                app_config=app_config,
                prompt=args.prompt,
                chat_session_id=args.chat_session,
                new_chat=args.new_chat,
                chat_title=args.chat_title,
            )
        except (KeyError, ProviderError) as exc:
            print(json.dumps({"error": "chat_error", "message": str(exc)}, indent=2), file=sys.stderr)
            raise SystemExit(2) from exc
        _print_chat_payload(payload, json_output=args.json_output)
        return

    response = moe.generate(args.prompt)
    _print_response(response, json_output=args.json_output)


def _interactive(
    moe: LocalMoE,
    *,
    app_config: object,
    chat_session_id: str | None,
    new_chat: bool,
    chat_title: str | None,
    json_output: bool,
) -> None:
    chat_store = _chat_store(app_config)
    memory_store = _memory_store(app_config)
    run_log_store = RunLogStore(_run_log_path(app_config))
    context_policy = _context_policy(app_config)
    session_id = _resolve_interactive_session(
        chat_store,
        session_id=chat_session_id,
        new_chat=new_chat,
        title=chat_title,
    )
    print("myMoE interactive shell. Type /help for commands or /exit to quit.", file=sys.stderr)
    print(f"Active chat session: {session_id}", file=sys.stderr)
    while True:
        try:
            prompt = input("mymoe> ")
        except EOFError:
            print(file=sys.stderr)
            return

        if prompt.strip() in {"/exit", "/quit"}:
            return
        if prompt.strip() == "/help":
            print("Commands: /sessions, /session <id>, /new [title], /summary, /exit", file=sys.stderr)
            continue
        if prompt.strip() == "/sessions":
            _print_chat_summaries(chat_store)
            continue
        if prompt.strip().startswith("/session "):
            requested = prompt.strip().split(maxsplit=1)[1].strip()
            if chat_store.get_session(requested) is None:
                print(f"Chat session not found: {requested}", file=sys.stderr)
                continue
            session_id = requested
            print(f"Active chat session: {session_id}", file=sys.stderr)
            continue
        if prompt.strip().startswith("/new"):
            title = prompt.strip()[4:].strip() or chat_title or "CLI chat"
            session_id = chat_store.create_session(title=title).id
            print(f"Active chat session: {session_id}", file=sys.stderr)
            continue
        if prompt.strip() == "/summary":
            session = chat_store.get_session(session_id)
            if session is None:
                print(f"Chat session not found: {session_id}", file=sys.stderr)
                continue
            _print_session_summary(session)
            continue
        if not prompt.strip():
            continue

        try:
            payload = generate_chat_turn(
                moe=moe,
                chat_store=chat_store,
                memory_store=memory_store,
                run_log_store=run_log_store,
                context_policy=context_policy,
                prompt=prompt.strip(),
                session_id=session_id,
                mode="cli-interactive",
            )
            session_id = str(payload["session_id"])
        except (KeyError, ProviderError) as exc:
            print(f"Generation failed: {exc}", file=sys.stderr)
            continue
        _print_chat_payload(payload, json_output=json_output)


def _generate_persistent_prompt(
    moe: LocalMoE,
    *,
    app_config: object,
    prompt: str,
    chat_session_id: str | None,
    new_chat: bool,
    chat_title: str | None,
) -> dict[str, object]:
    chat_store = _chat_store(app_config)
    session_id = chat_session_id
    if new_chat or (chat_title and not chat_session_id):
        session_id = chat_store.create_session(title=chat_title or "CLI chat").id
    return generate_chat_turn(
        moe=moe,
        chat_store=chat_store,
        memory_store=_memory_store(app_config),
        run_log_store=RunLogStore(_run_log_path(app_config)),
        context_policy=_context_policy(app_config),
        prompt=prompt,
        session_id=session_id,
        mode="cli-prompt",
    )


def _resolve_interactive_session(
    chat_store: FileChatStore,
    *,
    session_id: str | None,
    new_chat: bool,
    title: str | None,
) -> str:
    if session_id:
        if chat_store.get_session(session_id) is None:
            raise SystemExit(f"Chat session not found: {session_id}")
        return session_id
    if new_chat or title:
        return chat_store.create_session(title=title or "CLI chat").id
    return chat_store.create_session(title="CLI chat").id


def _print_chat_summaries(chat_store: FileChatStore) -> None:
    summaries = chat_store.list_sessions(limit=20)
    if not summaries:
        print("No chat sessions.", file=sys.stderr)
        return
    for summary in summaries:
        print(
            f"{summary.id}  {summary.title}  messages={summary.message_count}  updated={summary.updated_at}",
            file=sys.stderr,
        )


def _print_session_summary(session: object) -> None:
    payload = chat_session_payload(session)
    print(
        json.dumps(
            {
                "id": payload["id"],
                "title": payload["title"],
                "message_count": payload["message_count"],
                "summary": payload["summary"],
            },
            indent=2,
        ),
        file=sys.stderr,
    )


def _conversation_turns(session: ChatSession) -> tuple[ConversationTurn, ...]:
    return tuple(
        ConversationTurn(role=message.role, content=message.content)
        for message in session.messages
        if message.role in {"user", "assistant"}
    )


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


def _print_chat_payload(payload: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return

    print(payload["content"])
    print()
    print(json.dumps(_chat_metadata(payload), indent=2))


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


def _chat_metadata(payload: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": payload.get("session_id", ""),
        "correlation_id": payload.get("correlation_id", ""),
        "selected": payload.get("route", {}).get("selected", []) if isinstance(payload.get("route"), dict) else [],
        "fallback_order": (
            payload.get("route", {}).get("fallback_order", []) if isinstance(payload.get("route"), dict) else []
        ),
        "errors": payload.get("errors", []),
        "disagreement": payload.get("disagreement"),
        "context": payload.get("context", {}),
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


def _run_log_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/runs.jsonl"


def _chat_store(app_config: object) -> FileChatStore:
    return FileChatStore(f"{app_config.runtime.work_dir.rstrip('/')}/chats.json")


def _memory_store(app_config: object) -> FileMemoryStore:
    return FileMemoryStore(f"{app_config.runtime.work_dir.rstrip('/')}/memory.jsonl")


def _context_policy(app_config: object) -> object:
    return load_context_policy(
        app_config.runtime.context_policy_config,
        app_config.runtime.context_policy_profile,
    )


if __name__ == "__main__":
    main()
