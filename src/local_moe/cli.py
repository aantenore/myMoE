from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

from .audit import AuditLogStore
from .agent_loop import AgentLoopBudget, AgentRunResult, build_local_agent_loop
from .agent_provider import validate_local_agent_endpoints
from .agent_tools import AgentPermissionPolicy, ApprovalDecision, ApprovalRequest
from .app_config import load_app_config
from .assistant_lifecycle_cli import (
    AssistantBridgeCliError,
    canonical_json as assistant_bridge_canonical_json,
    lifecycle_mode as assistant_bridge_lifecycle_mode,
    load_cli_assistant_task,
    run_lifecycle_cli,
)
from .bootstrap import build_runtime_plan, runtime_plan_payload
from .browser_capability import (
    BrowserToolRunner,
    CompositeToolRunner,
    prefetch_browser_provider,
)
from .browser_setup import materialize_browser_workspace
from .desktop_capability import DesktopToolRunner
from .desktop_setup import materialize_desktop_workspace
from .chat_runtime import generate_chat_turn
from .chat_store import (
    ChatSession,
    FileChatStore,
    chat_session_payload,
    chat_summary_payload,
)
from .compaction import LocalCompactionProvider
from .config import load_config
from .config_profiles import recommend_config_profile
from .context import ConversationTurn
from .context_policy import load_context_policy
from .doctor import build_doctor_report, render_doctor_report_markdown
from .environment import build_environment_report, render_environment_report_markdown
from .evaluator import evaluate_router, load_eval_cases
from .execution_scope import ScopePolicyError
from .extensions import (
    create_plugin_scaffold,
    load_extension_registry,
    load_mcp_servers,
    registry_payload,
)
from .memory import FileMemoryStore
from .model_servers import (
    ModelServerManager,
    model_server_action_payload,
    wait_for_managed_processes,
)
from .orchestrator import LocalMoE
from .performance_report import (
    build_performance_report,
    render_performance_report_markdown,
)
from .providers import ProviderError
from .profile_activation import (
    activate_config_profile,
    activate_recommended_config_profile,
)
from .run_log import RunLogStore, run_log_payload, run_log_prune_payload
from .runtime_optimizer import (
    build_runtime_optimizer_report,
    render_runtime_optimizer_markdown,
)
from .scheduler import cron_status, cron_summary_payload, run_due_jobs
from .security_audit import build_security_audit_report, render_security_audit_markdown
from .setup_status import inspect_setup_status, setup_status_payload
from .setup_runner import run_runtime_setup, setup_run_payload
from .smoke import DEFAULT_SMOKE_PROMPT, build_generation_smoke_report
from .startup import run_startup_readiness
from .support_bundle import build_support_bundle, support_bundle_filename
from .tool_runner import LocalToolRunner, ToolExecutionError, tool_result_payload


if TYPE_CHECKING:
    from .assistant_bridge import AssistantTaskEnvelope, BridgeRunResult


_ASSISTANT_BRIDGE_EXTRA_MODULES = frozenset(
    {
        "cryptography",
        "detect_secrets",
        "platformdirs",
        "psutil",
        "rfc8785",
    }
)
_CODING_CANARY_EXTRA_MODULES = frozenset({"rfc8785"})


def _run_coding_canary_command(argv: list[str]) -> int:
    try:
        module = import_module(".coding_canary", package=__package__)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing not in _CODING_CANARY_EXTRA_MODULES:
            raise
        print(
            "The coding-canary extra is required. Install it with "
            "`uv sync --extra coding-canary` or "
            "`pip install 'local-moe-orchestrator[coding-canary]'`.",
            file=sys.stderr,
        )
        return 2
    return int(module.main(argv))


class _MymoeArgumentParser(argparse.ArgumentParser):
    """Preserve legacy errors while redacting lifecycle invocation failures."""

    def error(self, message: str) -> None:
        lifecycle_flags = {
            "--assistant-bridge-stage": "stage",
            "--assistant-bridge-status": "status",
            "--assistant-bridge-resume-plan": "resume_plan",
            "--assistant-bridge-resume": "resume",
        }
        selected = [
            mode
            for flag, mode in lifecycle_flags.items()
            if any(
                value == flag or value.startswith(f"{flag}=") for value in sys.argv[1:]
            )
        ]
        if not selected:
            abbreviated = [
                mode
                for flag, mode in lifecycle_flags.items()
                if any(
                    flag.startswith(value.split("=", 1)[0])
                    and value.startswith("--assistant-bridge-")
                    for value in sys.argv[1:]
                )
            ]
            selected = abbreviated
        if selected:
            mode = selected[0] if len(selected) == 1 else "lifecycle"
            failure = AssistantBridgeCliError(
                mode=mode,
                code="invocation_invalid",
                exit_code=2,
            )
            self.exit(2, f"{assistant_bridge_canonical_json(failure.payload())}\n")
        super().error(message)


def main() -> None:
    if sys.argv[1:2] == ["advisor-init"]:
        from .advisor_setup import materialize_advisor_workspace

        advisor_init_parser = argparse.ArgumentParser(
            prog="mymoe advisor-init",
            description=(
                "Create a no-clobber, self-contained Adaptive Advisor starter."
            ),
            allow_abbrev=False,
        )
        advisor_init_parser.add_argument("--out", required=True)
        advisor_init_args = advisor_init_parser.parse_args(sys.argv[2:])
        try:
            result = materialize_advisor_workspace(advisor_init_args.out)
        except (FileExistsError, OSError, TypeError, ValueError):
            print(
                json.dumps(
                    {
                        "error": "advisor_init_failed",
                        "message": "Advisor workspace was not created.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from None
        print(json.dumps(result, indent=2))
        return

    if sys.argv[1:2] == ["advisor"]:
        from .adaptive_advisor_cli import main as run_advisor

        raise SystemExit(run_advisor(sys.argv[2:]))

    if sys.argv[1:2] == ["cell-exec"]:
        from .adaptive_execution_cli import main as run_cell_execution

        raise SystemExit(run_cell_execution(sys.argv[2:]))

    if sys.argv[1:2] == ["desktop-init"]:
        desktop_parser = argparse.ArgumentParser(
            prog="mymoe desktop-init",
            description=(
                "Bind an opt-in desktop workspace to one current process and window."
            ),
            allow_abbrev=False,
        )
        desktop_parser.add_argument("--out", required=True)
        desktop_parser.add_argument("--target-id", required=True)
        desktop_parser.add_argument("--target-pid", required=True, type=int)
        desktop_parser.add_argument("--window-id", required=True, type=int)
        desktop_parser.add_argument("--provider-binary")
        desktop_args = desktop_parser.parse_args(sys.argv[2:])
        try:
            result = materialize_desktop_workspace(
                desktop_args.out,
                target_id=desktop_args.target_id,
                target_pid=desktop_args.target_pid,
                window_id=desktop_args.window_id,
                provider_binary=desktop_args.provider_binary,
            )
        except (FileExistsError, OSError, ValueError, ToolExecutionError) as exc:
            print(
                json.dumps(
                    {"error": "desktop_init_failed", "message": str(exc)},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        print(json.dumps(result, indent=2))
        return

    if sys.argv[1:2] == ["browser-init"]:
        browser_parser = argparse.ArgumentParser(
            prog="mymoe browser-init",
            description="Create an opt-in, self-contained local browser workspace.",
            allow_abbrev=False,
        )
        browser_parser.add_argument("--out", required=True)
        browser_args = browser_parser.parse_args(sys.argv[2:])
        try:
            result = materialize_browser_workspace(browser_args.out)
        except (OSError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "error": "browser_init_failed",
                        "message": str(exc),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        print(json.dumps(result, indent=2))
        return

    if sys.argv[1:2] == ["browser-prefetch"]:
        browser_parser = argparse.ArgumentParser(
            prog="mymoe browser-prefetch",
            description=(
                "Cache one pinned browser provider and its dependencies without "
                "executing the provider package."
            ),
            allow_abbrev=False,
        )
        browser_parser.add_argument("--mcp-config", required=True)
        browser_parser.add_argument("--server", default="browser-local")
        browser_args = browser_parser.parse_args(sys.argv[2:])
        try:
            servers = load_mcp_servers(browser_args.mcp_config)
            matches = [item for item in servers if item.name == browser_args.server]
            if len(matches) != 1:
                raise ValueError(
                    f"Browser MCP config must contain exactly one server named "
                    f"{browser_args.server}."
                )
            result = prefetch_browser_provider(matches[0])
        except (OSError, ValueError, ToolExecutionError) as exc:
            print(
                json.dumps(
                    {
                        "error": "browser_prefetch_failed",
                        "message": str(exc),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        print(json.dumps(result, indent=2))
        return

    if sys.argv[1:2] == ["coding-canary"]:
        raise SystemExit(_run_coding_canary_command(sys.argv[2:]))

    if sys.argv[1:2] == ["assistant-probe"]:
        from .assistant_provider_probe import main as run_assistant_provider_probe

        raise SystemExit(run_assistant_provider_probe(sys.argv[2:]))

    parser = _MymoeArgumentParser(
        description="Local MoE orchestrator",
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  advisor          Recommend a verified offline cell without changing runtime state.\n"
            "  advisor-init     Create a no-clobber Adaptive Advisor starter.\n"
            "  assistant-probe  Check local Codex tool compatibility in a disposable workspace.\n"
            "  browser-init     Create packaged config files for the local browser cell.\n"
            "  browser-prefetch Cache a pinned browser provider without executing its package.\n"
            "  cell-exec        Preview exact local-cell admission without executing it.\n"
            "  desktop-init     Bind packaged desktop-cell config to one app window.\n"
            "  coding-canary    Qualify one local Cline coding cell with an isolated edit and test."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_distribution_version()}",
    )
    parser.add_argument("--config")
    parser.add_argument("--app-config", default="configs/app.json")
    assistant_task_group = parser.add_mutually_exclusive_group()
    assistant_task_group.add_argument(
        "--assistant-task",
        help="Plan or run one local-first assistant task.",
    )
    assistant_task_group.add_argument(
        "--assistant-task-file",
        help="Load an AssistantTaskEnvelope JSON document.",
    )
    parser.add_argument(
        "--assistant-bridge-config",
        default="configs/assistant-bridge.json",
    )
    parser.add_argument(
        "--assistant-profile",
        choices=["economy", "balanced", "quality", "privacy", "offline"],
    )
    parser.add_argument("--assistant-capability", action="append")
    parser.add_argument("--assistant-required-tool", action="append")
    parser.add_argument(
        "--assistant-risk",
        choices=[
            "read_only",
            "compute_only",
            "write_local",
            "write_external",
            "destructive",
            "privileged",
        ],
    )
    parser.add_argument("--assistant-constraint", action="append")
    remote_group = parser.add_mutually_exclusive_group()
    remote_group.add_argument("--assistant-allow-remote", action="store_true")
    remote_group.add_argument("--assistant-deny-remote", action="store_true")
    parser.add_argument("--assistant-allow-remote-workspace", action="store_true")
    parser.add_argument("--assistant-max-premium-calls", type=int)
    parser.add_argument("--assistant-workspace")
    parser.add_argument("--assistant-local-provider", choices=["ollama", "lmstudio"])
    parser.add_argument("--assistant-verification")
    parser.add_argument("--assistant-include-diff", action="store_true")
    parser.add_argument("--assistant-capsule-out")
    assistant_mode_group = parser.add_mutually_exclusive_group()
    assistant_mode_group.add_argument("--assistant-bridge-execute", action="store_true")
    assistant_mode_group.add_argument("--assistant-bridge-stage", action="store_true")
    assistant_mode_group.add_argument(
        "--assistant-bridge-status",
        metavar="WORKFLOW_ID",
    )
    assistant_mode_group.add_argument(
        "--assistant-bridge-resume-plan",
        metavar="WORKFLOW_ID",
    )
    assistant_mode_group.add_argument(
        "--assistant-bridge-resume",
        metavar="WORKFLOW_ID",
    )
    parser.add_argument(
        "--assistant-workflow-config",
        help="Two-phase durable-state and public-trust configuration.",
    )
    parser.add_argument("--assistant-idempotency-key")
    parser.add_argument(
        "--assistant-attestation-file",
        action="append",
        help="Independent DSSE envelope; repeat for multiple verifiers.",
    )
    parser.add_argument("--assistant-resume-plan-id")
    parser.add_argument(
        "--assistant-confirm-receipt",
        help="Exact confirmation_id emitted by a prior plan for the unchanged task and workspace.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument(
        "--agent-prompt",
        help="Run one bounded local tool-calling agent task.",
    )
    parser.add_argument("--agent-expert")
    parser.add_argument(
        "--agent-tool",
        action="append",
        help="Expose one configured strict-schema tool; repeat to expose more.",
    )
    parser.add_argument(
        "--agent-approve",
        action="append",
        metavar="TOOL:ARGUMENTS_SHA256",
        help="Approve only a tool name and exact argument hash returned by a prior run.",
    )
    parser.add_argument(
        "--agent-browser-server",
        help=(
            "Expose the narrow local-only browser capability from one explicitly "
            "configured MCP server."
        ),
    )
    parser.add_argument(
        "--agent-desktop-server",
        help=(
            "Expose read-only semantic state from one explicitly configured "
            "desktop capability server."
        ),
    )
    parser.add_argument(
        "--agent-interactive-approvals",
        action="store_true",
        default=None,
        help=(
            "Keep a stateful agent session open and require the exact displayed "
            "TOOL:ARGUMENTS_SHA256 token before each protected call."
        ),
    )
    parser.add_argument("--agent-correlation-id")
    parser.add_argument("--agent-max-model-turns", type=int)
    parser.add_argument("--agent-max-tool-calls", type=int)
    parser.add_argument("--agent-max-proposed-tool-calls", type=int)
    parser.add_argument("--agent-max-tool-result-chars", type=int)
    parser.add_argument("--agent-max-task-chars", type=int)
    parser.add_argument("--agent-max-tool-argument-chars", type=int)
    parser.add_argument(
        "--agent-soft-wall-time-seconds",
        type=float,
        help=(
            "Soft run deadline; remaining time also caps built-in HTTP and MCP "
            "operation timeouts."
        ),
    )
    parser.add_argument(
        "--agent-max-wall-time-seconds",
        type=float,
        dest="agent_deprecated_max_wall_time_seconds",
        help="Deprecated alias for --agent-soft-wall-time-seconds.",
    )
    parser.add_argument(
        "--browser-canary",
        metavar="SERVER",
        help="Qualify one configured local-only browser provider on a disposable fixture.",
    )
    parser.add_argument(
        "--browser-canary-confirm",
        action="store_true",
        help="Confirm the deterministic local browser process and fixture interactions.",
    )
    parser.add_argument(
        "--desktop-canary",
        metavar="SERVER",
        help="Qualify one configured read-only semantic desktop provider.",
    )
    parser.add_argument(
        "--desktop-canary-confirm",
        action="store_true",
        help="Confirm the read-only observation of the configured app window.",
    )
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
    parser.add_argument(
        "--performance-report-format", choices=["json", "markdown"], default="json"
    )
    parser.add_argument("--performance-report-out")
    parser.add_argument("--runtime-optimizer", action="store_true")
    parser.add_argument(
        "--runtime-optimizer-format", choices=["json", "markdown"], default="json"
    )
    parser.add_argument("--runtime-optimizer-out")
    parser.add_argument("--runtime-optimizer-runs-limit", type=int, default=100)
    parser.add_argument("--security-audit", action="store_true")
    parser.add_argument(
        "--security-audit-format", choices=["json", "markdown"], default="json"
    )
    parser.add_argument("--security-audit-out")
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
    prepare_profile_group.add_argument(
        "--prepare-recommended-profile", action="store_true"
    )
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
    _validate_assistant_bridge_cli_mode(parser, args)
    _validate_agent_cli_mode(parser, args)
    _validate_browser_canary_cli_mode(parser, args)
    _validate_desktop_canary_cli_mode(parser, args)

    lifecycle_cli_mode = assistant_bridge_lifecycle_mode(args)
    if lifecycle_cli_mode is not None:
        _run_assistant_lifecycle_mode(args)
        return

    app_config = load_app_config(args.app_config)

    if args.browser_canary:
        browser_runner: BrowserToolRunner | None = None
        try:
            allow_process_execution = bool(
                getattr(
                    getattr(app_config, "permissions", None),
                    "allow_process_execution",
                    False,
                )
            )
            if not allow_process_execution:
                raise ToolExecutionError(
                    "Browser canary is disabled by the app process-execution policy."
                )
            browser_runner = BrowserToolRunner.from_registry(
                _registry(app_config),
                args.browser_canary,
                allow_process_execution=True,
            )
            result = browser_runner.canary()
        except (ToolExecutionError, ValueError) as exc:
            print(
                json.dumps(
                    {"error": "browser_canary_error", "message": str(exc)},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        finally:
            if browser_runner is not None:
                browser_runner.close()
        print(json.dumps(result, indent=2))
        if result.get("status") != "passed":
            raise SystemExit(2)
        return

    if args.desktop_canary:
        desktop_runner: DesktopToolRunner | None = None
        try:
            allow_process_execution = bool(
                getattr(
                    getattr(app_config, "permissions", None),
                    "allow_process_execution",
                    False,
                )
            )
            if not allow_process_execution:
                raise ToolExecutionError(
                    "Desktop canary is disabled by the app process-execution policy."
                )
            desktop_runner = DesktopToolRunner.from_registry(
                _registry(app_config),
                args.desktop_canary,
                allow_process_execution=True,
            )
            result = desktop_runner.canary()
        except (ToolExecutionError, ValueError) as exc:
            print(
                json.dumps(
                    {"error": "desktop_canary_error", "message": str(exc)},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        finally:
            if desktop_runner is not None:
                desktop_runner.close()
        print(json.dumps(result, indent=2))
        if result.get("status") != "passed":
            raise SystemExit(2)
        return

    if args.assistant_task is not None or args.assistant_task_file is not None:
        from .assistant_bridge import (
            AssistantBridgeError,
            AssistantBridgeRunner,
            load_assistant_bridge_config,
            load_verification_evidence,
        )

        assistant_execution_started = False
        try:
            task = load_cli_assistant_task(args)
            assistant_workspace = args.assistant_workspace or "."
            bridge = AssistantBridgeRunner(
                load_assistant_bridge_config(args.assistant_bridge_config)
            )
            external_evidence = (
                load_verification_evidence(args.assistant_verification)
                if args.assistant_verification
                else ()
            )
            if not args.assistant_bridge_execute:
                print(
                    json.dumps(
                        bridge.plan(
                            task,
                            workspace=assistant_workspace,
                            local_provider_override=args.assistant_local_provider,
                            external_evidence=external_evidence,
                            include_diff=args.assistant_include_diff,
                            capsule_out=args.assistant_capsule_out,
                        ),
                        indent=2,
                    )
                )
                return
            try:
                _validate_assistant_bridge_policy_preflight(app_config)
            except AssistantBridgeError as exc:
                _record_assistant_bridge_denied(app_config, task, exc)
                raise
            authority_receipt = bridge.inspect_route(
                task,
                workspace=assistant_workspace,
                local_provider_override=args.assistant_local_provider,
                external_evidence=external_evidence,
                include_diff=args.assistant_include_diff,
                capsule_out=args.assistant_capsule_out,
            )
            try:
                _validate_assistant_bridge_authority(
                    app_config, task, authority_receipt.route
                )
            except AssistantBridgeError as exc:
                _record_assistant_bridge_denied(app_config, task, exc)
                raise
            _record_assistant_bridge_started(
                app_config,
                task,
                args.assistant_confirm_receipt or "",
            )
            assistant_execution_started = True
            result = bridge.run(
                task,
                workspace=assistant_workspace,
                confirmation=args.assistant_confirm_receipt or "",
                local_provider_override=args.assistant_local_provider,
                external_evidence=external_evidence,
                include_diff=args.assistant_include_diff,
                capsule_out=args.assistant_capsule_out,
            )
            _record_assistant_bridge_control_plane(app_config, task, result)
        except (AssistantBridgeError, OSError) as exc:
            if assistant_execution_started:
                try:
                    _record_assistant_bridge_failed(app_config, task, exc)
                except OSError:
                    pass
            print(
                json.dumps(
                    {"error": "assistant_bridge_error", "message": str(exc)},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        print(json.dumps(result.user_payload(), indent=2))
        if result.status != "completed":
            raise SystemExit(2)
        return

    config_path = args.config or app_config.default_moe_config
    config = load_config(config_path)

    if args.agent_prompt is not None:
        browser_runner: BrowserToolRunner | None = None
        desktop_runner: DesktopToolRunner | None = None
        try:
            if (
                str(getattr(app_config, "mode", "")).strip().lower()
                == "local_model_required"
            ):
                validate_local_agent_endpoints(config)
            registry = _registry(app_config)
            local_runner = LocalToolRunner(
                registry,
                app_config=app_config,
                moe_config=config,
                app_config_path=args.app_config,
                active_config_path=config_path,
            )
            runner: object = local_runner
            specialized_runners: list[object] = []
            if args.agent_browser_server:
                browser_runner = BrowserToolRunner.from_registry(
                    registry,
                    args.agent_browser_server,
                    allow_process_execution=bool(
                        getattr(
                            getattr(app_config, "permissions", None),
                            "allow_process_execution",
                            False,
                        )
                    ),
                )
                specialized_runners.append(browser_runner)
            if args.agent_desktop_server:
                desktop_runner = DesktopToolRunner.from_registry(
                    registry,
                    args.agent_desktop_server,
                    allow_process_execution=bool(
                        getattr(
                            getattr(app_config, "permissions", None),
                            "allow_process_execution",
                            False,
                        )
                    ),
                )
                specialized_runners.append(desktop_runner)
            if specialized_runners:
                runner = CompositeToolRunner(local_runner, *specialized_runners)
            additional_specs = tuple(
                spec
                for specialized in specialized_runners
                for spec in getattr(specialized, "specs", ())
            )
            agent = build_local_agent_loop(
                config,
                runner,
                registry,
                expert_id=args.agent_expert,
                visible_tools=args.agent_tool or (),
                additional_specs=additional_specs,
                permission_policy=_agent_permission_policy(app_config),
                budget=_agent_budget(args),
            )
            result = agent.run(
                args.agent_prompt,
                correlation_id=args.agent_correlation_id,
                approval_handler=_agent_approval_handler(
                    args.agent_approve,
                    interactive=args.agent_interactive_approvals,
                ),
            )
        except ScopePolicyError as exc:
            _exit_scope_blocked(exc)
        except (ProviderError, ToolExecutionError, ValueError) as exc:
            print(
                json.dumps(
                    {"error": "agent_error", "message": str(exc)},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        finally:
            for specialized in (desktop_runner, browser_runner):
                if specialized is not None:
                    specialized.close()
        _print_agent_result(result, json_output=args.json_output)
        if result.status != "completed":
            raise SystemExit(2)
        return

    if args.doctor or args.doctor_out:
        report = build_doctor_report(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=args.app_config,
        )
        rendered = (
            render_doctor_report_markdown(report)
            if args.doctor_format == "markdown"
            else json.dumps(report, indent=2)
        )
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
        rendered = (
            render_environment_report_markdown(payload)
            if args.about_format == "markdown"
            else json.dumps(payload, indent=2)
        )
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
            print(
                json.dumps(
                    run_log_prune_payload(store.prune(keep=args.runs_keep)), indent=2
                )
            )
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

    if args.security_audit or args.security_audit_out:
        payload = build_security_audit_report(
            config_path=config_path,
            config=config,
            app_config=app_config,
            app_config_path=args.app_config,
            registry=_registry(app_config),
        )
        rendered = (
            render_security_audit_markdown(payload)
            if args.security_audit_format == "markdown"
            else json.dumps(payload, indent=2)
        )
        if args.security_audit_out:
            out = Path(args.security_audit_out)
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
            summaries = chat_store.search_sessions(
                args.chat_query, limit=args.chats_limit
            )
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
            print(
                json.dumps(
                    {"error": "not_found", "message": "Chat session not found."},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        return

    if args.rename_chat:
        if not args.chat_title:
            print(
                json.dumps(
                    {
                        "error": "bad_request",
                        "message": "--chat-title is required with --rename-chat.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            session = _chat_store(app_config).rename_session(
                args.rename_chat, args.chat_title
            )
        except KeyError as exc:
            print(
                json.dumps(
                    {"error": "not_found", "message": "Chat session not found."},
                    indent=2,
                ),
                file=sys.stderr,
            )
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
            print(
                json.dumps(
                    {"error": "not_found", "message": "Chat session not found."},
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2)
        turns = _conversation_turns(session)
        if not turns:
            print(
                json.dumps(
                    {
                        "error": "bad_request",
                        "message": "Chat session has no turns to compact.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            compacted = LocalCompactionProvider(
                config, expert_id=args.compact_expert
            ).compact(
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
        except ScopePolicyError as exc:
            _exit_scope_blocked(exc)
        except (KeyError, ProviderError, ValueError) as exc:
            print(
                json.dumps({"error": "compact_error", "message": str(exc)}, indent=2),
                file=sys.stderr,
            )
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
            print(
                json.dumps(
                    {"error": "not_found", "message": "Chat session not found."},
                    indent=2,
                ),
                file=sys.stderr,
            )
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
                profile_roots=(Path(args.activate_profile).parent,),
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
                print(
                    json.dumps(
                        {
                            "error": "profile_error",
                            "message": "No recommended runtime profile is available.",
                        },
                        indent=2,
                    )
                )
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
        if not result.ok and result.status not in {
            "planned",
            "needs_setup",
            "confirmation_required",
        }:
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
        if payload["status"] in {
            "confirmation_required",
            "error",
            "manual_required",
            "needs_setup",
            "needs_attention",
        }:
            raise SystemExit(2)
        return

    if args.bootstrap:
        print(
            json.dumps(
                runtime_plan_payload(
                    build_runtime_plan(config, app_config.runtime.preferred_backends)
                ),
                indent=2,
            )
        )
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
            action = manager.start(
                confirm=args.models_confirm, only_first=args.models_only_first
            )
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
        path = create_plugin_scaffold(
            args.create_plugin, root=app_config.extensions.plugins_dir
        )
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
        except ScopePolicyError as exc:
            _exit_scope_blocked(exc)
        except (json.JSONDecodeError, ToolExecutionError, ProviderError) as exc:
            print(
                json.dumps({"error": "tool_error", "message": str(exc)}, indent=2),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        print(json.dumps(tool_result_payload(result), indent=2))
        return

    if args.eval:
        try:
            cases = load_eval_cases(args.eval)
            result = evaluate_router(config, cases)
        except ScopePolicyError as exc:
            _exit_scope_blocked(exc)
        print(json.dumps(result, indent=2))
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
        except ScopePolicyError as exc:
            _exit_scope_blocked(exc)
        except (KeyError, ProviderError) as exc:
            print(
                json.dumps({"error": "chat_error", "message": str(exc)}, indent=2),
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        _print_chat_payload(payload, json_output=args.json_output)
        return

    try:
        response = moe.generate(args.prompt)
    except ScopePolicyError as exc:
        _exit_scope_blocked(exc)
    except ProviderError as exc:
        print(
            json.dumps({"error": "provider_error", "message": str(exc)}, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
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
    print(
        "myMoE interactive shell. Type /help for commands or /exit to quit.",
        file=sys.stderr,
    )
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
            print(
                "Commands: /sessions, /session <id>, /new [title], /summary, /exit",
                file=sys.stderr,
            )
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
        except ScopePolicyError as exc:
            print(
                json.dumps(_scope_blocked_payload(exc), indent=2),
                file=sys.stderr,
            )
            continue
        except (KeyError, ProviderError) as exc:
            print(f"Generation failed: {exc}", file=sys.stderr)
            continue
        _print_chat_payload(payload, json_output=json_output)


def _scope_blocked_payload(exc: ScopePolicyError) -> dict[str, str]:
    return {"error": exc.reason_code, "message": str(exc)}


def _exit_scope_blocked(exc: ScopePolicyError) -> None:
    print(json.dumps(_scope_blocked_payload(exc), indent=2), file=sys.stderr)
    raise SystemExit(2) from exc


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


def _print_agent_result(result: AgentRunResult, *, json_output: bool) -> None:
    payload = _agent_result_payload(result)
    if json_output or result.final_answer is None:
        print(json.dumps(payload, indent=2))
        return

    print(result.final_answer)
    print()
    metadata = dict(payload)
    metadata.pop("final_answer", None)
    print(json.dumps(metadata, indent=2))


def _agent_result_payload(result: AgentRunResult) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "mode": "agent",
        "status": result.status,
        "reason": result.reason,
        "final_answer": result.final_answer,
        "correlation_id": result.correlation_id,
        "model_turns": result.model_turns,
        "tool_calls": result.tool_calls,
        "grounded_in_tool_results": result.grounded_in_tool_results,
        "grounded_tool_call_ids": list(result.grounded_tool_call_ids),
        "tool_results": [item.payload() for item in result.tool_results],
        "approval_requests": [
            {
                **asdict(item),
                "arguments": dict(item.arguments),
                "approval_token": f"{item.tool_name}:{item.arguments_sha256}",
            }
            for item in result.approval_requests
        ],
        "trace_policy": "metadata_only",
        "trace": [asdict(item) for item in result.trace],
    }


def _agent_budget(args: argparse.Namespace) -> AgentLoopBudget:
    defaults = AgentLoopBudget()
    return AgentLoopBudget(
        max_model_turns=(
            args.agent_max_model_turns
            if args.agent_max_model_turns is not None
            else defaults.max_model_turns
        ),
        max_tool_calls=(
            args.agent_max_tool_calls
            if args.agent_max_tool_calls is not None
            else defaults.max_tool_calls
        ),
        max_proposed_tool_calls_per_turn=(
            args.agent_max_proposed_tool_calls
            if args.agent_max_proposed_tool_calls is not None
            else defaults.max_proposed_tool_calls_per_turn
        ),
        max_tool_result_chars=(
            args.agent_max_tool_result_chars
            if args.agent_max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        ),
        max_task_chars=(
            args.agent_max_task_chars
            if args.agent_max_task_chars is not None
            else defaults.max_task_chars
        ),
        max_tool_argument_chars=(
            args.agent_max_tool_argument_chars
            if args.agent_max_tool_argument_chars is not None
            else defaults.max_tool_argument_chars
        ),
        soft_wall_time_seconds=(
            args.agent_soft_wall_time_seconds
            if args.agent_soft_wall_time_seconds is not None
            else args.agent_deprecated_max_wall_time_seconds
            if args.agent_deprecated_max_wall_time_seconds is not None
            else defaults.soft_wall_time_seconds
        ),
    )


def _agent_permission_policy(app_config: object) -> AgentPermissionPolicy:
    """Apply app policy as an additional deny layer over exact agent approvals."""

    denied: set[str] = set()
    denied_tools: set[str] = set()
    permissions = getattr(app_config, "permissions", None)
    if not bool(getattr(permissions, "allow_process_execution", False)):
        denied.add("process_execution")
    if (
        str(getattr(permissions, "external_communication_policy", "draft_only"))
        != "approval_required"
    ):
        denied.update(("communication", "write_external"))
    if (
        str(getattr(permissions, "default_write_policy", "approval_required"))
        != "approval_required"
    ):
        denied.update(("write_local", "write_internal"))
    if (
        str(getattr(permissions, "connector_install_policy", "approval_required"))
        .strip()
        .lower()
        != "approval_required"
    ):
        denied_tools.add("extension.configure")
    return AgentPermissionPolicy(
        denied_risks=tuple(sorted(denied)),
        denied_tools=tuple(sorted(denied_tools)),
    )


def _agent_approval_handler(
    raw_approvals: list[str] | None,
    *,
    interactive: bool = False,
):
    if not raw_approvals and not interactive:
        return None

    approved: set[tuple[str, str]] = set()
    pattern = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{0,127}):([0-9a-fA-F]{64})$")
    for value in raw_approvals or ():
        match = pattern.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                "--agent-approve must be TOOL:ARGUMENTS_SHA256 using a 64-character hexadecimal hash."
            )
        approved.add((match.group(1), match.group(2).lower()))

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        key = (request.tool_name, request.arguments_sha256.lower())
        exact = key in approved
        if exact:
            approved.remove(key)
        elif interactive:
            token = f"{request.tool_name}:{request.arguments_sha256.lower()}"
            print(
                json.dumps(
                    {
                        "approval_required": {
                            "tool": request.tool_name,
                            "risk_class": request.risk_class,
                            "side_effects": request.side_effects,
                            "summary": _approval_summary(request),
                            "arguments": request.arguments,
                            "exact_token": token,
                        }
                    },
                    indent=2,
                ),
                file=sys.stderr,
                flush=True,
            )
            print(
                "Type y or paste the exact token to approve this one bound call; "
                "press Enter to deny:",
                file=sys.stderr,
                flush=True,
            )
            response = sys.stdin.readline().strip()
            exact = response.lower() == "y" or response == token
        return ApprovalDecision(
            approved=exact,
            reason=(
                "Exact CLI approval matched."
                if exact
                else "No exact CLI approval matched this tool call."
            ),
        )

    return decide


def _approval_summary(request: ApprovalRequest) -> str:
    arguments = request.arguments
    if request.tool_name == "browser.navigate":
        return f"Open local URL {arguments.get('url', '')}"
    if request.tool_name == "browser.observe":
        return f"Read accessible state from {arguments.get('origin', '')}"
    if request.tool_name == "browser.click":
        return f"Click {arguments.get('target_label', arguments.get('target', ''))}"
    if request.tool_name == "browser.type":
        text = str(arguments.get("text", ""))
        preview = text if len(text) <= 80 else f"{text[:77]}..."
        return (
            f"Type {preview!r} into "
            f"{arguments.get('target_label', arguments.get('target', ''))}"
        )
    if request.tool_name == "desktop.observe":
        target = str(arguments.get("target_id", "configured-target"))
        binding = str(arguments.get("binding_sha256", ""))[:12]
        return f"Read semantic state from desktop target {target} (binding {binding})"
    return f"Run {request.tool_name} with the shown exact arguments"


def _run_assistant_lifecycle_mode(
    args: argparse.Namespace,
    *,
    app_config: object | None = None,
) -> None:
    mode = assistant_bridge_lifecycle_mode(args) or "lifecycle"
    if app_config is None and mode == "stage":
        try:
            app_config = load_app_config(args.app_config)
        except (OSError, ValueError):
            failure = AssistantBridgeCliError(
                mode=mode,
                code="application_config_invalid",
                exit_code=2,
            )
            print(assistant_bridge_canonical_json(failure.payload()), file=sys.stderr)
            raise SystemExit(failure.exit_code) from None
    try:
        outcome = run_lifecycle_cli(args, app_config=app_config)
    except AssistantBridgeCliError as exc:
        print(assistant_bridge_canonical_json(exc.payload()), file=sys.stderr)
        raise SystemExit(exc.exit_code) from None
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        is_bridge_dependency = missing_module in _ASSISTANT_BRIDGE_EXTRA_MODULES
        failure = AssistantBridgeCliError(
            mode=mode,
            code=(
                "assistant_bridge_dependency_missing"
                if is_bridge_dependency
                else "unexpected_runtime_failure"
            ),
            exit_code=2 if is_bridge_dependency else 4,
        )
        print(assistant_bridge_canonical_json(failure.payload()), file=sys.stderr)
        raise SystemExit(failure.exit_code) from None
    except Exception:
        failure = AssistantBridgeCliError(
            mode=mode,
            code="unexpected_runtime_failure",
            exit_code=4,
        )
        print(assistant_bridge_canonical_json(failure.payload()), file=sys.stderr)
        raise SystemExit(failure.exit_code) from None
    print(assistant_bridge_canonical_json(outcome.payload))
    if outcome.exit_code:
        raise SystemExit(outcome.exit_code)


def _validate_assistant_bridge_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    has_task = args.assistant_task is not None or args.assistant_task_file is not None
    construction_options = (
        "assistant_profile",
        "assistant_capability",
        "assistant_required_tool",
        "assistant_risk",
        "assistant_constraint",
        "assistant_max_premium_calls",
    )
    lifecycle_cli_mode = assistant_bridge_lifecycle_mode(args)
    if lifecycle_cli_mode is not None:
        _validate_assistant_lifecycle_cli_mode(
            parser,
            args,
            mode=lifecycle_cli_mode,
            has_task=has_task,
            construction_options=construction_options,
        )
        return
    bridge_options_active = (
        any(getattr(args, name) is not None for name in construction_options)
        or any(
            bool(getattr(args, name))
            for name in (
                "assistant_allow_remote",
                "assistant_allow_remote_workspace",
                "assistant_deny_remote",
                "assistant_verification",
                "assistant_include_diff",
                "assistant_capsule_out",
                "assistant_bridge_execute",
                "assistant_bridge_stage",
                "assistant_bridge_status",
                "assistant_bridge_resume_plan",
                "assistant_bridge_resume",
                "assistant_confirm_receipt",
                "assistant_local_provider",
                "assistant_workflow_config",
                "assistant_idempotency_key",
                "assistant_attestation_file",
                "assistant_resume_plan_id",
            )
        )
        or any(
            getattr(args, name) != parser.get_default(name)
            for name in (
                "assistant_bridge_config",
                "assistant_workspace",
            )
        )
    )
    if not has_task and bridge_options_active:
        parser.error(
            "--assistant-* options require --assistant-task or --assistant-task-file"
        )
    if not has_task:
        return

    if args.assistant_task_file is not None and (
        any(getattr(args, name) is not None for name in construction_options)
        or args.assistant_allow_remote
        or args.assistant_allow_remote_workspace
        or args.assistant_deny_remote
    ):
        parser.error(
            "Assistant task construction options cannot override --assistant-task-file"
        )
    if args.assistant_bridge_execute and not args.assistant_confirm_receipt:
        parser.error("--assistant-bridge-execute requires --assistant-confirm-receipt")
    if args.assistant_confirm_receipt and not args.assistant_bridge_execute:
        parser.error("--assistant-confirm-receipt requires --assistant-bridge-execute")
    assistant_options = {
        "app_config",
        "assistant_allow_remote",
        "assistant_allow_remote_workspace",
        "assistant_bridge_config",
        "assistant_bridge_execute",
        "assistant_capability",
        "assistant_capsule_out",
        "assistant_confirm_receipt",
        "assistant_constraint",
        "assistant_deny_remote",
        "assistant_include_diff",
        "assistant_local_provider",
        "assistant_max_premium_calls",
        "assistant_profile",
        "assistant_required_tool",
        "assistant_risk",
        "assistant_task",
        "assistant_task_file",
        "assistant_verification",
        "assistant_workspace",
        "json_output",
    }
    _reject_assistant_mode_conflicts(parser, args, assistant_options)


def _validate_assistant_lifecycle_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    mode: str,
    has_task: bool,
    construction_options: tuple[str, ...],
) -> None:
    if args.assistant_workflow_config is None:
        parser.error("Lifecycle modes require --assistant-workflow-config")
    if mode != "stage":
        workflow_id = getattr(args, f"assistant_bridge_{mode}")
        if (
            not isinstance(workflow_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", workflow_id) is None
        ):
            parser.error("Lifecycle workflow id must be a safe identifier")
    if mode == "stage":
        if not has_task:
            parser.error(
                "--assistant-bridge-stage requires --assistant-task or --assistant-task-file"
            )
        if args.assistant_workspace is None:
            parser.error("--assistant-bridge-stage requires --assistant-workspace")
        if args.assistant_idempotency_key is None:
            parser.error(
                "--assistant-bridge-stage requires --assistant-idempotency-key"
            )
    else:
        if has_task:
            parser.error(
                f"--assistant-bridge-{mode.replace('_', '-')} forbids task input"
            )
        if any(getattr(args, name) is not None for name in construction_options) or any(
            (
                args.assistant_allow_remote,
                args.assistant_allow_remote_workspace,
                args.assistant_deny_remote,
            )
        ):
            parser.error("Task construction options require stage or legacy task mode")

    if args.assistant_task_file is not None and (
        any(getattr(args, name) is not None for name in construction_options)
        or args.assistant_allow_remote
        or args.assistant_allow_remote_workspace
        or args.assistant_deny_remote
    ):
        parser.error(
            "Assistant task construction options cannot override --assistant-task-file"
        )

    if args.assistant_idempotency_key is not None:
        encoded = args.assistant_idempotency_key.encode("utf-8")
        if not 16 <= len(encoded) <= 1024:
            parser.error("--assistant-idempotency-key length is outside safe bounds")
    if args.assistant_confirm_receipt is not None and (
        not args.assistant_confirm_receipt
        or len(args.assistant_confirm_receipt) > 4096
        or "\x00" in args.assistant_confirm_receipt
    ):
        parser.error("--assistant-confirm-receipt is outside safe bounds")

    common = {"assistant_workflow_config", "json_output"}
    if mode == "stage":
        allowed = common | {
            "app_config",
            "assistant_allow_remote",
            "assistant_allow_remote_workspace",
            "assistant_attestation_file",
            "assistant_bridge_config",
            "assistant_bridge_stage",
            "assistant_capability",
            "assistant_confirm_receipt",
            "assistant_constraint",
            "assistant_deny_remote",
            "assistant_idempotency_key",
            "assistant_include_diff",
            "assistant_local_provider",
            "assistant_max_premium_calls",
            "assistant_profile",
            "assistant_required_tool",
            "assistant_risk",
            "assistant_task",
            "assistant_task_file",
            "assistant_verification",
            "assistant_workspace",
        }
        if args.assistant_attestation_file:
            parser.error("Independent attestations are accepted only by resume-plan")
    elif mode == "status":
        allowed = common | {"assistant_bridge_status"}
    elif mode == "resume_plan":
        if args.assistant_workspace is None:
            parser.error(
                "--assistant-bridge-resume-plan requires --assistant-workspace"
            )
        if args.assistant_idempotency_key is None:
            parser.error(
                "--assistant-bridge-resume-plan requires --assistant-idempotency-key"
            )
        allowed = common | {
            "assistant_attestation_file",
            "assistant_bridge_config",
            "assistant_bridge_resume_plan",
            "assistant_idempotency_key",
            "assistant_workspace",
        }
    else:
        if args.assistant_workspace is None:
            parser.error("--assistant-bridge-resume requires --assistant-workspace")
        if args.assistant_resume_plan_id is None:
            parser.error(
                "--assistant-bridge-resume requires --assistant-resume-plan-id"
            )
        if re.fullmatch(r"[0-9a-f]{64}", args.assistant_resume_plan_id) is None:
            parser.error(
                "--assistant-resume-plan-id must be a lowercase SHA-256 digest"
            )
        if args.assistant_confirm_receipt is None:
            parser.error(
                "--assistant-bridge-resume requires --assistant-confirm-receipt"
            )
        allowed = common | {
            "app_config",
            "assistant_bridge_config",
            "assistant_bridge_resume",
            "assistant_confirm_receipt",
            "assistant_resume_plan_id",
            "assistant_workspace",
        }
    _reject_assistant_mode_conflicts(parser, args, allowed)


def _reject_assistant_mode_conflicts(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    allowed: set[str],
) -> None:
    explicit = _explicit_argument_destinations(parser)
    active_conflicts = [
        name
        for name, value in vars(args).items()
        if name not in allowed
        and (value != parser.get_default(name) or name in explicit)
    ]
    if active_conflicts:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in active_conflicts)
        parser.error(f"Assistant bridge mode cannot be combined with {rendered}")


def _explicit_argument_destinations(
    parser: argparse.ArgumentParser,
) -> set[str]:
    destinations: set[str] = set()
    actions = parser._option_string_actions
    for value in sys.argv[1:]:
        option = value.split("=", 1)[0]
        action = actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


def _validate_agent_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    agent_option_names = (
        "agent_expert",
        "agent_tool",
        "agent_approve",
        "agent_browser_server",
        "agent_desktop_server",
        "agent_interactive_approvals",
        "agent_correlation_id",
        "agent_max_model_turns",
        "agent_max_tool_calls",
        "agent_max_proposed_tool_calls",
        "agent_max_tool_result_chars",
        "agent_max_task_chars",
        "agent_max_tool_argument_chars",
        "agent_soft_wall_time_seconds",
        "agent_deprecated_max_wall_time_seconds",
    )
    if args.agent_prompt is None and any(
        getattr(args, name) is not None for name in agent_option_names
    ):
        parser.error("--agent-* options require --agent-prompt")
    if args.agent_prompt is None:
        return
    if (
        not args.agent_tool
        and not args.agent_browser_server
        and not args.agent_desktop_server
    ):
        parser.error(
            "--agent-prompt requires at least one explicit --agent-tool or "
            "specialized browser/desktop server; use --prompt for tool-free generation"
        )
    if (
        args.agent_soft_wall_time_seconds is not None
        and args.agent_deprecated_max_wall_time_seconds is not None
    ):
        parser.error(
            "--agent-soft-wall-time-seconds cannot be combined with deprecated --agent-max-wall-time-seconds"
        )
    if args.agent_deprecated_max_wall_time_seconds is not None:
        print(
            "warning: --agent-max-wall-time-seconds is deprecated; use --agent-soft-wall-time-seconds",
            file=sys.stderr,
        )

    if (
        args.agent_correlation_id is not None
        and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
            args.agent_correlation_id,
        )
        is None
    ):
        parser.error(
            "--agent-correlation-id must contain 1-96 safe identifier characters"
        )

    conflicting_actions = (
        "interactive",
        "eval",
        "chat_session",
        "new_chat",
        "chat_title",
        "chat_query",
        "list_chats",
        "export_chat",
        "compact_chat",
        "compact_expert",
        "delete_chat",
        "rename_chat",
        "chat_confirm",
        "doctor",
        "doctor_out",
        "about",
        "about_out",
        "runs",
        "runs_prune",
        "runs_confirm",
        "performance_report",
        "performance_report_out",
        "runtime_optimizer",
        "runtime_optimizer_out",
        "security_audit",
        "security_audit_out",
        "support_bundle",
        "support_bundle_out",
        "bootstrap",
        "setup",
        "recommend_profile",
        "activate_profile",
        "activate_recommended_profile",
        "profile_confirm",
        "prepare_runtime",
        "prepare_profile",
        "prepare_recommended_profile",
        "prepare_execute",
        "prepare_download_models",
        "prepare_confirm",
        "startup",
        "startup_prepare",
        "startup_download_models",
        "startup_start_models",
        "startup_confirm",
        "startup_only_first",
        "models_status",
        "models_logs",
        "models_log_expert",
        "start_models",
        "stop_models",
        "models_confirm",
        "models_only_first",
        "list_extensions",
        "create_plugin",
        "run_tool",
        "cron_status",
        "run_cron",
        "cron_dry_run",
        "cron_confirm_writes",
        "smoke_generate",
    )
    active_conflicts = [
        name for name in conflicting_actions if bool(getattr(args, name))
    ]
    if active_conflicts:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in active_conflicts)
        parser.error(f"--agent-prompt cannot be combined with {rendered}")


def _validate_browser_canary_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.browser_canary is None:
        if args.browser_canary_confirm:
            parser.error("--browser-canary-confirm requires --browser-canary")
        return
    if not args.browser_canary_confirm:
        parser.error("--browser-canary requires --browser-canary-confirm")
    allowed = {
        "app_config",
        "browser_canary",
        "browser_canary_confirm",
        "json_output",
    }
    explicit = _explicit_argument_destinations(parser)
    conflicts = sorted(
        name
        for name in explicit
        if name not in allowed
    )
    if conflicts:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in conflicts)
        parser.error(f"--browser-canary cannot be combined with {rendered}")


def _validate_desktop_canary_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.desktop_canary is None:
        if args.desktop_canary_confirm:
            parser.error("--desktop-canary-confirm requires --desktop-canary")
        return
    if not args.desktop_canary_confirm:
        parser.error("--desktop-canary requires --desktop-canary-confirm")
    allowed = {
        "app_config",
        "desktop_canary",
        "desktop_canary_confirm",
        "json_output",
    }
    explicit = _explicit_argument_destinations(parser)
    conflicts = sorted(name for name in explicit if name not in allowed)
    if conflicts:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in conflicts)
        parser.error(f"--desktop-canary cannot be combined with {rendered}")


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
        "disagreement": asdict(response.disagreement)
        if response.disagreement
        else None,
    }


def _chat_metadata(payload: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": payload.get("session_id", ""),
        "correlation_id": payload.get("correlation_id", ""),
        "selected": payload.get("route", {}).get("selected", [])
        if isinstance(payload.get("route"), dict)
        else [],
        "fallback_order": (
            payload.get("route", {}).get("fallback_order", [])
            if isinstance(payload.get("route"), dict)
            else []
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


def _record_assistant_bridge_control_plane(
    app_config: object,
    task: AssistantTaskEnvelope,
    result: BridgeRunResult,
) -> None:
    AuditLogStore(_audit_log_path(app_config)).record(
        "assistant_bridge.execute",
        result.status,
        risk_class=task.capability_demand.risk_class,
        subject=task.task_id,
        metadata={
            "task_fingerprint": task.task_fingerprint,
            "receipt_id": result.receipt.receipt_id,
            "route": result.receipt.route,
            "code": result.code,
            "final_provider": result.final_provider,
            "premium_calls_used": result.premium_calls_used,
            "verification_count": len(result.verification),
            "process_execution_policy": (
                app_config.permissions.assistant_bridge_execution_policy
            ),
        },
    )
    RunLogStore(_run_log_path(app_config)).record_generation(
        mode="assistant_bridge",
        prompt=task.objective,
        response_payload={
            "correlation_id": task.task_id,
            "route": {
                "selected": [
                    {
                        "expert_id": result.final_provider or "none",
                    }
                ],
                "fallback_order": [],
            },
            "results": [
                {
                    "model": result.final_provider or "none",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                }
            ],
            "errors": [] if result.status == "completed" else [result.code],
        },
        context_payload={
            "token_estimate": None,
            "budget_tokens": None,
            "compaction_needed": False,
            "dropped_turns": 0,
            "sections": {},
            "memory_ids": [],
        },
    )


def _record_assistant_bridge_started(
    app_config: object,
    task: AssistantTaskEnvelope,
    confirmation: str,
) -> None:
    AuditLogStore(_audit_log_path(app_config)).record(
        "assistant_bridge.execute",
        "started",
        risk_class=task.capability_demand.risk_class,
        subject=task.task_id,
        metadata={
            "task_fingerprint": task.task_fingerprint,
            "confirmation_sha256": hashlib.sha256(
                confirmation.encode("utf-8")
            ).hexdigest(),
            "content_logged": False,
        },
    )


def _record_assistant_bridge_failed(
    app_config: object,
    task: AssistantTaskEnvelope,
    error: Exception,
) -> None:
    message_sha256 = hashlib.sha256(str(error).encode("utf-8")).hexdigest()
    AuditLogStore(_audit_log_path(app_config)).record(
        "assistant_bridge.execute",
        "failed",
        risk_class=task.capability_demand.risk_class,
        subject=task.task_id,
        metadata={
            "task_fingerprint": task.task_fingerprint,
            "error_type": type(error).__name__,
            "error_message_sha256": message_sha256,
            "content_logged": False,
        },
    )


def _record_assistant_bridge_denied(
    app_config: object,
    task: AssistantTaskEnvelope,
    error: Exception,
) -> None:
    AuditLogStore(_audit_log_path(app_config)).record(
        "assistant_bridge.execute",
        "denied",
        risk_class=task.capability_demand.risk_class,
        subject=task.task_id,
        metadata={
            "task_fingerprint": task.task_fingerprint,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "content_logged": False,
        },
    )


def _validate_assistant_bridge_authority(
    app_config: object,
    task: AssistantTaskEnvelope,
    route: str,
) -> None:
    from .assistant_bridge import AssistantBridgeError

    execution_policy = _validate_assistant_bridge_policy_preflight(app_config)
    if execution_policy == "local_only" and route in {
        "local_then_verify",
        "premium",
    }:
        raise AssistantBridgeError(
            "Application local_only policy forbids a route that can invoke premium execution."
        )
    if task.capability_demand.risk_class == "write_local":
        policy = str(app_config.permissions.default_write_policy).strip().lower()
        if policy in {"deny", "denied", "disabled", "forbidden"}:
            raise AssistantBridgeError("Application policy forbids local writes.")
    if task.capability_demand.risk_class in {
        "write_external",
        "destructive",
        "privileged",
    }:
        raise AssistantBridgeError(
            "Assistant Bridge cannot grant external, destructive, or privileged authority."
        )


def _validate_assistant_bridge_policy_preflight(app_config: object) -> str:
    """Reject process-wide bridge policy before any route preparation."""

    from .assistant_bridge import AssistantBridgeError

    execution_policy = (
        str(
            getattr(
                app_config.permissions,
                "assistant_bridge_execution_policy",
                "disabled",
            )
        )
        .strip()
        .lower()
    )
    if execution_policy == "disabled":
        raise AssistantBridgeError(
            "Application policy disables Assistant Bridge process execution."
        )
    if execution_policy not in {
        "local_only",
        "hybrid_receipt_confirmation",
    }:
        raise AssistantBridgeError("Application Assistant Bridge policy is invalid.")
    return execution_policy


def _cron_state_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/cron-state.json"


def _run_log_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/runs.jsonl"


def _audit_log_path(app_config: object) -> str:
    return f"{app_config.runtime.work_dir.rstrip('/')}/audit.jsonl"


def _chat_store(app_config: object) -> FileChatStore:
    return FileChatStore(f"{app_config.runtime.work_dir.rstrip('/')}/chats.json")


def _memory_store(app_config: object) -> FileMemoryStore:
    return FileMemoryStore(f"{app_config.runtime.work_dir.rstrip('/')}/memory.jsonl")


def _distribution_version() -> str:
    try:
        return importlib_metadata.version("local-moe-orchestrator")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _context_policy(app_config: object) -> object:
    return load_context_policy(
        app_config.runtime.context_policy_config,
        app_config.runtime.context_policy_profile,
    )


if __name__ == "__main__":
    main()
