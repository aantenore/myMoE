from __future__ import annotations

from dataclasses import dataclass, field
import platform
import shutil
from typing import Any, Callable
from urllib import error
from urllib.parse import urlparse

from .config import MoEConfig
from .http_boundary import open_model_endpoint


@dataclass(frozen=True)
class ExpertRuntimeCommand:
    expert_id: str
    backend: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class RuntimePlan:
    """Runtime bootstrap commands with expert-aware and legacy projections.

    ``model_commands`` remains an initializer field so legacy callers keep
    working. Expert-aware plans preserve their exact id mapping across
    :func:`dataclasses.replace`; use :meth:`with_model_commands` to explicitly
    convert one to the legacy positional form.
    """

    platform_key: str
    backend: str
    install_commands: tuple[tuple[str, ...], ...]
    model_commands: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...]
    expert_commands: tuple[ExpertRuntimeCommand, ...] = field(
        default=(),
        kw_only=True,
    )

    def __post_init__(self) -> None:
        expert_commands = tuple(self.expert_commands)
        if expert_commands and self.model_commands != tuple(
            command.argv for command in expert_commands
        ):
            raise ValueError(
                "Expert-aware RuntimePlan commands must stay aligned; use "
                "with_model_commands() for an explicit legacy projection."
            )
        object.__setattr__(self, "expert_commands", expert_commands)

    def with_expert_commands(
        self,
        commands: tuple[ExpertRuntimeCommand, ...],
    ) -> RuntimePlan:
        expert_commands = tuple(commands)
        plan = RuntimePlan(
            platform_key=self.platform_key,
            backend=self.backend,
            install_commands=self.install_commands,
            model_commands=tuple(command.argv for command in expert_commands),
            notes=self.notes,
            expert_commands=expert_commands,
        )
        return plan

    def with_model_commands(
        self,
        commands: tuple[tuple[str, ...], ...],
    ) -> RuntimePlan:
        return RuntimePlan(
            platform_key=self.platform_key,
            backend=self.backend,
            install_commands=self.install_commands,
            model_commands=tuple(commands),
            notes=self.notes,
        )


def detect_platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "darwin_arm64"
    if system.startswith("win"):
        return "windows"
    if system == "linux":
        return "linux"
    return "fallback"


def build_runtime_plan(
    config: MoEConfig, preferred_backends: dict[str, str] | None = None
) -> RuntimePlan:
    platform_key = detect_platform_key()
    preferred = preferred_backends or {}
    experts = tuple(
        expert for expert in config.experts if expert.provider == "openai_compatible"
    )
    configured_backends = {
        str(expert.params.get("runtime_backend"))
        for expert in experts
        if expert.params.get("runtime_backend")
    }
    backend = _runtime_backend(configured_backends, preferred, platform_key)
    if backend == "mixed":
        venv_python = _venv_python(platform_key)
        default_backend = _default_backend(platform_key)
        effective_backends = {
            _expert_runtime_backend(expert, default_backend) for expert in experts
        }
        install = _mixed_install_commands(effective_backends, platform_key, experts)
        mixed_commands: list[ExpertRuntimeCommand] = []
        for index, expert in enumerate(experts):
            expert_backend = _expert_runtime_backend(expert, default_backend)
            mixed_commands.append(
                ExpertRuntimeCommand(
                    expert_id=expert.id,
                    backend=expert_backend,
                    argv=_expert_server_command(
                        expert,
                        expert_backend,
                        _expert_port(expert, 8101 + index),
                        venv_python,
                    ),
                )
            )
        expert_commands = tuple(mixed_commands)
        notes = (
            "Mixed runtime plan. Keep only the models needed for the current workflow resident.",
        )
    elif backend in {"mlx_lm", "mlx_vlm"}:
        venv_python = _venv_python(platform_key)
        install = (
            ("uv", "venv", "--python", "3.12", ".venv"),
            (
                "uv",
                "pip",
                "install",
                "--python",
                venv_python,
                _mlx_extra(experts, default_backend=backend),
            ),
        )
        mlx_commands: list[ExpertRuntimeCommand] = []
        for index, expert in enumerate(experts):
            expert_backend = _expert_runtime_backend(expert, backend)
            mlx_commands.append(
                ExpertRuntimeCommand(
                    expert_id=expert.id,
                    backend=expert_backend,
                    argv=_mlx_server_command(
                        venv_python,
                        expert,
                        _expert_port(expert, 8101 + index),
                        runtime_backend=expert_backend,
                    ),
                )
            )
        expert_commands = tuple(mlx_commands)
        notes = (
            "Best path for Apple Silicon. Keep one heavy model resident; add small fallback only if memory allows.",
            "Gemma 4 E4B is validated with the pinned .[mlx] profile because newer MLX packages can reject its current artifact.",
        )
    elif backend == "ollama":
        install = (("install", "ollama", "from", "https://ollama.com/download"),)
        expert_commands = tuple(
            ExpertRuntimeCommand(
                expert_id=expert.id,
                backend="ollama",
                argv=("ollama", "pull", _ollama_model_name(expert.model)),
            )
            for expert in experts
        )
        notes = (
            "Cross-platform fallback for Windows/Linux/macOS. Uses Ollama OpenAI-compatible API at port 11434.",
        )
    else:
        venv_python = _venv_python(platform_key)
        install = (
            ("uv", "venv", "--python", "3.12", ".venv"),
            ("uv", "pip", "install", "--python", venv_python, ".[gguf]"),
            _llama_cpp_install_command(platform_key),
        )
        expert_commands = tuple(
            ExpertRuntimeCommand(
                expert_id=expert.id,
                backend=backend,
                argv=_llama_server_command(
                    expert,
                    _expert_port(expert, 8101 + index),
                ),
            )
            for index, expert in enumerate(experts)
        )
        notes = (
            "GGUF backend through llama.cpp. Prefer quantized models that fit local RAM/VRAM headroom.",
        )

    legacy_plan = RuntimePlan(
        platform_key=platform_key,
        backend=backend,
        install_commands=install,
        model_commands=tuple(command.argv for command in expert_commands),
        notes=notes,
    )
    return legacy_plan.with_expert_commands(expert_commands)


def endpoint_is_reachable(
    base_url: str,
    timeout_seconds: float = 2.0,
    *,
    opener: Callable[..., Any] = open_model_endpoint,
) -> bool:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return False
    root = f"{parsed.scheme}://{parsed.netloc}"
    for suffix in ("/v1/models", "/health"):
        try:
            with opener(root + suffix, timeout=timeout_seconds):
                return True
        except (OSError, ValueError, error.URLError):
            continue
    return False


def runtime_plan_payload(plan: RuntimePlan) -> dict[str, object]:
    return {
        "platform_key": plan.platform_key,
        "backend": plan.backend,
        "available_commands": {
            "uv": shutil.which("uv") is not None,
            "ollama": shutil.which("ollama") is not None,
            "llama-server": shutil.which("llama-server") is not None,
            "python": shutil.which("python") is not None
            or shutil.which("python3") is not None,
        },
        "install_commands": [list(item) for item in plan.install_commands],
        "model_commands": [list(item) for item in plan.model_commands],
        "expert_commands": [
            {
                "expert_id": item.expert_id,
                "backend": item.backend,
                "argv": list(item.argv),
            }
            for item in plan.expert_commands
        ],
        "notes": list(plan.notes),
    }


def _default_backend(platform_key: str) -> str:
    if platform_key == "darwin_arm64":
        return "mlx_lm"
    if platform_key in {"windows", "linux"}:
        return "ollama"
    return "llama_cpp"


def _runtime_backend(
    configured_backends: set[str], preferred: dict[str, str], platform_key: str
) -> str:
    if len(configured_backends) == 1:
        return next(iter(configured_backends))
    if len(configured_backends) > 1:
        return "mixed"
    return (
        preferred.get(platform_key)
        or preferred.get("fallback")
        or _default_backend(platform_key)
    )


def _venv_python(platform_key: str) -> str:
    if platform_key == "windows":
        return ".venv\\Scripts\\python.exe"
    return ".venv/bin/python"


def _mlx_server_command(
    venv_python: str,
    expert: object,
    port: int,
    *,
    runtime_backend: str | None = None,
) -> tuple[str, ...]:
    params = getattr(expert, "params", {})
    runtime_backend = runtime_backend or str(params.get("runtime_backend", "mlx_lm"))
    module = "mlx_vlm.server" if runtime_backend == "mlx_vlm" else "mlx_lm.server"
    runtime_executable = str(params.get("runtime_executable", venv_python))
    command = [
        runtime_executable,
        "-m",
        module,
        "--model",
        str(expert.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    max_tokens = params.get("max_tokens")
    if max_tokens is not None:
        command.extend(
            ["--max-tokens", str(_positive_runtime_int(max_tokens, "max_tokens"))]
        )
    for key, flag in (
        ("decode_concurrency", "--decode-concurrency"),
        ("prompt_concurrency", "--prompt-concurrency"),
        ("prefill_step_size", "--prefill-step-size"),
        ("prompt_cache_size", "--prompt-cache-size"),
        ("prompt_cache_bytes", "--prompt-cache-bytes"),
    ):
        value = params.get(key)
        if value is not None and runtime_backend == "mlx_lm":
            command.extend([flag, str(_positive_runtime_int(value, key))])
    max_kv_size = params.get("max_kv_size")
    if max_kv_size is not None and runtime_backend == "mlx_vlm":
        command.extend(
            ["--max-kv-size", str(_positive_runtime_int(max_kv_size, "max_kv_size"))]
        )
    return tuple(command)


def _positive_runtime_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(
            f"Expert runtime parameter {label} must be a positive integer."
        )
    return value


def _expert_server_command(
    expert: object,
    backend: str,
    port: int,
    venv_python: str,
) -> tuple[str, ...]:
    if backend in {"mlx_lm", "mlx_vlm"}:
        return _mlx_server_command(
            venv_python,
            expert,
            port,
            runtime_backend=backend,
        )
    if backend == "llama_cpp":
        return _llama_server_command(expert, port)
    if backend == "ollama":
        return ("ollama", "pull", _ollama_model_name(str(expert.model)))
    return _llama_server_command(expert, port)


def _llama_server_command(expert: object, port: int) -> tuple[str, ...]:
    params = getattr(expert, "params", {})
    runtime_executable = str(params.get("runtime_executable", "llama-server"))
    model_source = str(params.get("runtime_model_source", "huggingface"))
    if model_source == "local":
        model_argument = "-m"
    elif model_source == "huggingface":
        model_argument = "-hf"
    else:
        raise ValueError(
            "Expert runtime parameter runtime_model_source must be local or huggingface."
        )
    command = [
        runtime_executable,
        model_argument,
        str(expert.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    context_size = params.get("context_size")
    if context_size is not None:
        command.extend(["--ctx-size", str(context_size)])
    gpu_layers = params.get("gpu_layers")
    if gpu_layers is not None:
        command.extend(["--n-gpu-layers", str(gpu_layers)])
    return tuple(command)


def _llama_cpp_install_command(platform_key: str) -> tuple[str, ...]:
    if platform_key == "windows":
        return (
            "install",
            "llama.cpp",
            "from",
            "https://github.com/ggml-org/llama.cpp/releases",
        )
    return ("install", "llama.cpp", "from", "https://github.com/ggml-org/llama.cpp")


def _mixed_install_commands(
    configured_backends: set[str],
    platform_key: str,
    experts: tuple[object, ...],
) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    if {"mlx_lm", "mlx_vlm"} & configured_backends:
        venv_python = _venv_python(platform_key)
        commands.extend(
            [
                ("uv", "venv", "--python", "3.12", ".venv"),
                (
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    venv_python,
                    _mlx_extra(
                        experts,
                        default_backend=_default_backend(platform_key),
                    ),
                ),
            ]
        )
    if "llama_cpp" in configured_backends:
        if not ({"mlx_lm", "mlx_vlm"} & configured_backends):
            venv_python = _venv_python(platform_key)
            commands.extend(
                [
                    ("uv", "venv", "--python", "3.12", ".venv"),
                    ("uv", "pip", "install", "--python", venv_python, ".[gguf]"),
                ]
            )
        commands.append(_llama_cpp_install_command(platform_key))
    if "ollama" in configured_backends:
        commands.append(("install", "ollama", "from", "https://ollama.com/download"))
    return tuple(commands)


def _expert_runtime_backend(expert: object, default: str) -> str:
    return str(getattr(expert, "params", {}).get("runtime_backend", default))


def _mlx_extra(
    experts: tuple[object, ...],
    *,
    default_backend: str = "mlx_lm",
) -> str:
    runtime_backends = {
        _expert_runtime_backend(expert, default_backend) for expert in experts
    }
    mlx_backends = runtime_backends & {"mlx_lm", "mlx_vlm"}
    if mlx_backends == {"mlx_vlm"}:
        return ".[mlx-vlm]"
    return ".[mlx]"


def _expert_port(expert: object, default: int) -> int:
    base_url = getattr(expert, "base_url", None)
    if not base_url:
        return default
    try:
        parsed = urlparse(str(base_url))
        if parsed.port:
            return int(parsed.port)
    except ValueError:
        return default
    return default


def _ollama_model_name(model: str) -> str:
    if "qwen3-4b" in model.lower() or "qwen3:4b" in model.lower():
        return "qwen3:4b"
    if "qwen3" in model.lower():
        return "qwen3:4b"
    return model
