from __future__ import annotations

from dataclasses import dataclass
import platform
import shutil
from typing import Any, Callable
from urllib import error
from urllib.parse import urlparse

from .config import MoEConfig
from .http_boundary import open_model_endpoint


@dataclass(frozen=True)
class RuntimePlan:
    platform_key: str
    backend: str
    install_commands: tuple[tuple[str, ...], ...]
    model_commands: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...]


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


def build_runtime_plan(config: MoEConfig, preferred_backends: dict[str, str] | None = None) -> RuntimePlan:
    platform_key = detect_platform_key()
    preferred = preferred_backends or {}
    experts = tuple(expert for expert in config.experts if expert.provider == "openai_compatible")
    configured_backends = {
        str(expert.params.get("runtime_backend"))
        for expert in experts
        if expert.params.get("runtime_backend")
    }
    backend = _runtime_backend(configured_backends, preferred, platform_key)
    models = tuple(expert.model for expert in experts)

    if backend == "mixed":
        venv_python = _venv_python(platform_key)
        install = _mixed_install_commands(configured_backends, platform_key, experts)
        commands = tuple(
            _expert_server_command(
                expert,
                _expert_runtime_backend(expert, _default_backend(platform_key)),
                _expert_port(expert, 8101 + index),
                venv_python,
            )
            for index, expert in enumerate(experts)
        )
        notes = ("Mixed runtime plan. Keep only the models needed for the current workflow resident.",)
    elif backend in {"mlx_lm", "mlx_vlm"}:
        venv_python = _venv_python(platform_key)
        install = (
            ("uv", "venv", "--python", "3.12", ".venv"),
            ("uv", "pip", "install", "--python", venv_python, _mlx_extra(experts)),
        )
        commands = tuple(_mlx_server_command(venv_python, expert, _expert_port(expert, 8101 + index)) for index, expert in enumerate(experts))
        notes = (
            "Best path for Apple Silicon. Keep one heavy model resident; add small fallback only if memory allows.",
            "Gemma 4 E4B is validated with the pinned .[mlx] profile because newer MLX packages can reject its current artifact.",
        )
    elif backend == "ollama":
        install = (("install", "ollama", "from", "https://ollama.com/download"),)
        commands = tuple(("ollama", "pull", _ollama_model_name(model)) for model in models)
        notes = ("Cross-platform fallback for Windows/Linux/macOS. Uses Ollama OpenAI-compatible API at port 11434.",)
    else:
        venv_python = _venv_python(platform_key)
        install = (
            ("uv", "venv", "--python", "3.12", ".venv"),
            ("uv", "pip", "install", "--python", venv_python, ".[gguf]"),
            _llama_cpp_install_command(platform_key),
        )
        commands = tuple(_llama_server_command(expert, _expert_port(expert, 8101 + index)) for index, expert in enumerate(experts))
        notes = ("GGUF backend through llama.cpp. Prefer quantized models that fit local RAM/VRAM headroom.",)

    return RuntimePlan(
        platform_key=platform_key,
        backend=backend,
        install_commands=install,
        model_commands=commands,
        notes=notes,
    )


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
            "python": shutil.which("python") is not None or shutil.which("python3") is not None,
        },
        "install_commands": [list(item) for item in plan.install_commands],
        "model_commands": [list(item) for item in plan.model_commands],
        "notes": list(plan.notes),
    }


def _default_backend(platform_key: str) -> str:
    if platform_key == "darwin_arm64":
        return "mlx_lm"
    if platform_key in {"windows", "linux"}:
        return "ollama"
    return "llama_cpp"


def _runtime_backend(configured_backends: set[str], preferred: dict[str, str], platform_key: str) -> str:
    if len(configured_backends) == 1:
        return next(iter(configured_backends))
    if len(configured_backends) > 1:
        return "mixed"
    return preferred.get(platform_key) or preferred.get("fallback") or _default_backend(platform_key)


def _venv_python(platform_key: str) -> str:
    if platform_key == "windows":
        return ".venv\\Scripts\\python.exe"
    return ".venv/bin/python"


def _mlx_server_command(venv_python: str, expert: object, port: int) -> tuple[str, ...]:
    runtime_backend = str(getattr(expert, "params", {}).get("runtime_backend", "mlx_lm"))
    module = "mlx_vlm.server" if runtime_backend == "mlx_vlm" else "mlx_lm.server"
    command = [
        venv_python,
        "-m",
        module,
        "--model",
        str(expert.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    max_kv_size = getattr(expert, "params", {}).get("max_kv_size")
    if max_kv_size is not None and runtime_backend == "mlx_vlm":
        command.extend(["--max-kv-size", str(max_kv_size)])
    return tuple(command)


def _expert_server_command(
    expert: object,
    backend: str,
    port: int,
    venv_python: str,
) -> tuple[str, ...]:
    if backend in {"mlx_lm", "mlx_vlm"}:
        return _mlx_server_command(venv_python, expert, port)
    if backend == "llama_cpp":
        return _llama_server_command(expert, port)
    if backend == "ollama":
        return ("ollama", "pull", _ollama_model_name(str(expert.model)))
    return _llama_server_command(expert, port)


def _llama_server_command(expert: object, port: int) -> tuple[str, ...]:
    command = [
        "llama-server",
        "-hf",
        str(expert.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    params = getattr(expert, "params", {})
    context_size = params.get("context_size")
    if context_size is not None:
        command.extend(["--ctx-size", str(context_size)])
    gpu_layers = params.get("gpu_layers")
    if gpu_layers is not None:
        command.extend(["--n-gpu-layers", str(gpu_layers)])
    return tuple(command)


def _llama_cpp_install_command(platform_key: str) -> tuple[str, ...]:
    if platform_key == "windows":
        return ("install", "llama.cpp", "from", "https://github.com/ggml-org/llama.cpp/releases")
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
                ("uv", "pip", "install", "--python", venv_python, _mlx_extra(experts)),
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


def _mlx_extra(experts: tuple[object, ...]) -> str:
    runtime_backends = {str(getattr(expert, "params", {}).get("runtime_backend", "mlx_lm")) for expert in experts}
    if runtime_backends == {"mlx_vlm"}:
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
