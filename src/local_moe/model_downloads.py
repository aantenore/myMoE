from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MoEConfig


@dataclass(frozen=True)
class ModelDownloadRequest:
    kind: str
    backend: str
    model: str
    repo_id: str | None = None
    allow_patterns: tuple[str, ...] = ()
    command: tuple[str, ...] = ()


def build_model_download_requests(
    config: MoEConfig,
    default_backend: str,
) -> tuple[ModelDownloadRequest, ...]:
    requests: list[ModelDownloadRequest] = []
    seen: set[tuple[str, str]] = set()
    for expert in config.experts:
        if expert.provider != "openai_compatible":
            continue
        backend = _expert_backend(expert, default_backend)
        key = (backend, expert.model)
        if key in seen:
            continue
        seen.add(key)

        if backend in {"mlx_lm", "mlx_vlm"}:
            requests.append(
                ModelDownloadRequest(
                    kind="huggingface_snapshot",
                    backend=backend,
                    model=expert.model,
                    repo_id=expert.model,
                )
            )
            continue

        if backend == "llama_cpp":
            requests.append(_llama_cpp_request(expert.model, backend))
            continue

        if backend == "ollama":
            requests.append(
                ModelDownloadRequest(
                    kind="ollama_pull",
                    backend=backend,
                    model=expert.model,
                    command=("ollama", "pull", _ollama_model_name(expert.model)),
                )
            )
            continue

        raise ValueError(f"Unsupported runtime backend for model download: {backend}")

    return tuple(requests)


def _expert_backend(expert: object, default_backend: str) -> str:
    backend = str(getattr(expert, "params", {}).get("runtime_backend") or default_backend)
    if backend == "mixed":
        raise ValueError("Mixed runtime downloads require each expert to declare params.runtime_backend.")
    return backend


def _llama_cpp_request(model: str, backend: str) -> ModelDownloadRequest:
    if _looks_like_local_model_path(model):
        return ModelDownloadRequest(kind="local_file", backend=backend, model=model)

    repo_id, selector = _split_huggingface_selector(model)
    patterns = _gguf_allow_patterns(selector)
    return ModelDownloadRequest(
        kind="huggingface_snapshot",
        backend=backend,
        model=model,
        repo_id=repo_id,
        allow_patterns=patterns,
    )


def _looks_like_local_model_path(model: str) -> bool:
    value = model.strip()
    if value.endswith(".gguf"):
        return True
    if len(value) > 2 and value[1:3] in {":\\", ":/"}:
        return True
    return value.startswith(("./", "../", "/", "~"))


def _split_huggingface_selector(model: str) -> tuple[str, str | None]:
    if ":" not in model:
        return model, None
    repo_id, selector = model.rsplit(":", 1)
    if "/" not in repo_id:
        return model, None
    return repo_id, selector or None


def _gguf_allow_patterns(selector: str | None) -> tuple[str, ...]:
    if not selector:
        return ("*.gguf",)
    if selector.endswith(".gguf"):
        return (selector,)
    return (f"*{selector}*.gguf",)


def _ollama_model_name(model: str) -> str:
    lower = model.lower()
    if "qwen3-4b" in lower or "qwen3:4b" in lower:
        return "qwen3:4b"
    if "qwen3" in lower:
        return "qwen3:4b"
    return model


def validate_local_file_request(request: ModelDownloadRequest) -> None:
    if request.kind != "local_file":
        return
    path = Path(request.model).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configured local model file does not exist: {path}")
