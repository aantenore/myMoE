from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .app_config import AppConfig
from .config import ExpertConfig, MoEConfig, load_config
from .hardware import HardwareProfile, detect_hardware
from .setup_status import inspect_setup_status, setup_status_payload

DEFAULT_CANDIDATE_PATHS = (
    Path("configs/model-candidates.json"),
    Path("configs/model-benchmark.json"),
)


def discover_config_profiles(
    *,
    active_config_path: str,
    app_config: AppConfig,
    app_config_path: str = "configs/app.json",
    config_dir: str | Path = "configs",
    hardware_profile: HardwareProfile | None = None,
    candidate_paths: tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    """Return read-only metadata for runnable local MoE profiles."""

    paths = list(_candidate_paths(config_dir))
    active_path = Path(active_config_path)
    if active_path.exists() and not _contains_path(paths, active_path):
        paths.insert(0, active_path)
    hardware = hardware_profile or detect_hardware()
    candidate_index = _candidate_index(DEFAULT_CANDIDATE_PATHS if candidate_paths is None else candidate_paths)

    profiles = [
        _profile_payload(
            path,
            active_config_path=active_config_path,
            default_config_path=app_config.default_moe_config,
            app_config=app_config,
            app_config_path=app_config_path,
            hardware_profile=hardware,
            candidate_index=candidate_index,
        )
        for path in paths
    ]
    recommendation = _profile_recommendation(profiles)
    recommended_path = recommendation.get("profile_path")
    if recommended_path:
        for profile in profiles:
            profile["recommended"] = profile.get("path") == recommended_path
    return {
        "schema_version": "1.0",
        "active_config_path": _display_path(active_config_path),
        "default_config_path": _display_path(app_config.default_moe_config),
        "config_dir": _display_path(config_dir),
        "count": len(profiles),
        "hardware": _hardware_payload(hardware),
        "recommendation": recommendation,
        "profiles": profiles,
    }


def recommend_config_profile(
    *,
    active_config_path: str,
    app_config: AppConfig,
    app_config_path: str = "configs/app.json",
    config_dir: str | Path = "configs",
    hardware_profile: HardwareProfile | None = None,
    candidate_paths: tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    profiles = discover_config_profiles(
        active_config_path=active_config_path,
        app_config=app_config,
        app_config_path=app_config_path,
        config_dir=config_dir,
        hardware_profile=hardware_profile,
        candidate_paths=candidate_paths,
    )
    return {
        "schema_version": "1.0",
        "active_config_path": profiles["active_config_path"],
        "default_config_path": profiles["default_config_path"],
        "hardware": profiles["hardware"],
        "profiles_considered": profiles["count"],
        "recommendation": profiles["recommendation"],
    }


def _candidate_paths(config_dir: str | Path) -> tuple[Path, ...]:
    root = Path(config_dir)
    if not root.exists():
        return ()
    paths = [
        path
        for path in root.glob("*.json")
        if path.name.startswith(("moe.", "single."))
    ]
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _profile_payload(
    path: Path,
    *,
    active_config_path: str,
    default_config_path: str,
    app_config: AppConfig,
    app_config_path: str,
    hardware_profile: HardwareProfile,
    candidate_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    display_path = _display_path(path)
    payload: dict[str, Any] = {
        "path": display_path,
        "name": path.stem,
        "active": _same_path(path, active_config_path),
        "default": _same_path(path, default_config_path),
        "status": "invalid",
        "error": "",
    }
    try:
        config = load_config(path)
        setup = inspect_setup_status(
            display_path,
            config,
            app_config,
            app_config_path=app_config_path,
        )
    except Exception as exc:
        payload["error"] = str(exc)
        return payload

    setup_payload = setup_status_payload(setup)
    model_status_counts = Counter(str(item["status"]) for item in setup_payload["models"])
    runtime_backends = sorted(
        {
            str(expert.params.get("runtime_backend") or "provider_default")
            for expert in config.experts
        }
    )
    payload.update(
        {
            "status": "valid",
            "expert_count": len(config.experts),
            "provider_count": len({expert.provider for expert in config.experts}),
            "backend": setup.runtime_plan.backend,
            "runtime_backends": runtime_backends,
            "routing": {
                "strategy": config.routing.strategy,
                "aggregation": config.routing.aggregation,
                "top_k": config.routing.top_k,
                "semantic": config.routing.semantic.enabled,
                "distilled": config.routing.distilled.enabled,
            },
            "experts": [
                {
                    "id": expert.id,
                    "provider": expert.provider,
                    "model": expert.model,
                    "role": expert.role,
                    "runtime_backend": str(expert.params.get("runtime_backend") or "provider_default"),
                    "base_url": expert.base_url,
                }
                for expert in config.experts
            ],
            "setup": {
                "status": setup_payload["status"],
                "model_count": len(setup_payload["models"]),
                "model_status_counts": dict(sorted(model_status_counts.items())),
                "download_command_display": setup_payload["download_command_display"],
                "error": setup_payload["error"],
            },
            "hardware_fit": _hardware_fit(config, hardware_profile, candidate_index),
            "launch_commands": _launch_commands(
                display_path,
                app_config_path=app_config_path,
            ),
        }
    )
    return payload


def _launch_commands(config_path: str, *, app_config_path: str) -> list[dict[str, Any]]:
    python = ".venv/bin/python"
    env = {"PYTHONPATH": "src"}
    commands = [
        {
            "id": "inspect_setup",
            "label": "Inspect setup",
            "description": "Preview setup readiness for this profile without side effects.",
            "argv": [
                python,
                "-m",
                "local_moe.cli",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--setup",
            ],
            "side_effects": "none",
            "requires_confirmation": False,
        },
        {
            "id": "prepare_runtime",
            "label": "Prepare runtime",
            "description": "Install runtime dependencies and download configured model assets.",
            "argv": [
                python,
                "scripts/bootstrap_runtime.py",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--execute",
                "--download-models",
            ],
            "side_effects": "installs_dependencies_and_downloads_models",
            "requires_confirmation": True,
        },
        {
            "id": "start_models",
            "label": "Start models",
            "description": "Start the model servers configured by this profile in the foreground.",
            "argv": [
                python,
                "scripts/start_local_models.py",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
            ],
            "side_effects": "starts_local_model_processes",
            "requires_confirmation": True,
        },
        {
            "id": "start_ui",
            "label": "Start UI",
            "description": "Run the web UI with this profile.",
            "argv": [
                python,
                "-m",
                "local_moe.web",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--port",
                "8089",
            ],
            "side_effects": "starts_local_web_server",
            "requires_confirmation": False,
        },
        {
            "id": "open_cli",
            "label": "Open CLI",
            "description": "Open an interactive CLI session with this profile.",
            "argv": [
                python,
                "-m",
                "local_moe.cli",
                "--app-config",
                app_config_path,
                "--config",
                config_path,
                "--interactive",
            ],
            "side_effects": "starts_interactive_cli",
            "requires_confirmation": False,
        },
    ]
    return [{**command, "env": env, "display": _display_command(command["argv"], env=env)} for command in commands]


def _profile_recommendation(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    invalid_count = 0
    for profile in profiles:
        if profile.get("status") != "valid":
            invalid_count += 1
            continue
        score, rationale = _profile_score(profile)
        scored.append(
            {
                "score": score,
                "profile": profile,
                "rationale": rationale,
            }
        )

    if not scored:
        return {
            "status": "unavailable",
            "profile_path": "",
            "profile_name": "",
            "score": 0,
            "confidence": 0,
            "reason": "No valid runtime profiles were found.",
            "rationale": [f"{invalid_count} invalid profile(s) skipped."] if invalid_count else [],
            "requires_setup": False,
            "next_actions": [],
            "alternatives": [],
        }

    scored.sort(
        key=lambda item: (
            item["score"],
            bool(item["profile"].get("active")),
            bool(item["profile"].get("default")),
            str(item["profile"].get("path", "")),
        ),
        reverse=True,
    )
    selected = scored[0]
    profile = selected["profile"]
    setup_status = str(profile.get("setup", {}).get("status") or "unknown")
    fit_status = str(profile.get("hardware_fit", {}).get("status") or "unknown")
    status = _recommendation_status(setup_status, fit_status)
    second_score = scored[1]["score"] if len(scored) > 1 else selected["score"]
    return {
        "status": status,
        "profile_path": profile.get("path", ""),
        "profile_name": profile.get("name", ""),
        "score": selected["score"],
        "confidence": _recommendation_confidence(selected["score"], second_score),
        "reason": _recommendation_reason(status, setup_status, fit_status, profile),
        "setup_status": setup_status,
        "hardware_fit_status": fit_status,
        "active": bool(profile.get("active")),
        "default": bool(profile.get("default")),
        "requires_setup": setup_status != "ready",
        "rationale": selected["rationale"],
        "next_actions": _recommendation_actions(profile, setup_status),
        "alternatives": [
            {
                "profile_path": item["profile"].get("path", ""),
                "profile_name": item["profile"].get("name", ""),
                "score": item["score"],
                "setup_status": item["profile"].get("setup", {}).get("status") or "unknown",
                "hardware_fit_status": item["profile"].get("hardware_fit", {}).get("status") or "unknown",
            }
            for item in scored[1:4]
        ],
    }


def _profile_score(profile: dict[str, Any]) -> tuple[int, list[str]]:
    setup_status = str(profile.get("setup", {}).get("status") or "unknown")
    fit_status = str(profile.get("hardware_fit", {}).get("status") or "unknown")
    routing = profile.get("routing", {})
    score = 0
    rationale: list[str] = []

    fit_scores = {
        "recommended": 90,
        "fits": 78,
        "compatible": 70,
        "stretch": 35,
        "unknown": 18,
        "too_large": -240,
    }
    fit_score = fit_scores.get(fit_status, 0)
    score += fit_score
    rationale.append(f"hardware_fit={fit_status} contributed {fit_score} point(s).")

    setup_scores = {
        "ready": 45,
        "needs_setup": 12,
    }
    setup_score = setup_scores.get(setup_status, 0)
    score += setup_score
    rationale.append(f"setup={setup_status} contributed {setup_score} point(s).")

    if _has_general_expert(profile):
        score += 14
        rationale.append("A general-purpose expert is present.")
    if profile.get("expert_count", 0) > 1:
        score += 5
        rationale.append("Multiple experts are configured for fallback or specialization.")
    if routing.get("distilled"):
        score += 6
        rationale.append("Distilled local routing is enabled.")
    if routing.get("semantic"):
        score += 4
        rationale.append("Multilingual semantic routing is enabled.")
    if profile.get("active"):
        score += 3
        rationale.append("The profile is currently active.")
    if profile.get("default"):
        score += 2
        rationale.append("The profile is the configured default.")
    return score, rationale


def _has_general_expert(profile: dict[str, Any]) -> bool:
    for expert in profile.get("experts", []):
        role = str(expert.get("role") or "").lower()
        expert_id = str(expert.get("id") or "").lower()
        if "general" in role or expert_id == "general":
            return True
    return False


def _recommendation_status(setup_status: str, fit_status: str) -> str:
    if fit_status == "too_large":
        return "unavailable"
    if setup_status != "ready":
        return "needs_setup"
    return "ready"


def _recommendation_reason(
    status: str,
    setup_status: str,
    fit_status: str,
    profile: dict[str, Any],
) -> str:
    name = str(profile.get("name") or profile.get("path") or "profile")
    if status == "ready":
        return f"{name} is the best ready local profile for the detected machine."
    if status == "needs_setup":
        return f"{name} is the best fit, but setup is {setup_status}; prepare runtime before use."
    return f"No recommended profile is currently usable because the best candidate is {fit_status}."


def _recommendation_confidence(score: int, second_score: int) -> float:
    if score <= 0:
        return 0.0
    if score == second_score:
        return 0.5
    margin = max(0, score - second_score)
    return round(min(0.99, 0.55 + margin / 100.0), 2)


def _recommendation_actions(profile: dict[str, Any], setup_status: str) -> list[dict[str, Any]]:
    wanted = {"inspect_setup", "prepare_runtime"} if setup_status != "ready" else {"start_models", "start_ui", "open_cli"}
    return [
        command
        for command in profile.get("launch_commands", [])
        if command.get("id") in wanted
    ]


def _hardware_payload(profile: HardwareProfile) -> dict[str, Any]:
    return {
        "machine": profile.machine,
        "cpu_brand": profile.cpu_brand,
        "memory_bytes": profile.memory_bytes,
        "memory_gib": profile.memory_gib,
        "recommended_strategy": profile.recommended_strategy,
        "rationale": list(profile.rationale),
    }


def build_hardware_fit(
    config: MoEConfig,
    *,
    hardware_profile: HardwareProfile | None = None,
    candidate_paths: tuple[str | Path, ...] | None = None,
) -> dict[str, Any]:
    hardware = hardware_profile or detect_hardware()
    candidate_index = _candidate_index(DEFAULT_CANDIDATE_PATHS if candidate_paths is None else candidate_paths)
    return _hardware_fit(config, hardware, candidate_index)


def _hardware_fit(
    config: MoEConfig,
    hardware_profile: HardwareProfile,
    candidate_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if all(expert.provider == "synthetic" for expert in config.experts):
        return {
            "status": "compatible",
            "summary": "Synthetic fixture; no local model memory required.",
            "estimated_memory_gb": 0.0,
            "memory_gib": hardware_profile.memory_gib,
            "headroom_gb": hardware_profile.memory_gib,
            "resident_large_experts": 0,
            "matched_models": [],
            "unknown_models": [],
            "rationale": ["Synthetic providers are confined to tests and do not allocate model weights."],
        }

    estimates = [_expert_fit(expert, candidate_index) for expert in config.experts]
    known = [item for item in estimates if item["estimated_memory_gb"] is not None]
    unknown = [item for item in estimates if item["estimated_memory_gb"] is None]
    matched = [item for item in estimates if item["source"] != "heuristic" and item["estimated_memory_gb"] is not None]
    memory_gib = float(hardware_profile.memory_gib or 0.0)
    estimated_memory_gb = round(sum(float(item["estimated_memory_gb"]) for item in known), 2)
    headroom_gb = round(memory_gib - estimated_memory_gb, 2) if memory_gib else None
    large_experts = sum(1 for item in known if float(item["estimated_memory_gb"]) >= 12.0)

    rationale: list[str] = []
    if matched:
        rationale.append(f"Matched {len(matched)} expert model estimate(s) from the model candidate manifests.")
    heuristic = [item for item in estimates if item["source"] == "heuristic"]
    if heuristic:
        rationale.append(f"Estimated {len(heuristic)} expert model(s) from model name and backend heuristics.")
    if unknown:
        rationale.append(f"{len(unknown)} expert model(s) have no memory estimate; run a benchmark before relying on this profile.")
    if large_experts > 1:
        rationale.append("More than one large resident expert is configured; prefer cold-loading specialists on 24 GiB class machines.")

    status = _fit_status(
        estimated_memory_gb=estimated_memory_gb,
        memory_gib=memory_gib,
        resident_large_experts=large_experts,
        unknown_models=len(unknown),
        estimates=estimates,
        strategy=hardware_profile.recommended_strategy,
    )
    summary = _fit_summary(status, estimated_memory_gb, memory_gib, large_experts, len(unknown))
    return {
        "status": status,
        "summary": summary,
        "estimated_memory_gb": estimated_memory_gb if known else None,
        "memory_gib": hardware_profile.memory_gib,
        "headroom_gb": headroom_gb,
        "resident_large_experts": large_experts,
        "matched_models": [
            {
                "expert_id": item["expert_id"],
                "model": item["model"],
                "candidate_id": item.get("candidate_id"),
                "estimated_memory_gb": item["estimated_memory_gb"],
                "role": item.get("role", ""),
                "source": item["source"],
            }
            for item in known
        ],
        "unknown_models": [{"expert_id": item["expert_id"], "model": item["model"]} for item in unknown],
        "rationale": rationale,
    }


def _fit_status(
    *,
    estimated_memory_gb: float,
    memory_gib: float,
    resident_large_experts: int,
    unknown_models: int,
    estimates: list[dict[str, Any]],
    strategy: str,
) -> str:
    if memory_gib <= 0 or unknown_models:
        return "unknown"
    if any(str(item.get("role", "")).lower() == "not_viable_on_24gb" for item in estimates):
        return "too_large"
    if any(float(item.get("minimum_memory_gb") or 0.0) > memory_gib for item in estimates):
        return "too_large"
    if estimated_memory_gb > memory_gib * 1.12:
        return "too_large"
    if estimated_memory_gb > memory_gib or resident_large_experts > 1:
        return "stretch"
    if strategy == "general_purpose_moe_single_resident" and resident_large_experts == 1:
        return "recommended"
    return "fits"


def _fit_summary(
    status: str,
    estimated_memory_gb: float,
    memory_gib: float,
    resident_large_experts: int,
    unknown_models: int,
) -> str:
    if status == "compatible":
        return "No local model memory required."
    if status == "unknown":
        return f"{unknown_models} model estimate(s) missing; benchmark before using this profile."
    memory = f"{estimated_memory_gb:.1f} GiB estimated / {memory_gib:.1f} GiB RAM"
    if status == "recommended":
        return f"Recommended for this machine: {memory}, {resident_large_experts} large resident expert."
    if status == "fits":
        return f"Fits this machine: {memory}."
    if status == "stretch":
        return f"Stretch profile: {memory}; monitor memory and avoid extra resident specialists."
    return f"Too large for this machine: {memory}."


def _expert_fit(expert: ExpertConfig, candidate_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = _match_candidate(expert.model, candidate_index)
    if candidate:
        return {
            "expert_id": expert.id,
            "model": expert.model,
            "candidate_id": candidate.get("id", ""),
            "role": candidate.get("role", ""),
            "minimum_memory_gb": candidate.get("minimum_memory_gb"),
            "estimated_memory_gb": candidate["estimated_memory_gb"],
            "source": candidate["source"],
        }
    estimate = _heuristic_memory_estimate(expert)
    return {
        "expert_id": expert.id,
        "model": expert.model,
        "candidate_id": "",
        "role": expert.role,
        "minimum_memory_gb": None,
        "estimated_memory_gb": estimate,
        "source": "heuristic" if estimate is not None else "unknown",
    }


def _match_candidate(model: str, candidate_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    keys = _model_keys(model)
    for key in keys:
        if key in candidate_index:
            return candidate_index[key]
    return None


def _candidate_index(paths: tuple[str | Path, ...]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in raw.get("candidates", []):
            estimate = _candidate_memory_estimate(item)
            if estimate is None:
                continue
            candidate = {
                "id": str(item.get("id", "")),
                "role": str(item.get("role", "")),
                "repo": str(item.get("repo", "")),
                "minimum_memory_gb": _float_or_none(item.get("minimum_memory_gb")),
                "estimated_memory_gb": estimate,
                "source": "manifest",
            }
            for key in _candidate_keys(item):
                index.setdefault(key, candidate)
    return index


def _candidate_memory_estimate(item: dict[str, Any]) -> float | None:
    for key in ("estimated_memory_gb", "recommended_memory_gb", "minimum_memory_gb"):
        value = _float_or_none(item.get(key))
        if value is not None:
            return round(value, 2)
    approx_size = _float_or_none(item.get("approx_size_gb"))
    if approx_size is not None:
        return round(approx_size + 2.5, 2)
    return None


def _candidate_keys(item: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for value in (item.get("id"), item.get("repo"), item.get("label"), item.get("config")):
        if value:
            keys.update(_model_keys(str(value)))
    repo = str(item.get("repo") or "")
    file_alias = str(item.get("file_alias") or "")
    if repo and file_alias:
        keys.update(_model_keys(f"{repo}:{file_alias}"))
    return {key for key in keys if key}


def _model_keys(model: str) -> set[str]:
    normalized = _normalize_model_key(model)
    keys = {normalized}
    if "/" in normalized and ":" in normalized:
        keys.add(normalized.split(":", 1)[0])
    return keys


def _normalize_model_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _heuristic_memory_estimate(expert: ExpertConfig) -> float | None:
    model = expert.model.lower()
    if expert.provider == "synthetic":
        return 0.0
    e_match = re.search(r"e(\d+(?:\.\d+)?)b", model)
    b_match = re.search(r"(\d+(?:\.\d+)?)b", model)
    size_b = float((e_match or b_match).group(1)) if (e_match or b_match) else None
    if size_b is None:
        return None
    backend = str(expert.params.get("runtime_backend") or "").lower()
    if "q4" in model or "4bit" in model or "4-bit" in model:
        factor = 0.58
    else:
        factor = 1.2
    overhead = 2.0 if backend == "mlx_lm" else 1.5
    return round(max(1.0, size_b * factor + overhead), 2)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains_path(paths: list[Path], target: Path) -> bool:
    return any(_same_path(path, target) for path in paths)


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left).as_posix() == Path(right).as_posix()


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _display_command(argv: list[str], *, env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={value}" for key, value in env.items())
    body = " ".join(_quote_arg(item) for item in argv)
    return f"{prefix} {body}".strip()


def _quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return json.dumps(value)
    return value
