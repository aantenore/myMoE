from __future__ import annotations

from dataclasses import dataclass
import json
import platform
import subprocess
from pathlib import Path


@dataclass(frozen=True)
class HardwareProfile:
    machine: str
    cpu_brand: str
    memory_bytes: int
    memory_gib: float
    recommended_strategy: str
    rationale: tuple[str, ...]


def detect_hardware() -> HardwareProfile:
    machine = platform.machine()
    cpu_brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    mem_raw = _run(["sysctl", "-n", "hw.memsize"])
    memory_bytes = int(mem_raw) if mem_raw and mem_raw.isdigit() else 0
    memory_gib = memory_bytes / (1024**3) if memory_bytes else 0.0
    strategy, rationale = recommend_strategy(machine, cpu_brand, memory_gib)
    return HardwareProfile(
        machine=machine,
        cpu_brand=cpu_brand,
        memory_bytes=memory_bytes,
        memory_gib=round(memory_gib, 2),
        recommended_strategy=strategy,
        rationale=tuple(rationale),
    )


def recommend_strategy(machine: str, cpu_brand: str, memory_gib: float) -> tuple[str, list[str]]:
    rationale: list[str] = []
    apple_silicon = machine == "arm64" and "Apple" in cpu_brand
    if apple_silicon:
        rationale.append("Apple Silicon detected: llama.cpp Metal is the preferred runtime.")

    if memory_gib >= 48:
        rationale.append("Memory is high enough for larger sparse MoE candidates.")
        return "moe_or_large_a3b", rationale
    if memory_gib >= 24:
        rationale.append("24 GiB class memory: use a MoE harness with one strong resident general expert.")
        rationale.append("Run small summarizer/router experts resident; cold-load large specialists only when eval wins.")
        return "general_purpose_moe_single_resident", rationale
    if memory_gib >= 16:
        rationale.append("16 GiB class memory: prefer 1.5B-7B dense experts and avoid multi-model residency.")
        return "small_single_expert", rationale

    rationale.append("Memory is limited: use tiny models and remote teacher only for offline data generation.")
    return "tiny_expert_only", rationale


def write_hardware_report(path: str | Path) -> HardwareProfile:
    profile = detect_hardware()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(profile.__dict__, indent=2), encoding="utf-8")
    return profile


def _run(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return ""
    return completed.stdout.strip()
