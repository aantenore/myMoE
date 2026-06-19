from __future__ import annotations

import json
from pathlib import Path

from .context import ContextPolicy


class ContextPolicyError(ValueError):
    """Raised when a context policy file is invalid."""


def load_context_policy(path: str | Path, profile: str = "default") -> ContextPolicy:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContextPolicyError("Context policy file must contain a JSON object.")
    selected = raw.get(profile)
    if selected is None and profile != "default":
        selected = raw.get("default")
    if not isinstance(selected, dict):
        raise ContextPolicyError(f"Context policy profile not found: {profile}")
    return ContextPolicy(
        context_limit_tokens=int(selected.get("context_limit_tokens", 32768)),
        reserved_output_tokens=int(selected.get("reserved_output_tokens", 2048)),
        compaction_trigger_ratio=float(selected.get("compaction_trigger_ratio", 0.75)),
        max_recent_turns=int(selected.get("max_recent_turns", 12)),
        max_memory_items=int(selected.get("max_memory_items", 8)),
    )
