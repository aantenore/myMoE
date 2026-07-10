from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class AgentToolCall:
    """Provider-neutral request emitted by a model for a client-side tool."""

    id: str
    name: str
    arguments: object


@dataclass(frozen=True)
class AgentMessage:
    """Provider-neutral chat message used by the manual agent loop."""

    role: str
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()


@dataclass(frozen=True)
class AgentModelUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class AgentModelOutput:
    """Normalized model output: either a final answer or one or more tool calls."""

    final_answer: str | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()
    assistant_content: str = ""
    model: str = ""
    usage: AgentModelUsage = field(default_factory=AgentModelUsage)


@dataclass(frozen=True)
class AgentToolSpec:
    """Strict model-visible contract mapped to one canonical local tool name."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    risk_class: str
    side_effects: str
    model_name: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.exposed_name):
            raise ValueError(
                f"Model-visible tool name must match [A-Za-z0-9_-] and be <= 64 chars: "
                f"{self.exposed_name}"
            )
        if self.input_schema.get("type") != "object":
            raise ValueError(f"Tool {self.name} input schema must have type=object")
        if self.input_schema.get("additionalProperties") is not False:
            raise ValueError(
                f"Tool {self.name} input schema must reject unknown root properties"
            )
        if not self.risk_class.strip():
            raise ValueError(f"Tool {self.name} requires a risk class")

    @property
    def exposed_name(self) -> str:
        # OpenAI-compatible function names commonly reject dots. Double
        # underscores keep the mapping readable and reversible by the registry.
        return self.model_name or self.name.replace(".", "__")

    def openai_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": (
                    f"{self.description} Risk: {self.risk_class}. "
                    f"Side effects: {self.side_effects}. "
                    "Do not add confirmation fields; the trusted harness handles approvals."
                ),
                "parameters": dict(self.input_schema),
            },
        }


class AgentModelAdapter(Protocol):
    """Stable adapter boundary for provider-specific model APIs."""

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[AgentToolSpec],
        *,
        correlation_id: str,
    ) -> AgentModelOutput: ...
