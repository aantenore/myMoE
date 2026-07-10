from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from typing import Sequence
import unittest
import warnings

from local_moe.agent_loop import AgentLoop, AgentLoopBudget, build_local_agent_loop
from local_moe.agent_tools import (
    AgentToolRegistry,
    ApprovalDecision,
)
from local_moe.agent_types import (
    AgentMessage,
    AgentModelOutput,
    AgentToolCall,
    AgentToolSpec,
)
from local_moe.config import parse_config
from local_moe.extensions import load_extension_registry
from local_moe.memory import FileMemoryStore
from local_moe.tool_runner import LocalToolRunner, ToolRunResult


class ScriptedModel:
    def __init__(self, outputs: Sequence[AgentModelOutput]):
        self.outputs = list(outputs)
        self.requests: list[
            tuple[tuple[AgentMessage, ...], tuple[AgentToolSpec, ...]]
        ] = []
        self.timeouts: list[float | None] = []

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[AgentToolSpec],
        *,
        correlation_id: str,
        timeout_seconds: float | None = None,
    ) -> AgentModelOutput:
        self.requests.append((tuple(messages), tuple(tools)))
        self.timeouts.append(timeout_seconds)
        return self.outputs.pop(0)


class GroundingModel:
    def __init__(self):
        self.turn = 0
        self.observation: dict[str, object] | None = None

    def generate(
        self,
        messages: Sequence[AgentMessage],
        tools: Sequence[AgentToolSpec],
        *,
        correlation_id: str,
        timeout_seconds: float | None = None,
    ) -> AgentModelOutput:
        self.turn += 1
        if self.turn == 1:
            return AgentModelOutput(
                tool_calls=(
                    AgentToolCall(
                        id="memory-call-1",
                        name="memory__search",
                        arguments={"query": "configurable routing", "scope": "project"},
                    ),
                ),
                model="scripted",
            )
        self.observation = json.loads(messages[-1].content)
        records = self.observation["data"]["records"]  # type: ignore[index]
        evidence = records[0]["text"]  # type: ignore[index]
        return AgentModelOutput(
            final_answer=f"Grounded evidence: {evidence}",
            model="scripted",
        )


class RecordingRunner:
    def __init__(self, payload: dict[str, object] | None = None):
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.timeouts: list[float | None] = []
        self.payload = payload or {"value": "ok"}

    def run(
        self,
        name: str,
        payload: dict[str, object] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolRunResult:
        self.calls.append((name, dict(payload or {})))
        self.timeouts.append(timeout_seconds)
        return ToolRunResult(
            name=name,
            status="ok",
            risk_class="read_only",
            side_effects="none",
            message="Completed.",
            payload=self.payload,
        )


class InvalidResultRunner:
    def run(self, name, payload=None, *, timeout_seconds=None):
        return object()


def test_soft_wall_time_is_propagated_and_deprecated_max_alias_is_supported() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall("call-1", "read__value", {"query": "value"}),
                )
            ),
            AgentModelOutput(final_answer="Done."),
        )
    )
    result = AgentLoop(
        model,
        registry,
        budget=AgentLoopBudget(soft_wall_time_seconds=1.0),
    ).run("Read a value.")

    assert result.status == "completed"
    assert all(timeout is not None and 0 < timeout <= 1 for timeout in model.timeouts)
    assert len(runner.timeouts) == 1
    assert runner.timeouts[0] is not None
    assert 0 < runner.timeouts[0] <= 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = AgentLoopBudget(max_wall_time_seconds=2.0)
    assert legacy.soft_wall_time_seconds == 2.0
    assert any(item.category is DeprecationWarning for item in caught)


def test_final_answer_is_grounded_in_structured_local_tool_result() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.jsonl"
        FileMemoryStore(memory_path).add(
            "Routing remains configurable through expert profiles.",
            scope="project",
        )
        runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)
        registry = AgentToolRegistry.from_local_tools(
            runner,
            load_extension_registry(),
            visible_tools=("memory.search",),
        )
        model = GroundingModel()

        result = AgentLoop(model, registry).run(
            "Use local memory to explain how routing is configured.",
            correlation_id="run-grounded",
        )

    assert result.status == "completed"
    assert result.final_answer == (
        "Grounded evidence: Routing remains configurable through expert profiles."
    )
    assert result.grounded_in_tool_results is True
    assert result.grounded_tool_call_ids == ("memory-call-1",)
    assert model.observation is not None
    assert model.observation["status"] == "success"
    assert result.tool_results[0].status == "success"

    trace_json = json.dumps([asdict(event) for event in result.trace])
    assert "Use local memory" not in trace_json
    assert "Routing remains configurable" not in trace_json
    assert "prompt_sha256" in trace_json
    assert "arguments_sha256" in trace_json


def test_unknown_tool_gets_structured_result_and_model_can_recover() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(AgentToolCall("call-1", "not_registered", {}),),
                model="scripted",
            ),
            AgentModelOutput(
                final_answer="That capability is unavailable.", model="scripted"
            ),
        )
    )

    result = AgentLoop(model, registry).run("Try an unavailable capability.")

    assert result.status == "completed"
    assert result.tool_results[0].code == "unknown_tool"
    assert result.grounded_in_tool_results is False
    assert runner.calls == []
    observation = json.loads(model.requests[1][0][-1].content)
    assert observation["status"] == "error"
    assert observation["code"] == "unknown_tool"


def test_visible_assistant_tool_call_content_is_replayed_without_hidden_reasoning() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall("call-1", "read__value", {"query": "one"}),
                ),
                assistant_content="Visible tool preface.",
            ),
            AgentModelOutput(final_answer="Done."),
        )
    )

    result = AgentLoop(model, registry).run("Read one value.")

    assert result.status == "completed"
    assistant_message = model.requests[1][0][-2]
    assert assistant_message.role == "assistant"
    assert assistant_message.content == "Visible tool preface."


def test_invalid_arguments_and_model_confirmation_are_rejected_before_execution() -> (
    None
):
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_write_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall(
                        "call-1",
                        "write__note",
                        {"content": "hello", "confirm": True},
                    ),
                )
            ),
            AgentModelOutput(final_answer="The invalid action was not executed."),
        )
    )

    result = AgentLoop(model, registry).run("Write a note.")

    assert result.status == "completed"
    assert result.tool_results[0].code == "invalid_arguments"
    assert "confirm is not allowed" in result.tool_results[0].message
    assert runner.calls == []


def test_risky_tool_pauses_for_exact_approval_without_side_effect() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.jsonl"
        runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)
        registry = AgentToolRegistry.from_local_tools(
            runner,
            load_extension_registry(),
            visible_tools=("knowledge.ingest",),
        )
        model = ScriptedModel(
            (
                AgentModelOutput(
                    tool_calls=(
                        AgentToolCall(
                            "write-1",
                            "knowledge__ingest",
                            {
                                "title": "Router",
                                "content": "Keep routing configurable.",
                            },
                        ),
                    )
                ),
            )
        )

        result = AgentLoop(model, registry).run("Remember this project decision.")
        records = FileMemoryStore(memory_path).list()

    assert result.status == "approval_required"
    assert result.final_answer is None
    assert result.tool_results[0].status == "approval_required"
    assert result.tool_results[0].code == "approval_required"
    assert len(result.approval_requests) == 1
    request = result.approval_requests[0]
    assert request.call_id == "write-1"
    assert request.tool_name == "knowledge.ingest"
    assert request.scope == "single_tool_call"
    assert len(request.arguments_sha256) == 64
    assert records == []


def test_external_approval_injects_runner_confirmation_and_completes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.jsonl"
        runner = LocalToolRunner(load_extension_registry(), memory_path=memory_path)
        registry = AgentToolRegistry.from_local_tools(
            runner,
            load_extension_registry(),
            visible_tools=("knowledge.ingest",),
        )
        model = ScriptedModel(
            (
                AgentModelOutput(
                    tool_calls=(
                        AgentToolCall(
                            "write-1",
                            "knowledge__ingest",
                            {
                                "title": "Router",
                                "content": "Keep routing configurable.",
                            },
                        ),
                    )
                ),
                AgentModelOutput(final_answer="Saved after explicit approval."),
            )
        )
        approvals = []

        def approve(request):
            approvals.append(request)
            return ApprovalDecision(approved=True)

        result = AgentLoop(model, registry).run(
            "Remember this project decision.",
            approval_handler=approve,
        )
        records = FileMemoryStore(memory_path).list()

    assert result.status == "completed"
    assert result.tool_results[0].status == "success"
    assert len(approvals) == 1
    assert len(records) == 1
    assert records[0].text.endswith("Keep routing configurable.")
    assert result.grounded_tool_call_ids == ("write-1",)
    assert any(
        event.event == "approval_decision" and event.status == "approved"
        for event in result.trace
    )


def test_non_boolean_approval_decision_fails_closed() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_write_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall("write-1", "write__note", {"content": "hello"}),
                )
            ),
        )
    )

    result = AgentLoop(model, registry).run(
        "Write a note.",
        approval_handler=lambda request: ApprovalDecision(approved="false"),
    )

    assert result.status == "approval_required"
    assert result.tool_results[0].code == "approval_handler_error"
    assert runner.calls == []


def test_invalid_runner_result_still_receives_structured_tool_observation() -> None:
    registry = AgentToolRegistry(InvalidResultRunner(), (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(AgentToolCall("call-1", "read__value", {"query": "value"}),)
            ),
            AgentModelOutput(final_answer="The result was invalid."),
        )
    )

    result = AgentLoop(model, registry).run("Read a value.")

    assert result.status == "completed"
    assert result.tool_results[0].status == "error"
    assert result.tool_results[0].code == "invalid_tool_result"
    assert result.grounded_in_tool_results is False
    observation = json.loads(model.requests[1][0][-1].content)
    assert observation["code"] == "invalid_tool_result"


def test_tool_call_budget_stops_extra_calls_and_results_every_proposal() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall("call-1", "read__value", {"query": "one"}),
                    AgentToolCall("call-2", "read__value", {"query": "two"}),
                )
            ),
        )
    )

    result = AgentLoop(
        model,
        registry,
        budget=AgentLoopBudget(max_model_turns=3, max_tool_calls=1),
    ).run("Read two values.")

    assert result.status == "stopped"
    assert result.reason == "budget_exceeded"
    assert result.tool_calls == 1
    assert len(result.tool_results) == 2
    assert result.tool_results[0].status == "success"
    assert result.tool_results[1].code == "tool_call_budget_exceeded"
    assert [call[1]["query"] for call in runner.calls] == ["one"]


def test_last_model_turn_does_not_execute_a_tool_without_observation_turn() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(AgentToolCall("call-1", "read__value", {"query": "one"}),)
            ),
        )
    )

    result = AgentLoop(
        model,
        registry,
        budget=AgentLoopBudget(max_model_turns=1, max_tool_calls=2),
    ).run("Read one value.")

    assert result.status == "stopped"
    assert result.tool_results[0].code == "model_turn_budget_exceeded"
    assert runner.calls == []


def test_tool_result_is_redacted_and_bounded_before_model_context() -> None:
    runner = RecordingRunner(
        {
            "api_token": "must-not-leak",
            "blob": "x" * 5_000,
        }
    )
    registry = AgentToolRegistry(runner, (_read_tool_spec(),))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(AgentToolCall("call-1", "read__value", {"query": "large"}),)
            ),
            AgentModelOutput(final_answer="The bounded observation was consumed."),
        )
    )

    result = AgentLoop(
        model,
        registry,
        budget=AgentLoopBudget(max_tool_result_chars=512),
    ).run("Read a large value.")

    content = model.requests[1][0][-1].content
    parsed = json.loads(content)
    assert len(content) <= 512
    assert parsed["data"]["truncated"] is True
    assert "must-not-leak" not in content
    assert result.tool_results[0].data["truncated"] is True


def test_secret_like_nested_arguments_are_never_passed_to_runner() -> None:
    runner = RecordingRunner()
    spec = AgentToolSpec(
        name="connector.call",
        model_name="connector__call",
        description="Call a connector.",
        input_schema={
            "type": "object",
            "properties": {
                "arguments": {"type": "object", "additionalProperties": True},
            },
            "required": ["arguments"],
            "additionalProperties": False,
        },
        risk_class="read_only",
        side_effects="none",
    )
    registry = AgentToolRegistry(runner, (spec,))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall(
                        "call-1",
                        "connector__call",
                        {"arguments": {"api_key": "secret"}},
                    ),
                )
            ),
            AgentModelOutput(final_answer="Credentials were rejected."),
        )
    )

    result = AgentLoop(model, registry).run("Call the connector.")

    assert result.tool_results[0].code == "secret_argument_forbidden"
    assert runner.calls == []
    assert "secret" not in json.dumps([asdict(event) for event in result.trace])


def test_non_finite_numbers_are_rejected_even_inside_open_objects() -> None:
    runner = RecordingRunner()
    spec = AgentToolSpec(
        name="connector.call",
        model_name="connector__call",
        description="Call a connector.",
        input_schema={
            "type": "object",
            "properties": {
                "arguments": {"type": "object", "additionalProperties": True},
            },
            "required": ["arguments"],
            "additionalProperties": False,
        },
        risk_class="read_only",
        side_effects="none",
    )
    registry = AgentToolRegistry(runner, (spec,))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall(
                        "call-1",
                        "connector__call",
                        {"arguments": {"score": float("nan")}},
                    ),
                )
            ),
            AgentModelOutput(final_answer="The invalid number was rejected."),
        )
    )

    result = AgentLoop(model, registry).run("Call the connector.")

    assert result.tool_results[0].code == "invalid_arguments"
    assert "finite JSON numbers" in result.tool_results[0].message
    assert runner.calls == []


def test_final_answer_redacts_secret_patterns_and_hidden_reasoning() -> None:
    runner = RecordingRunner()
    registry = AgentToolRegistry(runner, ())
    model = ScriptedModel(
        (
            AgentModelOutput(
                final_answer=(
                    "<think>private chain</think>Use token=super-secret-value only once."
                )
            ),
        )
    )

    result = AgentLoop(model, registry).run("Return a safe answer.")

    assert result.status == "completed"
    assert result.final_answer == "Use [redacted] only once."
    assert "private chain" not in result.final_answer
    assert "super-secret-value" not in result.final_answer


def test_untrusted_model_label_is_redacted_and_bounded_in_trace() -> None:
    registry = AgentToolRegistry(RecordingRunner(), ())
    model = ScriptedModel(
        (
            AgentModelOutput(
                final_answer="Safe answer.",
                model="token=trace-secret" + "x" * 500,
            ),
        )
    )

    result = AgentLoop(model, registry).run("Return an answer.")

    model_labels = [event.model for event in result.trace if event.model]
    assert all("trace-secret" not in label for label in model_labels)
    assert all(len(label) <= 160 for label in model_labels)


def test_legitimate_author_and_token_estimate_fields_are_not_secret_false_positives() -> None:
    runner = RecordingRunner(
        {"prompt_token_estimate": 42, "metadata": {"author": "Antonio"}}
    )
    spec = AgentToolSpec(
        name="read.metadata",
        model_name="read__metadata",
        description="Read metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "metadata": {"type": "object", "additionalProperties": True},
            },
            "required": ["metadata"],
            "additionalProperties": False,
        },
        risk_class="read_only",
        side_effects="none",
    )
    registry = AgentToolRegistry(runner, (spec,))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall(
                        "call-1",
                        "read__metadata",
                        {"metadata": {"author": "Antonio"}},
                    ),
                )
            ),
            AgentModelOutput(final_answer="Metadata read."),
        )
    )

    result = AgentLoop(model, registry).run("Read author metadata.")

    assert result.tool_results[0].status == "success"
    assert result.tool_results[0].data["prompt_token_estimate"] == 42
    assert result.tool_results[0].data["metadata"]["author"] == "Antonio"
    assert runner.calls[0][1]["metadata"]["author"] == "Antonio"


def test_mcp_application_error_is_not_promoted_to_success_or_grounding() -> None:
    runner = RecordingRunner(
        {"is_error": True, "content": [{"type": "text", "text": "failed"}]}
    )
    spec = AgentToolSpec(
        name="mcp.call_tool",
        model_name="mcp__call_tool",
        description="Call an MCP tool.",
        input_schema={
            "type": "object",
            "properties": {"server": {"type": "string"}},
            "required": ["server"],
            "additionalProperties": False,
        },
        risk_class="process_execution",
        side_effects="starts_process_and_calls_tool",
    )
    registry = AgentToolRegistry(runner, (spec,))
    model = ScriptedModel(
        (
            AgentModelOutput(
                tool_calls=(
                    AgentToolCall("call-1", "mcp__call_tool", {"server": "docs"}),
                )
            ),
            AgentModelOutput(final_answer="The MCP tool reported an error."),
        )
    )

    result = AgentLoop(model, registry).run(
        "Call MCP.",
        approval_handler=lambda request: True,
    )

    assert result.tool_results[0].status == "error"
    assert result.tool_results[0].code == "tool_reported_error"
    assert result.grounded_in_tool_results is False


def test_all_enabled_local_tools_have_strict_model_schemas() -> None:
    extension_registry = load_extension_registry()
    registry = AgentToolRegistry.from_local_tools(
        LocalToolRunner(extension_registry),
        extension_registry,
    )

    enabled = {tool.name for tool in extension_registry.tools if tool.enabled}
    exposed = {spec.name for spec in registry.specs}
    assert exposed == enabled
    assert all(
        spec.input_schema["additionalProperties"] is False for spec in registry.specs
    )


def test_factory_runs_openai_compatible_model_tool_observation_loop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        memory_path = Path(tmp) / "memory.jsonl"
        FileMemoryStore(memory_path).add("Factory evidence.", scope="project")
        extension_registry = load_extension_registry()
        runner = LocalToolRunner(extension_registry, memory_path=memory_path)
        config = parse_config(
            {
                "routing": {"top_k": 1},
                "experts": [
                    {
                        "id": "general",
                        "provider": "openai_compatible",
                        "model": "local-agent",
                        "role": "general",
                        "base_url": "http://127.0.0.1:1234/v1",
                    }
                ],
                "rules": [{"expert_id": "general", "keywords": ["evidence"]}],
            }
        )
        requests = []

        def transport(http_request, timeout):
            payload = json.loads(http_request.data.decode("utf-8"))
            requests.append(payload)
            if len(requests) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "factory-call",
                                        "function": {
                                            "name": "memory__search",
                                            "arguments": (
                                                '{"query":"factory","scope":"project"}'
                                            ),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            observation = json.loads(payload["messages"][-1]["content"])
            text = observation["data"]["records"][0]["text"]
            return {"choices": [{"message": {"content": f"Observed: {text}"}}]}

        loop = build_local_agent_loop(
            config,
            runner,
            extension_registry,
            visible_tools=("memory.search",),
            transport=transport,
        )
        result = loop.run("Find factory evidence.")

    assert result.status == "completed"
    assert result.final_answer == "Observed: Factory evidence."
    assert result.grounded_tool_call_ids == ("factory-call",)
    assert requests[1]["messages"][-2]["tool_calls"][0]["id"] == "factory-call"
    assert requests[1]["messages"][-1]["tool_call_id"] == "factory-call"


def _read_tool_spec() -> AgentToolSpec:
    return AgentToolSpec(
        name="read.value",
        model_name="read__value",
        description="Read one value.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
        risk_class="read_only",
        side_effects="none",
    )


def _write_tool_spec() -> AgentToolSpec:
    return AgentToolSpec(
        name="write.note",
        model_name="write__note",
        description="Write one note.",
        input_schema={
            "type": "object",
            "properties": {"content": {"type": "string", "minLength": 1}},
            "required": ["content"],
            "additionalProperties": False,
        },
        risk_class="write_local",
        side_effects="writes_note",
    )


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
