from __future__ import annotations

import json
from typing import Any
import unittest

from local_moe.agent_provider import (
    OpenAICompatibleAgentAdapter,
    is_loopback_base_url,
    parse_openai_compatible_output,
    select_agent_expert,
    validate_local_agent_endpoints,
)
from local_moe.agent_types import AgentMessage, AgentToolSpec
from local_moe.config import ExpertConfig, parse_config
from local_moe.providers import ProviderError


def test_openai_compatible_adapter_serializes_tools_and_preserves_control_plane_fields() -> (
    None
):
    captured: dict[str, Any] = {}

    def transport(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        return {
            "model": "local-agent",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "memory__search",
                                    "arguments": '{"query":"router"}',
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }

    expert = ExpertConfig(
        id="general",
        provider="openai_compatible",
        model="local-agent",
        role="general",
        base_url="http://127.0.0.1:1234/v1/",
        timeout_seconds=12,
        params={
            "temperature": 0.1,
            "messages": [{"role": "user", "content": "override"}],
            "tools": [{"type": "evil"}],
        },
    )
    adapter = OpenAICompatibleAgentAdapter(expert, transport=transport)
    spec = AgentToolSpec(
        name="memory.search",
        description="Search memory.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        risk_class="read_only",
        side_effects="none",
    )

    output = adapter.generate(
        (
            AgentMessage(role="system", content="Trusted instructions."),
            AgentMessage(role="user", content="Find router notes."),
        ),
        (spec,),
        correlation_id="correlation-1",
    )

    request_payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert captured["timeout"] == 12
    assert request_payload["messages"][0]["content"] == "Trusted instructions."
    assert request_payload["messages"][1]["content"] == "Find router notes."
    assert request_payload["tools"][0]["function"]["name"] == "memory__search"
    assert request_payload["parallel_tool_calls"] is False
    assert request_payload["temperature"] == 0.1
    assert output.tool_calls[0].arguments == {"query": "router"}
    assert output.usage.prompt_tokens == 42
    assert output.usage.completion_tokens == 7


def test_parser_keeps_malformed_arguments_for_local_structured_validation() -> None:
    output = parse_openai_compatible_output(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "memory__search",
                                    "arguments": "{not-json",
                                },
                            }
                        ]
                    }
                }
            ]
        },
        model="local-agent",
        correlation_id="correlation-1",
    )

    assert output.tool_calls[0].arguments == "{not-json"
    assert output.final_answer is None


def test_adapter_caps_http_timeout_to_remaining_soft_deadline() -> None:
    captured: dict[str, float] = {}

    def transport(_http_request, timeout):
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "done"}}]}

    expert = ExpertConfig(
        id="general",
        provider="openai_compatible",
        model="local-agent",
        role="general",
        base_url="http://127.0.0.1:1234/v1",
        timeout_seconds=12,
    )
    adapter = OpenAICompatibleAgentAdapter(expert, transport=transport)

    adapter.generate(
        (AgentMessage(role="user", content="Finish quickly."),),
        (),
        correlation_id="timeout-cap",
        timeout_seconds=0.25,
    )

    assert captured["timeout"] == 0.25


def test_parser_rejects_non_standard_nan_json_arguments() -> None:
    output = parse_openai_compatible_output(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "score__record",
                                    "arguments": '{"score":NaN}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        model="local-agent",
        correlation_id="correlation-1",
    )

    assert output.tool_calls[0].arguments == '{"score":NaN}'


def test_parser_strips_hidden_reasoning_and_ignores_reasoning_field() -> None:
    output = parse_openai_compatible_output(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>private chain</think>Public answer.",
                        "reasoning_content": "private provider reasoning",
                    }
                }
            ]
        },
        model="local-agent",
        correlation_id="correlation-1",
    )

    assert output.final_answer == "Public answer."
    assert "private" not in output.final_answer


def test_parser_rejects_non_text_content_instead_of_stringifying_reasoning() -> None:
    try:
        parse_openai_compatible_output(
            {
                "choices": [
                    {
                        "message": {
                            "content": {
                                "reasoning": "private chain",
                                "final": "Public answer.",
                            }
                        }
                    }
                ]
            },
            model="local-agent",
            correlation_id="correlation-1",
        )
    except ProviderError as exc:
        assert "non-text assistant content" in str(exc)
    else:
        raise AssertionError("Expected non-text assistant content to be rejected")


def test_parser_preserves_only_visible_content_alongside_tool_calls() -> None:
    output = parse_openai_compatible_output(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>private chain</think>I will inspect memory.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "memory__search",
                                    "arguments": '{"query":"router"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        model="local-agent",
        correlation_id="correlation-1",
    )

    assert output.final_answer is None
    assert output.assistant_content == "I will inspect memory."
    assert "private" not in output.assistant_content


def test_tool_result_message_requires_matching_call_id() -> None:
    expert = ExpertConfig(
        id="general",
        provider="openai_compatible",
        model="local-agent",
        role="general",
        base_url="http://127.0.0.1:1234/v1",
    )
    adapter = OpenAICompatibleAgentAdapter(expert, transport=lambda *_: {})

    try:
        adapter.request_payload((AgentMessage(role="tool", content="{}"),), ())
    except ProviderError as exc:
        assert "tool_call_id" in str(exc)
    else:
        raise AssertionError(
            "Expected ProviderError for a tool message without tool_call_id"
        )


def test_select_agent_expert_prefers_general_and_validates_explicit_id() -> None:
    config = parse_config(
        {
            "routing": {"top_k": 1},
            "experts": [
                {
                    "id": "fast",
                    "provider": "openai_compatible",
                    "model": "fast-model",
                    "role": "fast",
                    "base_url": "http://127.0.0.1:1234/v1",
                },
                {
                    "id": "general",
                    "provider": "openai_compatible",
                    "model": "general-model",
                    "role": "general",
                    "base_url": "http://127.0.0.1:1235/v1",
                },
            ],
            "rules": [{"expert_id": "general", "keywords": ["analyze"]}],
        }
    )

    assert select_agent_expert(config).id == "general"
    assert select_agent_expert(config, expert_id="fast").id == "fast"
    try:
        select_agent_expert(config, expert_id="missing")
    except ProviderError as exc:
        assert "Unknown agent expert" in str(exc)
    else:
        raise AssertionError(
            "Expected ProviderError for an unknown explicit agent expert"
        )


def test_local_agent_endpoint_policy_accepts_loopback_and_rejects_remote() -> None:
    assert is_loopback_base_url("http://localhost:8101/v1")
    assert is_loopback_base_url("http://localhost.:8101/v1")
    assert is_loopback_base_url("http://127.0.0.2:8101/v1")
    assert is_loopback_base_url("http://[::1]:8101/v1")
    assert is_loopback_base_url("http://[::ffff:127.0.0.1]:8101/v1")
    assert not is_loopback_base_url("https://models.example.com/v1")
    assert not is_loopback_base_url("http://0.0.0.0:8101/v1")

    config = parse_config(
        {
            "routing": {"top_k": 1},
            "experts": [
                {
                    "id": "general",
                    "provider": "openai_compatible",
                    "model": "general-model",
                    "role": "general",
                    "base_url": "http://127.0.0.1:8101/v1",
                },
                {
                    "id": "remote-fallback",
                    "provider": "openai_compatible",
                    "model": "remote-model",
                    "role": "fallback",
                    "base_url": "https://models.example.com/v1",
                },
            ],
            "rules": [],
        }
    )

    try:
        validate_local_agent_endpoints(config)
    except ProviderError as exc:
        assert "remote-fallback" in str(exc)
        assert "non-loopback" in str(exc)
    else:
        raise AssertionError("Expected remote fallback endpoint to be rejected")


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
