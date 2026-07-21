from __future__ import annotations

from typing import Any
import unittest

from local_moe.agent_types import AgentToolSpec
from local_moe.browser_capability import CompositeToolRunner
from local_moe.tool_runner import ToolExecutionError, ToolRunResult


class CompositeToolRunnerTests(unittest.TestCase):
    def test_routes_each_specialized_tool_and_falls_back_to_default(self) -> None:
        events: list[str] = []
        default = _RecordingRunner("default", ("local.inspect",), events)
        browser = _RecordingRunner("browser", ("browser.observe",), events)
        desktop = _RecordingRunner("desktop", ("desktop.observe",), events)
        composite = CompositeToolRunner(default, browser, desktop)

        browser_result = composite.run(
            "browser.observe",
            {"revision": 1},
            timeout_seconds=2.5,
        )
        desktop_result = composite.run(
            "desktop.observe",
            {"target_id": "editor"},
            timeout_seconds=3.5,
        )
        default_result = composite.run("local.inspect", {"path": "."})

        self.assertEqual(browser_result.payload["runner"], "browser")
        self.assertEqual(desktop_result.payload["runner"], "desktop")
        self.assertEqual(default_result.payload["runner"], "default")
        self.assertEqual(
            browser.calls,
            [("browser.observe", {"revision": 1}, 2.5)],
        )
        self.assertEqual(
            desktop.calls,
            [("desktop.observe", {"target_id": "editor"}, 3.5)],
        )
        self.assertEqual(default.calls, [("local.inspect", {"path": "."}, None)])

    def test_rejects_duplicate_specialized_tool_names(self) -> None:
        events: list[str] = []
        first = _RecordingRunner("first", ("shared.observe",), events)
        second = _RecordingRunner("second", ("shared.observe",), events)

        with self.assertRaisesRegex(ToolExecutionError, "duplicated canonical tool"):
            CompositeToolRunner(_RecordingRunner("default", (), events), first, second)

    def test_closes_specialized_runners_in_reverse_order(self) -> None:
        events: list[str] = []
        first = _RecordingRunner("first", ("first.observe",), events)
        second = _RecordingRunner("second", ("second.observe",), events)
        composite = CompositeToolRunner(
            _RecordingRunner("default", (), events),
            first,
            second,
        )

        composite.close()

        self.assertEqual(events, ["close:second", "close:first"])


class _RecordingRunner:
    def __init__(
        self,
        name: str,
        tool_names: tuple[str, ...],
        events: list[str],
    ) -> None:
        self.name = name
        self.specs = tuple(_spec(tool_name) for tool_name in tool_names)
        self.events = events
        self.calls: list[tuple[str, dict[str, Any] | None, float | None]] = []

    def run(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolRunResult:
        self.calls.append((name, payload, timeout_seconds))
        return ToolRunResult(
            name=name,
            status="ok",
            risk_class="read_only",
            side_effects="none",
            message=f"handled by {self.name}",
            payload={"runner": self.name},
        )

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


def _spec(name: str) -> AgentToolSpec:
    return AgentToolSpec(
        name=name,
        description=f"Synthetic {name} tool.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        risk_class="read_only",
        side_effects="none",
    )


if __name__ == "__main__":
    unittest.main()
