"""Agent core tests with a fake LLM + fake provider (no API, no real data).

Locks the ReAct loop wiring: tool-call round-trip, reasoning_content echo through
the real ContextManager, final-answer parsing into a RootCause.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from rca_agent.agent.core import RcaAgent
from rca_agent.context.manager import build_context_manager
from rca_agent.contracts import (
    AlertFilter,
    CloudEvent,
    LogFilter,
    LogLine,
    MetricFilter,
    MetricSeries,
    Modality,
    TimeWindow,
    TopologyFilter,
    TopologySubgraph,
    TraceFilter,
    Trace,
    EventFilter,
    K8sEvent,
)
from rca_agent.memory.inmemory_store import InMemoryStore
from rca_agent.tools.registry import build_default_tools
from datetime import datetime, timezone


class FakeProvider:
    case_id = "t001"
    window = TimeWindow(
        start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=timezone.utc),
        end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=timezone.utc),
    )

    def query_alerts(self, f: AlertFilter) -> list[CloudEvent]:
        return [CloudEvent(id="a1", type="ALERT", severity="CRITICAL", subject="checkout 错误次数告警")]

    def query_logs(self, f: LogFilter) -> list[LogLine]:
        return [LogLine(pod="payment-abc", content="Invalid token. app.loyalty.level=gold")]

    def query_metrics(self, f: MetricFilter) -> list[MetricSeries]:
        return []

    def query_traces(self, f: TraceFilter) -> list[Trace]:
        return []

    def query_events(self, f: EventFilter) -> list[K8sEvent]:
        return []

    def query_topology(self, f: TopologyFilter) -> TopologySubgraph:
        return TopologySubgraph(
            entities=[{"id": "p", "type": "apm.service", "name": "payment"}], edges=[]
        )

    def modalities(self) -> list[Modality]:
        return [Modality.LOGS, Modality.ALERTS]


class FakeLLM:
    """Returns a tool_call on turn 1, a JSON final answer on turn 2."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, req) -> tuple[Any, Any, Any, Any]:
        self.calls += 1
        if self.calls == 1:
            return (
                "Let me check the alerts first.",
                "The alert is about checkout errors; I should read the alert and then look at logs.",
                [{"id": "call_1", "type": "function",
                  "function": {"name": "query_alerts", "arguments": json.dumps({})}}],
                {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            )
        # turn 2: final JSON answer
        answer = (
            "```json\n"
            + json.dumps({
                "summary": "payment service charge.js:65 rejects gold payments",
                "fault_type": "app.exception",
                "entity_refs": [{"entity_name": "payment", "entity_type": "apm.service", "entity_domain": "apm"}],
                "evidence": ["query_logs: Invalid token. app.loyalty.level=gold"],
                "confidence": 0.85,
                "contributing_factors": [],
                "recommended_actions": ["rollback payment deploy"],
            }) + "\n```"
        )
        return (answer, "Evidence is sufficient to conclude.", None,
                {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280})

    async def stream(self, req):  # pragma: no cover - not used by complete()
        raise NotImplementedError


def _build_agent() -> RcaAgent:
    return RcaAgent(
        provider=FakeProvider(),
        llm=FakeLLM(),
        memory=InMemoryStore(),
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), InMemoryStore()),
        max_steps=5,
    )


@pytest.mark.asyncio
async def test_agent_loop_produces_report(sample_case):
    agent = _build_agent()
    events = [e async for e in agent.run(sample_case)]
    from rca_agent.contracts import RcaReport, RcaStep, StepKind

    report = [e for e in events if isinstance(e, RcaReport)]
    steps = [e for e in events if isinstance(e, RcaStep)]
    assert report, "no report produced"
    rep = report[-1]
    assert rep.status == "completed"
    rc = rep.root_cause
    assert rc.fault_type == "app.exception"
    assert rc.confidence == pytest.approx(0.85)
    assert any(e.get("entity_name") == "payment" for e in rc.entity_refs)
    # The loop must have emitted a tool_call + tool_result for query_alerts.
    kinds = [s.step_kind for s in steps]
    assert StepKind.TOOL_CALL in kinds
    assert StepKind.TOOL_RESULT in kinds
    assert StepKind.CONCLUDE in kinds


@pytest.mark.asyncio
async def test_agent_truncates_on_max_steps(sample_case):
    """An LLM that always calls tools hits the step cap and yields a truncated report."""
    from rca_agent.contracts import RcaReport

    class AlwaysTools(FakeLLM):
        async def complete(self, req):
            self.calls += 1
            return ("", "thinking", [{"id": f"c{self.calls}", "type": "function",
                    "function": {"name": "query_alerts", "arguments": "{}"}}], {})

    agent = RcaAgent(
        provider=FakeProvider(), llm=AlwaysTools(), memory=InMemoryStore(),
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), InMemoryStore()), max_steps=2,
    )
    events = [e async for e in agent.run(sample_case)]
    report = [e for e in events if isinstance(e, RcaReport)]
    assert report and report[-1].status == "truncated"
