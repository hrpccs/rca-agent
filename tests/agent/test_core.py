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


# --------------------------------------------------------------------------- #
# Hardening: non-fatal failures must never kill the ReAct loop.
# --------------------------------------------------------------------------- #


class _ExplodingProvider(FakeProvider):
    """Provider whose query_alerts blows up — used to exercise tool-handler error
    surfacing without touching real data."""

    def query_alerts(self, f):
        raise RuntimeError("provider exploded: DB connection refused")


class _ExplodingMemory(InMemoryStore):
    """Memory store whose retrieve_for_context always raises — exercises that a
    seed/prior load failure cannot abort the run."""

    def retrieve_for_context(self, case_id, query, top_k=8):
        raise OSError("memory backend unavailable")


@pytest.mark.asyncio
async def test_tool_handler_error_is_surfaced_and_run_continues(sample_case):
    """A tool handler that raises must yield a tool_result describing the error,
    and the loop must continue so the next LLM turn can still conclude."""
    from rca_agent.contracts import RcaReport, RcaStep, StepKind

    class ToolsThenConclude(FakeLLM):
        async def complete(self, req):
            self.calls += 1
            if self.calls == 1:
                return (
                    "Let me check alerts.",
                    "checking alerts",
                    [{"id": "c1", "type": "function",
                      "function": {"name": "query_alerts", "arguments": "{}"}}],
                    {"total_tokens": 10},
                )
            return (
                "```json\n" + json.dumps({
                    "summary": "investigated; provider error was surfaced as evidence",
                    "fault_type": "unknown",
                    "confidence": 0.4,
                    "evidence": ["query_alerts raised: provider exploded"],
                    "contributing_factors": [],
                    "recommended_actions": [],
                    "entity_refs": [],
                }) + "\n```",
                "Evidence collected (incl. the tool error); concluding.",
                None,
                {"total_tokens": 20},
            )

    agent = RcaAgent(
        provider=_ExplodingProvider(),
        llm=ToolsThenConclude(),
        memory=InMemoryStore(),
        context_manager=build_context_manager(),
        tools=build_default_tools(_ExplodingProvider(), InMemoryStore()),
        max_steps=4,
    )

    events = [e async for e in agent.run(sample_case)]
    steps = [e for e in events if isinstance(e, RcaStep)]
    reports = [e for e in events if isinstance(e, RcaReport)]
    assert reports, "no report produced — tool error crashed the run"
    assert reports[-1].status == "completed"

    # The tool_result for query_alerts must carry the error payload.
    tool_results = [
        s for s in steps
        if s.step_kind == StepKind.TOOL_RESULT and s.tool_name == "query_alerts"
    ]
    assert tool_results, "no tool_result emitted for the failing call"
    res = tool_results[-1].tool_result
    assert isinstance(res, dict) and "error" in res
    assert "RuntimeError" in res["error"]
    assert "provider exploded" in res["error"]

    # The LLM must have been called again after the error (loop continued).
    assert agent.llm.calls >= 2


@pytest.mark.asyncio
async def test_malformed_final_answer_yields_completed_report(sample_case):
    """When the LLM returns a non-JSON, tool-call-free final answer, the
    parse_root_cause fallback must still yield a completed RcaReport (low/default
    confidence) and never raise."""
    from rca_agent.contracts import RcaReport

    class GarbageFinal(FakeLLM):
        async def complete(self, req):
            self.calls += 1
            # No tool_calls, no JSON — just prose. parse_root_cause must fall back.
            return (
                "I am not sure what is wrong; the logs look noisy and nothing"
                " clearly points to a fault. Maybe a deploy went out.",
                "could not converge on a single root cause",
                None,
                {"total_tokens": 5},
            )

    agent = RcaAgent(
        provider=FakeProvider(),
        llm=GarbageFinal(),
        memory=InMemoryStore(),
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), InMemoryStore()),
        max_steps=3,
    )
    events = [e async for e in agent.run(sample_case)]
    reports = [e for e in events if isinstance(e, RcaReport)]
    assert reports, "no report produced — parse raised"
    rep = reports[-1]
    # Must complete (not truncate) with a default/low confidence — never raise.
    assert rep.status == "completed"
    assert rep.root_cause.confidence <= 0.5
    assert rep.root_cause.summary  # non-empty fallback summary


@pytest.mark.asyncio
async def test_memory_seed_failure_does_not_crash_run(sample_case):
    """If memory.retrieve_for_context raises at startup, the agent must still run
    to completion using the FakeLLM's normal flow (no priors, but alive)."""
    from rca_agent.contracts import RcaReport

    agent = RcaAgent(
        provider=FakeProvider(),
        llm=FakeLLM(),  # tool_call turn 1, JSON final answer turn 2
        memory=_ExplodingMemory(),
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), _ExplodingMemory()),
        max_steps=4,
    )
    events = [e async for e in agent.run(sample_case)]
    reports = [e for e in events if isinstance(e, RcaReport)]
    assert reports, "no report produced — memory failure aborted the run"
    assert reports[-1].status == "completed"
    assert reports[-1].root_cause.confidence == pytest.approx(0.85)


# --------------------------------------------------------------------------- #
# T3: memory-retrieval surfaced as a display-only trace step.
# --------------------------------------------------------------------------- #


class _RecordingLLM(FakeLLM):
    """FakeLLM that records every LLMRequest it is handed, so tests can assert
    the display-only memory step never leaks into the model's context."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[Any] = []

    async def complete(self, req):
        self.received.append(req)
        return await super().complete(req)


def _seeded_memory(*items):
    """Build an InMemoryStore preloaded with the given MemoryItems."""
    store = InMemoryStore()
    store.index(list(items))
    return store


@pytest.mark.asyncio
async def test_memory_retrieval_yields_display_step_before_first_reasoning(sample_case):
    """When memory returns >=1 hit, run() yields a REASONING step that mentions
    memory/priors, carrying the retrieved entities, BEFORE any LLM-driven
    reasoning step appears."""
    from rca_agent.contracts import MemoryItem, RcaReport, RcaStep, StepKind

    mem = _seeded_memory(
        MemoryItem(
            id="rb-1",
            case_id="__global__",
            content="Runbook: checkout 5xx — check payment service tokens.",
            kind="runbook",
            entities=["payment", "checkout", "loyalty"],
        ),
    )
    agent = RcaAgent(
        provider=FakeProvider(),
        llm=_RecordingLLM(),
        memory=mem,
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), mem),
        max_steps=4,
    )

    events = [e async for e in agent.run(sample_case)]
    steps = [e for e in events if isinstance(e, RcaStep)]

    mem_steps = [
        s for s in steps
        if s.step_kind == StepKind.REASONING and s.thought and "memory" in s.thought
    ]
    assert mem_steps, "no memory step yielded despite hits"
    mem_step = mem_steps[0]
    assert "prior" in mem_step.thought
    assert sample_case.task.alert_title in mem_step.thought
    # Entities from the retrieved MemoryItem are surfaced as context.
    assert "payment" in mem_step.entities
    assert "loyalty" in mem_step.entities

    # The memory step must come BEFORE the first LLM-driven reasoning step (the
    # one whose thought is NOT the memory line).
    llm_reason_idx = next(
        i for i, s in enumerate(steps)
        if s.step_kind == StepKind.REASONING and not (s.thought or "").startswith("memory:")
    )
    assert steps.index(mem_step) < llm_reason_idx

    # Regression: the run still completes and the memory step is in report.steps.
    reports = [e for e in events if isinstance(e, RcaReport)]
    assert reports and reports[-1].status == "completed"
    assert mem_step.step_id in {s.step_id for s in reports[-1].steps}


@pytest.mark.asyncio
async def test_no_memory_step_when_hits_empty(sample_case):
    """When memory returns no hits, NO memory step is yielded (but the run
    still completes normally)."""
    from rca_agent.contracts import RcaReport, RcaStep, StepKind

    # Empty InMemoryStore -> retrieve_for_context returns [].
    mem = InMemoryStore()
    agent = RcaAgent(
        provider=FakeProvider(),
        llm=FakeLLM(),
        memory=mem,
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), mem),
        max_steps=4,
    )
    events = [e async for e in agent.run(sample_case)]
    steps = [e for e in events if isinstance(e, RcaStep)]
    assert not any(
        s.step_kind == StepKind.REASONING and s.thought and "memory" in s.thought
        for s in steps
    ), "memory step yielded despite empty hits"
    reports = [e for e in events if isinstance(e, RcaReport)]
    assert reports and reports[-1].status == "completed"


@pytest.mark.asyncio
async def test_memory_step_not_in_llm_context(sample_case):
    """The display-only memory step must NEVER be fed to the LLM. Assert the
    memory thought text does not appear anywhere in the messages handed to the
    model, so the ReAct loop / token usage / final answer are byte-for-byte
    unaffected by this display step."""
    from rca_agent.contracts import MemoryItem

    mem = _seeded_memory(
        MemoryItem(
            id="rb-1",
            case_id="__global__",
            content="SOP: checkout errors — verify payment token validation.",
            kind="sop",
            entities=["payment"],
        ),
    )
    rec_llm = _RecordingLLM()
    agent = RcaAgent(
        provider=FakeProvider(),
        llm=rec_llm,
        memory=mem,
        context_manager=build_context_manager(),
        tools=build_default_tools(FakeProvider(), mem),
        max_steps=4,
    )
    # Drain the generator so every LLM call is recorded.
    _ = [e async for e in agent.run(sample_case)]

    assert rec_llm.received, "LLM was never called"
    # Concatenate all message contents the model saw and assert the memory
    # display-text is absent (the memory PRIORS are legitimately in the brief;
    # only the synthetic 'memory: retrieved ...' thought must not leak).
    forbidden = "memory: retrieved"
    for req in rec_llm.received:
        blob = json.dumps(
            [getattr(m, "model_dump", lambda: {})() for m in req.messages],
            ensure_ascii=False, default=str,
        )
        assert forbidden not in blob, (
            f"display-only memory step leaked into LLM messages: {blob[:200]}"
        )


class _BrokenMetrics:
    """Module shim whose every attribute lookup raises ImportError."""

    def __getattr__(self, name):
        raise ImportError("simulated broken observability")


def test_safe_otel_no_ops_on_missing_recorder_and_bad_import(monkeypatch):
    """_safe_otel must never raise: a missing recorder or a broken metrics
    module is swallowed with a log line and the loop is unaffected."""
    import rca_agent.observability as _obs_pkg
    from rca_agent.agent import core as core_mod

    # 1) Recorder not found on the module surface -> warning, no raise.
    core_mod._safe_otel("this_recorder_does_not_exist", "x", y=1)

    # 2) The metrics module attribute is a broken shim whose __getattr__ raises
    #    ImportError -> the deferred import inside _safe_otel fails; must not
    #    raise. monkeypatch auto-reverts the package attribute, so no manual
    #    cleanup is needed.
    monkeypatch.setattr(_obs_pkg, "metrics", _BrokenMetrics(), raising=False)
    core_mod._safe_otel("record_run", "completed")  # must not raise
