"""Contract conformance gate.

These tests validate the FROZEN contracts themselves (not any implementation).
They must pass in the foundation before any worker branches. They also encode
the invariants every implementation must satisfy:
  * tools schema/validate round-trip from a single args_model source of truth
  * DataProvider / MemoryStore / ContextManager / LLMClient are Protocols
  * SSE wire format is stable
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from rca_agent.contracts import (
    AlertFilter,
    Case,
    CloudEvent,
    ContextManager,
    ContextState,
    DataProvider,
    LLMClient,
    LogFilter,
    MemoryItem,
    MemoryQuery,
    MemoryStore,
    MetricFilter,
    Modality,
    RegisteredTool,
    RcaReport,
    RcaStep,
    RootCause,
    SSEEvent,
    SSEEventKind,
    Task,
    TimeWindow,
    ToolCall,
    ToolSpec,
    TraceFilter,
    build_openai_tools,
    sse_format,
    validate_tool_call,
)
from rca_agent.contracts.tools import ToolHandler


# --------------------------------------------------------------------------- #
# Protocols exist and are runtime-checkable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "proto",
    [DataProvider, MemoryStore, ContextManager, LLMClient],
)
def test_protocols_are_runtime_checkable(proto):
    # A bare object should NOT match (negative control)
    assert not isinstance(object(), proto)


def test_modality_values():
    assert {m.value for m in Modality} == {
        "metrics",
        "logs",
        "traces",
        "events",
        "alerts",
        "topology",
    }


# --------------------------------------------------------------------------- #
# Dataset models round-trip through JSON
# --------------------------------------------------------------------------- #
def test_task_roundtrip(time_window):
    t = Task(
        task_id="t001",
        alert_title="x",
        alert_window=time_window,
        prompt_text="p",
        available_modalities=[Modality.LOGS, Modality.METRICS],
        alert_entity={"entity_id": None, "entity_type": "apm.operation"},
    )
    js = t.model_dump_json()
    t2 = Task.model_validate_json(js)
    assert t2.task_id == "t001"
    assert t2.available_modalities == [Modality.LOGS, Modality.METRICS]


def test_case_serializable(sample_case: Case):
    js = sample_case.model_dump_json()
    assert Case.model_validate_json(js).task.task_id == "t001"


# --------------------------------------------------------------------------- #
# Filters carry window + limit defaults
# --------------------------------------------------------------------------- #
def test_filters_require_window(time_window):
    for F in [MetricFilter, LogFilter, TraceFilter, AlertFilter]:
        f = F(window=time_window)
        assert f.limit > 0


# --------------------------------------------------------------------------- #
# Tools: schema build + validate come from one args_model
# --------------------------------------------------------------------------- #
class _PingArgs(__import__("pydantic").BaseModel):
    """Trivial args model used to exercise the tool pipeline."""

    entity: str
    hops: int = 1


def _ping_handler(args: Any, provider: Any, memory: Any):
    return {"echo": getattr(args, "entity", ""), "hops": getattr(args, "hops", 1)}


def _make_tool() -> RegisteredTool:
    spec = ToolSpec(name="ping", description="ping tool", args_model=_PingArgs)
    return RegisteredTool(spec=spec, handler=_ping_handler)


def test_build_openai_tools_shape():
    tools = build_openai_tools([_make_tool()])
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "ping"
    assert fn["parameters"]["type"] == "object"
    assert "entity" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["entity"]


def test_validate_tool_call_roundtrip():
    tool = _make_tool()
    call = ToolCall(name="ping", arguments={"entity": "svc", "hops": 2})
    reg, validated = validate_tool_call(call, [tool])
    assert reg is tool
    assert isinstance(validated, _PingArgs)
    assert validated.entity == "svc" and validated.hops == 2


def test_validate_tool_call_unknown_raises():
    with pytest.raises(KeyError):
        validate_tool_call(ToolCall(name="nope", arguments={}), [_make_tool()])


def test_validate_tool_call_bad_args_raises():
    call = ToolCall(name="ping", arguments={"hops": 2})  # missing required entity
    with pytest.raises(Exception):
        validate_tool_call(call, [_make_tool()])


def test_tool_handler_protocol_accepts_callable():
    assert isinstance(_ping_handler, type(ToolHandler)) or callable(_ping_handler)


# --------------------------------------------------------------------------- #
# RCA report schema
# --------------------------------------------------------------------------- #
def test_rca_report_required_fields():
    rc = RootCause(summary="bad pod", confidence=0.8)
    r = RcaReport(
        case_id="t001",
        task_id="t001",
        alert_title="x",
        root_cause=rc,
    )
    js = r.model_dump_json()
    assert RcaReport.model_validate_json(js).root_cause.summary == "bad pod"


def test_rca_step_serializable():
    s = RcaStep(step_id="s1", case_id="t001", step_kind="tool_call", tool_name="query_logs")
    assert s.model_dump(mode="json")["step_kind"] == "tool_call"


# --------------------------------------------------------------------------- #
# SSE wire format stability (server <-> frontend contract)
# --------------------------------------------------------------------------- #
def test_sse_format_stable():
    ev = SSEEvent(event=SSEEventKind.STEP, case_id="t001", data={"a": 1}, seq=3)
    wire = sse_format(ev)
    assert wire.startswith("event: step\n")
    assert "data: " in wire
    assert wire.endswith("\n\n")
    line = wire.split("data: ", 1)[1].strip()
    payload = json.loads(line)
    assert payload["event"] == "step"
    assert payload["case_id"] == "t001"
    assert payload["seq"] == 3


# --------------------------------------------------------------------------- #
# ContextState is a plain model (impls satisfy ContextManager)
# --------------------------------------------------------------------------- #
def test_context_state_defaults():
    st = ContextState(case_id="t001", system="sys")
    assert st.messages == [] and st.turns == []


def test_memory_models():
    it = MemoryItem(id="m1", content="rtt rise may be cpu contention", kind="domain_fact")
    q = MemoryQuery(text="rtt")
    assert it.id == "m1"
    assert q.top_k == 8
