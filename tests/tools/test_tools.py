"""Tests for the SRE investigation tools (builtin + registry + prompts).

Uses a FakeProvider defined locally — never imports the real parquet provider.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from rca_agent.contracts import (
    CloudEvent,
    K8sEvent,
    LogLine,
    MemoryItem,
    MetricSeries,
    TimeWindow,
    TopologySubgraph,
    Trace,
    Span,
    ToolCall,
    build_openai_tools,
    validate_tool_call,
)
from rca_agent.tools import builtin
from rca_agent.tools.registry import build_default_tools
from rca_agent.tools.prompts import SYSTEM_PROMPT, to_final_answer_guidance


# --------------------------------------------------------------------------- #
# Fake provider — covers all DataProvider methods with controlled data.
# --------------------------------------------------------------------------- #
class FakeProvider:
    case_id = "t001"
    window = TimeWindow(
        start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=timezone.utc),
        end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=timezone.utc),
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def query_alerts(self, f):
        self.calls.append(("query_alerts", f))
        return [
            CloudEvent(
                id="a1",
                type="ALERT",
                severity="CRITICAL",
                subtype="error_rate",
                subject="checkout 错误次数告警",
                ts=datetime(2026, 4, 25, 5, 20, 0, tzinfo=timezone.utc),
                data={"service": "checkout", "threshold": 0.1},
            )
        ]

    def query_events(self, f):
        self.calls.append(("query_events", f))
        return [
            K8sEvent(
                ts=datetime(2026, 4, 25, 5, 19, 0, tzinfo=timezone.utc),
                level="Warning",
                pod="checkout-abc",
                hostname="node-1",
                reason="BackOff",
                message="Back-off restarting failed container",
            )
        ]

    def query_metrics(self, f):
        self.calls.append(("query_metrics", f))
        return [
            MetricSeries(
                entity_id="e1",
                entity_name="checkout",
                entity_type="apm.service",
                domain="apm",
                metric="cpu_usage",
                points=[(1, 0.9), (2, 0.95), (3, 0.97)],
            )
        ]

    def query_logs(self, f):
        self.calls.append(("query_logs", f))
        return [
            LogLine(
                ts=datetime(2026, 4, 25, 5, 19, 30, tzinfo=timezone.utc),
                pod="checkout-abc",
                namespace="prod",
                container="app",
                host="node-1",
                content="OOMKilled",
            )
        ]

    def query_traces(self, f):
        self.calls.append(("query_traces", f))
        return [
            Trace(
                trace_id="tr1",
                spans=[
                    Span(trace_id="tr1", span_id="s1", name="GET /checkout", service="checkout",
                         duration_ns=2_000_000_000, status_code="ERROR", status_message="timeout"),
                    Span(trace_id="tr1", span_id="s2", parent_span_id="s1", name="db.query",
                         service="postgres", duration_ns=1_800_000_000),
                ],
            )
        ]

    def query_topology(self, f):
        self.calls.append(("query_topology", f))
        return TopologySubgraph(
            entities=[
                {"id": "checkout", "type": "apm.service", "name": "checkout", "lang": "go"},
                {"id": "postgres", "type": "db", "name": "postgres"},
            ],
            edges=[{"source": "checkout", "target": "postgres", "relation": "calls"}],
        )

    def modalities(self):
        return []


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def tools(provider, fake_memory):
    return build_default_tools(provider, fake_memory)


# --------------------------------------------------------------------------- #
# Schema + registry
# --------------------------------------------------------------------------- #
def test_build_default_tools_count_and_names(tools):
    names = [t.spec.name for t in tools]
    assert names == [
        "query_alerts",
        "query_events",
        "query_metrics",
        "query_logs",
        "query_traces",
        "get_topology",
        "inspect_entity",
        "store_observation",
    ]


def test_openai_schema_is_valid_object(tools):
    schema = build_openai_tools(tools)
    assert len(schema) == len(tools)
    for entry, tool in zip(schema, tools):
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] == tool.spec.name
        assert fn["description"]
        params = fn["parameters"]
        # every args model is a JSON-schema object with properties (even if empty)
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
    # json-serializable end to end
    json.dumps(schema)


def test_validate_tool_call_round_trip(tools):
    # pick query_metrics and validate a partial args dict
    call = ToolCall(name="query_metrics", arguments={"service": "checkout", "limit": 10})
    tool, validated = validate_tool_call(call, tools)
    assert tool.spec.name == "query_metrics"
    assert isinstance(validated, builtin.QueryMetricsArgs)
    assert validated.service == "checkout"
    assert validated.limit == 10
    # defaults applied
    assert validated.metric is None


def test_validate_tool_call_rejects_bad_args(tools):
    from pydantic import ValidationError

    call = ToolCall(name="query_alerts", arguments={"limit": 0})  # ge=1
    with pytest.raises(ValidationError):
        validate_tool_call(call, tools)


def test_validate_tool_call_unknown_tool(tools):
    with pytest.raises(KeyError):
        validate_tool_call(ToolCall(name="nope", arguments={}), tools)


# --------------------------------------------------------------------------- #
# Handler behaviour
# --------------------------------------------------------------------------- #
def _invoke(tools, name, args, provider, memory):
    tool = next(t for t in tools if t.spec.name == name)
    validated = tool.spec.args_model.model_validate(args)
    return tool.handler(validated, provider, memory)


def test_query_alerts_handler(provider, fake_memory, tools):
    res = _invoke(tools, "query_alerts", {"limit": 5}, provider, fake_memory)
    assert res["tool"] == "query_alerts"
    assert res["count"] == 1
    assert "checkout" in res["text"]
    assert "CRITICAL" in res["text"]
    assert isinstance(res["raw"], list) and res["raw"][0]["id"] == "a1"
    # window derived from provider
    fname, f = provider.calls[0]
    assert fname == "query_alerts"
    assert f.window is provider.window


def test_query_events_handler(provider, fake_memory, tools):
    res = _invoke(tools, "query_events", {"pod": "checkout-abc", "level": "Warning"}, provider, fake_memory)
    assert res["count"] == 1
    assert "BackOff" in res["text"]
    fname, f = provider.calls[0]
    assert f.window is provider.window
    assert f.pod_names == ["checkout-abc"]
    assert f.levels == ["Warning"]


def test_query_metrics_handler(provider, fake_memory, tools):
    res = _invoke(tools, "query_metrics", {"service": "checkout", "metric": "cpu_usage"}, provider, fake_memory)
    assert res["count"] == 1
    assert "cpu_usage" in res["text"]
    # raw keeps summary stats, not points
    raw = res["raw"][0]
    assert raw["metric"] == "cpu_usage"
    assert "max" in raw["stats"]
    fname, f = provider.calls[0]
    assert f.services == ["checkout"] and f.metrics == ["cpu_usage"]


def test_query_logs_handler(provider, fake_memory, tools):
    res = _invoke(tools, "query_logs", {"pod": "checkout-abc", "contains": "OOM"}, provider, fake_memory)
    assert res["count"] == 1
    assert "OOMKilled" in res["text"]
    fname, f = provider.calls[0]
    assert f.contains == "OOM" and f.pod_names == ["checkout-abc"]


def test_query_traces_handler_duration_conversion(provider, fake_memory, tools):
    res = _invoke(
        tools, "query_traces",
        {"service": "checkout", "min_duration_ms": 1000, "status": "ERROR"},
        provider, fake_memory,
    )
    assert res["count"] == 1
    assert "timeout" in res["text"]
    fname, f = provider.calls[0]
    assert f.min_duration_ns == 1_000_000_000  # 1000ms -> 1e9 ns
    assert f.status_codes == ["ERROR"]


def test_get_topology_handler(provider, fake_memory, tools):
    res = _invoke(tools, "get_topology", {"entity_name": "checkout", "hops": 1}, provider, fake_memory)
    assert res["count"] == 2
    assert "checkout" in res["text"]
    assert "calls" in res["text"]  # edge relation rendered
    fname, f = provider.calls[0]
    assert f.entity_names == ["checkout"] and f.hops == 1


def test_inspect_entity_handler_by_name(provider, fake_memory, tools):
    res = _invoke(tools, "inspect_entity", {"entity_name": "checkout"}, provider, fake_memory)
    assert res["count"] == 2
    assert "checkout" in res["text"]
    assert "props" in res["text"]
    assert "neighbors" in res["text"]
    assert res["raw"]["entity"]["id"] == "checkout"
    assert any(e["id"] == "postgres" for e in res["raw"]["neighbors"])


def test_inspect_entity_requires_key(provider, fake_memory, tools):
    res = _invoke(tools, "inspect_entity", {}, provider, fake_memory)
    assert res["count"] == 0
    assert "requires" in res["text"]


def test_inspect_entity_not_found_does_not_fallback(provider, fake_memory, tools):
    """A missing entity must NOT silently adopt a neighbor as the answer."""
    res = _invoke(tools, "inspect_entity", {"entity_name": "does-not-exist"}, provider, fake_memory)
    assert res["count"] == 0
    assert "not found" in res["text"]
    # must not have reported one of the real entities as the inspected entity
    assert res["raw"] is None


def test_store_observation_handler(provider, fake_memory, tools):
    res = _invoke(
        tools, "store_observation",
        {"content": "checkout cpu saturated", "kind": "metric_obs", "entities": ["checkout"]},
        provider, fake_memory,
    )
    assert res["raw"]["stored"] is True
    assert res["raw"]["id"]
    # actually indexed in memory under the provider's case_id
    items = fake_memory._items.get("t001", [])
    assert any(isinstance(it, MemoryItem) and it.content == "checkout cpu saturated" for it in items)
    indexed = next(it for it in items if it.content == "checkout cpu saturated")
    assert indexed.kind == "metric_obs"
    assert indexed.entities == ["checkout"]
    assert indexed.source_tool == "store_observation"


def test_store_observation_tolerates_none_memory(provider, tools):
    # memory=None must not crash
    res = _invoke(tools, "store_observation", {"content": "x"}, provider, None)
    assert res["raw"]["stored"] is True


# --------------------------------------------------------------------------- #
# Empty-result rendering (evidence-by-absence)
# --------------------------------------------------------------------------- #
class EmptyProvider(FakeProvider):
    def query_alerts(self, f):
        return []

    def query_logs(self, f):
        return []


def test_empty_alerts_render(provider_empty, fake_memory, tools_empty):
    res = _invoke(tools_empty, "query_alerts", {}, provider_empty, fake_memory)
    assert res["count"] == 0
    assert "no alerts" in res["text"]


@pytest.fixture
def provider_empty():
    return EmptyProvider()


@pytest.fixture
def tools_empty(provider_empty, fake_memory):
    return build_default_tools(provider_empty, fake_memory)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def test_system_prompt_is_substantial():
    assert "SRE" in SYSTEM_PROMPT
    assert "根因" in SYSTEM_PROMPT
    assert "tool" in SYSTEM_PROMPT.lower()


def test_final_answer_guidance_covers_root_cause_fields():
    g = to_final_answer_guidance()
    for key in ["summary", "fault_type", "entity_refs", "evidence", "confidence",
                "contributing_factors", "recommended_actions"]:
        assert key in g
    assert "entity" in g.lower()
