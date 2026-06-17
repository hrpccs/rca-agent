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


# --------------------------------------------------------------------------- #
# validate_tool_call edge cases
# --------------------------------------------------------------------------- #
def test_validate_tool_call_malformed_json_arguments(tools):
    """A tool call whose ``arguments`` is not a dict (e.g. a malformed JSON
    string that failed to parse upstream) cannot even construct a ToolCall —
    pydantic rejects it at the contract seam with ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ToolCall(name="query_alerts", arguments="{not valid json")  # type: ignore[arg-type]


def test_validate_tool_call_unknown_tool_name(tools):
    """An unknown tool name surfaces as KeyError (matches the contract doc)."""
    with pytest.raises(KeyError):
        validate_tool_call(ToolCall(name="does_not_exist", arguments={}), tools)


def test_validate_tool_call_missing_required_arg(tools):
    """store_observation requires ``content``; omitting it must fail validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        validate_tool_call(
            ToolCall(name="store_observation", arguments={}), tools
        )


def test_validate_tool_call_wrong_type_arg(tools):
    """A clearly-wrong type for a typed field must fail validation (no silent
    coercion of e.g. a non-numeric string to int)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        validate_tool_call(
            ToolCall(name="query_alerts", arguments={"limit": "definitely-not-an-int"}),
            tools,
        )


def test_validate_tool_call_valid_returns_parsed_args(tools):
    """A well-formed call returns the (tool, validated-args-model) pair with
    defaults applied."""
    call = ToolCall(
        name="query_logs",
        arguments={"pod": "checkout-abc", "contains": "OOM"},
    )
    tool, validated = validate_tool_call(call, tools)
    assert tool.spec.name == "query_logs"
    assert isinstance(validated, builtin.QueryLogsArgs)
    assert validated.pod == "checkout-abc"
    assert validated.contains == "OOM"
    # default applied
    assert validated.limit == 50


def test_validate_tool_call_extra_unknown_args_are_ignored(tools):
    """The current contract: args models do NOT set ``extra='forbid'``, so
    unknown keys are silently dropped (pydantic default). Asserting the
    documented behavior — do not change it."""
    call = ToolCall(
        name="query_alerts",
        arguments={"limit": 5, "bogus_field": "should-be-ignored"},
    )
    tool, validated = validate_tool_call(call, tools)
    assert tool.spec.name == "query_alerts"
    assert validated.limit == 5
    # the bogus key is not promoted to an attribute on the validated model
    assert not hasattr(validated, "bogus_field")
    assert "bogus_field" not in type(validated).model_fields


# --------------------------------------------------------------------------- #
# Handler error paths — a failing provider must yield a structured error RESULT
# (not an exception), so the agent loop can keep investigating.
# --------------------------------------------------------------------------- #
class _ExplodingProvider(FakeProvider):
    """Provider whose every query_* raises — simulates a backend outage."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def query_alerts(self, f):
        raise self._exc

    def query_events(self, f):
        raise self._exc

    def query_metrics(self, f):
        raise self._exc

    def query_logs(self, f):
        raise self._exc

    def query_traces(self, f):
        raise self._exc

    def query_topology(self, f):
        raise self._exc


@pytest.fixture
def boom_provider() -> _ExplodingProvider:
    return _ExplodingProvider(RuntimeError("backend down"))


@pytest.fixture
def tools_boom(boom_provider, fake_memory):
    return build_default_tools(boom_provider, fake_memory)


def _assert_error_result(res: dict, tool_name: str) -> None:
    """Shared shape assertions for an error-path tool result."""
    assert res["tool"] == tool_name
    assert res["count"] == 0
    assert res["raw"] is None
    # text is non-empty, mentions the tool and the failure
    assert isinstance(res["text"], str) and res["text"]
    assert tool_name in res["text"]
    assert "failed" in res["text"]
    # error carries the exception type + message
    assert "error" in res
    assert "RuntimeError" in res["error"]
    assert "backend down" in res["error"]


@pytest.mark.parametrize(
    "name, args",
    [
        ("query_alerts", {"limit": 5}),
        ("query_events", {"pod": "x"}),
        ("query_metrics", {"service": "x"}),
        ("query_logs", {"pod": "x"}),
        ("query_traces", {"service": "x"}),
        ("get_topology", {"entity_name": "x"}),
        ("inspect_entity", {"entity_name": "x"}),
    ],
)
def test_handler_returns_error_result_on_provider_failure(name, args, tools_boom, boom_provider, fake_memory):
    """Each query handler converts a provider exception into a structured error
    RESULT rather than raising — the agent loop must stay alive."""
    res = _invoke(tools_boom, name, args, boom_provider, fake_memory)
    _assert_error_result(res, name)


def test_error_result_helper_shape():
    """The shared error-result builder produces the canonical shape."""
    res = builtin._error_result("query_logs", ValueError("boom"))
    assert res == {
        "tool": "query_logs",
        "count": 0,
        "text": "(query_logs failed: ValueError: boom)",
        "raw": None,
        "error": "ValueError: boom",
    }


# --------------------------------------------------------------------------- #
# Formatter / renderer shape — canned data round-trips into structured text
# containing the returned entities.
# --------------------------------------------------------------------------- #
def test_query_alerts_text_contains_entity_and_severity(provider, fake_memory, tools):
    res = _invoke(tools, "query_alerts", {}, provider, fake_memory)
    assert res["count"] == 1
    txt = res["text"]
    # entity + severity + data payload key are rendered
    assert "checkout" in txt
    assert "CRITICAL" in txt
    assert "threshold" in txt  # from CloudEvent.data


def test_query_events_text_contains_pod_and_reason(provider, fake_memory, tools):
    res = _invoke(tools, "query_events", {}, provider, fake_memory)
    txt = res["text"]
    assert "checkout-abc" in txt  # pod
    assert "BackOff" in txt  # reason
    assert "Back-off" in txt  # message body


def test_query_metrics_text_contains_entity_metric_and_stats(provider, fake_memory, tools):
    res = _invoke(tools, "query_metrics", {}, provider, fake_memory)
    txt = res["text"]
    assert "checkout" in txt  # entity_name
    assert "cpu_usage" in txt  # metric
    # summary stats are rendered
    assert "n=" in txt and "max=" in txt and "avg=" in txt


def test_query_logs_text_contains_pod_and_content(provider, fake_memory, tools):
    res = _invoke(tools, "query_logs", {}, provider, fake_memory)
    txt = res["text"]
    assert "checkout-abc" in txt  # pod
    assert "OOMKilled" in txt  # content
    assert "prod" in txt  # namespace


def test_query_traces_text_contains_trace_and_spans(provider, fake_memory, tools):
    res = _invoke(tools, "query_traces", {}, provider, fake_memory)
    txt = res["text"]
    assert "trace=tr1" in txt
    assert "spans=2" in txt
    # slowest span + error status rendered
    assert "GET /checkout" in txt
    assert "timeout" in txt  # status_message
    assert "errors=1" in txt


def test_get_topology_text_contains_entities_and_edge(provider, fake_memory, tools):
    res = _invoke(tools, "get_topology", {}, provider, fake_memory)
    txt = res["text"]
    assert "entities(2)" in txt
    assert "checkout" in txt
    assert "postgres" in txt
    assert "calls" in txt  # edge relation
    assert "checkout --calls--> postgres" in txt


def test_inspect_entity_text_contains_props_and_neighbors(provider, fake_memory, tools):
    res = _invoke(tools, "inspect_entity", {"entity_name": "checkout"}, provider, fake_memory)
    txt = res["text"]
    assert "entity checkout" in txt
    assert "props:" in txt
    assert "lang=go" in txt  # prop from the entity dict
    assert "neighbors(1)" in txt
    assert "postgres" in txt  # neighbor rendered
