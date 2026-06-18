"""Exhaustive unit tests for ``rca_agent.agent.prompts``.

Covers every ``parse_root_cause`` strategy (fenced json, bare object, markdown
sections, free-text fallback) plus confidence clamping, entity/evidence/fault
extraction, and ``build_initial_brief`` rendering / graceful degradation.

All fixtures are inline; no live API, no DB.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rca_agent.agent.prompts import (
    _clean_section_lead,
    _md_table_rows,
    _split_md_sections,
    build_initial_brief,
    parse_root_cause,
)
from rca_agent.contracts import (
    MemoryItem,
    Modality,
    Task,
    TimeWindow,
    Topology,
    TopologyEdge,
    TopologyEntity,
)

# --------------------------------------------------------------------------- #
# Inline fixtures
# --------------------------------------------------------------------------- #
_WINDOW = TimeWindow(
    start=datetime(2026, 6, 17, 0, 0, tzinfo=UTC),
    end=datetime(2026, 6, 17, 0, 10, tzinfo=UTC),
)


def _task(
    *,
    alert_entity: dict[str, Any] | None = None,
    modalities: list[Modality] | None = None,
) -> Task:
    return Task(
        task_id="t-fixture",
        alert_title="checkout 错误率告警",
        alert_window=_WINDOW,
        alert_entity=dict(alert_entity or {}),
        prompt_text="帮我定位根因。",
        available_modalities=modalities if modalities is not None
        else [Modality.LOGS, Modality.METRICS],
    )


def _topology(
    *,
    entities: list[TopologyEntity] | None = None,
    edges: list[TopologyEdge] | None = None,
) -> Topology:
    return Topology(
        case_id="t-fixture",
        window=_WINDOW,
        entities=list(entities or []),
        edges=list(edges or []),
    )


def _mem(content: str, *, kind: str = "runbook") -> MemoryItem:
    return MemoryItem(id=f"m-{abs(hash(content)) % 10**6}", content=content, kind=kind)


# --------------------------------------------------------------------------- #
# parse_root_cause — strategy 1: fenced ```json block
# --------------------------------------------------------------------------- #
def test_parse_fenced_json_full_fields():
    payload = {
        "summary": "payment rejects gold-tier loyalty tokens",
        "fault_type": "app.exception",
        "entity_refs": [
            {"entity_name": "payment", "entity_type": "apm.service", "entity_domain": "apm"}
        ],
        "evidence": ["query_logs: Invalid token. app.loyalty.level=gold"],
        "confidence": 0.85,
        "contributing_factors": ["feature flag rollout"],
        "recommended_actions": ["rollback payment deploy"],
    }
    content = "```json\n" + __import__("json").dumps(payload) + "\n```"
    rc = parse_root_cause(content)
    assert rc.summary == payload["summary"]
    assert rc.fault_type == "app.exception"
    assert rc.confidence == pytest.approx(0.85)
    assert rc.entity_refs[0]["entity_name"] == "payment"
    assert rc.entity_refs[0]["entity_type"] == "apm.service"
    assert rc.entity_refs[0]["entity_domain"] == "apm"
    assert rc.evidence == ["query_logs: Invalid token. app.loyalty.level=gold"]
    assert rc.contributing_factors == ["feature flag rollout"]
    assert rc.recommended_actions == ["rollback payment deploy"]


def test_parse_fenced_json_without_lang_tag():
    """Fenced block with no language tag still parses."""
    content = "```\n{\"summary\": \"x\", \"confidence\": 0.4}\n```"
    rc = parse_root_cause(content)
    assert rc.summary == "x"
    assert rc.confidence == pytest.approx(0.4)


def test_parse_fenced_json_malformed_falls_through():
    """Invalid JSON in a fenced block must not raise; falls through to next
    strategy (here: bare-object miss → free-text)."""
    rc = parse_root_cause("```json\n{not valid json}\n```")
    assert isinstance(rc.summary, str)
    assert rc.summary  # non-empty


# --------------------------------------------------------------------------- #
# strategy 2: bare {...}
# --------------------------------------------------------------------------- #
def test_parse_bare_object_with_surrounding_prose():
    content = (
        "After analysis I conclude:\n"
        '{"summary": "db pool exhausted", "fault_type": "db.connection_pool", '
        '"confidence": 0.7}\n'
        "That is the root cause."
    )
    rc = parse_root_cause(content)
    assert rc.summary == "db pool exhausted"
    assert rc.fault_type == "db.connection_pool"
    assert rc.confidence == pytest.approx(0.7)


def test_parse_bare_object_missing_optional_fields_defaults():
    rc = parse_root_cause('{"summary": "only summary"}')
    assert rc.summary == "only summary"
    assert rc.fault_type is None
    assert rc.entity_refs == []
    assert rc.evidence == []
    assert rc.contributing_factors == []
    assert rc.recommended_actions == []
    # confidence missing → default 0.0
    assert rc.confidence == pytest.approx(0.0)


def test_parse_bare_object_empty_summary_falls_back_to_content():
    """summary empty/missing → fall back to ``content.strip()[:1200]`` (verbatim
    for short input, truncated at 1200 chars for long input)."""
    content = '{"confidence": 0.5, "fault_type": "x"}'
    rc = parse_root_cause(content)
    assert rc.summary  # non-empty — sliced from content
    assert "confidence" in rc.summary


def test_parse_bare_object_empty_summary_truncates_long_content():
    """When the fallback content exceeds 1200 chars, the summary is truncated."""
    long_body = '"x": "' + ("a" * 2000) + '"'
    rc = parse_root_cause("{" + long_body + "}")
    assert len(rc.summary) <= 1200


# --------------------------------------------------------------------------- #
# strategy 3: markdown sections
# --------------------------------------------------------------------------- #
_MD_DOC = """\
### 1. summary
(根因总结) payment service at charge.js:65 rejects gold loyalty tokens.

### 2. fault_type
app.exception

### 3. entity_refs
| name | type | domain |
|------|------|--------|
| payment | apm.service | apm |
| checkout | apm.service | apm |

### 4. evidence
- query_logs: Invalid token. app.loyalty.level=gold
- query_metrics: error_rate spike 0.2→0.9

### 5. confidence
0.82

### 6. recommended_actions
- rollback payment deploy
- toggle feature flag off
"""


def test_parse_markdown_sections_full():
    rc = parse_root_cause(_MD_DOC)
    # summary: bilingual parenthetical dropped
    assert "根因总结" not in rc.summary
    assert "charge.js:65" in rc.summary
    # fault_type extracted
    assert rc.fault_type == "app.exception"
    # entity_refs: header row dropped, 2 entities parsed
    assert len(rc.entity_refs) == 2
    assert rc.entity_refs[0] == {
        "entity_name": "payment", "entity_type": "apm.service", "entity_domain": "apm"
    }
    assert rc.entity_refs[1]["entity_name"] == "checkout"
    # evidence bullets parsed
    assert any("Invalid token" in e for e in rc.evidence)
    assert any("error_rate" in e for e in rc.evidence)
    # confidence from section
    assert rc.confidence == pytest.approx(0.82)
    # recommended actions
    assert rc.recommended_actions == ["rollback payment deploy", "toggle feature flag off"]


def test_parse_markdown_header_row_dropped_single_row():
    """Even with one data row, the header row is dropped (not returned)."""
    md = """\
### entity_refs
| name | type | domain |
|------|------|--------|
| payment | apm.service | apm |
"""
    rc = parse_root_cause(md)
    assert rc.entity_refs == [
        {"entity_name": "payment", "entity_type": "apm.service", "entity_domain": "apm"}
    ]


def test_parse_markdown_evidence_table_form():
    """Evidence rendered as a markdown table also feeds the evidence list."""
    md = """\
### evidence
| source | finding |
|--------|---------|
| query_logs | Invalid token |
"""
    rc = parse_root_cause(md)
    joined = " ".join(rc.evidence)
    assert "query_logs" in joined
    assert "Invalid token" in joined
    # header row must NOT leak into evidence
    assert "source" not in joined


def test_parse_markdown_evidence_table_and_bullets_both_emitted():
    """When the evidence section contains BOTH a table and bullets, the table
    rows and the bullets are each emitted (the markdown path appends table rows
    then extends with bullets — an item present in both forms is emitted twice).
    This test documents that current behavior so a future change is intentional."""
    md = """\
### evidence
| source | finding |
|--------|---------|
| query_logs | Invalid token |
- query_logs: Invalid token
"""
    rc = parse_root_cause(md)
    # table row present
    assert any("query_logs" in e and "Invalid token" in e for e in rc.evidence)
    # bullet present
    assert any(e == "query_logs: Invalid token" for e in rc.evidence)
    # both forms emitted → the shared finding appears twice
    assert sum("Invalid token" in e for e in rc.evidence) == 2


def test_parse_markdown_confidence_default_when_absent():
    md = "### summary\nSome root cause sentence here.\n"
    rc = parse_root_cause(md)
    assert rc.confidence == pytest.approx(0.5)  # md-path default


# --------------------------------------------------------------------------- #
# strategy 4: free-text fallback
# --------------------------------------------------------------------------- #
def test_parse_free_text_fallback():
    rc = parse_root_cause("Looks like the payment service is rejecting tokens.")
    assert "payment service" in rc.summary
    assert rc.fault_type is None
    assert rc.confidence == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# never raises — empty / garbage / None
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("content", [None, ""])
def test_parse_empty_or_whitespace_never_raises(content):
    rc = parse_root_cause(content)
    assert rc.summary  # non-empty placeholder
    assert rc.confidence == pytest.approx(0.0)


def test_parse_whitespace_only_falls_to_freetext():
    """Non-empty whitespace-only string is truthy at the `if not content` guard,
    so it falls through to the free-text fallback (confidence 0.3), not the
    empty-conclusion branch (0.0)."""
    rc = parse_root_cause("   \n\t  ")
    assert isinstance(rc.summary, str)
    assert rc.confidence == pytest.approx(0.3)


def test_parse_pure_garbage_never_raises():
    rc = parse_root_cause("!!! @#$ no structure here at all")
    assert isinstance(rc.summary, str)
    assert rc.summary  # non-empty


def test_parse_returns_root_cause_instance_always():
    from rca_agent.contracts import RootCause

    for content in [None, "", "x", "{}", "```json\n{}\n```", _MD_DOC]:
        assert isinstance(parse_root_cause(content), RootCause)


# --------------------------------------------------------------------------- #
# confidence clamping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        (-0.5, 0.0),
        (-1, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.0),
        (12, 1.0),
    ],
)
def test_confidence_clamp_json_path(raw, expected):
    rc = parse_root_cause(f'{{"summary": "x", "confidence": {raw}}}')
    assert rc.confidence == pytest.approx(expected)


def test_confidence_non_numeric_json_defaults_to_zero():
    rc = parse_root_cause('{"summary": "x", "confidence": "high"}')
    assert rc.confidence == pytest.approx(0.0)


def test_confidence_missing_json_defaults_to_zero():
    rc = parse_root_cause('{"summary": "x"}')
    assert rc.confidence == pytest.approx(0.0)


def test_confidence_clamp_markdown_path():
    md = "### summary\nx\n### confidence\n1.7\n"
    rc = parse_root_cause(md)
    assert rc.confidence == pytest.approx(1.0)


def test_confidence_negative_markdown_sign_ignored():
    """The markdown-path confidence regex does not capture a leading sign, so
    `-0.3` parses as 0.3 (the documented best-effort behavior)."""
    md = "### summary\nx\n### confidence\n-0.3\n"
    rc = parse_root_cause(md)
    assert rc.confidence == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# entity_refs / fault_type / evidence extraction (JSON path variants)
# --------------------------------------------------------------------------- #
def test_entity_refs_extraction_fields():
    rc = parse_root_cause(
        '{"summary": "x", "entity_refs": ['
        '{"entity_name": "a", "entity_type": "apm.service", "entity_domain": "apm"},'
        '{"entity_name": "b", "entity_type": "k8s.pod", "entity_domain": "k8s"}'
        "]}"
    )
    assert len(rc.entity_refs) == 2
    assert {e["entity_name"] for e in rc.entity_refs} == {"a", "b"}
    assert all({"entity_name", "entity_type", "entity_domain"} <= set(e) for e in rc.entity_refs)


def test_entity_refs_non_list_treated_as_empty():
    rc = parse_root_cause('{"summary": "x", "entity_refs": "notalist"}')
    assert rc.entity_refs == []


def test_fault_type_extraction_json():
    rc = parse_root_cause('{"summary": "x", "fault_type": "k8s.pod_crashloop"}')
    assert rc.fault_type == "k8s.pod_crashloop"


def test_fault_type_missing_json_is_none():
    rc = parse_root_cause('{"summary": "x"}')
    assert rc.fault_type is None


def test_evidence_list_coerced_to_strings():
    rc = parse_root_cause(
        '{"summary": "x", "evidence": [42, {"k": "v"}, "plain"]}'
    )
    assert rc.evidence == ["42", "{'k': 'v'}", "plain"]


def test_contributing_and_actions_coerced_to_strings():
    rc = parse_root_cause(
        '{"summary": "x",'
        '"contributing_factors": [7],'
        '"recommended_actions": [true]}'
    )
    assert rc.contributing_factors == ["7"]
    assert rc.recommended_actions == ["True"]


# --------------------------------------------------------------------------- #
# helper functions (internal — locked for refactor-safety)
# --------------------------------------------------------------------------- #
def test_md_table_rows_drops_header_and_separator():
    block = "| name | type |\n|------|------|\n| a | b |\n| c | d |"
    rows = _md_table_rows(block)
    assert rows == [["a", "b"], ["c", "d"]]


def test_md_table_rows_single_row_kept():
    """When the table has exactly one row, the header-drop guard
    (``len(rows) > 1``) does not trigger, so the row is returned as-is."""
    assert _md_table_rows("| name | type |") == [["name", "type"]]


def test_clean_section_lead_drops_bilingual_parenthetical():
    assert _clean_section_lead("(根因总结) actual text") == "actual text"
    assert _clean_section_lead("（中文括号） text") == "text"
    assert _clean_section_lead(":  -  text") == "text"


def test_split_md_sections_returns_canonical_fields():
    secs = _split_md_sections(_MD_DOC)
    assert set(secs) >= {
        "summary", "fault_type", "entity_refs", "evidence",
        "confidence", "recommended_actions",
    }


# --------------------------------------------------------------------------- #
# core.py compatibility — fenced json path producing app.exception / 0.85
# --------------------------------------------------------------------------- #
def test_fenced_json_path_matches_core_expectation():
    """The exact final-answer shape produced by tests/agent/test_core.FakeLLM
    must round-trip to fault_type=app.exception, confidence=0.85, payment
    entity_ref."""
    import json as _json
    payload = _json.dumps({
        "summary": "payment service charge.js:65 rejects gold payments",
        "fault_type": "app.exception",
        "entity_refs": [{"entity_name": "payment", "entity_type": "apm.service",
                         "entity_domain": "apm"}],
        "evidence": ["query_logs: Invalid token. app.loyalty.level=gold"],
        "confidence": 0.85,
        "contributing_factors": [],
        "recommended_actions": ["rollback payment deploy"],
    })
    rc = parse_root_cause("```json\n" + payload + "\n```")
    assert rc.fault_type == "app.exception"
    assert rc.confidence == pytest.approx(0.85)
    assert any(e.get("entity_name") == "payment" for e in rc.entity_refs)


# --------------------------------------------------------------------------- #
# build_initial_brief
# --------------------------------------------------------------------------- #
def test_brief_includes_task_summary():
    task = _task()
    brief = build_initial_brief(task, None, [])
    assert task.task_id in brief
    assert task.alert_title in brief
    assert task.prompt_text in brief
    # window rendered as ISO UTC
    assert _WINDOW.start.isoformat() in brief
    assert "UTC" in brief
    # modalities
    assert "logs" in brief and "metrics" in brief


def test_brief_includes_topology_render():
    topo = _topology(
        entities=[
            TopologyEntity(id="p", type="apm.service", name="payment"),
            TopologyEntity(id="c", type="apm.service", name="checkout"),
            TopologyEntity(id="db", type="db", name="orders"),
        ],
        edges=[TopologyEdge(src="c", src_type="apm.service", dst="p",
                            dst_type="apm.service", relation="calls")],
    )
    brief = build_initial_brief(_task(), topo, [])
    assert "topology" in brief.lower()
    assert "3 entities" in brief
    assert "2 services" in brief
    assert "1 relations" in brief


def test_brief_includes_memory_hits():
    hits = [
        _mem("Runbook: if loyalty token invalid, check feature flag X.", kind="runbook"),
        _mem("Domain fact: apm.service 5xx spikes correlate with deploys.", kind="domain_fact"),
    ]
    brief = build_initial_brief(_task(), None, hits)
    assert "relevant knowledge" in brief.lower()
    assert "Runbook" in brief
    assert "Domain fact" in brief
    assert "[runbook]" in brief
    assert "[domain_fact]" in brief


def test_brief_truncates_memory_content():
    long = "x" * 1000
    hits = [_mem(long)]
    brief = build_initial_brief(_task(), None, hits)
    # content[:240]
    assert "x" * 240 in brief
    assert "x" * 241 not in brief


def test_brief_caps_memory_hits_to_six():
    hits = [_mem(f"item {i}") for i in range(12)]
    brief = build_initial_brief(_task(), None, hits)
    for i in range(6):
        assert f"item {i}" in brief
    for i in range(6, 12):
        assert f"item {i}" not in brief


def test_brief_degrades_when_memory_empty():
    brief = build_initial_brief(_task(), None, [])
    assert "relevant knowledge" not in brief.lower()


def test_brief_degrades_when_topology_none():
    brief = build_initial_brief(_task(), None, [])
    assert "topology" not in brief.lower()


def test_brief_degrades_when_topology_empty():
    """A Topology with zero entities is still rendered but with zero counts."""
    brief = build_initial_brief(_task(), _topology(), [])
    # topology section present (Topology is not None), with 0 counts
    assert "0 entities" in brief


def test_brief_alert_entity_name_and_type():
    task = _task(alert_entity={
        "entity_name": "checkout", "entity_type": "apm.operation",
        "entity_domain": "apm",
    })
    brief = build_initial_brief(task, None, [])
    assert "checkout" in brief
    assert "apm.operation" in brief


def test_brief_alert_entity_only_id():
    task = _task(alert_entity={"entity_id": "abc123"})
    brief = build_initial_brief(task, None, [])
    assert "abc123" in brief
    assert "entity_id" in brief


def test_brief_alert_entity_missing():
    """No entity fields at all → unknown-entity hint."""
    brief = build_initial_brief(_task(alert_entity={}), None, [])
    assert "unknown" in brief.lower()


def test_brief_modalities_default_all_when_empty():
    task = _task(modalities=[])
    brief = build_initial_brief(task, None, [])
    # empty modalities list → "all"
    assert "modalities: all" in brief.lower()


# --------------------------------------------------------------------------- #
# S4: skill_name pointer (keyword-only, backward-compatible)
# --------------------------------------------------------------------------- #
def test_brief_skill_name_pointer_prepended():
    """When skill_name is set, a one-line 'Loaded SOP' pointer is prepended."""
    task = _task()
    brief = build_initial_brief(task, None, [], skill_name="myskill")
    assert brief.startswith("已加载排查 SOP: myskill")
    assert "Loaded SOP: myskill (see system prompt)" in brief
    # The rest of the brief is unchanged.
    assert task.alert_title in brief
    assert task.task_id in brief


def test_brief_skill_name_none_is_byte_identical_to_omitted():
    """Passing skill_name=None (the default) must produce byte-identical output
    to not passing the arg at all — so every existing caller is unaffected."""
    task = _task()
    a = build_initial_brief(task, None, [])
    b = build_initial_brief(task, None, [], skill_name=None)
    assert a == b
    assert "Loaded SOP" not in a
    assert "已加载排查 SOP" not in a


def test_brief_skill_name_keyword_only():
    """skill_name is keyword-only: passing it positionally must TypeError."""
    task = _task()
    with pytest.raises(TypeError):
        build_initial_brief(task, None, [], "myskill")  # type: ignore[misc]
