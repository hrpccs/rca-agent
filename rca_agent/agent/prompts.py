"""Agent-level prompt assembly: the initial investigation brief and the
final-answer parser that turns the model's conclusion into a RootCause.

The system prompt + final-answer guidance live in :mod:`rca_agent.tools.prompts`
(authored by the tools unit); this module adds the case-specific framing and the
parse logic.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import RootCause, Task, Topology


def build_initial_brief(task: Task, topology: Topology | None, memory_hits: list[Any]) -> str:
    """The first user message: the alert under investigation + context.

    Includes the alert title/window/entity, the available modalities, a one-line
    topology summary, and any retrieved memory (runbook/domain knowledge).
    """
    w = task.alert_window
    ent = task.alert_entity or {}
    ent_str = ""
    if ent.get("entity_name") or ent.get("entity_type"):
        ent_str = (
            f"告警主体 / alert entity: {ent.get('entity_name') or '?'} "
            f"({ent.get('entity_type') or '?'}, {ent.get('entity_domain') or '?'})"
        )
    elif ent.get("entity_id"):
        ent_str = f"告警主体 entity_id: {ent['entity_id']} (name/type unknown — discover via topology)"
    else:
        ent_str = "告警主体缺失（entity 未知）—— 需通过告警标题与拓扑推断主体。/ Alert entity unknown."

    topo_str = ""
    if topology is not None:
        n_svc = sum(1 for e in topology.entities if e.type == "apm.service")
        n_ent = len(topology.entities)
        n_edge = len(topology.edges)
        topo_str = (
            f"拓扑 / topology: {n_ent} entities ({n_svc} services), {n_edge} relations. "
            "Use get_topology / inspect_entity to traverse dependencies."
        )

    mem_str = ""
    if memory_hits:
        lines = []
        for m in memory_hits[:6]:
            tag = f"[{m.kind}]" if getattr(m, "kind", None) else ""
            lines.append(f"- {tag} {m.content[:240]}")
        mem_str = "相关知识 / relevant knowledge from memory:\n" + "\n".join(lines)

    modalities = ", ".join(m.value for m in (task.available_modalities or [])) or "all"

    return (
        f"请分析以下告警的根因。\n"
        f"任务 / task_id: {task.task_id}\n"
        f"告警 / alert: {task.alert_title}\n"
        f"告警窗口 / alert_window: {w.start.isoformat()} ~ {w.end.isoformat()} (UTC)\n"
        f"{ent_str}\n"
        f"可用数据模态 / available modalities: {modalities}\n"
        f"{topo_str}\n"
        f"用户原始请求 / user prompt: {task.prompt_text}\n"
        f"{mem_str}\n"
        f"\n请按系统提示的调查方法，使用工具收集证据并收敛到根因。"
        f"完成调查后，不要再调用工具，直接返回结构化最终结论（字段见系统提示末尾的 final-answer guidance）。"
    )


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_MD_HEADING_RE = re.compile(
    r"^#{1,4}\s*\d*\.?\s*([a-zA-Z_]+)", re.MULTILINE
)

# Map heading words (canonical) to RootCause fields, incl. bilingual/aliases.
_HEADING_ALIASES = {
    "summary": "summary", "根因": "summary", "summ": "summary",
    "fault_type": "fault_type", "fault": "fault_type", "故障": "fault_type",
    "entity_refs": "entity_refs", "entities": "entity_refs", "实体": "entity_refs",
    "evidence": "evidence", "证据": "evidence",
    "confidence": "confidence", "置信": "confidence",
    "contributing_factors": "contributing_factors", "contributing": "contributing_factors",
    "促成": "contributing_factors",
    "recommended_actions": "recommended_actions", "actions": "recommended_actions",
    "action": "recommended_actions", "建议": "recommended_actions", "处置": "recommended_actions",
}


def _strip_md(text: str) -> str:
    """Strip common markdown bold/code markers; collapse whitespace."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _md_table_rows(block: str) -> list[list[str]]:
    """Parse a markdown table block into rows of cell text. Drops the header
    row (markdown tables always have one) when more than one row is present."""
    rows: list[list[str]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [_strip_md(c) for c in line.strip("|").split("|")]
        # skip separator rows like |---|---|
        if all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()):
            continue
        rows.append(cells)
    if len(rows) > 1:
        rows = rows[1:]  # drop the header row
    return rows


def _clean_section_lead(s: str) -> str:
    """Drop a leading bilingual parenthetical like '(根因总结)' that follows the
    heading word, plus leading punctuation/whitespace."""
    s = re.sub(r"^\s*[\(（][^\)）]*[\)）]\s*", "", s)
    s = re.sub(r"^[\s:：\-—•]+", "", s)
    return s.strip()


def _bullets(block: str) -> list[str]:
    """Extract bullet/numbered list items from a block."""
    out: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        m = re.match(r"^[-*•]\s+(.*)", line) or re.match(r"^\d+[.)]\s+(.*)", line)
        if m:
            item = _strip_md(m.group(1))
            if item:
                out.append(item)
    return out


def _split_md_sections(text: str) -> dict[str, str]:
    """Split markdown into {canonical_field: section_text} by headings."""
    matches = list(_MD_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        word = m.group(1).lower()
        field = _HEADING_ALIASES.get(word)
        if not field:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[field] = text[start:end].strip()
    return sections


def parse_root_cause(content: str | None) -> RootCause:
    """Best-effort parse of the model's final answer into a RootCause.

    Strategy:
      1. fenced ```json block → json.loads
      2. first balanced {...} blob → json.loads
      3. markdown sections (### 1. summary …) → structured extraction
      4. fallback: treat the whole text as a low-confidence summary
    """
    if not content:
        return RootCause(summary="(空结论 / empty conclusion)", confidence=0.0)

    # --- 1/2. JSON extraction ---
    data: dict[str, Any] | None = None
    m = _JSON_BLOCK_RE.search(content)
    if m:
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            data = None
    if data is None:
        m = _BARE_OBJ_RE.search(content)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        def _str(k: str, default: str = "") -> str:
            v = data.get(k, default)
            return str(v).strip() if v is not None else default

        def _list(k: str) -> list[Any]:
            v = data.get(k)
            return list(v) if isinstance(v, list) else []

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        return RootCause(
            summary=_str("summary") or content.strip()[:1200],
            fault_type=_str("fault_type") or None,
            entity_refs=_list("entity_refs"),
            evidence=[str(x) for x in _list("evidence")],
            confidence=confidence,
            contributing_factors=[str(x) for x in _list("contributing_factors")],
            recommended_actions=[str(x) for x in _list("recommended_actions")],
        )

    # --- 3. markdown sections fallback ---
    secs = _split_md_sections(content)
    if secs:
        def first_line(s: str) -> str:
            s = _strip_md(s)
            return s.split("。")[0].split(". ")[0].split("—")[0].strip() if s else ""

        summary = _strip_md(_clean_section_lead(secs.get("summary", "")))[:1500]
        fault_type = first_line(_clean_section_lead(secs.get("fault_type", ""))) or None

        entity_refs: list[Any] = []
        ent_block = secs.get("entity_refs", "")
        for row in _md_table_rows(ent_block):
            if len(row) >= 3:
                entity_refs.append({
                    "entity_name": row[0], "entity_type": row[1], "entity_domain": row[2],
                })
            elif len(row) == 2:
                entity_refs.append({"entity_name": row[0], "entity_type": row[1]})

        evidence: list[str] = []
        ev_rows = _md_table_rows(secs.get("evidence", ""))
        if ev_rows:
            for row in ev_rows:
                evidence.append(": ".join(c for c in row if c))
        evidence.extend(_bullets(secs.get("evidence", "")))

        actions = _bullets(secs.get("recommended_actions", "")) or [
            _strip_md(l) for l in secs.get("recommended_actions", "").splitlines() if l.strip()
        ]
        contrib = _bullets(secs.get("contributing_factors", ""))

        confidence = 0.5
        cm = re.search(r"([0-9]*\.?[0-9]+)", secs.get("confidence", ""))
        if cm:
            try:
                confidence = max(0.0, min(1.0, float(cm.group(1))))
            except ValueError:
                pass

        return RootCause(
            summary=summary or content.strip()[:1200],
            fault_type=fault_type,
            entity_refs=entity_refs,
            evidence=evidence,
            confidence=confidence,
            contributing_factors=contrib,
            recommended_actions=actions,
        )

    # --- 4. fallback ---
    return RootCause(summary=content.strip()[:1500], confidence=0.3)


__all__ = ["build_initial_brief", "parse_root_cause"]
