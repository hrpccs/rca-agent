"""RCA agent core — the LLM-driven ReAct investigation loop.

Wires the frozen-contract building blocks together:
  LLMClient (DeepSeek thinking) + ContextManager (reasoning_content echo) +
  Tools (SRE toolkit over a DataProvider) + MemoryStore.

``RcaAgent.run(case)`` is an async generator that yields each :class:`RcaStep`
(thought / tool_call / tool_result / conclude) as it happens and finally yields
the :class:`RcaReport`. This is what the SSE server streams and what the CLI
prints.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..cases import load_case
from ..config import Settings, get_settings
from ..context.manager import ContextManager, build_context_manager
from ..contracts import (
    Case,
    LLMRequest,
    RcaReport,
    RcaStep,
    RootCause,
    StepKind,
    ToolCall,
    ToolMessage,
    build_openai_tools,
    validate_tool_call,
)
from ..llm.deepseek_client import default_client
from ..memory.inmemory_store import InMemoryStore
from ..tools.prompts import SYSTEM_PROMPT, to_final_answer_guidance
from ..tools.registry import build_default_tools
from .prompts import build_initial_brief, parse_root_cause

GLOBAL = "__global__"
_SEED_DIR = Path(__file__).resolve().parents[2] / "memory" / "seed"

logger = logging.getLogger(__name__)

# S4 — Skills integration env gate.
# The skills engine (S1 store + S2 recall + S3 content) is wired into the
# ReAct loop via this knob. Defaults ON (``"1"``) so a fresh checkout benefits
# from per-fault-type SOP injection; OFF values (``0``/``false``/``no``/``off``)
# reproduce the EXACT pre-skills agent behavior — essential for before/after
# ablation runs. Parsed via the same ``os.environ.get`` idiom used by
# ``RCA_FORCE_CONCLUDE`` above and sibling numeric knobs; NOT a config.py field
# so the frozen config surface is untouched.
_SKILLS_ENABLED_ENV = "RCA_SKILLS_ENABLED"


@runtime_checkable
class SkillLibrary(Protocol):
    """Structural seam for the skills engine the agent recalls SOPs from.

    ``SkillRecaller`` (from :mod:`rca_agent.skills.recall`) already satisfies
    this Protocol structurally — it exposes both ``catalog()`` and
    ``best_for()``. Declared as a Protocol (not a concrete import) so the agent
    core stays decoupled from the skills package at the type level: tests can
    inject any duck-typed fake, and a misconfigured/missing engine never breaks
    the import graph. Runtime-checkable so ``isinstance`` works in tests.
    """

    def catalog(self) -> list[tuple[str, str]]:
        """Return ``[(name, description), ...]`` for the catalog disclosure."""
        ...

    def best_for(
        self, alert_title: str, signals: list[str] | None = None
    ) -> tuple[str, str] | None:
        """Return ``(skill_name, sop_body)`` for the best-matched SOP, or None."""
        ...


def _skills_enabled() -> bool:
    """Read ``RCA_SKILLS_ENABLED``; default ON, falsy values disable.

    Mirrors ``_force_conclude_enabled``'s idiom: OFF when the value is empty or
    one of ``0``/``false``/``no``/``off`` (case-insensitive). Anything else
    (including the unset default) enables skills — the safer default, since an
    unset env var should not silently regress SOP injection.
    """
    raw = os.environ.get(_SKILLS_ENABLED_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _default_skill_library() -> SkillLibrary | None:
    """Build the default skills library from the real engine, or None.

    Lazily imports the skills package so a missing/broken engine never breaks
    agent construction. Returns ``None`` (logged at warning) on ANY failure —
    import error, construction error, or env-gate OFF — so callers can treat a
    falsy return as "skills unavailable; run proceeds exactly as before skills".
    Never raises.
    """
    if not _skills_enabled():
        return None
    try:
        from ..skills.recall import SkillRecaller
        from ..skills.store import SkillStore

        return SkillRecaller(SkillStore())
    except _NONFATAL_EXC as e:
        logger.warning(
            "skills library construction failed (%s: %s); running without SOP injection",
            type(e).__name__, e,
            extra={"component": "skills", "error": str(e)},
        )
        return None
    except Exception as e:  # noqa: BLE001 — engine is pluggable; never fatal
        logger.warning(
            "skills library construction raised unexpected %s; running without SOP injection: %s",
            type(e).__name__, e,
            extra={"component": "skills", "error": str(e)},
        )
        return None


# I2 — force-conclude fallback at the step cap.
# When the ReAct loop exhausts ``max_steps`` without the model emitting a final
# answer, we make ONE extra forced-conclusion LLM call to recover a best-effort
# root cause instead of yielding a placeholder summary + confidence 0.0 (which
# the 8-case eval showed burning 768K tokens for zero usable output). The gate
# defaults ON (``"1"``) but can be turned OFF (``"0"``) to restore the exact
# prior truncation behavior — fully reversible. Parsed via the same
# ``os.environ.get`` idiom used elsewhere (cli.py, deepseek_client.py); NOT a
# config.py field so the frozen config surface is untouched.
_FORCE_CONCLUDE_ENV = "RCA_FORCE_CONCLUDE"


def _force_conclude_enabled() -> bool:
    """Read ``RCA_FORCE_CONCLUDE``; default ON, falsy values disable.

    A value is OFF when it is empty or one of ``0``/``0.0``/``false``/``no``/
    ``off``/``disable``/``disabled`` (case-insensitive). Numeric-zero spellings
    are accepted because sibling knobs (``RCA_LLM_MAX_RETRIES``,
    ``RCA_MEMORY_MAX_PER_BUCKET``) are numeric, so an operator may reasonably
    write ``RCA_FORCE_CONCLUDE=0.0``. Anything else enables recovery — the
    safer default for a root-cause agent (an unset/typo'd env var still
    recovers a best-effort answer instead of truncating to nothing).
    """
    raw = os.environ.get(_FORCE_CONCLUDE_ENV, "1").strip().lower()
    return raw not in {"0", "0.0", "false", "no", "off", "disable", "disabled", ""}

# Non-fatal failures from pluggable collaborators (memory backends, OTel
# exporters) that must never kill the ReAct loop. Kept broad on purpose: a
# backend can wrap arbitrary I/O, so we enumerate the realistic concrete
# failures and fall back to a last-resort Exception guard (always logged).
_NONFATAL_EXC: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    LookupError,
    AttributeError,
    TypeError,
)


def _safe_otel(fn_name: str, *args, **kwargs) -> None:
    """Best-effort observability call — never let telemetry break the agent.

    OTel is a pluggable surface: the metrics module import, the named recorder
    attribute, and the recorder call itself can each fail in isolation (exporter
    network errors, SDK version drift, misconfiguration). We catch the specific
    import/attribute failures up front with a structured warning so a
    misconfiguration is diagnosable, then keep a last-resort guard on the actual
    instrument call (logged at warning) so an exporter error can never propagate
    into the investigation loop.
    """
    try:
        from ..observability import metrics as _m
    except ImportError as e:  # observability package missing/misconfigured
        logger.warning(
            "otel metrics import failed; disabling telemetry: %s", e,
            extra={"component": "otel", "recorder": fn_name, "error": str(e)},
        )
        return
    # getattr(..., default) does NOT suppress errors raised by a module's own
    # ``__getattr__`` (it only supplies the default for a plain AttributeError),
    # so guard the lookup explicitly — a broken observability shim must never
    # leak into the ReAct loop.
    try:
        fn = getattr(_m, fn_name, None)
    except _NONFATAL_EXC as e:
        logger.warning(
            "otel metrics lookup for %r raised %s; skipping (non-fatal): %s",
            fn_name, type(e).__name__, e,
            extra={"component": "otel", "recorder": fn_name, "error": str(e)},
        )
        return
    except Exception as e:  # noqa: BLE001 — last-resort: telemetry must not kill the loop
        logger.warning(
            "otel metrics lookup for %r raised unexpected %s; skipping (non-fatal): %s",
            fn_name, type(e).__name__, e,
            extra={"component": "otel", "recorder": fn_name, "error": str(e)},
        )
        return
    if fn is None or not callable(fn):
        logger.warning(
            "otel metrics recorder %r not found; skipping", fn_name,
            extra={"component": "otel", "recorder": fn_name},
        )
        return
    try:
        fn(*args, **kwargs)
    except _NONFATAL_EXC as e:
        logger.warning(
            "otel recorder %s raised %s; skipping (non-fatal): %s",
            fn_name, type(e).__name__, e,
            extra={"component": "otel", "recorder": fn_name, "error": str(e)},
        )
    except Exception as e:  # noqa: BLE001 — last-resort: telemetry must not kill the loop
        logger.warning(
            "otel recorder %s raised unexpected %s; skipping (non-fatal): %s",
            fn_name, type(e).__name__, e,
            extra={"component": "otel", "recorder": fn_name, "error": str(e)},
        )


class RcaAgent:
    """LLM-core RCA agent. Stateless except for the injected collaborators."""

    def __init__(
        self,
        provider: Any,
        llm: Any,
        memory: Any,
        context_manager: ContextManager | None = None,
        tools: list | None = None,
        settings: Settings | None = None,
        max_steps: int | None = None,
        model: str | None = None,
        skill_library: Any | None = None,
    ) -> None:
        self.provider = provider
        self.llm = llm
        self.memory = memory
        self.cm = context_manager or build_context_manager()
        self.tools = tools if tools is not None else build_default_tools(provider, memory)
        self.settings = settings or get_settings()
        self.max_steps = max_steps or self.settings.llm_max_steps
        self.model = model or self.settings.deepseek_model
        # S4: skills library for per-fault-type SOP recall. Falls back to the
        # default (real engine, env-gated) when no library is injected — so the
        # production path gets SOP injection for free, while tests can inject a
        # fake or pass None to exercise the env-gate-OFF path.
        self.skill_library = (
            skill_library if skill_library is not None else _default_skill_library()
        )

    def _step_id(self, case_id: str) -> str:
        return f"{case_id}-{uuid.uuid4().hex[:10]}"

    def _build_skill_block(self, case: Case) -> tuple[str, str | None]:
        """Recall the best SOP for this alert and assemble the system-prompt block.

        Returns ``(skill_block, loaded_skill_name)`` where ``skill_block`` is the
        string to append to the system prompt (possibly empty) and
        ``loaded_skill_name`` is the name of the matched skill (for the
        display-only trace step), or ``None`` when nothing was loaded.

        Never raises: every engine call is wrapped so a throwing recaller,
        empty catalog, or env-gate-OFF all yield ``("", None)`` and the run
        proceeds exactly as before skills. See ``run()`` for why the block is
        system-prompt-only (durable + compaction-protected).
        """
        if self.skill_library is None:
            return "", None

        # Recall the single best SOP. ``best_for`` already returns None below
        # the engine's score threshold, so we only inject a genuinely relevant
        # SOP — never force-inject noise.
        match: tuple[str, str] | None = None
        try:
            match = self.skill_library.best_for(case.task.alert_title)
        except _NONFATAL_EXC as e:
            logger.warning(
                "skill best_for(%r) failed (%s: %s); running without SOP injection",
                case.task.alert_title, type(e).__name__, e,
                extra={
                    "component": "skills", "case_id": case.task.task_id,
                    "alert": case.task.alert_title, "error": str(e),
                },
            )
            match = None
        except Exception as e:  # noqa: BLE001 — engine is pluggable; never fatal
            logger.warning(
                "skill best_for(%r) raised unexpected %s; running without SOP injection: %s",
                case.task.alert_title, type(e).__name__, e,
                extra={
                    "component": "skills", "case_id": case.task.task_id,
                    "alert": case.task.alert_title, "error": str(e),
                },
            )
            match = None

        parts: list[str] = []
        loaded_name: str | None = None
        if match is not None:
            # Defensive: a pluggable/fake SkillLibrary could return a wrong-
            # arity sequence (1-tuple, 3-tuple, list) or non-str elements. The
            # method's "Never raises" contract requires this unpack be guarded
            # — a ValueError/TypeError from a malformed match must not abort
            # the run. (The real SkillRecaller always returns a clean 2-tuple
            # or None, but the Protocol invites duck-typed fakes.)
            try:
                name, body = match  # type: ignore[misc]
                # Coerce so the injected block is always well-formed even if a
                # fake returns non-str elements.
                name = str(name) if name is not None else ""
                body = str(body) if body is not None else ""
            except (TypeError, ValueError) as e:
                logger.warning(
                    "skill best_for returned malformed match %r (%s: %s); "
                    "skipping SOP injection",
                    match, type(e).__name__, e,
                    extra={"component": "skills", "error": str(e)},
                )
                name, body = "", ""
            if name and body:
                loaded_name = name
                parts.append(
                    "\n\n# 已加载排查技能 / Loaded troubleshooting skill\n"
                    f"<loaded_skill name=\"{name}\">\n{body}\n</loaded_skill>"
                )

        # Tier-1 disclosure: a compact catalog so the model knows what SOPs
        # exist (without paying their full token cost). Kept small — one line
        # per skill. A throwing catalog() yields [] and is skipped.
        try:
            catalog = self.skill_library.catalog()
        except _NONFATAL_EXC as e:
            logger.warning(
                "skill catalog() failed (%s: %s); omitting catalog disclosure",
                type(e).__name__, e,
                extra={"component": "skills", "error": str(e)},
            )
            catalog = []
        except Exception as e:  # noqa: BLE001 — engine is pluggable; never fatal
            logger.warning(
                "skill catalog() raised unexpected %s; omitting catalog disclosure: %s",
                type(e).__name__, e,
                extra={"component": "skills", "error": str(e)},
            )
            catalog = []

        if catalog:
            lines = []
            for entry in catalog:
                try:
                    n, d = entry
                    n = str(n) if n is not None else ""
                    d = str(d) if d is not None else ""
                except (TypeError, ValueError):
                    continue
                if not n:
                    continue
                # Cap each description so the catalog stays compact even with
                # verbose skill metadata (the full body is already injected for
                # the winner above).
                lines.append(f"- {n}: {d[:120]}")
            if lines:
                parts.append(
                    "\n\n# 可用技能目录 / Available skills catalog\n"
                    "<available_skills>\n"
                    + "\n".join(lines)
                    + "\n</available_skills>"
                )

        return "".join(parts), loaded_name

    async def run(self, case: Case) -> AsyncIterator[RcaStep | RcaReport]:
        """Investigate ``case``; yield each step then the final report."""
        case_id = case.task.task_id
        _safe_otel("record_run", "started")

        # S4 — Recall the single best troubleshooting SOP for this alert and
        # inject it into the SYSTEM prompt. Injecting into the system message
        # (always messages[0]) is deliberate and load-bearing:
        #   * DURABLE: the system prompt is re-emitted on every turn by
        #     ``ContextManager.assemble_turn`` (see manager.py — it always
        #     prepends ``{"role": "system", "content": state.system}``), so the
        #     SOP survives the entire multi-step ReAct loop without being
        #     re-fetched.
        #   * COMPACTION-PROTECTED: ``ContextManager.compress`` ALWAYS keeps the
        #     system message (it is never in a droppable group — only assistant
        #     tool-call groups and user messages are summarizable), and the I4
        #     tool-message sliding window never touches the leading system
        #     prefix. So the SOP is never evicted under context pressure.
        #   * SINGLE BEST SOP (not all): the engine (S2) already picks the one
        #     highest-scoring SOP for the alert via the keyword router; loading
        #     every SOP would dilute the signal and bloat every prompt. The
        #     compact <available_skills> catalog below is the tier-1 disclosure
        #     that lets the model know other SOPs exist without paying their
        #     token cost.
        # Every skill call is try/except + logged: a malformed engine, a missing
        # skills dir, or a throwing recaller yields ``skill_block=""`` and the
        # run proceeds byte-identically to the pre-skills agent.
        skill_block, loaded_skill_name = self._build_skill_block(case)

        system = (
            SYSTEM_PROMPT
            + "\n\n# 最终结论结构 / Final-answer structure\n"
            + to_final_answer_guidance()
            + skill_block
        )
        state = self.cm.init(case_id, system)

        # Seed the agent's context with relevant prior knowledge, if any.
        # A memory-backend failure (corrupt store, bad retriever, etc.) must
        # never abort the investigation — the agent proceeds without priors.
        hits: list[Any] = []
        try:
            hits = self.memory.retrieve_for_context(GLOBAL, case.task.alert_title, top_k=6)
        except _NONFATAL_EXC as e:
            logger.warning(
                "memory retrieve_for_context failed for case %s; proceeding without priors (%s: %s)",
                case_id, type(e).__name__, e,
                extra={"component": "memory", "case_id": case_id, "error": str(e)},
            )
            hits = []
        except Exception as e:  # noqa: BLE001 — pluggable backend; never fatal
            logger.warning(
                "memory retrieve_for_context raised unexpected %s for case %s; proceeding without priors: %s",
                type(e).__name__, case_id, e,
                extra={"component": "memory", "case_id": case_id, "error": str(e)},
            )
            hits = []

        first_user = build_initial_brief(
            case.task, case.topology, hits, skill_name=loaded_skill_name
        )
        msgs = self.cm.assemble_turn(state, new_user=first_user)
        oa_tools = build_openai_tools(self.tools)

        steps: list[RcaStep] = []

        # Surface the memory module's interaction in the trace (DISPLAY ONLY).
        # This step records that priors were retrieved and what entities they
        # carried, so a persisted/replayed trace shows the memory module at work.
        # It is NEVER appended to the LLM's context: the only thing that feeds
        # the model below is `state = self.cm.append_tool_result(...)` (plus the
        # initial assemble_turn above, which already baked `hits` into the brief
        # independently of this step). Adding it to `steps` makes it land in the
        # final RcaReport for display/persistence; yielding it streams it live.
        # Defensive about pluggable backends: a malformed MemoryItem (missing,
        # None, or non-list `.entities` — e.g. a bare string, which would
        # otherwise iterate character-by-character) must not crash the loop or
        # emit garbage entities. `_mem_entities` only flattens lists-of-str and
        # skips anything else. The outer `isinstance(hits, list)` normalizes a
        # backend that hands back a single item/dict instead of a list —
        # `len(hits)` and the comprehension would otherwise raise.
        def _mem_entities() -> list[str]:
            out: list[str] = []
            for h in hits:
                ent_list = getattr(h, "entities", None)
                if not isinstance(ent_list, list):
                    continue
                for e in ent_list:
                    if isinstance(e, str) and e:
                        out.append(e)
            return sorted(set(out))[:20]

        if isinstance(hits, list) and hits:
            mem_step = RcaStep(
                step_id=self._step_id(case_id),
                case_id=case_id,
                step_kind=StepKind.REASONING,
                thought=(
                    f"memory: retrieved {len(hits)} prior(s) for "
                    f"{case.task.alert_title!r}"
                ),
                entities=_mem_entities(),
            )
            steps.append(mem_step)
            yield mem_step
            _safe_otel("record_step", "memory")

        # S4 — Surface the loaded skill in the trace (DISPLAY ONLY).
        # Mirrors the T3 memory step above: records that an SOP was recalled and
        # injected, so a persisted/replayed trace shows the skills engine at
        # work. NEVER fed to the LLM — the SOP body lives in the SYSTEM prompt
        # (assembled into ``state.system`` above), and the ONLY thing that feeds
        # the model below is ``self.cm.append_tool_result`` (plus the initial
        # ``assemble_turn``, which reads ``state.system`` + the brief — this
        # display step is not in either path). Adding it to ``steps`` lands it
        # in the final RcaReport for display/persistence; yielding it streams it
        # live. Wrapped in try/except so a malformed step object (or a yield
        # failure in an exotic async runner) can never abort the run.
        if loaded_skill_name:
            try:
                skill_step = RcaStep(
                    step_id=self._step_id(case_id),
                    case_id=case_id,
                    step_kind=StepKind.REASONING,
                    thought=(
                        f"loaded skill: {loaded_skill_name} "
                        f"(matched SOP for this alert)"
                    ),
                )
                steps.append(skill_step)
                yield skill_step
                _safe_otel("record_step", "skill")
            except _NONFATAL_EXC as e:
                logger.warning(
                    "skill display step emission failed (%s: %s); run continues",
                    type(e).__name__, e,
                    extra={"component": "skills", "case_id": case_id, "error": str(e)},
                )
            except Exception as e:  # noqa: BLE001 — display-only; never fatal
                logger.warning(
                    "skill display step emission raised unexpected %s; run continues: %s",
                    type(e).__name__, e,
                    extra={"component": "skills", "case_id": case_id, "error": str(e)},
                )

        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

        def _accum_usage(u: dict | None) -> None:
            if not u:
                return
            for k in usage_total:
                try:  # noqa: SIM105 - tolerate non-numeric usage fields
                    usage_total[k] += int(u.get(k, 0) or 0)
                except (TypeError, ValueError):
                    pass

        step_no = 0
        while step_no < self.max_steps:
            step_no += 1
            req = LLMRequest(
                messages=msgs,
                tools=oa_tools,
                model=self.model,
                reasoning_effort=self.settings.reasoning_effort,
                thinking_enabled=True,
                max_tokens=self.settings.llm_max_tokens,
            )
            content, reasoning, tool_calls, usage = await self.llm.complete(req)
            _accum_usage(usage)
            state = self.cm.append_assistant(state, content, reasoning, tool_calls)

            # Surface the thinking / narration for display (reasoning_content
            # is the DeepSeek thinking trace; fall back to content).
            thought = (reasoning or content or "").strip()
            if thought:
                s = RcaStep(
                    step_id=self._step_id(case_id),
                    case_id=case_id,
                    step_kind=StepKind.REASONING,
                    thought=thought[:800],
                )
                steps.append(s)
                yield s
                _safe_otel("record_step", "reasoning")

            if not tool_calls:
                # Final answer — parse + emit conclusion + report.
                rc = parse_root_cause(content)
                entities = [e.get("entity_name") or e.get("entity_id") or "" for e in rc.entity_refs if isinstance(e, dict)]
                s = RcaStep(
                    step_id=self._step_id(case_id),
                    case_id=case_id,
                    step_kind=StepKind.CONCLUDE,
                    hypothesis=rc.summary,
                    confidence=rc.confidence,
                    entities=[e for e in entities if e],
                )
                steps.append(s)
                yield s
                _safe_otel("record_step", "conclude")
                report = RcaReport(
                    case_id=case_id,
                    task_id=case.task.task_id,
                    alert_title=case.task.alert_title,
                    root_cause=rc,
                    steps=steps,
                    model=self.model,
                    token_usage=usage_total or None,
                    status="completed",
                )
                _safe_otel("record_run", "completed")
                yield report
                return

            # Execute each requested tool and feed results back.
            for tc in tool_calls:
                fn = (tc or {}).get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                args_str = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                tc_id = tc.get("id") if isinstance(tc, dict) else None

                s_call = RcaStep(
                    step_id=self._step_id(case_id),
                    case_id=case_id,
                    step_kind=StepKind.TOOL_CALL,
                    tool_name=name,
                    tool_args=args,
                )
                steps.append(s_call)
                yield s_call
                _safe_otel("record_step", "tool_call")

                ok = True
                try:
                    call = ToolCall(name=name, arguments=args, id=tc_id)
                    tool, vargs = validate_tool_call(call, self.tools)
                    result = tool.handler(vargs, self.provider, self.memory)
                    if not isinstance(result, dict):
                        result = {"result": result}
                except Exception as e:  # tool failure is evidence, not a crash
                    ok = False
                    result = {"error": f"{type(e).__name__}: {e}"}
                    logger.warning(
                        "tool %s raised %s for case %s; surfacing as evidence: %s",
                        name, type(e).__name__, case_id, e,
                        extra={
                            "component": "tool", "case_id": case_id,
                            "tool": name, "error": str(e),
                        },
                        exc_info=False,
                    )
                # A handler may also signal failure by returning a dict carrying
                # an "error" key (the builtin tools do this instead of raising,
                # so the agent loop keeps investigating) — count those as errors
                # in the tool-call metric too, not just raised exceptions.
                if ok and isinstance(result, dict) and result.get("error"):
                    ok = False
                _safe_otel("record_tool_call", name, "ok" if ok else "error")

                text = result.get("text") if isinstance(result, dict) else None
                if text is None:
                    text = json.dumps(result, ensure_ascii=False, default=str)[:4000]
                else:
                    text = str(text)

                s_res = RcaStep(
                    step_id=self._step_id(case_id),
                    case_id=case_id,
                    step_kind=StepKind.TOOL_RESULT,
                    tool_name=name,
                    tool_args=args,
                    tool_result=result,
                    tool_result_text=text,
                )
                steps.append(s_res)
                yield s_res
                _safe_otel("record_step", "tool_result")

                state = self.cm.append_tool_result(
                    state, [ToolMessage(tool_call_id=tc_id or "", name=name, content=text)]
                )

            msgs = self.cm.assemble_turn(state)

        # Out of steps without a final conclusion. I2: attempt a force-conclude
        # recovery so the run returns a usable root cause instead of a
        # placeholder + confidence 0.0 (the 8-case eval showed this path burning
        # 768K tokens for no useful output). Env-gated; when OFF we preserve the
        # exact prior behavior (placeholder summary, confidence 0.0, no extra
        # LLM call) so the change is fully reversible.
        if _force_conclude_enabled():
            # ``tools=None`` FORBIDS tool calls — at the cap the model has no
            # budget left to investigate further, so we want a direct answer
            # rather than another tool_call we'd have to ignore. thinking stays
            # enabled so the model can still reason over the evidence gathered.
            # The final-answer JSON shape is already in the system prompt
            # (assembled at the top of run()), so we only point at it here
            # instead of re-emitting the whole guidance block (saves ~600 tokens
            # on every forced call — this path exists to cut token waste).
            force_msg = (
                "你已达到本轮调查的步数上限（max_steps），不能再调用任何工具。\n"
                "请基于目前已收集的证据，立刻给出你**最可能**的单一根因结论。\n"
                "即使证据不完整，也必须按系统提示末尾的 final-answer guidance 输出一个结构化的 "
                "```json 最终答案（而不是继续调查）。\n\n"
                "You have reached the investigation step budget (max_steps). You MUST NOT call "
                "any more tools. Based on the evidence gathered so far, output your SINGLE best "
                "root-cause hypothesis now as a structured final answer — a single ```json block "
                "in EXACTLY the shape defined in the final-answer guidance at the end of the "
                "system prompt. If evidence is incomplete, state the gap and give the most likely "
                "hypothesis with a suitably low confidence — do NOT keep investigating."
            )
            forced_msgs = self.cm.assemble_turn(state, new_user=force_msg)

            rc: RootCause | None = None
            try:
                req = LLMRequest(
                    messages=forced_msgs,
                    tools=None,
                    model=self.model,
                    reasoning_effort=self.settings.reasoning_effort,
                    thinking_enabled=True,
                    max_tokens=self.settings.llm_max_tokens,
                )
                content, _reasoning, tool_calls, usage = await self.llm.complete(req)
                _accum_usage(usage)
                # A well-behaved model returns no tool_calls here (tools=None),
                # but some backends echo prior tool_calls — we only consume text.
                if tool_calls:
                    logger.warning(
                        "force-conclude call returned tool_calls despite tools=None; "
                        "ignoring (case %s)",
                        case_id,
                        extra={"component": "agent", "case_id": case_id},
                    )
                parsed = parse_root_cause(content)
                # Accept the parsed answer ONLY if it carries a real hypothesis:
                # a non-empty summary AND confidence > 0. parse_root_cause's own
                # fallbacks return either an empty/whitespace summary (strategy
                # 4 on whitespace input) or the "(空结论 / empty conclusion)"
                # placeholder at confidence 0.0 (strategy on None/empty input)
                # — both mean "the model gave us nothing", so we fall through to
                # the heuristic rather than ship a 0.0-confidence placeholder as
                # if it were a recovered answer. (A genuine model answer always
                # asserts confidence > 0, even if low.)
                if (parsed.summary or "").strip() and parsed.confidence > 0.0:
                    rc = parsed
            except _NONFATAL_EXC as e:
                logger.warning(
                    "force-conclude LLM call failed (%s: %s) for case %s; "
                    "using heuristic fallback",
                    type(e).__name__, e, case_id,
                    extra={"component": "agent", "case_id": case_id, "error": str(e)},
                )
            except Exception as e:  # noqa: BLE001 — pluggable LLM; must not kill recovery
                logger.warning(
                    "force-conclude LLM call raised unexpected %s for case %s; "
                    "using heuristic fallback: %s",
                    type(e).__name__, case_id, e,
                    extra={"component": "agent", "case_id": case_id, "error": str(e)},
                )

            if rc is None:
                # Heuristic fallback: synthesize from the last REASONING
                # thought so the trace at least shows the agent's working
                # hypothesis. Confidence is clamped LOW (0.3) because this path
                # means we could NOT get a clean answer — callers must not treat
                # it as high conviction. Matches parse_root_cause's own
                # prose-fallback level. Skip the display-only memory step (its
                # thought starts with "memory:") AND the display-only skill step
                # (its thought starts with "loaded skill:") — see run() above —
                # so telemetry text is never surfaced as the root-cause
                # hypothesis.
                last_thought = ""
                for s in reversed(steps):
                    thought = (s.thought or "").strip()
                    if (
                        s.step_kind == StepKind.REASONING
                        and thought
                        and not thought.startswith("memory:")
                        and not thought.startswith("loaded skill:")
                    ):
                        last_thought = thought
                        break
                rc = RootCause(
                    summary=last_thought[:800]
                    or "(达到步数上限且未能形成假设 / "
                    "step cap reached and no hypothesis could be formed)",
                    confidence=0.3,
                )

            # Emit exactly ONE CONCLUDE step so the trace/persisted report shows
            # the recovery attempt regardless of which branch produced ``rc``.
            # This block is only reachable AFTER the loop exits, so there is no
            # risk of double-emitting a CONCLUDE (the normal conclude path
            # ``return``s before the loop guard can fail) and no risk of
            # looping (we make at most one extra LLM call and never re-enter).
            entities = [
                e.get("entity_name") or e.get("entity_id") or ""
                for e in rc.entity_refs
                if isinstance(e, dict)
            ]
            conclude_step = RcaStep(
                step_id=self._step_id(case_id),
                case_id=case_id,
                step_kind=StepKind.CONCLUDE,
                hypothesis=rc.summary,
                confidence=rc.confidence,
                entities=[e for e in entities if e],
            )
            steps.append(conclude_step)
            yield conclude_step
            _safe_otel("record_step", "conclude")
        else:
            # Env-OFF: identical to the pre-I2 truncated report. No extra LLM
            # call, placeholder summary, confidence 0.0, no CONCLUDE step.
            rc = RootCause(
                summary=(
                    "(达到步数上限仍未给出结论 / "
                    "max steps reached without a final conclusion)"
                ),
                confidence=0.0,
            )

        report = RcaReport(
            case_id=case_id,
            task_id=case.task.task_id,
            alert_title=case.task.alert_title,
            root_cause=rc,
            steps=steps,
            model=self.model,
            token_usage=usage_total or None,
            status="truncated",
        )
        _safe_otel("record_run", "truncated")
        yield report


def build_agent_for_case(
    case_id: str,
    backend: str | None = None,
    settings: Settings | None = None,
    llm: Any | None = None,
    memory: Any | None = None,
) -> tuple[Case, RcaAgent]:
    """Construct a ready-to-run agent for a benchmark case.

    ``backend``: ``"parquet"`` (default; reads the dataset files directly) or
    ``"clickhouse"`` (queries imported data). Memory is seeded from
    ``memory/seed/`` if present.
    """
    s = settings or get_settings()
    case = load_case(case_id)
    backend = (backend or s.data_backend).strip().lower()

    if backend == "clickhouse":
        from ..providers.clickhouse_provider import ClickhouseProvider

        provider = ClickhouseProvider(case_id, window=case.task.alert_window)
    else:
        from ..providers.parquet_provider import ParquetProvider

        provider = ParquetProvider(case)

    if memory is None:
        try:
            memory = InMemoryStore.load_seed(_SEED_DIR) if _SEED_DIR.exists() else InMemoryStore()
        except _NONFATAL_EXC as e:
            logger.warning(
                "memory seed load from %s failed (%s: %s); starting with empty memory",
                _SEED_DIR, type(e).__name__, e,
                extra={"component": "memory", "seed_dir": str(_SEED_DIR), "error": str(e)},
            )
            memory = InMemoryStore()
        except Exception as e:  # noqa: BLE001 — pluggable backend; never fatal
            logger.warning(
                "memory seed load raised unexpected %s; starting with empty memory: %s",
                type(e).__name__, e,
                extra={"component": "memory", "seed_dir": str(_SEED_DIR), "error": str(e)},
            )
            memory = InMemoryStore()

    agent = RcaAgent(
        provider=provider,
        llm=llm or default_client(),
        memory=memory,
        settings=s,
    )
    return case, agent


__all__ = [
    "RcaAgent",
    "build_agent_for_case",
    "SkillLibrary",
    "_default_skill_library",
    "_skills_enabled",
]
