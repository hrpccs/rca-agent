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
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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
    ) -> None:
        self.provider = provider
        self.llm = llm
        self.memory = memory
        self.cm = context_manager or build_context_manager()
        self.tools = tools if tools is not None else build_default_tools(provider, memory)
        self.settings = settings or get_settings()
        self.max_steps = max_steps or self.settings.llm_max_steps
        self.model = model or self.settings.deepseek_model

    def _step_id(self, case_id: str) -> str:
        return f"{case_id}-{uuid.uuid4().hex[:10]}"

    async def run(self, case: Case) -> AsyncIterator[RcaStep | RcaReport]:
        """Investigate ``case``; yield each step then the final report."""
        case_id = case.task.task_id
        _safe_otel("record_run", "started")

        system = (
            SYSTEM_PROMPT
            + "\n\n# 最终结论结构 / Final-answer structure\n"
            + to_final_answer_guidance()
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

        first_user = build_initial_brief(case.task, case.topology, hits)
        msgs = self.cm.assemble_turn(state, new_user=first_user)
        oa_tools = build_openai_tools(self.tools)

        steps: list[RcaStep] = []
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

        # Out of steps — emit a truncated report.
        report = RcaReport(
            case_id=case_id,
            task_id=case.task.task_id,
            alert_title=case.task.alert_title,
            root_cause=RootCause(
                summary="(达到步数上限仍未给出结论 / max steps reached without a final conclusion)",
                confidence=0.0,
            ),
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


__all__ = ["RcaAgent", "build_agent_for_case"]
