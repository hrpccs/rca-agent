"""Context manager implementation.

Owns the DeepSeek ``reasoning_content`` echo invariant end-to-end: for any
assistant turn that produced ``tool_calls``, the matching ``reasoning_content``
MUST be present in the messages sent on every subsequent request (the API
returns HTTP 400 otherwise). For non-tool turns ``reasoning_content`` is kept
only in :attr:`ContextState.turns` (for UI replay) and omitted from
``messages`` since the API ignores it between user turns.

The agent loop never touches ``reasoning_content``; it only calls the methods
on this class.

I4 — optional context bounding (env-gated, OFF by default):
============================================================
An 8-case evaluation found prompt tokens balloon to ~831K (max 1.13M) at
~16K tokens/tool-call. The single biggest safe, in-scope lever is the ASSEMBLY
path here: what gets sent to the LLM. core.py already caps each tool-result
text at 4000 chars upstream (NOT changed here), so single results are not the
driver — the growth is cross-turn accumulation. These knobs bound that growth
without touching persisted state (``RcaStep.tool_result_text`` is recorded by
core.py BEFORE assembly, so nothing is lost). Both knobs are OFF by default so
the default output is byte-identical to today; enable them only when running
long, high-cost cases where truncation/cost is a concern.

  * ``RCA_CONTEXT_TOOL_RESULT_MAX_CHARS`` (default ``0`` = OFF/unbounded):
    when >0, truncate each ``role:"tool"`` message's ``content`` to this many
    chars at assembly time, appending a ``…[truncated: <K> chars; full text
    retained in the persisted trace]`` suffix. Only affects tool messages.

  * ``RCA_CONTEXT_MAX_TOOL_MESSAGES`` (default ``0`` = OFF/keep-all): when >0
    (N), keep the most recent N tool messages and drop the older ones,
    replacing them with a single summary note. Dropped tool messages are
    removed as part of an ATOMIC ``[assistant(tool_calls) + trailing tool
    responses]`` group, so dropping a tool result also drops its originating
    assistant tool_call — this preserves the OpenAI/DeepSeek contract (an
    assistant message carrying ``tool_calls`` MUST be followed by matching
    ``role:"tool"`` responses; an orphaned tool_call is rejected with HTTP
    400). The retained assistant tool turns therefore keep their
    ``reasoning_content`` echo intact.

Non-positive or unparseable env values fall back to OFF with a warning, so a
misconfigured env var never crashes the agent loop.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from rca_agent.contracts.context import (
    ContextState,
    ToolMessage,
    TurnRecord,
)

# How many of the most-recent turns compress() will never drop.
_MIN_RETAINED_TURNS = 4

# Env knob: per-tool-result content char cap at assembly (0 = OFF/unbounded).
_TOOL_RESULT_MAX_CHARS_ENV = "RCA_CONTEXT_TOOL_RESULT_MAX_CHARS"
# Env knob: keep only the N most recent tool messages at assembly (0 = OFF).
_MAX_TOOL_MESSAGES_ENV = "RCA_CONTEXT_MAX_TOOL_MESSAGES"

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token, never zero for non-empty)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class ContextManager:
    """Concrete :class:`rca_agent.contracts.context.ContextManager` impl."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def init(self, case_id: str, system_prompt: str) -> ContextState:
        state = ContextState(case_id=case_id, system=system_prompt)
        state.token_estimate = estimate_tokens(system_prompt)
        return state

    # ------------------------------------------------------------------ #
    # Recording turns
    # ------------------------------------------------------------------ #
    def append_assistant(
        self,
        state: ContextState,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list[dict[str, Any]] | None,
    ) -> ContextState:
        """Record an assistant turn and append the openai-shaped message.

        Invariant:
          * tool_calls present  -> message carries ``reasoning_content``.
          * no tool_calls       -> ``reasoning_content`` is stored in
            ``turns`` only (the API ignores it between user turns).
        """
        # Defensive copy so the caller cannot mutate recorded state by
        # mutating the list/dicts they passed in. (Pydantic also isolates the
        # TurnRecord copy, but messages are plain dicts appended after the
        # model_copy and would otherwise alias the caller's objects.)
        safe_tool_calls: list[dict[str, Any]] | None = (
            [dict(c) for c in tool_calls] if tool_calls else None
        )
        record = TurnRecord(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=safe_tool_calls,
        )
        new_state = state.model_copy(deep=True)
        new_state.turns.append(record)

        if safe_tool_calls:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": content or "",
                "reasoning_content": reasoning_content or "",
                "tool_calls": safe_tool_calls,
            }
        else:
            message = {
                "role": "assistant",
                "content": content or "",
            }
        new_state.messages.append(message)

        new_state.token_estimate = self._estimate_messages(new_state)
        return new_state

    def append_tool_result(
        self, state: ContextState, results: list[ToolMessage]
    ) -> ContextState:
        """Append one ``{"role": "tool", ...}`` message per result."""
        new_state = state.model_copy(deep=True)
        for r in results:
            message = {
                "role": "tool",
                "tool_call_id": r.tool_call_id,
                "name": r.name,
                "content": r.content,
            }
            new_state.messages.append(message)
        new_state.token_estimate = self._estimate_messages(new_state)
        return new_state

    # ------------------------------------------------------------------ #
    # Assembling the request payload
    # ------------------------------------------------------------------ #
    def assemble_turn(
        self, state: ContextState, new_user: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the messages array to send to the LLM this turn.

        Guarantees:
          * system message always first as ``{"role": "system", ...}``;
          * every assistant message with ``tool_calls`` also carries its
            ``reasoning_content`` — re-injected from ``state.turns`` if missing
            (e.g. after a partial load).

        I4 bounding (env-gated, OFF by default — see module docstring):
          * ``RCA_CONTEXT_TOOL_RESULT_MAX_CHARS`` truncates each
            ``role:"tool"`` message's ``content``;
          * ``RCA_CONTEXT_MAX_TOOL_MESSAGES`` keeps only the most recent N
            tool messages, dropping older ``[assistant(tool_calls)+tool
            responses]`` groups ATOMICALLY so no tool_call is ever orphaned.

        Both knobs are read from the env on EVERY call so tests/ops can flip
        them without re-instantiating. When both are OFF (the default) the
        returned list is byte-identical to the un-bounded output. State is
        never mutated — only the assembled-for-LLM list is bounded.
        """
        tool_max_chars = _parse_positive_int_env(_TOOL_RESULT_MAX_CHARS_ENV)
        max_tool_msgs = _parse_positive_int_env(_MAX_TOOL_MESSAGES_ENV)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": state.system}
        ]
        # Index assistant turns in chronological order for reasoning lookup.
        assistant_turns = [t for t in state.turns if t.role == "assistant"]

        asst_seen = 0
        for msg in state.messages:
            if msg.get("role") == "assistant":
                out = dict(msg)
                if out.get("tool_calls"):
                    if (
                        "reasoning_content" not in out
                        or not out["reasoning_content"]
                    ) and asst_seen < len(assistant_turns):
                        # Re-inject from the matching TurnRecord (best-effort,
                        # positional match by assistant-turn order).
                        rc = assistant_turns[asst_seen].reasoning_content
                        out["reasoning_content"] = rc or ""
                    # Final belt-and-braces guarantee: a tool-bearing assistant
                    # message MUST carry the key, even if no record was found.
                    out.setdefault("reasoning_content", "")
                    # Defensive copy of tool_calls so the truncation/drop path
                    # below never mutates the caller's recorded state.
                    out["tool_calls"] = [dict(c) for c in (out["tool_calls"] or [])]
                messages.append(out)
                asst_seen += 1
            else:
                messages.append(dict(msg))

        if new_user is not None:
            messages.append({"role": "user", "content": new_user})

        # ---- I4 bounding: only what's SENT to the LLM is bounded. ---------- #
        # Order matters: the sliding window drops whole atomic groups (so the
        # truncation cap, if also on, is applied AFTER the window to the
        # remaining tool messages). Both return early when their knob is OFF.
        if max_tool_msgs:
            messages = _apply_tool_message_window(messages, max_tool_msgs)
        if tool_max_chars:
            _apply_tool_result_char_cap(messages, tool_max_chars)
        return messages

    # ------------------------------------------------------------------ #
    # Compression
    # ------------------------------------------------------------------ #
    def compress(self, state: ContextState, max_tokens: int) -> ContextState:
        """Summarize/drop the oldest turns to fit ``max_tokens``.

        Strategy (budget-driven, invariant-preserving):
          * Always keep the system message.
          * Prefer to keep the last ``_MIN_RETAINED_TURNS`` messages, but if
            even that tail cannot fit the budget, keep shrinking it (oldest
            group first) until it does. The hard requirement is the budget and
            the reasoning_content echo invariant; the "last few turns" rule is
            best-effort.
          * Dropped messages are summarized into a single
            ``{"role": "system", "content": "Prior investigation summary: ..."}``
            placed right after the system prompt, using ``content`` only
            (``reasoning_content`` of dropped non-tool turns is discarded;
            dropped tool turns contribute their ``content`` text too).
          * Retained assistant turns that had ``tool_calls`` KEEP their
            ``reasoning_content`` — the echo invariant is never broken.
          * An assistant tool-call message and its immediately-following tool
            messages are treated as one atomic group (never split), so we never
            emit an orphan tool result.
          * ``token_estimate`` is recomputed; the result is ``<= max_tokens``
            whenever the system prompt alone fits, and never violates the
            invariant.
        """
        if state.token_estimate <= max_tokens:
            return state.model_copy(deep=True)

        new_state = state.model_copy(deep=True)

        # NOTE: the real system prompt is NEVER stored in state.messages — it
        # lives in state.system and is prepended by assemble_turn. This keeps
        # _estimate_messages (system + sum(messages)) free of double-counting
        # and prevents assemble_turn from emitting a duplicate system message
        # after compress. The summary is a SEPARATE {"role":"system",...} msg.

        # The system contribution to the token budget uses the same shape
        # _estimate_messages will (estimate_tokens(state.system)).
        system_tokens = estimate_tokens(new_state.system)

        # ---- Split messages into atomic groups -------------------------------- #
        # A group is either:
        #   * an assistant message with tool_calls + its trailing tool messages, or
        #   * any single other message (assistant-no-tools, user, prior-summary,
        #     ...). We never emit an orphan tool result.
        groups: list[list[dict[str, Any]]] = []
        i = 0
        msgs = new_state.messages
        while i < len(msgs):
            m = msgs[i]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                group = [m]
                j = i + 1
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    group.append(msgs[j])
                    j += 1
                groups.append(group)
                i = j
            else:
                groups.append([m])
                i += 1

        # ---- Decide how many newest groups to retain verbatim ----------------- #
        def group_tokens(group: list[dict[str, Any]]) -> int:
            return sum(estimate_tokens(_stringify(m)) for m in group)

        # Reserve a small floor for the summary message (role overhead + margin)
        # so the past isn't erased entirely when there's room.
        summary_floor = min(
            estimate_tokens(_stringify({"role": "system", "content": "Prior investigation summary: "})),
            max(0, max_tokens // 8),
        )

        # Greedily keep as many of the NEWEST groups as fit [system + retained + floor].
        best_count = 0
        for count in range(len(groups), -1, -1):
            retained = groups[len(groups) - count :] if count else []
            retained_tokens = sum(group_tokens(g) for g in retained)
            if system_tokens + retained_tokens + summary_floor <= max_tokens:
                best_count = count
                break

        retained_count = max(0, min(best_count, len(groups)))
        # Prefer to keep the desired minimum tail unless the budget truly forbids it.
        desired = min(_MIN_RETAINED_TURNS, len(groups))
        if retained_count < desired:
            desired_retained = groups[len(groups) - desired :]
            desired_tokens = sum(group_tokens(g) for g in desired_retained)
            if system_tokens + desired_tokens <= max_tokens:
                retained_count = desired

        retained_groups = (
            groups[len(groups) - retained_count :] if retained_count else []
        )
        dropped_groups = (
            groups[: len(groups) - retained_count] if retained_count else list(groups)
        )

        # ---- Build the summary from dropped messages (content only) ---------- #
        summary_pieces: list[str] = []
        for grp in dropped_groups:
            for m in grp:
                role = m.get("role")
                c = m.get("content")
                if role == "tool":
                    if c:
                        summary_pieces.append(f"[tool {m.get('name', '')}] {c}")
                elif role == "assistant":
                    if m.get("tool_calls"):
                        # dropped tool turn — content only (reasoning_content
                        # discarded; only RETAINED tool turns must keep it)
                        if c:
                            summary_pieces.append(f"[assistant/tool] {c}")
                    elif c:
                        summary_pieces.append(c)
                elif c:
                    summary_pieces.append(c)

        retained_tokens = sum(group_tokens(g) for g in retained_groups)
        remaining_budget = max(0, max_tokens - system_tokens - retained_tokens)

        # state.messages holds ONLY [summary?] + retained groups (no real system).
        kept_messages: list[dict[str, Any]] = []

        if summary_pieces:
            prefix = "Prior investigation summary: "
            summary_text = prefix + " | ".join(summary_pieces)
            # Estimate against the full {"role":"system","content":...} shape and
            # reserve a 1-token margin for len//4 rounding.
            summary_budget = max(0, remaining_budget - 1)

            def _summary_msg_est(text: str) -> int:
                return estimate_tokens(_stringify({"role": "system", "content": text}))

            if _summary_msg_est(summary_text) > summary_budget:
                if summary_budget <= _summary_msg_est(prefix):
                    summary_text = ""
                else:
                    body = summary_text[len(prefix) :]
                    max_chars = summary_budget * 4
                    while max_chars > 0 and _summary_msg_est(
                        prefix + body[:max_chars]
                    ) > summary_budget:
                        max_chars -= 1
                    summary_text = prefix + body[:max_chars] if max_chars > 0 else ""
            if summary_text:
                kept_messages.append({"role": "system", "content": summary_text})

        for grp in retained_groups:
            kept_messages.extend(grp)

        new_state.messages = kept_messages

        # ---- Defensive re-injection of reasoning_content for RETAINED tool turns #
        # Retained assistant tool messages already carry reasoning_content verbatim
        # from append_assistant; this only fires for partial-load / corruption. We
        # match by POSITION against only the turns that correspond to retained
        # assistant messages: count how many assistant turns were dropped and
        # offset the index so we never pair a surviving message with the wrong turn.
        assistant_turns = [t for t in new_state.turns if t.role == "assistant"]
        dropped_assistant_count = sum(
            1
            for grp in dropped_groups
            for m in grp
            if m.get("role") == "assistant"
        )
        asst_seen = 0
        for msg in new_state.messages:
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    turn_idx = dropped_assistant_count + asst_seen
                    if turn_idx < len(assistant_turns) and not msg.get("reasoning_content"):
                        rc = assistant_turns[turn_idx].reasoning_content
                        msg["reasoning_content"] = rc or ""
                    msg.setdefault("reasoning_content", "")
                asst_seen += 1

        # ---- Budget safety net ------------------------------------------------ #
        # The retained groups + system were selected to fit, but the defensive
        # re-injection above (partial-load path) can grow messages and push the
        # total over budget. If so, drop the prior-investigation summary (the
        # only non-invariant-bearing message we can safely remove). This loop
        # only runs when the over-budget culprit is the summary; if the system
        # prompt + retained turns alone exceed budget we cannot help and leave
        # the state as-is (documented carve-out).
        while (
            len(new_state.messages) > 1
            and self._estimate_messages(new_state) > max_tokens
        ):
            summary_idx = next(
                (
                    idx
                    for idx in range(len(new_state.messages))
                    if new_state.messages[idx].get("role") == "system"
                    and "Prior investigation summary"
                    in (new_state.messages[idx].get("content") or "")
                ),
                None,
            )
            if summary_idx is None:
                break
            del new_state.messages[summary_idx]

        new_state.token_estimate = self._estimate_messages(new_state)
        return new_state

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_messages(state: ContextState) -> int:
        total = estimate_tokens(state.system)
        for m in state.messages:
            total += estimate_tokens(_stringify(m))
        return total


def _stringify(message: dict[str, Any]) -> str:
    """Flatten an openai-shaped message to a string for token estimation."""
    parts: list[str] = []
    for v in message.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    # openai-shaped tool_call: {"id","type","function":{"name","arguments"}}
                    fn = item.get("function")
                    if isinstance(fn, dict):
                        inner = fn.get("arguments") or fn.get("name") or ""
                    else:
                        inner = item.get("arguments") or item.get("name") or ""
                    if isinstance(inner, str):
                        parts.append(inner)
                    else:
                        parts.append(str(inner))
                elif isinstance(item, str):
                    parts.append(item)
    return " ".join(parts)


def _parse_positive_int_env(name: str) -> int:
    """Parse an env var as a positive int; anything else -> 0 (OFF).

    Non-positive or unparseable values resolve to ``0`` = OFF, so a
    misconfigured env var never crashes the agent loop. A warning is logged
    only when a value was present but unparseable (NOT when it's absent or a
    benign ``0``/``""``, to avoid noise in the default-off path).
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return 0
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "env %s=%r is not an integer; disabling context bounding knob",
            name, raw,
            extra={"component": "context", "knob": name, "raw": raw},
        )
        return 0
    return value if value > 0 else 0


def _apply_tool_result_char_cap(
    messages: list[dict[str, Any]], max_chars: int
) -> None:
    """Truncate each ``role:"tool"`` content in-place to ``max_chars`` chars.

    Mutates the assembled list only (the messages here are already fresh dicts
    copied in ``assemble_turn``). Non-tool messages are never touched. The
    persisted ``RcaStep.tool_result_text`` was recorded by core.py BEFORE
    assembly, so truncation here loses nothing durable — it only bounds what
    the LLM sees this turn.
    """
    trimmed_total = 0
    trimmed_msgs = 0
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        dropped = len(content) - max_chars
        m["content"] = (
            content[:max_chars]
            + f"…[truncated: {dropped} chars; full text retained in the persisted trace]"
        )
        trimmed_total += dropped
        trimmed_msgs += 1
    if trimmed_msgs:
        logger.info(
            "context bounding: truncated %d tool message(s), %d chars total "
            "(cap=%d); full text retained in the persisted trace",
            trimmed_msgs, trimmed_total, max_chars,
            extra={
                "component": "context",
                "knob": _TOOL_RESULT_MAX_CHARS_ENV,
                "trimmed_msgs": trimmed_msgs,
                "trimmed_chars": trimmed_total,
                "cap": max_chars,
            },
        )


def _apply_tool_message_window(
    messages: list[dict[str, Any]], max_tool_msgs: int
) -> list[dict[str, Any]]:
    """Keep only the most recent ``max_tool_msgs`` tool messages.

    Pairing invariant (OpenAI/DeepSeek contract): an assistant message
    carrying ``tool_calls`` MUST be immediately followed by matching
    ``role:"tool"`` responses. Dropping a tool result while keeping its
    originating assistant ``tool_calls`` would leave a dangling reference that
    the API rejects with HTTP 400.

    Decision: drop whole ATOMIC groups. A group is ``[assistant(tool_calls) +
    its trailing contiguous tool messages]`` (identical grouping to
    ``compress``). We drop the OLDEST groups until the number of surviving
    tool messages is ``<= max_tool_msgs``, then insert ONE summary note just
    after the retained prefix. This guarantees:
      * no orphaned tool_call (the assistant tool_call is dropped WITH its
        responses), and
      * retained assistant tool turns keep their ``reasoning_content`` echo
        (they are not touched at all).

    The summary note is emitted as ``role:"system"`` (NOT ``role:"tool"``):
    the OpenAI/DeepSeek chat-completions contract requires every
    ``role:"tool"`` message to reference a ``tool_call_id`` from a preceding
    assistant ``tool_calls`` entry, and a standalone tool message with an
    unmatched sentinel id would be rejected with HTTP 400. ``compress`` uses
    the same ``role:"system"`` summary shape for the same reason. The note is
    placed after the preserved prefix (system prompt + leading non-tool
    context), adjacent to where the dropped tool results were.

    The leading system message (and any other leading non-tool context) is
    ALWAYS preserved: the window only ever drops ``[assistant(tool_calls) +
    tool responses]`` groups, never the system prompt.
    """
    total_tools = sum(1 for m in messages if m.get("role") == "tool")
    if total_tools <= max_tool_msgs:
        return messages  # nothing to drop

    n = len(messages)

    # Locate the leading PREFIX: everything up to (but not including) the first
    # assistant(tool_calls) group. This always includes the system message at
    # index 0 and is preserved verbatim (the window never touches it).
    first_tool_group_start = n
    for i in range(n):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            first_tool_group_start = i
            break
    if first_tool_group_start == n:
        return messages  # no assistant(tool_calls) groups -> nothing to drop

    body = messages[first_tool_group_start:]

    # Build atomic groups over the body.
    groups: list[tuple[int, int]] = []  # (start, end_exclusive) into body
    i = 0
    while i < len(body):
        m = body[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            start = i
            j = i + 1
            while j < len(body) and body[j].get("role") == "tool":
                j += 1
            groups.append((start, j))
            i = j
        else:
            groups.append((i, i + 1))
            i += 1

    # Greedily keep the NEWEST groups until we'd exceed the tool-message cap.
    # Walk groups from newest backwards, counting tool messages; the oldest
    # groups that fall outside the cap are dropped as whole atomic units.
    keep_from_group: int = len(groups)  # index of the first group to KEEP
    kept_tools = 0
    for gi in range(len(groups) - 1, -1, -1):
        g_start, g_end = groups[gi]
        g_tool_count = sum(
            1 for k in range(g_start, g_end) if body[k].get("role") == "tool"
        )
        if kept_tools + g_tool_count > max_tool_msgs:
            keep_from_group = gi + 1
            break
        kept_tools += g_tool_count
        keep_from_group = gi

    if keep_from_group == 0:
        return messages  # cap covers the whole body after all

    dropped_group_spans = groups[:keep_from_group]
    dropped_tool_msgs = sum(
        1
        for (gs, ge) in dropped_group_spans
        for k in range(gs, ge)
        if body[k].get("role") == "tool"
    )
    dropped_assistant_toolcalls = sum(
        1
        for (gs, ge) in dropped_group_spans
        for k in range(gs, ge)
        if body[k].get("role") == "assistant" and body[k].get("tool_calls")
    )

    note = {
        # role:"system" (NOT role:"tool"): the OpenAI/DeepSeek contract
        # requires every role:"tool" message to reference a tool_call_id from a
        # preceding assistant tool_calls entry; a standalone tool message with
        # an unmatched sentinel id would be rejected with HTTP 400. compress()
        # uses the same role:"system" summary shape for dropped turns.
        "role": "system",
        "content": (
            f"[context window: {dropped_tool_msgs} earlier tool result(s) "
            f"across {dropped_assistant_toolcalls} tool turn(s) omitted to "
            f"bound context; full trace persisted]"
        ),
    }

    # Reconstruct: preserved prefix + summary note + retained body groups.
    out: list[dict[str, Any]] = list(messages[:first_tool_group_start])
    out.append(note)
    for gi in range(keep_from_group, len(groups)):
        gs, ge = groups[gi]
        out.extend(body[gs:ge])

    logger.info(
        "context bounding: dropped %d tool message(s) and %d paired assistant "
        "tool_call turn(s) via sliding window (keep=%d); full trace persisted",
        dropped_tool_msgs, dropped_assistant_toolcalls, max_tool_msgs,
        extra={
            "component": "context",
            "knob": _MAX_TOOL_MESSAGES_ENV,
            "dropped_tool_msgs": dropped_tool_msgs,
            "dropped_assistant_toolcalls": dropped_assistant_toolcalls,
            "kept": max_tool_msgs,
        },
    )
    return out


def build_context_manager() -> ContextManager:
    """Factory used by the agent loop / DI containers."""
    return ContextManager()


__all__ = ["ContextManager", "build_context_manager", "estimate_tokens"]
