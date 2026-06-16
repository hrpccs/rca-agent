"""Context manager implementation.

Owns the DeepSeek ``reasoning_content`` echo invariant end-to-end: for any
assistant turn that produced ``tool_calls``, the matching ``reasoning_content``
MUST be present in the messages sent on every subsequent request (the API
returns HTTP 400 otherwise). For non-tool turns ``reasoning_content`` is kept
only in :attr:`ContextState.turns` (for UI replay) and omitted from
``messages`` since the API ignores it between user turns.

The agent loop never touches ``reasoning_content``; it only calls the methods
on this class.
"""
from __future__ import annotations

from typing import Any

from rca_agent.contracts.context import (
    ContextState,
    ToolMessage,
    TurnRecord,
)

# How many of the most-recent turns compress() will never drop.
_MIN_RETAINED_TURNS = 4


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
        """
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
                messages.append(out)
                asst_seen += 1
            else:
                messages.append(dict(msg))

        if new_user is not None:
            messages.append({"role": "user", "content": new_user})
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


def build_context_manager() -> ContextManager:
    """Factory used by the agent loop / DI containers."""
    return ContextManager()


__all__ = ["ContextManager", "build_context_manager", "estimate_tokens"]
