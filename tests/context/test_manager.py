"""Tests for the context manager — focused on the reasoning_content invariant."""
from __future__ import annotations

from rca_agent.context.manager import (
    ContextManager,
    _stringify,
    build_context_manager,
    estimate_tokens,
)
from rca_agent.contracts import ContextManager as ContextManagerProtocol
from rca_agent.contracts import ContextState, ToolMessage

TOOL_CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "query_logs", "arguments": "{}"},
}


def _fresh_state() -> ContextState:
    cm = ContextManager()
    return cm.init("t001", "You are an SRE.")


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_build_context_manager():
    cm = build_context_manager()
    assert isinstance(cm, ContextManager)
    # Satisfies the Protocol structurally (runtime_checkable).
    assert isinstance(cm, ContextManagerProtocol)


def test_init_state():
    cm = ContextManager()
    s = cm.init("t001", "You are an SRE.")
    assert s.case_id == "t001"
    assert s.system == "You are an SRE."
    assert s.messages == []
    assert s.turns == []
    assert s.token_estimate > 0


def test_protocol_conformance():
    cm = ContextManager()
    assert isinstance(cm, ContextManagerProtocol)


# --------------------------------------------------------------------------- #
# THE invariant: assemble_turn re-injects reasoning_content for tool turns
# --------------------------------------------------------------------------- #
def test_assemble_turn_reinjects_reasoning_for_tool_turns():
    cm = ContextManager()
    s = cm.init("t001", "sys")
    s = cm.append_assistant(
        s,
        content="let me check",
        reasoning_content="thinking...",
        tool_calls=[TOOL_CALL],
    )
    s = cm.append_tool_result(
        s,
        [ToolMessage(tool_call_id="c1", name="query_logs", content="{}")],
    )
    msgs = cm.assemble_turn(s, new_user="next?")
    asst_with_tools = [
        m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert asst_with_tools, "expected at least one tool-bearing assistant msg"
    assert all(
        "reasoning_content" in m for m in asst_with_tools
    ), "INVARIANT VIOLATED: reasoning_content missing on tool turn"
    assert asst_with_tools[0]["reasoning_content"] == "thinking..."


def test_assemble_turn_reinjects_from_turns_when_missing_in_messages():
    """Simulate a partial load: message lost its reasoning_content but turns has it."""
    cm = ContextManager()
    s = cm.init("t001", "sys")
    s = cm.append_assistant(
        s, content="c", reasoning_content="rc", tool_calls=[TOOL_CALL]
    )
    # Corrupt the message directly (drop reasoning_content) to mimic a bad load.
    for m in s.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m.pop("reasoning_content", None)
    msgs = cm.assemble_turn(s)
    asst = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst
    assert all("reasoning_content" in m for m in asst)
    assert asst[0]["reasoning_content"] == "rc"


def test_assemble_turn_system_first():
    cm = ContextManager()
    s = _fresh_state()
    s = cm.append_assistant(s, content="hi", reasoning_content=None, tool_calls=None)
    msgs = cm.assemble_turn(s, new_user="go")
    assert msgs[0] == {"role": "system", "content": "You are an SRE."}
    assert msgs[-1] == {"role": "user", "content": "go"}


def test_assemble_turn_user_appended_only_when_provided():
    cm = ContextManager()
    s = _fresh_state()
    s = cm.append_assistant(s, content="hi", reasoning_content=None, tool_calls=None)
    msgs_no_user = cm.assemble_turn(s)
    assert all(m.get("role") != "user" for m in msgs_no_user)
    msgs_user = cm.assemble_turn(s, new_user="q")
    assert msgs_user[-1]["role"] == "user"


# --------------------------------------------------------------------------- #
# Non-tool turns DROP reasoning_content from messages
# --------------------------------------------------------------------------- #
def test_non_tool_turn_drops_reasoning_from_messages():
    cm = ContextManager()
    s = cm.init("t001", "sys")
    s = cm.append_assistant(
        s, content="answer", reasoning_content="secret thoughts", tool_calls=None
    )
    # reasoning_content is kept in turns for UI replay...
    assert s.turns[-1].reasoning_content == "secret thoughts"
    # ...but NOT in messages.
    asst = [m for m in s.messages if m.get("role") == "assistant"]
    assert asst
    assert "reasoning_content" not in asst[0]
    # And assemble_turn keeps it dropped.
    msgs = cm.assemble_turn(s)
    asst = [m for m in msgs if m.get("role") == "assistant"]
    assert all("reasoning_content" not in m for m in asst)


# --------------------------------------------------------------------------- #
# Tool result shape
# --------------------------------------------------------------------------- #
def test_append_tool_result_shape():
    cm = ContextManager()
    s = _fresh_state()
    s = cm.append_assistant(
        s, content="c", reasoning_content="r", tool_calls=[TOOL_CALL]
    )
    s = cm.append_tool_result(
        s,
        [
            ToolMessage(tool_call_id="c1", name="query_logs", content='{"ok":1}'),
            ToolMessage(tool_call_id="c2", name="query_metrics", content='{"v":2}'),
        ],
    )
    tools = [m for m in s.messages if m.get("role") == "tool"]
    assert len(tools) == 2
    assert tools[0] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "query_logs",
        "content": '{"ok":1}',
    }
    assert tools[1]["tool_call_id"] == "c2"


# --------------------------------------------------------------------------- #
# append_assistant tool-call message keeps reasoning_content (echo)
# --------------------------------------------------------------------------- #
def test_append_assistant_tool_turn_keeps_reasoning_in_messages():
    cm = ContextManager()
    s = _fresh_state()
    s = cm.append_assistant(
        s, content="c", reasoning_content="rc", tool_calls=[TOOL_CALL]
    )
    asst = [m for m in s.messages if m.get("role") == "assistant"][-1]
    assert asst["reasoning_content"] == "rc"
    assert asst["tool_calls"] == [TOOL_CALL]
    assert asst["content"] == "c"


def test_append_assistant_empty_content_normalizes():
    cm = ContextManager()
    s = _fresh_state()
    s = cm.append_assistant(s, content=None, reasoning_content="r", tool_calls=[TOOL_CALL])
    asst = [m for m in s.messages if m.get("role") == "assistant"][-1]
    assert asst["content"] == ""


def test_append_does_not_mutate_input_state():
    cm = ContextManager()
    s0 = _fresh_state()
    before_msgs = len(s0.messages)
    out = cm.append_assistant(s0, content="x", reasoning_content=None, tool_calls=None)
    assert len(s0.messages) == before_msgs  # input untouched
    assert out is not s0


def test_append_assistant_does_not_alias_caller_tool_calls():
    """Caller mutating its tool_calls list must not corrupt recorded state."""
    cm = ContextManager()
    s = _fresh_state()
    tc = [
        {"id": "c1", "type": "function", "function": {"name": "q", "arguments": "{}"}}
    ]
    s = cm.append_assistant(s, content="c", reasoning_content="r", tool_calls=tc)
    recorded = s.messages[-1]["tool_calls"]
    # Mutate the caller's list/dicts after the call.
    tc[0]["id"] = "MUTATED"
    tc.append({"id": "extra"})
    assert recorded[0]["id"] == "c1", "inner dict aliasing corrupted recorded state"
    assert len(recorded) == 1, "caller list append leaked into recorded state"
    assert s.turns[-1].tool_calls[0]["id"] == "c1"


# --------------------------------------------------------------------------- #
# compress: preserves reasoning for retained tool turns; fits budget
# --------------------------------------------------------------------------- #
def test_compress_noop_when_under_budget():
    cm = ContextManager()
    s = _fresh_state()
    s = cm.append_assistant(s, content="hi", reasoning_content=None, tool_calls=None)
    out = cm.compress(s, max_tokens=10_000)
    assert out.token_estimate <= 10_000
    # Nothing dropped because we were already under budget.
    assert len(out.messages) == len(s.messages)


def test_compress_preserves_reasoning_for_retained_tool_turn_and_fits():
    cm = ContextManager()
    s = cm.init("t001", "sys")
    # Build a long history of many tool turns so compress is forced to act.
    for i in range(12):
        s = cm.append_assistant(
            s,
            content=f"step {i} " + "x" * 400,
            reasoning_content=f"reasoning-{i} " + "y" * 400,
            tool_calls=[
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "query", "arguments": "{}"},
                }
            ],
        )
        s = cm.append_tool_result(
            s,
            [ToolMessage(tool_call_id=f"c{i}", name="query", content="r" * 400)],
        )

    budget = 600
    out = cm.compress(s, max_tokens=budget)
    assert out.token_estimate <= budget, (
        f"compress exceeded budget: {out.token_estimate} > {budget}"
    )
    # Every retained assistant tool turn still carries reasoning_content.
    asst_tools = [
        m for m in out.messages if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert asst_tools, "expected retained tool turns"
    assert all("reasoning_content" in m for m in asst_tools), (
        "INVARIANT VIOLATED after compress"
    )
    assert all(m["reasoning_content"] for m in asst_tools)
    # compress actually dropped the oldest turns.
    assert len(out.messages) < len(s.messages), "compress did not drop any turns"
    # A summary system message exists when something was dropped.
    sys_msgs = [m for m in out.messages if m.get("role") == "system"]
    assert any(
        "Prior investigation summary" in (m.get("content") or "") for m in sys_msgs
    ), "expected a prior-investigation summary message"
    # The real system prompt is NOT stored in state.messages (no double-count /
    # no duplicate system when assemble_turn prepends it).
    assert all(
        not (m.get("role") == "system" and m.get("content") == "sys")
        for m in out.messages
    ), "real system prompt leaked into state.messages"
    # assemble_turn puts the single real system message first.
    assembled = cm.assemble_turn(out)
    assert assembled[0] == {"role": "system", "content": "sys"}
    assert sum(1 for m in assembled if m.get("role") == "system" and m.get("content") == "sys") == 1


def test_compress_system_first_and_summary_after_system():
    cm = ContextManager()
    s = cm.init("t001", "sys")
    for i in range(10):
        s = cm.append_assistant(
            s,
            content=f"plain {i} " + "z" * 500,
            reasoning_content=None,
            tool_calls=None,
        )
    out = cm.compress(s, max_tokens=400)
    assembled = cm.assemble_turn(out)
    # The real system prompt is always first in the assembled payload.
    assert assembled[0] == {"role": "system", "content": "sys"}
    # ... and appears exactly once.
    assert sum(1 for m in assembled if m.get("content") == "sys" and m.get("role") == "system") == 1
    # If a summary exists, it must come immediately after the system message.
    if len(assembled) > 1 and assembled[1].get("role") == "system":
        assert "Prior investigation summary" in assembled[1]["content"]


def test_compress_never_drops_reasoning_for_surviving_tool_turn():
    """Single tool turn that must survive: reasoning must remain."""
    cm = ContextManager()
    s = cm.init("t001", "sys")
    s = cm.append_assistant(
        s, content="only tool turn", reasoning_content="keep-me", tool_calls=[TOOL_CALL]
    )
    s = cm.append_tool_result(
        s, [ToolMessage(tool_call_id="c1", name="query_logs", content="{}")]
    )
    # Large budget -> nothing dropped.
    out = cm.compress(s, max_tokens=10_000)
    asst = [m for m in out.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst and asst[0]["reasoning_content"] == "keep-me"


# --------------------------------------------------------------------------- #
# Full round-trip invariant across many turns
# --------------------------------------------------------------------------- #
def test_invariant_holds_across_multi_turn_dialog():
    cm = ContextManager()
    s = cm.init("t001", "sys")
    # tool turn
    s = cm.append_assistant(s, "a", "ra", tool_calls=[TOOL_CALL])
    s = cm.append_tool_result(
        s, [ToolMessage(tool_call_id="c1", name="q", content="{}")]
    )
    # plain turn
    s = cm.append_assistant(s, "b", "rb", tool_calls=None)
    # another tool turn
    s = cm.append_assistant(
        s,
        "c",
        "rc",
        tool_calls=[
            {"id": "c2", "type": "function", "function": {"name": "q2", "arguments": "{}"}}
        ],
    )
    s = cm.append_tool_result(
        s, [ToolMessage(tool_call_id="c2", name="q2", content="{}")]
    )
    msgs = cm.assemble_turn(s, new_user="final?")
    tool_assistants = [
        m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(tool_assistants) == 2
    assert all("reasoning_content" in m for m in tool_assistants)
    # Plain assistant turn must NOT carry reasoning_content.
    plain = [
        m
        for m in msgs
        if m.get("role") == "assistant" and not m.get("tool_calls")
    ]
    assert plain and all("reasoning_content" not in m for m in plain)


def test_token_estimate_updates():
    cm = ContextManager()
    s = cm.init("t001", "sys")
    base = s.token_estimate
    s = cm.append_assistant(s, "hello world content", "r", tool_calls=None)
    assert s.token_estimate > base


# --------------------------------------------------------------------------- #
# Regressions found in code review
# --------------------------------------------------------------------------- #
def test_stringify_counts_tool_call_function_and_arguments():
    """_stringify must include the tool_call's function name + arguments."""
    tc = {
        "id": "c1",
        "type": "function",
        "function": {"name": "query_logs", "arguments": '{"filter":"error"}'},
    }
    with_tool = estimate_tokens(_stringify({"role": "assistant", "content": "x", "tool_calls": [tc]}))
    without = estimate_tokens(_stringify({"role": "assistant", "content": "x"}))
    assert with_tool > without, "tool_call function name/arguments not counted"


def test_compress_no_double_count_of_system_prompt():
    """token_estimate must not count the system prompt twice after compress."""
    cm = ContextManager()
    sys_prompt = "system prompt of moderate length here"
    s = cm.init("t001", sys_prompt)
    # Force compression by adding many large turns.
    for i in range(10):
        s = cm.append_assistant(s, content=f"plain {i} " + "z" * 400, reasoning_content=None, tool_calls=None)
    out = cm.compress(s, max_tokens=400)
    # Recompute independently: system once + each message once.
    expected = estimate_tokens(out.system) + sum(
        estimate_tokens(_stringify(m)) for m in out.messages
    )
    assert out.token_estimate == expected, "system prompt double-counted in token_estimate"


def test_compress_then_assemble_single_system_message():
    """After compress, assemble_turn emits exactly ONE real system message."""
    cm = ContextManager()
    s = cm.init("t001", "the real system prompt")
    for i in range(10):
        s = cm.append_assistant(
            s, content=f"plain {i} " + "z" * 400, reasoning_content=None, tool_calls=None
        )
    out = cm.compress(s, max_tokens=300)
    assembled = cm.assemble_turn(out, new_user="go")
    real_system = [
        m for m in assembled if m.get("role") == "system" and m.get("content") == "the real system prompt"
    ]
    assert len(real_system) == 1, f"expected 1 system message, got {len(real_system)}"
    assert assembled[0]["role"] == "system"


def test_compress_reinject_offset_after_dropped_turns():
    """Re-injection after compress must map retained turns to the CORRECT reasoning.

    Simulates a partial-load where retained assistant tool messages lost their
    reasoning_content; compress must restore each from its OWN turn, not an
    older dropped turn's.
    """
    cm = ContextManager()
    s = cm.init("t001", "sys")
    for i in range(8):
        s = cm.append_assistant(
            s,
            content=f"step-{i}",
            reasoning_content=f"REASON-{i}",
            tool_calls=[
                {"id": f"c{i}", "type": "function", "function": {"name": "q", "arguments": "{}"}}
            ],
        )
        s = cm.append_tool_result(
            s, [ToolMessage(tool_call_id=f"c{i}", name="q", content="r" * 300)]
        )
    # Corrupt reasoning_content on the assistant messages (turns kept intact),
    # mimicking a partial load where messages lost their reasoning echo.
    for m in s.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["reasoning_content"] = ""
    # Tight budget forces several oldest turns to be dropped.
    out = cm.compress(s, max_tokens=350)
    assert len(out.messages) < len(s.messages), "compress did not drop turns"
    retained = [m for m in out.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert retained, "expected retained tool turns"
    # Each retained message must carry ITS OWN turn's reasoning (offset correct).
    for m in retained:
        content = m.get("content") or ""
        rc = m.get("reasoning_content") or ""
        if content.startswith("step-"):
            idx = content.split("-", 1)[1]
            assert rc == f"REASON-{idx}", (
                f"reasoning_content misattributed: content={content!r} rc={rc!r}"
            )


def test_compress_then_assemble_invariant_round_trip():
    """compress followed by assemble_turn must still satisfy the echo invariant."""
    cm = ContextManager()
    s = cm.init("t001", "sys")
    for i in range(10):
        s = cm.append_assistant(
            s,
            content=f"step-{i}",
            reasoning_content=f"reasoning-{i}",
            tool_calls=[
                {"id": f"c{i}", "type": "function", "function": {"name": "q", "arguments": "{}"}}
            ],
        )
        s = cm.append_tool_result(
            s, [ToolMessage(tool_call_id=f"c{i}", name="q", content="r" * 200)]
        )
    out = cm.compress(s, max_tokens=500)
    assembled = cm.assemble_turn(out, new_user="next?")
    tool_assistants = [
        m for m in assembled if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert tool_assistants
    assert all("reasoning_content" in m for m in tool_assistants), "INVARIANT VIOLATED"


def test_empty_string_reasoning_on_tool_turn_keeps_key():
    """A tool turn with reasoning_content='' must still carry the key (invariant)."""
    cm = ContextManager()
    s = cm.init("t001", "sys")
    s = cm.append_assistant(s, content="c", reasoning_content="", tool_calls=[TOOL_CALL])
    msgs = cm.assemble_turn(s)
    asst = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst
    assert "reasoning_content" in asst[0]


def test_compress_drops_oldest_turns():
    """compress must reduce the message count (not just fit budget)."""
    cm = ContextManager()
    s = cm.init("t001", "sys")
    for i in range(10):
        s = cm.append_assistant(
            s, content=f"turn-{i} " + "x" * 400, reasoning_content=None, tool_calls=None
        )
    n_before = len(s.messages)
    out = cm.compress(s, max_tokens=300)
    assert len(out.messages) < n_before, "compress did not drop any turns"
    assert out.token_estimate <= 300
