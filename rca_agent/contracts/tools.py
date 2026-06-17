"""Tool registry + OpenAI function-schema builder.

Tools are the agent's interface to :class:`DataProvider` and
:class:`MemoryStore`. The OpenAI ``tools=[...]`` JSON schema and the runtime
argument validation are derived from the *same* ``args_model`` (a Pydantic
model), so they can never drift.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict

# A tool returns a JSON-serializable dict (callers render it to text for the LLM).
ToolResult: TypeAlias = dict[str, Any]


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]  # parsed JSON arguments
    id: str | None = None


class ToolSpec(BaseModel):
    """Static spec of a tool."""

    name: str
    description: str
    args_model: type[BaseModel]


@runtime_checkable
class ToolHandler(Protocol):
    """Stateless handler: (validated args, provider, memory) -> ToolResult.

    The handler receives the bound :class:`DataProvider` and :class:`MemoryStore`
    so it never needs global state.
    """

    def __call__(
        self,
        args: BaseModel,
        provider: Any,  # DataProvider, typed Any to avoid import cycle
        memory: Any,  # MemoryStore
    ) -> ToolResult: ...


class RegisteredTool(BaseModel):
    """A tool spec paired with its handler."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ToolSpec
    handler: Any  # ToolHandler (callable; arbitrary type allowed)


def build_openai_tools(tools: list[RegisteredTool]) -> list[dict]:
    """Build the ``tools=[...]`` array for openai ``chat.completions``.

    Each entry: ``{"type": "function",
                    "function": {"name", "description", "parameters": <json_schema>}}``.
    The parameters schema is derived from the tool's ``args_model``.
    """
    out: list[dict] = []
    for t in tools:
        schema = t.spec.args_model.model_json_schema()
        # openai expects JSON-schema "object" parameters
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.spec.name,
                    "description": t.spec.description,
                    "parameters": schema,
                },
            }
        )
    return out


def validate_tool_call(
    call: ToolCall, tools: list[RegisteredTool]
) -> tuple[RegisteredTool, BaseModel]:
    """Find the tool by name and validate ``call.arguments`` via its
    ``args_model``. Raises ``KeyError`` if unknown, ``ValidationError`` on bad args.
    Returns the (registered tool, validated args model)."""
    for t in tools:
        if t.spec.name == call.name:
            validated = t.spec.args_model.model_validate(call.arguments)
            return t, validated
    raise KeyError(f"Unknown tool: {call.name}")


__all__ = [
    "ToolResult",
    "ToolCall",
    "ToolSpec",
    "ToolHandler",
    "RegisteredTool",
    "build_openai_tools",
    "validate_tool_call",
]
