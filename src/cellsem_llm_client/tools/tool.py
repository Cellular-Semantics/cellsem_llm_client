"""Generic tool abstraction for LiteLLM tool-calling flows.

Provides a ``Tool`` frozen dataclass that bundles a tool schema with its handler,
and ``unpack_tools`` to convert ``list[Tool]`` into the legacy
``(list[dict], dict[str, Callable])`` format used by ``_run_tool_loop``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    """Immutable tool definition pairing schema metadata with an executable handler.

    Args:
        name: Unique tool name.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema dict describing the tool's input parameters.
        handler: Callable that receives parsed arguments and returns a result string
            (or ``None``).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str | None]

    def to_litellm_schema(self) -> dict[str, Any]:
        """Convert to the OpenAI-format dict expected by LiteLLM.

        Returns:
            A tool definition dict with ``type`` and ``function`` keys.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def unpack_tools(
    tools: list[Tool],
) -> tuple[list[dict[str, Any]], dict[str, Callable[[dict[str, Any]], str | None]]]:
    """Convert a list of Tool objects to the legacy tuple format.

    Args:
        tools: List of Tool objects to convert.

    Returns:
        A tuple of (tool_schemas, handler_map) suitable for ``_run_tool_loop``.
    """
    schemas: list[dict[str, Any]] = []
    handlers: dict[str, Callable[[dict[str, Any]], str | None]] = {}

    for tool in tools:
        schemas.append(tool.to_litellm_schema())
        handlers[tool.name] = tool.handler

    return schemas, handlers
