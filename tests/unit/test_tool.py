"""Unit tests for Tool dataclass and unpack_tools helper."""

from typing import Any

import pytest

from cellsem_llm_client.tools.tool import Tool, unpack_tools


class TestTool:
    """Tests for the Tool frozen dataclass."""

    @pytest.mark.unit
    def test_tool_creation(self) -> None:
        """Test that Tool stores all fields correctly."""

        def handler(args: dict[str, Any]) -> str:
            return "ok"

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            handler=handler,
        )

        assert tool.name == "my_tool"
        assert tool.description == "A test tool"
        assert tool.parameters["type"] == "object"
        assert tool.handler is handler

    @pytest.mark.unit
    def test_tool_frozen(self) -> None:
        """Test that Tool is immutable (frozen dataclass)."""
        tool = Tool(
            name="immutable",
            description="Cannot change",
            parameters={"type": "object"},
            handler=lambda args: None,
        )

        with pytest.raises(AttributeError):
            tool.name = "changed"  # type: ignore[misc]

    @pytest.mark.unit
    def test_tool_to_litellm_schema(self) -> None:
        """Test to_litellm_schema produces correct OpenAI-format dict."""
        tool = Tool(
            name="search",
            description="Search for items",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda args: "result",
        )

        schema = tool.to_litellm_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for items"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]

    @pytest.mark.unit
    def test_tool_handler_invocation(self) -> None:
        """Test that handler is callable with dict and returns string."""
        captured: dict[str, Any] = {}

        def handler(args: dict[str, Any]) -> str:
            captured.update(args)
            return f"result: {args['x']}"

        tool = Tool(
            name="calc",
            description="Calculator",
            parameters={"type": "object"},
            handler=handler,
        )

        result = tool.handler({"x": 42})
        assert result == "result: 42"
        assert captured == {"x": 42}

    @pytest.mark.unit
    def test_tool_handler_returns_none(self) -> None:
        """Test that handler can return None."""
        tool = Tool(
            name="void",
            description="Returns nothing",
            parameters={"type": "object"},
            handler=lambda args: None,
        )

        result = tool.handler({})
        assert result is None


class TestUnpackTools:
    """Tests for unpack_tools helper function."""

    @pytest.mark.unit
    def test_unpack_tools_basic(self) -> None:
        """Test converting list[Tool] to legacy tuple format."""

        def handler_a(args: dict[str, Any]) -> str:
            return "a"

        def handler_b(args: dict[str, Any]) -> str:
            return "b"

        tools = [
            Tool(
                name="tool_a",
                description="Tool A",
                parameters={"type": "object", "properties": {}},
                handler=handler_a,
            ),
            Tool(
                name="tool_b",
                description="Tool B",
                parameters={"type": "object", "properties": {}},
                handler=handler_b,
            ),
        ]

        schemas, handlers = unpack_tools(tools)

        assert len(schemas) == 2
        assert schemas[0]["function"]["name"] == "tool_a"
        assert schemas[1]["function"]["name"] == "tool_b"
        assert handlers["tool_a"] is handler_a
        assert handlers["tool_b"] is handler_b

    @pytest.mark.unit
    def test_unpack_tools_empty(self) -> None:
        """Test unpack_tools with empty list."""
        schemas, handlers = unpack_tools([])
        assert schemas == []
        assert handlers == {}

    @pytest.mark.unit
    def test_unpack_tools_preserves_parameters(self) -> None:
        """Test that parameters are preserved in schema output."""
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        tool = Tool(
            name="search",
            description="Search",
            parameters=params,
            handler=lambda args: "ok",
        )

        schemas, _ = unpack_tools([tool])
        assert schemas[0]["function"]["parameters"] == params
