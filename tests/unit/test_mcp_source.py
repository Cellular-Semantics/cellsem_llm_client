"""Unit tests for MCPToolSource."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from cellsem_llm_client.tools.mcp_source import MCPToolSource
from cellsem_llm_client.tools.tool import Tool


def _make_openai_tool(
    name: str, description: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Helper to create an OpenAI-format tool dict (as returned by load_mcp_tools)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


class TestMCPToolSourceToolDiscovery:
    """Test that MCPToolSource discovers and wraps MCP tools."""

    @pytest.mark.unit
    @patch("cellsem_llm_client.tools.mcp_source.load_mcp_tools")
    @patch("cellsem_llm_client.tools.mcp_source.ClientSession")
    @patch("cellsem_llm_client.tools.mcp_source.sse_client")
    def test_mcp_source_discovers_tools(
        self,
        mock_sse: Any,
        mock_session_cls: Any,
        mock_load: Any,
    ) -> None:
        """MCPToolSource should discover tools from MCP session and return list[Tool]."""
        mock_load.return_value = [
            _make_openai_tool(
                "search",
                "Search ontology",
                {"type": "object", "properties": {"q": {"type": "string"}}},
            ),
            _make_openai_tool(
                "lookup",
                "Look up term",
                {"type": "object", "properties": {"id": {"type": "string"}}},
            ),
        ]

        # Setup async context manager chain: sse_client -> (read, write) -> ClientSession
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.return_value = (mock_read, mock_write)
        mock_sse.return_value = mock_transport_cm

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cls.return_value = mock_session_cm

        with MCPToolSource("https://example.com/mcp") as source:
            tools = source.tools
            assert len(tools) == 2
            assert all(isinstance(t, Tool) for t in tools)
            assert tools[0].name == "search"
            assert tools[1].name == "lookup"
            assert tools[0].description == "Search ontology"
            assert tools[0].parameters == {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            }

    @pytest.mark.unit
    @patch("cellsem_llm_client.tools.mcp_source.load_mcp_tools")
    @patch("cellsem_llm_client.tools.mcp_source.ClientSession")
    @patch("cellsem_llm_client.tools.mcp_source.sse_client")
    def test_mcp_source_tool_handler_calls_mcp(
        self,
        mock_sse: Any,
        mock_session_cls: Any,
        mock_load: Any,
    ) -> None:
        """Tool handler should dispatch call_tool to the MCP session."""
        mock_load.return_value = [
            _make_openai_tool(
                "search",
                "Search",
                {"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ]

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.return_value = (mock_read, mock_write)
        mock_sse.return_value = mock_transport_cm

        # Mock call_tool to return a CallToolResult-like object
        mock_call_result = Mock()
        mock_call_result.isError = False
        text_content = Mock()
        text_content.type = "text"
        text_content.text = "CL:0000001 | cell"
        mock_call_result.content = [text_content]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cls.return_value = mock_session_cm

        with MCPToolSource("https://example.com/mcp") as source:
            tool = source.tools[0]
            result = tool.handler({"q": "cell"})

            assert result == "CL:0000001 | cell"
            mock_session.call_tool.assert_awaited_once_with("search", {"q": "cell"})


class TestMCPToolSourceContextManager:
    """Test MCPToolSource lifecycle management."""

    @pytest.mark.unit
    @patch("cellsem_llm_client.tools.mcp_source.load_mcp_tools")
    @patch("cellsem_llm_client.tools.mcp_source.ClientSession")
    @patch("cellsem_llm_client.tools.mcp_source.sse_client")
    def test_mcp_source_context_manager(
        self,
        mock_sse: Any,
        mock_session_cls: Any,
        mock_load: Any,
    ) -> None:
        """Context manager should connect on enter and clean up on exit."""
        mock_load.return_value = []

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.return_value = (mock_read, mock_write)
        mock_sse.return_value = mock_transport_cm

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cls.return_value = mock_session_cm

        with MCPToolSource("https://example.com/mcp") as source:
            assert source.tools is not None
            mock_session.initialize.assert_awaited_once()

        # After exit, tools should still be accessible (they're just data)
        assert source.tools is not None


class TestMCPToolSourceTransportDetection:
    """Test transport auto-detection logic."""

    @pytest.mark.unit
    def test_auto_transport_http_url(self) -> None:
        """HTTP/HTTPS URLs should be detected as SSE transport."""
        source = MCPToolSource.__new__(MCPToolSource)
        source._server = "https://example.com/mcp"
        source._transport = "auto"
        assert source._detect_transport() == "sse"

    @pytest.mark.unit
    def test_auto_transport_http_prefix(self) -> None:
        """http:// prefix should also be detected as SSE."""
        source = MCPToolSource.__new__(MCPToolSource)
        source._server = "http://localhost:8080/mcp"
        source._transport = "auto"
        assert source._detect_transport() == "sse"

    @pytest.mark.unit
    def test_auto_transport_file_path(self) -> None:
        """File paths should be detected as stdio transport."""
        source = MCPToolSource.__new__(MCPToolSource)
        source._server = "./my-server.py"
        source._transport = "auto"
        assert source._detect_transport() == "stdio"

    @pytest.mark.unit
    def test_auto_transport_command(self) -> None:
        """Commands without URL prefix should be detected as stdio."""
        source = MCPToolSource.__new__(MCPToolSource)
        source._server = "node server.js"
        source._transport = "auto"
        assert source._detect_transport() == "stdio"

    @pytest.mark.unit
    def test_explicit_transport_override(self) -> None:
        """Explicit transport= overrides auto-detection."""
        source = MCPToolSource.__new__(MCPToolSource)
        source._server = "https://example.com/mcp"
        source._transport = "stdio"
        assert source._detect_transport() == "stdio"

    @pytest.mark.unit
    def test_auto_transport_streamable_http(self) -> None:
        """Explicit streamable_http transport should be returned."""
        source = MCPToolSource.__new__(MCPToolSource)
        source._server = "https://example.com/mcp"
        source._transport = "streamable_http"
        assert source._detect_transport() == "streamable_http"


class TestMCPToolSourceSchemaMapping:
    """Test that MCP tool definitions map correctly to Tool.parameters."""

    @pytest.mark.unit
    @patch("cellsem_llm_client.tools.mcp_source.load_mcp_tools")
    @patch("cellsem_llm_client.tools.mcp_source.ClientSession")
    @patch("cellsem_llm_client.tools.mcp_source.sse_client")
    def test_mcp_source_tool_schema_mapping(
        self,
        mock_sse: Any,
        mock_session_cls: Any,
        mock_load: Any,
    ) -> None:
        """Tool.parameters should reflect the MCP tool's input schema."""
        complex_params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "ontology": {"type": "string", "description": "Ontology filter"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }
        mock_load.return_value = [
            _make_openai_tool("complex_search", "Complex search", complex_params),
        ]

        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__.return_value = (mock_read, mock_write)
        mock_sse.return_value = mock_transport_cm

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cls.return_value = mock_session_cm

        with MCPToolSource("https://example.com/mcp") as source:
            tool = source.tools[0]
            assert tool.parameters == complex_params
            # Verify round-trip: to_litellm_schema should reconstruct the original
            schema = tool.to_litellm_schema()
            assert schema["function"]["parameters"] == complex_params
