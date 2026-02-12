"""Integration tests for MCPToolSource with real MCP servers."""

import os

import pytest

from cellsem_llm_client.agents.agent_connection import LiteLLMAgent
from cellsem_llm_client.tools.mcp_source import MCPToolSource

OLS4_MCP_URL = "https://www.ebi.ac.uk/ols4/api/mcp"


@pytest.mark.integration
class TestMCPSourceOLS4Real:
    """Integration tests connecting to the real OLS4 MCP endpoint (streamable HTTP)."""

    def test_mcp_source_ols4_real(self) -> None:
        """Connect to real OLS4 MCP endpoint, discover tools, and call one."""
        with MCPToolSource(OLS4_MCP_URL, transport="streamable_http") as ols:
            # Should discover at least one tool
            assert len(ols.tools) > 0

            # Each tool should be a proper Tool object
            for tool in ols.tools:
                assert tool.name
                assert tool.description
                assert isinstance(tool.parameters, dict)
                assert callable(tool.handler)

            tool_names = [t.name for t in ols.tools]
            print(f"Discovered tools: {tool_names}")

            # Find a search-related tool and try calling it
            search_tools = [t for t in ols.tools if "search" in t.name.lower()]
            if search_tools:
                result = search_tools[0].handler({"query": "cell"})
                assert result is not None
                assert isinstance(result, str)
                print(f"Search result: {result[:200]}")

    def test_mcp_source_with_query_unified(self) -> None:
        """End-to-end: discover tools via MCP, use them in query_unified."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.fail("OPENAI_API_KEY required for this integration test")

        agent = LiteLLMAgent(model="gpt-4o-mini", api_key=api_key, max_tokens=500)

        with MCPToolSource(OLS4_MCP_URL, transport="streamable_http") as ols:
            if not ols.tools:
                pytest.skip("No tools discovered from OLS4 MCP endpoint")

            result = agent.query_unified(
                system_message="You are an ontology expert. Use the available tools to answer.",
                message="Search for 'neuron' in the Cell Ontology and tell me the top result.",
                tools=ols.tools,
                max_turns=5,
            )

            assert result.text is not None
            assert len(result.text) > 0
            print(f"Agent response: {result.text[:300]}")
