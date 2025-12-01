"""Integration test for OLS4 MCP tool."""

import pytest

from cellsem_llm_client.tools import build_ols4_search_tool


@pytest.mark.integration
def test_ols4_search_real_endpoint() -> None:
    """Ensure the OLS4 MCP search endpoint returns real ontology hits."""
    tools, handlers = build_ols4_search_tool()
    handler = handlers["ols4_search"]

    result = handler(
        {
            "query": "Bergmann glial cell",
            "ontology": "cl",
            "rows": 2,
        }
    )

    assert "Bergmann glial cell" in result
    # Ensure we received Cell Ontology IRIs
    assert "purl.obolibrary.org/obo/CL_" in result
