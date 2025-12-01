"""Unit tests for the OLS4 MCP tool helper."""

import pytest
import requests

from cellsem_llm_client.tools.ols_mcp import build_ols4_search_tool


@pytest.mark.unit
def test_build_ols4_tool_definition() -> None:
    """Tool definition should expose the expected schema."""
    tools, handlers = build_ols4_search_tool()

    assert len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "ols4_search"

    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"query"}
    assert set(params["properties"].keys()) == {"query", "ontology", "rows"}
    assert "ols4_search" in handlers
    assert callable(handlers["ols4_search"])


@pytest.mark.unit
def test_ols4_handler_calls_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler should call OLS4 search API and format the top hits."""
    tools, handlers = build_ols4_search_tool()
    handler = handlers["ols4_search"]

    captured: dict[str, object] = {}

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "response": {
                    "docs": [
                        {
                            "label": "cell",
                            "iri": "http://purl.obolibrary.org/obo/CL_0000000",
                            "description": "A basic biological unit.",
                        }
                    ]
                }
            }

    def fake_get(url: str, params: dict[str, object], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("cellsem_llm_client.tools.ols_mcp.requests.get", fake_get)

    result = handler({"query": "cell", "rows": 1})

    assert "cell" in result
    assert "http://purl.obolibrary.org/obo/CL_0000000" in result
    assert captured["params"] == {"q": "cell", "rows": 1, "type": "class"}
    assert captured["url"] == "https://www.ebi.ac.uk/ols4/api/mcp/search"


@pytest.mark.unit
def test_ols4_handler_fallback_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler should fall back to legacy /api/search when MCP path is missing."""
    tools, handlers = build_ols4_search_tool()
    handler = handlers["ols4_search"]

    calls: list[str] = []

    class NotFoundResponse:
        status_code = 404

        def raise_for_status(self) -> None:  # noqa: D401
            resp = requests.Response()
            resp.status_code = 404
            raise requests.HTTPError(response=resp)

    class OkResponse:
        status_code = 200

        def raise_for_status(self) -> None:  # noqa: D401
            return None

        def json(self) -> dict[str, object]:
            return {"response": {"docs": [{"label": "fallback", "iri": "iri:1"}]}}

    def fake_get(url: str, params: dict[str, object], timeout: float) -> object:
        calls.append(url)
        if "mcp" in url:
            return NotFoundResponse()
        return OkResponse()

    monkeypatch.setattr("cellsem_llm_client.tools.ols_mcp.requests.get", fake_get)

    result = handler({"query": "cell"})

    assert result.startswith("fallback")
    assert calls == [
        "https://www.ebi.ac.uk/ols4/api/mcp/search",
        "https://www.ebi.ac.uk/ols4/api/search",
    ]
