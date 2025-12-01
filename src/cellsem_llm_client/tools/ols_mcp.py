"""OLS4 Model Context Protocol (MCP) tool helpers.

This module exposes a helper that builds a LiteLLM-compatible tool definition plus
an executable handler for searching the OLS4 ontology service.
"""

from collections.abc import Callable
from typing import Any

import requests

OLS4_MCP_BASE_URL = "https://www.ebi.ac.uk/ols4/api/mcp"
OLS4_SEARCH_FALLBACK_URL = "https://www.ebi.ac.uk/ols4/api/search"


def build_ols4_search_tool(
    base_url: str = OLS4_MCP_BASE_URL,
    default_rows: int = 3,
    timeout: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Callable[[dict[str, Any]], str]]]:
    """Build the OLS4 search tool definition and handler.

    Args:
        base_url: Base URL for the OLS4 search endpoint.
        default_rows: Default number of rows to request when not specified.
        timeout: Timeout for outbound HTTP requests in seconds.

    Returns:
        A tuple of (tools_list, handlers_map) suitable for ``query_with_tools``.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ols4_search",
                "description": "Search the OLS4 ontology service for terms.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query string to search for.",
                        },
                        "ontology": {
                            "type": "string",
                            "description": "Optional ontology identifier to scope the search.",
                        },
                        "rows": {
                            "type": "integer",
                            "description": "Number of results to return (default: 3).",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    def _handler(args: dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "query is required."

        ontology = args.get("ontology")
        rows = int(args.get("rows") or default_rows)

        search_url = f"{base_url.rstrip('/')}/search"
        params: dict[str, Any] = {"q": query, "rows": rows, "type": "class"}
        if ontology:
            params["ontology"] = ontology

        def _fetch(url: str) -> dict[str, Any]:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()

        try:
            data = _fetch(search_url)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # Fallback to the legacy search endpoint if MCP path is missing
            if status == 404 and "mcp" in base_url:
                try:
                    data = _fetch(OLS4_SEARCH_FALLBACK_URL)
                except requests.RequestException as exc_inner:
                    return f"OLS4 request failed after fallback: {exc_inner}"
            else:
                return f"OLS4 request failed: {exc}"
        except requests.RequestException as exc:
            return f"OLS4 request failed: {exc}"

        docs = (data.get("response") or {}).get("docs") or []

        if not docs:
            return "No results found."

        snippets: list[str] = []
        for doc in docs[:rows]:
            label = str(doc.get("label") or "Unknown")
            iri = str(doc.get("iri") or "")
            description = doc.get("description")
            if isinstance(description, list):
                description = description[0] if description else ""
            desc_text = str(description) if description else ""

            snippet = f"{label} — {iri}"
            if desc_text:
                snippet = f"{snippet} :: {desc_text}"
            snippets.append(snippet)

        return " | ".join(snippets)

    handlers: dict[str, Callable[[dict[str, Any]], str]] = {"ols4_search": _handler}
    return tools, handlers
