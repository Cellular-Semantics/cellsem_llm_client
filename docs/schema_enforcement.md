# Schema Enforcement (JSON-First)

This page shows how to enforce structured outputs using JSON Schema across providers, with optional Pydantic helpers for validation and retries.

## Core Concepts

- **JSON-First**: Supply a JSON Schema `dict` directly. Pydantic models are optional helpers (use `MyModel.model_json_schema()`).
- **Provider-Aware**:
  - **OpenAI**: Uses native strict `response_format`.
  - **Anthropic**: Converts the schema to a single enforced tool call.
  - **Other providers**: Prompt hint plus post-validation fallback.
- **Validation & Retry**: Responses are parsed and validated against a derived Pydantic model with lightweight retries; hard failures raise `SchemaValidationException`.
- **Runtime Model Generation**: `SchemaManager` loads schemas from dict/file/URL, generates Pydantic models on the fly, and caches them.

## Quick Example (JSON-First)

```python
from cellsem_llm_client.agents import LiteLLMAgent

schema = {
    "type": "object",
    "properties": {
        "term": {"type": "string", "description": "Cell type name"},
        "iri": {"type": "string", "format": "uri"},
    },
    "required": ["term", "iri"],
    "additionalProperties": False,
}

agent = LiteLLMAgent(model="gpt-4o", api_key="your-key")
result = agent.query_with_schema(
    message="Return a cell type name and IRI.",
    schema=schema,  # JSON-first
)

print(result.model_dump())  # Pydantic model generated at runtime
```

## Schema Inputs

- **JSON Schema dict** (preferred): Pass directly via `schema=...`.
- **Pydantic model**: Pass the class or `model_json_schema()`; the schema is derived for enforcement.
- **Schema name**: Place `<name>.json` in your schema directory and use `schema="name"`; `SchemaManager` will load, validate, and cache it.

## Under the Hood

- `SchemaManager`: Loads schemas (dict/file/URL) and generates Pydantic models.
- `SchemaAdapterFactory`: Picks a provider adapter (OpenAI strict, Anthropic tool, fallback prompt hint).
- `SchemaValidator`: Parses/validates responses with retries; raises `SchemaValidationException` on exhaustion.

## Notes

- For OpenAI strict mode, the schema is tightened (e.g., `additionalProperties=False`) for better enforcement.
- Anthropic responses are returned from the first enforced tool call’s JSON arguments.
- You can add custom schema directories by configuring `SchemaManager` if needed.
