# Roadmap — CellSem LLM Client

## Completed Features

### Multi-Provider LLM Agents (beta)
- LiteLLM-based agents with seamless switching between OpenAI, Anthropic, and other providers
- Environment-based configuration with dotenv integration
- Abstract base classes with provider-specific implementations

### Token Tracking & Cost Monitoring (beta)
- Real-time cost tracking via OpenAI and Anthropic usage APIs (aggregate per-key)
- Detailed token metrics: input, output, cached, and thinking tokens
- Fallback cost calculation with rate database (per-request precision)
- Enabled by default when `track_usage=True`

### JSON Schema Compliance (beta)
- OpenAI native structured outputs with `strict=true`
- Anthropic schema enforcement via tool use pattern
- Runtime Pydantic model generation from JSON Schema dicts
- Validate-and-retry logic with `SchemaValidationException` on hard failure
- JSON-first API: pass plain dicts, Pydantic models optional

### Tool Calling & MCP Integration (alpha)
- Generic `Tool` abstraction (`tools/tool.py`) for uniform tool definitions
- `LiteLLMAgent.query_with_tools` / `query_unified` — LiteLLM tool loop with automatic tool execution
- `MCPToolSource` — discovers tools from any MCP server via stdio, bridging async MCP SDK to sync callers

---

## Upcoming / Planned

### Async Support
Add async variants of agent methods to unblock async callers and simplify MCP internals.

- `aquery()`, `aquery_unified()`, etc. wrapping `litellm.acompletion()`
- Collapse `MCPToolSource` background-thread bridge to direct `await` calls
- Full backward compatibility — existing sync API unchanged
- Requires `pytest-asyncio` for new async tests
- See: [async-support-analysis.md](async-support-analysis.md)
- Downstream guidance: [async-downstream-guidance.md](async-downstream-guidance.md) (to publish to template repo when ready)

### File Attachment Support
Multi-format file inputs (images, PDFs, documents) via LiteLLM file APIs.

- Provider capability detection (`litellm.supports_pdf_input()`)
- Unified `query_with_files()` method on `LiteLLMAgent`
- Base64, URL, and file-path inputs
- Stub modules exist: `files/capabilities.py`, `files/processors.py`

### LiteLLM Feature Optimization
Reduce custom code by leveraging more of LiteLLM's built-in capabilities.

- **Cost calculation**: Use `litellm.completion_cost()` as primary, keep `FallbackCostCalculator` for edge cases only
- **Provider detection**: Use `supports_response_schema()` instead of manual model-name parsing
- **Automatic tracking**: Explore `CustomLogger` integration for usage logging
- See: [archive/litellm-feature-optimization-analysis.md](archive/litellm-feature-optimization-analysis.md)

### Model Recommendations (exploratory)
AI-powered advisory for model selection and resource estimation.

- Task complexity analysis
- Model selection advice based on cost/capability trade-offs
- Token requirement estimation for input and output
- Stub modules exist: `advisors/`

---

## Archive

Completed and superseded planning documents are in [`planning/archive/`](archive/).
