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
- Built-in `ols4_search` helper for EBI OLS4 ontology lookups (MCP with legacy fallback)
- Live integration test coverage against OLS4

---

## Upcoming / Planned

### Cyberian Integration (experimental, high priority)
Replace direct LLM API calls with local agent execution via agentapi/cyberian.

**Use Case**: Run queries through local CLI agents (Claude Code, Aider, etc.) instead of direct API calls, enabling access to agent-specific capabilities (file operations, tool use, etc.) within the library's unified interface.

**Key Differences from Deep-Research Pattern**:
- Simple query execution, not iterative research workflows
- Fast execution comparable to API calls (not 10-30 minute deep-research loops)
- Drop-in replacement for standard `query()` / `query_unified()` methods

**Implementation Approach**:
- Optional execution mode: `agent.query(..., via_cyberian=True, agent_type="claude")`
- Wraps agentapi HTTP API for communication with local agent servers
- Maps LLM responses from agent terminal output
- Maintains full compatibility with existing features (schema enforcement, tool calling, cost tracking)

**Benefits**:
- Leverage agent capabilities beyond raw LLM APIs (code execution, file operations, extended tool ecosystems)
- Test workflows locally before deploying to production API endpoints
- Enable hybrid local/cloud execution patterns

**Requirements**:
- Local CLI agents installed (Claude Code, Aider, etc.)
- agentapi server running (or managed by library)
- `pip install cellsem-llm-client[cyberian]` for optional dependencies

**Considerations**:
- Adds dependency on external CLI tools and agentapi
- Requires process/server lifecycle management
- Terminal output parsing may be less reliable than structured API responses
- Limited to agents with CLI interfaces

### Async Support
Add async variants of agent methods to unblock async callers and simplify MCP internals.

- `aquery()`, `aquery_unified()`, etc. wrapping `litellm.acompletion()`
- Collapse `MCPToolSource` background-thread bridge to direct `await` calls
- Full backward compatibility — existing sync API unchanged
- Requires `pytest-asyncio` for new async tests
- See: [async-support-analysis.md](async-support-analysis.md)

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
