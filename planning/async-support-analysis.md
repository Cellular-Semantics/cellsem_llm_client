# Async Support Analysis

## Current State

The library is **entirely synchronous**. `MCPToolSource` is the sole exception — it hides async behind a sync context manager using a background thread + event loop bridge.

### Sync throughout

- `LiteLLMAgent` — all `query*` methods are sync. Uses `litellm.completion()` (sync variant).
- `SchemaManager`, `SchemaValidator`, adapters — all sync.
- `FallbackCostCalculator`, `UsageMetrics` — pure data, no I/O.
- `build_ols4_search_tool()` — uses `requests.get()` (sync HTTP).
- All tests — no `pytest-asyncio`, no `async def test_*`.

### Async only in `mcp_source.py`

- The `mcp` SDK is async-only, so `MCPToolSource` runs a background thread with its own event loop and bridges back to sync via `concurrent.futures.Future`. Callers never see async.

### Implications

- LiteLLM *does* have `litellm.acompletion()` (async counterpart) but the repo doesn't use it.
- There's no `async def query()` anywhere — callers who are themselves async would block their event loop on every LLM call.
- The `MCPToolSource` bridge pattern (background thread + event loop) works but adds complexity. If the rest of the library were async-native, the MCP integration would be much simpler — just `await` directly.

## Next Goal

Add async support while maintaining full backward compatibility for existing sync callers in downstream projects.

### Approach: async variants alongside sync

If async support becomes a goal, the main effort would be adding `async` variants of the agent methods (e.g. `async def aquery_unified()` wrapping `litellm.acompletion()`), at which point the MCP bridge could collapse to direct `await` calls. The existing sync API stays unchanged — downstream projects continue working without modification.
