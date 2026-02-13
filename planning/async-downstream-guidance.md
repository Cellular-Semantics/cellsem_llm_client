# Async vs Sync Guidance for Downstream Users

> **TODO:** When async support is implemented (see [ROADMAP.md](ROADMAP.md#async-support)),
> add this guidance to the cellular-semantics template repo.

## Recommendation

**Default to sync unless you have a concrete concurrency need.** The sync API will remain first-class and fully supported. If you find yourself writing loops that make multiple independent LLM calls, or building a web service, switch those specific workflows to async. You don't need to go all-in — async and sync can coexist since the sync API won't be deprecated.

## Use async when:

- **Multiple LLM calls in a workflow** — e.g. querying 3 providers in parallel for comparison, or fan-out across ontology terms. Async turns sequential 3x latency into ~1x
- **Web applications / API servers** — if the downstream project serves HTTP requests (FastAPI, aiohttp), blocking on LLM calls ties up the thread pool. Async is the natural fit
- **MCP-heavy workflows** — MCPToolSource wraps an async SDK; with native async support the bridge complexity vanishes and you get cleaner error propagation
- **Batch processing** — running many schema-enforced queries can be parallelised with `asyncio.gather()` with concurrency control via semaphores

## Stick with sync when:

- **Scripts and notebooks** — Jupyter has its own event loop complications. Simple sequential scripts gain nothing from async and lose readability
- **Single-query CLI tools** — if you make one LLM call and exit, async is just ceremony
- **Team familiarity** — async Python has real footguns (`await` missing silently, accidental blocking in async context, debugger confusion). If the team isn't comfortable with it, the bugs will cost more than the latency savings
- **Integration with sync-only libraries** — if your downstream pipeline is built on synchronous libraries (e.g. some bioinformatics tooling), mixing in async creates friction

## Key insight

LLM API calls are **high-latency** (seconds, not milliseconds), so the concurrency benefit of async is proportional to how many you run in parallel. One call = no benefit. Five concurrent calls = significant benefit.
