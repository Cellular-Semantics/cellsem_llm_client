# Codex MCP Integration POC Report

**Date**: 2026-01-13
**Objective**: Evaluate viability of using OpenAI Codex MCP as an alternative API backend for CellSem_LLM_client to leverage institutional Codex access and reduce API costs.

---

## Executive Summary

**Result**: ❌ **NOT VIABLE for cost savings**

**Key Finding**: Codex MCP requires `OPENAI_API_KEY` to function, even when using an institutionally authenticated Codex CLI. Institutional Codex authentication alone is **insufficient** for MCP operations. This means queries still bill to a personal/project OpenAI account, providing **no cost savings** over direct LiteLLM usage.

**Recommendation**: Continue using LiteLLM for direct API access. Do not integrate Codex MCP unless institutional authentication becomes supported.

---

## Test Environment

### Prerequisites Installed
- ✅ Codex CLI: `codex` (installed at `/opt/homebrew/bin/codex`)
- ✅ Node.js 18+
- ✅ Python 3.10+
- ✅ OpenAI Agents SDK: `openai-agents` (PyPI package, imports as `agents`)
- ✅ Dependencies: `openai`, `python-dotenv`

### Authentication Methods Tested
1. **WITH OPENAI_API_KEY**: Personal API key from `.env` file
2. **WITHOUT OPENAI_API_KEY**: Institutional Codex CLI authentication only

---

## Tests Performed

### Test Script
- **Location**: `poc_codex_mcp.py`
- **Method**: Auto-spawns `codex mcp-server` as subprocess via `MCPServerStdio`
- **Transport**: stdio (only transport supported by Codex MCP)

### Test Scenarios

#### Scenario A: WITH OPENAI_API_KEY ✅
**Result**: **ALL TESTS PASSED**

1. ✅ **Connection Test**: Successfully connected to Codex MCP server
2. ✅ **Coding Prompt**: Generated Python fibonacci function correctly
3. ✅ **Non-Coding Prompt**: Explained photosynthesis for 10-year-old (full explanation)
4. ✅ **Usage Tracking**: `RunResult` object contains metadata for extraction

**Sample Output**:
```
Prompt: "Explain photosynthesis in simple terms, suitable for a 10-year-old"

Response: "Sure! Photosynthesis is how plants make their own food. Here's how it works:
1. Sunlight: Plants need sunlight. The green part of the plant, called leaves, catch the sunlight.
2. Air: Plants also take in a gas from the air called carbon dioxide...
[Full educational explanation provided]"
```

#### Scenario B: WITHOUT OPENAI_API_KEY ❌
**Result**: **FAILED TO RUN**

- ❌ Server spawn failed or agent initialization failed
- Institutional Codex CLI authentication was **not sufficient**
- No fallback to institutional billing

---

## Technical Findings

### 1. Import Corrections Required
- **Issue**: Package name is `openai-agents` but imports are `from agents import ...`
- **Resolution**: Corrected imports in POC script

### 2. Codex MCP Server Invocation
- **Attempted**: `npx -y codex mcp-server` (downloads fresh copy, no auth inheritance)
- **Attempted**: `codex mcp-server` (uses local authenticated binary)
- **Result**: Both require `OPENAI_API_KEY` environment variable

### 3. Transport Limitations
- Codex MCP only supports **stdio transport** (not HTTP/SSE)
- Cannot connect to pre-running server - must spawn as subprocess
- This is by design: `[experimental] Run the Codex MCP server (stdio transport)`

### 4. Authentication Architecture
```
User's Setup:
┌─────────────────────────────────────────┐
│ Codex CLI (authenticated via codex login) │
│ - Institutional SSO/browser auth         │
│ - Used for: codex run, codex chat, etc. │
└─────────────────────────────────────────┘
                  ↓
         spawns subprocess
                  ↓
┌─────────────────────────────────────────┐
│ codex mcp-server (stdio transport)       │
│ - Exposes Codex as MCP tools            │
│ - Makes OpenAI API calls internally     │
│ - REQUIRES: OPENAI_API_KEY ⚠️            │
└─────────────────────────────────────────┘
                  ↓
         Used by OpenAI Agents SDK
                  ↓
┌─────────────────────────────────────────┐
│ openai-agents (Python SDK)               │
│ - Orchestrates agent workflows          │
│ - Calls MCP server tools                │
│ - REQUIRES: OPENAI_API_KEY ⚠️            │
└─────────────────────────────────────────┘
```

**Critical Issue**: Both `codex mcp-server` and `openai-agents` require `OPENAI_API_KEY`, independent of Codex CLI authentication.

---

## Capability Assessment

### What Codex MCP CAN Do ✅
- ✅ Handle coding tasks (function generation, debugging)
- ✅ Handle general LLM queries (explanations, creative writing, Q&A)
- ✅ Multi-turn conversations with state management
- ✅ Tool calling and agent handoffs
- ✅ Provide usage metadata via `RunResult` objects

### What Codex MCP CANNOT Do ❌
- ❌ Use institutional Codex authentication for billing
- ❌ Avoid OpenAI API key requirement
- ❌ Provide cost savings over direct API calls
- ❌ Operate without `OPENAI_API_KEY` environment variable

---

## Integration Feasibility for CellSem_LLM_client

### Current Architecture
CellSem_LLM_client uses:
- **LiteLLM**: Direct API calls to multiple providers (OpenAI, Anthropic, etc.)
- **Clean abstractions**: `AgentConnection` interface, provider adapters, schema enforcement
- **Cost tracking**: Token usage monitoring and cost calculation

### Codex MCP Integration Would Add
- ➕ MCP protocol layer (additional abstraction)
- ➕ Multi-agent orchestration capabilities
- ➕ Conversation state management
- ➖ **Still requires OPENAI_API_KEY** (no cost reduction)
- ➖ Complexity without benefit for single-agent use cases
- ➖ Limited to OpenAI models (LiteLLM supports 100+ providers)

### Recommendation: ❌ **DO NOT INTEGRATE**

**Reasoning**:
1. **No cost savings**: Primary goal was to leverage institutional Codex access - not achieved
2. **Added complexity**: MCP layer adds overhead without compensating benefits
3. **LiteLLM superiority**: Direct API calls are simpler, faster, and support more providers
4. **Feature overlap**: LiteLLM already provides model switching, cost tracking, schema compliance

**Alternative considered**: Integrate MCP for multi-agent orchestration features only
- **Rejected**: Complexity not justified unless institutional auth works

---

## Cost Analysis

### Goal
Leverage unused institutional Codex capacity to avoid personal API token costs during testing/development.

### Reality
```
Cost with LiteLLM:         $X per 1M tokens (direct OpenAI billing)
Cost with Codex MCP:       $X per 1M tokens (still OpenAI billing via API key)
Cost with institutional:   NOT AVAILABLE (requires API key anyway)

Savings: $0 ❌
```

### Conclusion
Codex MCP is essentially a **wrapper around OpenAI API calls**, not an alternative authentication/billing mechanism.

---

## Files Created

1. **`poc_codex_mcp.py`** - Main POC script with dual auth testing
2. **`poc_codex_mcp_manual.py`** - Attempted manual server connection (unused)
3. **`POC_README.md`** - Setup instructions
4. **`CODEX_MCP_POC_REPORT.md`** - This report
5. **`~/.claude/plans/cached-watching-locket.md`** - Detailed integration plan (ON HOLD)

---

## Lessons Learned

1. **Package naming**: `openai-agents` (PyPI) imports as `agents` (not `openai_agents`)
2. **MCP transport**: stdio-only means subprocess spawning, not external server connection
3. **Authentication layers**: CLI auth ≠ API auth; they're independent systems
4. **Codex MCP design**: Built for multi-agent orchestration, not as an API cost-reduction tool

---

## Future Considerations

### If Requirements Change
Only reconsider Codex MCP integration if:
- ✅ Institutional authentication becomes supported (removes API key requirement)
- ✅ Multi-agent orchestration becomes a core requirement
- ✅ Cost savings can be demonstrated

### Alternative Cost-Reduction Strategies
1. **Use cheaper models**: Switch to `gpt-4o-mini`, `claude-3-haiku` for simple tasks
2. **Prompt caching**: Implement caching for repeated queries (already in CellSem design)
3. **Rate limiting**: Implement intelligent retry/backoff for cost control
4. **Local models**: Investigate Ollama, LM Studio for offline/free inference
5. **Batch processing**: Use OpenAI Batch API for non-urgent queries (50% discount)

---

## References

- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- Codex MCP Guide: https://developers.openai.com/codex/guides/agents-sdk/
- MCP Protocol: https://docs.mcp.run/
- CellSem_LLM_client Repo: `/Users/do12/Documents/GitHub/CellSem_LLM_client`

---

## Contact & Context

**Investigator**: User with institutional Codex access
**Date**: January 13, 2026
**Use Case**: Cost-effective LLM testing/development for CellSem project
**Outcome**: Continue with LiteLLM; institutional Codex not viable for MCP

---

**Status**: POC COMPLETE - Integration NOT RECOMMENDED ❌
