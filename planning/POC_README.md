# Codex MCP Proof of Concept

Quick test to evaluate if OpenAI Codex MCP can be used as an alternative API backend for CellSem_LLM_client.

## Goal

Test whether institutional Codex access can be leveraged for:
- General LLM queries (not just coding)
- Cost-effective testing and development
- Reusing unused institutional Codex capacity

## Prerequisites

1. **Node.js 18+** - Check: `node --version`
2. **Codex CLI** - Install: `npm install -g codex`
3. **Python 3.10+** - Check: `python --version`
4. **Authentication** (pick one):
   - **Institutional Codex**: Authenticate once via `codex login` (browser-based SSO)
   - **Personal OpenAI**: Set `OPENAI_API_KEY` in `.env` file

## Quick Start

### Option A: Institutional Codex (Recommended - No Token Cost!)

```bash
# 1. Authenticate Codex with institutional login (one-time setup)
codex login
# This will open browser for SSO authentication

# 2. Install Python dependencies
uv add openai openai-agents python-dotenv

# 3. Remove or rename .env file (so it doesn't use personal API key)
mv .env .env.backup  # Or just delete OPENAI_API_KEY from .env

# 4. Run the POC script - it will use your institutional Codex auth!
uv run python poc_codex_mcp.py
```

### Option B: Personal OpenAI API Key

```bash
# 1. Install Python dependencies
uv add openai openai-agents python-dotenv

# 2. Set up API key
echo "OPENAI_API_KEY=your-key-here" >> .env

# 3. Run the POC script
uv run python poc_codex_mcp.py
```

## What the Script Tests

1. **Connection**: Can we connect to Codex MCP server?
2. **Coding Prompt**: Does it work for code generation (baseline)?
3. **Non-Coding Prompt**: Can it handle general queries? (CRITICAL)
4. **Usage Tracking**: Can we extract token/cost information?

## Expected Outcomes

### ✅ Success (Viable for Integration)
- All 4 tests pass
- Codex handles both coding and non-coding prompts
- Proceed with MCPAgent integration

### ⚠️ Limited Success
- Coding works, non-coding doesn't
- Could integrate for code-specific features only
- Adds complexity (hybrid approach)

### ❌ Failure (Not Viable)
- Cannot connect or queries fail
- Document why and explore alternatives

## Critical Questions to Answer

- [ ] Can we authenticate with institutional login (no personal API key)?
- [ ] Does Codex accept non-coding prompts?
- [ ] Is usage billed to institutional account or personal API?
- [ ] What are the token/quota limits?
- [ ] Can we extract usage metrics for cost tracking?

## Next Steps After POC

See `/Users/do12/.claude/plans/cached-watching-locket.md` for:
- Full integration plan (on hold until POC completes)
- Architecture design for MCPAgent
- Open questions to research

## Troubleshooting

**Error: `openai_agents not found`**
```bash
uv add openai openai-agents python-dotenv
```

**Error: `codex: command not found`**
```bash
npm install -g codex
codex --version
```

**Error: `Cannot connect to MCP server`**
- Check Node.js version: `node --version` (needs 18+)
- Try running manually: `npx -y codex mcp-server`
- Check for firewall/network issues

**Institutional Login Questions**
- Contact your institution's IT support for Codex access details
- May need different authentication than OPENAI_API_KEY
- POC will test if API key is required or optional

## Files

- `poc_codex_mcp.py` - Main test script
- `/Users/do12/.claude/plans/cached-watching-locket.md` - Full integration plan
- `POC_README.md` - This file
