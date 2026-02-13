# CellSem LLM Client

[![Tests](https://github.com/Cellular-Semantics/cellsem_llm_client/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Cellular-Semantics/cellsem_llm_client/actions/workflows/test.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Cellular-Semantics/cellsem_llm_client/main/.github/badges/coverage.json)](https://github.com/Cellular-Semantics/cellsem_llm_client/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A flexible LLM client with multi-provider support, built using LiteLLM + Pydantic for seamless integration across OpenAI, Anthropic, and other providers.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Cellular-Semantics/cellsem_llm_client.git
cd cellsem_llm_client

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync --dev

# Set up pre-commit hooks (optional but recommended)
uv run pre-commit install
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Basic Usage

```python
from cellsem_llm_client import create_openai_agent, create_anthropic_agent

# Create agents with automatic environment configuration
openai_agent = create_openai_agent()
anthropic_agent = create_anthropic_agent()

# Simple queries
openai_response = openai_agent.query("Explain quantum computing in 50 words")
claude_response = anthropic_agent.query("What are the benefits of renewable energy?")

# Custom configuration
custom_agent = create_openai_agent(
    model="gpt-4",
    max_tokens=2000
)
```



## 📚 Documentation

**Full Documentation**: [https://cellular-semantics.github.io/cellsem_llm_client/](https://cellular-semantics.github.io/cellsem_llm_client/)

The documentation is built automatically from the `docs/` folder on each push to main using GitHub Actions.

Quick links:
- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [Development Guidelines](docs/contributing.md)
- [API Reference](docs/api/cellsem_llm_client/index.rst) (auto-generated)
- [Schema Enforcement](docs/schema_enforcement.md)
- [Cost Tracking](docs/cost_tracking.md)

## ✨ Current Features

### LLM client

STATUS - beta

- ✅ **Multi-Provider Support**: Seamless switching between OpenAI, Anthropic, and other LiteLLM-supported providers
- ✅ **Environment-Based Configuration**: Automatic API key loading from environment variables
- ✅ **Type Safety**: Full MyPy type checking with strict configuration
- ✅ **Abstract Base Classes**: Clean architecture with provider-specific implementations
- ✅ **Error Handling**: Robust validation and error management
- ✅ **Configuration Utilities**: Helper functions for quick agent setup

###  Token Tracking & Cost Monitoring

STATUS - beta

- ✅ **Real-time Cost Tracking**: Direct integration with OpenAI and Anthropic usage APIs (aggregate per-key)
- ✅ **Token Usage Metrics**: Detailed tracking of input, output, cached, and thinking tokens
- ✅ **Cost Calculation**: Automated cost computation with fallback rate database (per-request precision); enabled by default when `track_usage=True` (opt-out available)
- ✅ **Usage Analytics**: Comprehensive reporting and cost optimization insights

### JSON Schema Compliance

STATUS - beta

- ✅ **Native Schema Support**: OpenAI structured outputs with `strict=true` enforcement
- ✅ **Tool Use Integration**: Anthropic schema validation via tool use patterns
- ✅ **Pydantic Integration**: Automatic model validation and retry logic
- ✅ **Cross-Provider Compatibility**: Unified schema interface across all providers
- ✅ **JSON-First Inputs**: Prefer plain JSON Schema dicts; Pydantic models are optional helpers

### Tool Calling & MCP Integration

STATUS - alpha

- ✅ **Generic Tool Abstraction**: Uniform `Tool` dataclass for defining tools with JSON-schema parameters and Python handlers
- ✅ **LiteLLM Tool Loop**: `LiteLLMAgent.query_with_tools` and `query_unified` execute tool calls and resume the conversation automatically
- ✅ **MCP Tool Discovery**: `MCPToolSource` discovers tools from any MCP server (stdio transport), bridging the async MCP SDK to sync callers via a background thread
- ✅ **OLS4 Ontology Search**: Built-in `ols4_search` helper targeting the EBI OLS4 MCP (with legacy REST fallback) for ontology lookups
- ✅ **Integration Coverage**: Live test hits OLS4 for "Bergmann glial cell" to verify real responses

### Schema Enforcement (JSON-first)

STATUS - beta

- 🎯 **Use JSON Schema Directly**: Pass a JSON Schema `dict` to enforce structure; optionally derive it from a Pydantic model via `model_json_schema()` or by schema name (resolves `<name>.json` in your schema directory).
- 🔌 **Provider-Aware**: OpenAI uses strict `response_format`; Anthropic converts the schema to a single enforced tool call; other providers get prompt hints plus post-validation.
- 🔁 **Validate & Retry**: Responses are parsed and validated against the derived Pydantic model with lightweight retries; hard failures raise `SchemaValidationException`.
- 🧰 **Runtime Model Generation**: `SchemaManager` loads schemas from dict/file/URL and generates/caches Pydantic models on the fly.
- 🪄 **Example (JSON-first)**:
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

## Planned / Under Development

See [`planning/ROADMAP.md`](planning/ROADMAP.md) for the full roadmap.

### Async Support
- ⏳ **Async Agent Methods**: `aquery()`, `aquery_unified()` wrapping `litellm.acompletion()`
- ⏳ **Simplified MCP Bridge**: Collapse background-thread bridge to direct `await` calls
- ⏳ **Full Backward Compatibility**: Existing sync API unchanged

### File Attachment Support
- ⏳ **Multi-Format Support**: Images (PNG, JPEG, WebP), PDFs, and documents
- ⏳ **Provider Capability Detection**: Automatic validation via `litellm.supports_pdf_input()`
- ⏳ **Unified API**: `query_with_files()` method with base64, URL, and file-path inputs

### AI-Powered Model Recommendations
- ⏳ **Task Complexity Analysis**: AI-powered prompt difficulty assessment
- ⏳ **Model Selection**: Intelligent recommendations based on task requirements
- ⏳ **Cost Optimization**: Balance performance and cost for optimal model choice

## 🏗️ Architecture

```
cellsem_llm_client/
├── agents/          # Core LLM agent implementations
├── tools/           # Generic Tool abstraction and MCP tool discovery
├── utils/           # Configuration and helper utilities
├── tracking/        # Token usage and cost monitoring
├── schema/          # JSON schema validation and compliance
├── files/           # File attachment processing (stub)
└── advisors/        # AI-powered model recommendations (stub)
```

## 📋 Requirements

- **Python**: 3.11+
- **Dependencies**: LiteLLM, Pydantic, python-dotenv
- **API Keys**: OpenAI and/or Anthropic API keys for full functionality

## 🤝 Contributing

1. Follow the TDD workflow defined in `CLAUDE.md`
2. Write tests first, implement features to pass tests
3. Ensure all quality checks pass: `ruff`, `mypy`, `pytest`
4. Maintain >85% test coverage
5. Use conventional commit messages

See [`planning/ROADMAP.md`](planning/ROADMAP.md) for detailed implementation plans for pending features.

### 🧪 Testing Strategy

- **Unit Tests**: Fast, isolated tests (no external dependencies) — run in CI
- **Integration Tests**: Real API validation in local development — require API keys
- **Coverage**: >90% code coverage maintained across all modules

### Development Workflow

```bash
# Run tests
uv run pytest                    # All tests
uv run pytest -m unit           # Unit tests only
uv run pytest -m integration    # Integration tests only

# Code quality checks
uv run ruff check --fix src/ tests/   # Lint and auto-fix
uv run ruff format src/ tests/        # Format code
uv run mypy src/                      # Type checking

# Run with coverage
uv run pytest --cov=cellsem_llm_client --cov-report=html
```

## 📄 License

MIT License - see LICENSE file for details.
