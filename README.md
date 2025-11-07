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

## ✨ Current Features (Phase 2 - Complete)

- ✅ **Multi-Provider Support**: Seamless switching between OpenAI, Anthropic, and other LiteLLM-supported providers
- ✅ **Environment-Based Configuration**: Automatic API key loading from environment variables
- ✅ **Type Safety**: Full MyPy type checking with strict configuration
- ✅ **Comprehensive Testing**: Dual testing strategy (real APIs for local dev, mocks for CI)
- ✅ **Abstract Base Classes**: Clean architecture with provider-specific implementations
- ✅ **Error Handling**: Robust validation and error management
- ✅ **Configuration Utilities**: Helper functions for quick agent setup
- ✅ **Security**: No API key exposure in logs or test output

## 🔮 Planned Features (Phase 3 - In Development)

### 🔥 Priority 1: Token Tracking & Cost Monitoring
- ⏳ **Real-time Cost Tracking**: Direct integration with OpenAI and Anthropic usage APIs
- ⏳ **Token Usage Metrics**: Detailed tracking of input, output, cached, and thinking tokens
- ⏳ **Cost Calculation**: Automated cost computation with fallback rate database
- ⏳ **Usage Analytics**: Comprehensive reporting and cost optimization insights

### 🔥 Priority 2: JSON Schema Compliance
- ⏳ **Native Schema Support**: OpenAI structured outputs with `strict=true` enforcement
- ⏳ **Tool Use Integration**: Anthropic schema validation via tool use patterns
- ⏳ **Pydantic Integration**: Automatic model validation and retry logic
- ⏳ **Cross-Provider Compatibility**: Unified schema interface across all providers

### 🔥 Priority 3: File Attachment Support
- ⏳ **Multi-Format Support**: Images (PNG, JPEG, WebP), PDFs, and documents
- ⏳ **Provider Abstraction**: Unified file API across different LLM providers
- ⏳ **Capability Detection**: Automatic model file support validation
- ⏳ **Flexible Input**: Base64, URL, and file path support

### 🚀 Priority 4: AI-Powered Model Recommendations
- ⏳ **Task Complexity Analysis**: AI-powered prompt difficulty assessment
- ⏳ **Model Selection**: Intelligent recommendations based on task requirements
- ⏳ **Cost Optimization**: Balance performance and cost for optimal model choice
- ⏳ **Token Estimation**: Predict token usage for better planning

## 🏗️ Architecture

```
cellsem_llm_client/
├── agents/          # Core LLM agent implementations
├── utils/           # Configuration and helper utilities
├── tracking/        # Token usage and cost monitoring (Phase 3)
├── schema/          # JSON schema validation and compliance (Phase 3)
├── files/           # File attachment processing (Phase 3)
└── advisors/        # AI-powered model recommendations (Phase 3)
```

## 🧪 Testing Strategy

- **Unit Tests**: Fast, isolated tests with mocked dependencies
- **Integration Tests**: Real API validation in development, controlled mocks in CI
- **Environment-Based**: `USE_MOCKS=true` for CI, real APIs for local development
- **Coverage**: >90% code coverage maintained across all modules

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

See `ROADMAP.md` for detailed Phase 3 implementation plans.

## 📄 License

MIT License - see LICENSE file for details.
