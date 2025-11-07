# Installation

## Requirements

- **Python**: 3.11+
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Quick Installation

### Using uv (Recommended)

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

### Using pip

```bash
# Clone the repository
git clone https://github.com/Cellular-Semantics/cellsem_llm_client.git
cd cellsem_llm_client

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

:::{note}
You need at least one API key to use the client. Both providers are supported, but you only need the keys for the providers you plan to use.
:::

## Verify Installation

Test your installation by running the test suite:

```bash
# Run all tests
uv run pytest

# Run only unit tests (no API calls)
uv run pytest -m unit

# Run with API keys (integration tests)
uv run pytest -m integration
```

:::{warning}
Integration tests require real API keys and will make actual API calls, which may incur costs. Unit tests are free and use mocks.
:::

## Development Installation

For contributing to the project:

```bash
# Install with all development dependencies
uv sync --dev --extra docs

# Install pre-commit hooks
uv run pre-commit install

# Verify code quality tools
uv run ruff check src/ tests/
uv run mypy src/
```

## Docker Installation (Optional)

A Dockerfile is planned for Phase 3. For now, use the Python installation methods above.

## Troubleshooting

### Common Issues

**Import Error**: If you get import errors, ensure you've activated the virtual environment and installed the package in editable mode.

**API Key Issues**: Verify your `.env` file is in the project root and contains valid API keys.

**Permission Errors**: On some systems, you may need to use `python3` and `pip3` instead of `python` and `pip`.

### Getting Help

- Check the [GitHub Issues](https://github.com/Cellular-Semantics/cellsem_llm_client/issues)
- Read the [Contributing Guide](contributing.md)
- Review the [API Documentation](api/index.md)