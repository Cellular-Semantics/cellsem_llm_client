# Contributing

Thank you for your interest in contributing to CellSem LLM Client! This guide will help you get started with our development workflow.

## Development Philosophy

This project follows **strict Test-Driven Development (TDD)** as outlined in our [CLAUDE.md](https://github.com/Cellular-Semantics/cellsem_llm_client/blob/main/CLAUDE.md) development guidelines.

::::{grid} 2

:::{grid-item-card} TDD Workflow
:class-header: bg-light

1. Write **tests first** (they must fail initially)
2. Write **minimal code** to pass tests
3. **Refactor** while keeping tests green
4. **Commit** at each stage
:::

:::{grid-item-card} Quality Standards
:class-header: bg-light

- **>85% test coverage** required
- **Type safety** with MyPy
- **Code formatting** with Ruff
- **Real API integration** for local testing
:::

::::

## Getting Started

### 1. Fork and Clone

```bash
git fork https://github.com/Cellular-Semantics/cellsem_llm_client.git
git clone https://github.com/YOUR-USERNAME/cellsem_llm_client.git
cd cellsem_llm_client
```

### 2. Set Up Development Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install all dependencies
uv sync --dev --extra docs

# Set up pre-commit hooks
uv run pre-commit install
```

### 3. Configure Environment

Create a `.env` file with your API keys for testing:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Development Workflow

### Branch Strategy

:::{important}
**All development must be done in feature branches.** Never commit directly to `main`.
:::

```bash
# Create a new feature branch
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Work on your feature...

# Push and create PR
git push origin feature/your-feature-name
```

### TDD Cycle

```bash
# 1. Write failing tests first
uv run pytest tests/unit/test_your_feature.py -v
# Tests should FAIL initially

# 2. Write minimal code to pass tests
# Edit src/cellsem_llm_client/...

# 3. Verify tests pass
uv run pytest tests/unit/test_your_feature.py -v

# 4. Run full test suite
uv run pytest

# 5. Code quality checks
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

### Testing Strategy

We use a **dual testing strategy**:

::::{tab-set}

:::{tab-item} Unit Tests
```bash
# Fast, isolated tests with mocks
uv run pytest -m unit
```

- No external API calls
- Mock all dependencies
- Test individual components
- Should run in <10 seconds
:::

:::{tab-item} Integration Tests
```bash
# Real API tests (local development)
uv run pytest -m integration

# Mocked tests (CI environment)
USE_MOCKS=true uv run pytest -m integration
```

- Real API calls in development
- Mocks in CI for reliability
- Test actual provider integration
- Require valid API keys locally
:::

::::

### Code Quality Standards

All code must pass these checks:

```bash
# Linting and formatting
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Test coverage
uv run pytest --cov=cellsem_llm_client --cov-report=html
# Must achieve >85% coverage
```

## Contributing Guidelines

### 1. Documentation

- **Google-style docstrings** for all public functions/classes
- **Type hints** for all parameters and return values
- **Usage examples** in docstrings when helpful
- **Update documentation** if changing public APIs

```python
def example_function(param: str, optional: int = 100) -> str:
    """Example function with proper documentation.

    Args:
        param: Description of the parameter
        optional: Optional parameter with default value

    Returns:
        Description of what the function returns

    Raises:
        ValueError: When param is invalid

    Example:
        ```python
        result = example_function("hello", 200)
        print(result)
        ```
    """
    pass
```

### 2. Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add token tracking for OpenAI API calls
fix: resolve API key validation error
docs: update installation instructions
test: add integration tests for Anthropic agent
```

### 3. Pull Request Process

1. **Create feature branch** from `main`
2. **Implement using TDD** (tests first!)
3. **Ensure all quality checks pass**
4. **Update documentation** if needed
5. **Submit PR** with clear description
6. **Address review feedback**

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Changes
- [ ] Feature implementation
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints added

## Testing
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage >85%
- [ ] Manual testing completed

## Checklist
- [ ] Follows TDD workflow
- [ ] Code quality checks pass
- [ ] Documentation updated
- [ ] Breaking changes documented
```

## Development Commands

### Core Commands

```bash
# Install dependencies
uv sync --dev --extra docs

# Run specific test types
uv run pytest -m unit           # Unit tests only
uv run pytest -m integration    # Integration tests only
uv run pytest --cov             # With coverage

# Code quality
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run mypy src/

# Documentation
cd docs && uv run sphinx-build . _build/html
```

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Pytest**: Quick smoke tests

```bash
# Manual pre-commit run
uv run pre-commit run --all-files

# Skip hooks (only if necessary)
git commit --no-verify -m "message"
```

## Architecture Guidelines

### Adding New Features

When adding new features (especially Phase 3 features):

1. **Follow the roadmap** in [planning/ROADMAP.md](https://github.com/Cellular-Semantics/cellsem_llm_client/blob/main/planning/ROADMAP.md)
2. **Create feature branch** for each major component
3. **Write tests first** that demonstrate the feature
4. **Implement minimally** to pass tests
5. **Refactor** for clean, maintainable code

### Code Organization

```
src/cellsem_llm_client/
├── agents/          # Core agent implementations
├── utils/           # Helper functions and configuration
├── tracking/        # Phase 3: Token/cost tracking
├── schema/          # Phase 3: JSON schema validation
├── files/           # Phase 3: File attachment support
└── advisors/        # Phase 3: Model recommendations
```

### Design Principles

- **Abstract interfaces** for extensibility
- **Dependency injection** for testability
- **Environment-based configuration**
- **Provider-agnostic design** where possible
- **Graceful error handling**

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or propose ideas
- **Code Review**: Learn from PR feedback
- **Documentation**: Refer to API docs and examples

## Recognition

Contributors will be recognized in:

- **Commit co-authorship** for significant contributions
- **CONTRIBUTORS.md** file (coming soon)
- **Release notes** for feature contributions

Thank you for contributing to CellSem LLM Client! 🚀