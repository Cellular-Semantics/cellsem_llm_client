# Project Setup Plan: CellSem LLM Client

## Phase 1: Foundation Setup
1. **Create pyproject.toml** - Adapt from lit_agent with:
   - Project metadata for `cellsem_llm_client`
   - Core dependencies: `litellm`, `pydantic`, `python-dotenv`
   - Dev dependencies: `pytest`, `ruff`, `mypy`, `black`, `pre-commit`
   - Python 3.11+ requirement
   - Pytest configuration with unit/integration markers

2. **Set up Python 3.11 environment**
   - Create Python 3.11 virtual environment
   - Install uv in the new environment
   - Run `uv sync --dev` to set up project dependencies

3. **Update GitHub workflow badges** in README.md
   - Fix repository URLs to use correct `Cellular-Semantics/cellsem_llm_client`
   - Update project title
   - Update test matrix to Python 3.11 and 3.12

## Phase 2: Core Architecture (TDD Approach)
4. **Create package structure**
   - Add `__init__.py` files for proper Python packaging
   - Set up basic module structure in `src/cellsem_llm_client/`

5. **Write initial tests** (following TDD mandate)
   - Create `tests/unit/` and `tests/integration/` directories
   - Port and adapt test patterns from lit_agent
   - Write failing tests for core LLM client functionality

6. **Implement core LLM client**
   - Adapt clean architecture from lit_agent
   - Port non-domain-specific functionality from SCELLECTOR
   - Focus on: multi-provider support, token tracking, cost calculation
   - Make tests pass

## Phase 3: Essential Features
7. **Add environment configuration**
   - Implement dotenv-based configuration
   - API key management for multiple providers

8. **Integrate pre-commit hooks**
   - Run `uv run pre-commit install`
   - Ensure code quality pipeline works

This follows the TDD workflow: tests first, minimal implementation, then refactor while keeping tests green.