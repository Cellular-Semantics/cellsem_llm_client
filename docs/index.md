# CellSem LLM Client Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
schema_enforcement
cost_tracking
contributing
```

## Welcome

CellSem LLM Client is a flexible LLM client with multi-provider support, built using LiteLLM + Pydantic for seamless integration across OpenAI, Anthropic, and other providers.

## Key Features

::::{grid} 2

:::{grid-item-card} Multi-Provider Support
:class-header: bg-light

Seamless switching between OpenAI, Anthropic, and other LiteLLM-supported providers
:::

:::{grid-item-card} Type Safety
:class-header: bg-light

Full MyPy type checking with strict configuration for reliable code
:::

:::{grid-item-card} Comprehensive Testing
:class-header: bg-light

Dual testing strategy: real APIs for local development, mocks for CI
:::

:::{grid-item-card} Environment Configuration
:class-header: bg-light

Automatic API key loading and configuration from environment variables
:::

::::

## Quick Links

- {doc}`installation` - Get started with installation and setup
- {doc}`quickstart` - Basic usage examples and workflows
- {doc}`api/cellsem_llm_client/index` - Complete API reference documentation
- {doc}`contributing` - Development guidelines and TDD workflow

## Current Status

**Phase 2 - Complete**: Core multi-provider functionality with comprehensive testing

**Phase 3 - In Development**: Advanced features including:
- Token tracking & cost monitoring
- JSON schema compliance
- File attachment support
- AI-powered model recommendations

See our [GitHub repository](https://github.com/Cellular-Semantics/cellsem_llm_client) for the latest updates and [ROADMAP.md](https://github.com/Cellular-Semantics/cellsem_llm_client/blob/main/planning/ROADMAP.md) for detailed implementation plans.
