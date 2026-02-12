"""CellSem LLM Client - A flexible LLM client with multi-provider support."""

__version__ = "0.1.0"

# Core agent classes
from .agents import AgentConnection, AnthropicAgent, LiteLLMAgent, OpenAIAgent

# Custom exceptions
from .exceptions import (
    CellSemLLMException,
    ConfigurationException,
    CostCalculationException,
    ProviderException,
    SchemaValidationException,
)

# Tool helpers
from .tools import MCPToolSource, Tool, build_ols4_search_tool, unpack_tools

# Configuration utilities
from .utils import (
    create_anthropic_agent,
    create_litellm_agent,
    create_openai_agent,
    get_available_providers,
    get_default_models,
    load_environment,
)

__all__ = [
    "__version__",
    "AgentConnection",
    "LiteLLMAgent",
    "OpenAIAgent",
    "AnthropicAgent",
    "build_ols4_search_tool",
    "MCPToolSource",
    "Tool",
    "unpack_tools",
    "load_environment",
    "create_openai_agent",
    "create_anthropic_agent",
    "create_litellm_agent",
    "get_available_providers",
    "get_default_models",
    # Exceptions
    "CellSemLLMException",
    "ConfigurationException",
    "CostCalculationException",
    "ProviderException",
    "SchemaValidationException",
]
