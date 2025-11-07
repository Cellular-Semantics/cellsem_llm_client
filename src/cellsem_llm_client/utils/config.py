"""Configuration utilities for environment-based setup."""

import os

from dotenv import load_dotenv

from cellsem_llm_client.agents.agent_connection import (
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)


def load_environment() -> None:
    """Load environment variables from .env file if it exists."""
    load_dotenv()


def create_openai_agent(
    model: str = "gpt-3.5-turbo",
    api_key: str | None = None,
    max_tokens: int = 1000,
) -> OpenAIAgent:
    """Create an OpenAI agent with environment-based configuration.

    Args:
        model: OpenAI model name (default: 'gpt-3.5-turbo')
        api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
        max_tokens: Maximum tokens for response

    Returns:
        Configured OpenAIAgent

    Raises:
        ValueError: If no API key is found in parameter or environment
    """
    load_environment()

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return OpenAIAgent(model=model, api_key=api_key, max_tokens=max_tokens)


def create_anthropic_agent(
    model: str = "claude-3-haiku-20240307",
    api_key: str | None = None,
    max_tokens: int = 1000,
) -> AnthropicAgent:
    """Create an Anthropic agent with environment-based configuration.

    Args:
        model: Anthropic model name (default: 'claude-3-haiku-20240307')
        api_key: Anthropic API key (if None, loads from ANTHROPIC_API_KEY env var)
        max_tokens: Maximum tokens for response

    Returns:
        Configured AnthropicAgent

    Raises:
        ValueError: If no API key is found in parameter or environment
    """
    load_environment()

    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key is None:
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return AnthropicAgent(model=model, api_key=api_key, max_tokens=max_tokens)


def create_litellm_agent(
    model: str,
    api_key: str | None = None,
    max_tokens: int = 1000,
) -> LiteLLMAgent:
    """Create a LiteLLM agent with environment-based configuration.

    Args:
        model: Model name (e.g., 'gpt-3.5-turbo', 'claude-3-haiku-20240307')
        api_key: API key (if None, tries to infer from model and environment)
        max_tokens: Maximum tokens for response

    Returns:
        Configured LiteLLMAgent

    Raises:
        ValueError: If no API key is found and cannot be inferred
    """
    load_environment()

    if api_key is None:
        # Try to infer API key based on model name
        if model.startswith("gpt"):
            api_key = os.getenv("OPENAI_API_KEY")
        elif model.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key is None:
        raise ValueError(
            f"API key not found for model '{model}'. Set appropriate environment "
            "variable or pass api_key parameter."
        )

    return LiteLLMAgent(model=model, api_key=api_key, max_tokens=max_tokens)


def get_available_providers() -> dict[str, bool]:
    """Check which providers have API keys available.

    Returns:
        Dictionary mapping provider names to availability status
    """
    load_environment()

    return {
        "openai": os.getenv("OPENAI_API_KEY") is not None,
        "anthropic": os.getenv("ANTHROPIC_API_KEY") is not None,
    }


def get_default_models() -> dict[str, str]:
    """Get default models for each provider.

    Returns:
        Dictionary mapping provider names to default model names
    """
    return {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
    }
