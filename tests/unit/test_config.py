"""Unit tests for configuration utilities."""

from typing import Any
from unittest.mock import patch

import pytest

from cellsem_llm_client.agents.agent_connection import (
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)
from cellsem_llm_client.utils.config import (
    create_anthropic_agent,
    create_litellm_agent,
    create_openai_agent,
    get_available_providers,
    get_default_models,
    load_environment,
)


class TestConfigurationUtils:
    """Test configuration utility functions."""

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_load_environment(self, mock_load_dotenv: Any) -> None:
        """Test loading environment variables."""
        load_environment()
        mock_load_dotenv.assert_called_once()

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_openai_agent_with_env_key(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test creating OpenAI agent with environment API key."""
        mock_getenv.return_value = "test-openai-key"

        agent = create_openai_agent()

        assert isinstance(agent, OpenAIAgent)
        assert agent.model == "gpt-3.5-turbo"
        assert agent.api_key == "test-openai-key"
        assert agent.max_tokens == 1000
        mock_getenv.assert_called_with("OPENAI_API_KEY")

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_openai_agent_with_explicit_key(self, mock_load_dotenv: Any) -> None:
        """Test creating OpenAI agent with explicit API key."""
        agent = create_openai_agent(
            model="gpt-4", api_key="explicit-key", max_tokens=2000
        )

        assert isinstance(agent, OpenAIAgent)
        assert agent.model == "gpt-4"
        assert agent.api_key == "explicit-key"
        assert agent.max_tokens == 2000

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_openai_agent_no_key_raises_error(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test that missing OpenAI API key raises ValueError."""
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            create_openai_agent()

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_anthropic_agent_with_env_key(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test creating Anthropic agent with environment API key."""
        mock_getenv.return_value = "test-anthropic-key"

        agent = create_anthropic_agent()

        assert isinstance(agent, AnthropicAgent)
        assert agent.model == "claude-3-haiku-20240307"
        assert agent.api_key == "test-anthropic-key"
        mock_getenv.assert_called_with("ANTHROPIC_API_KEY")

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_anthropic_agent_with_explicit_key(
        self, mock_load_dotenv: Any
    ) -> None:
        """Test creating Anthropic agent with explicit API key."""
        agent = create_anthropic_agent(
            model="claude-3-sonnet-20240229", api_key="explicit-key"
        )

        assert isinstance(agent, AnthropicAgent)
        assert agent.model == "claude-3-sonnet-20240229"
        assert agent.api_key == "explicit-key"

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_anthropic_agent_no_key_raises_error(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test that missing Anthropic API key raises ValueError."""
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="Anthropic API key not found"):
            create_anthropic_agent()

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_litellm_agent_openai_model(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test creating LiteLLM agent with OpenAI model inference."""
        mock_getenv.side_effect = (
            lambda key: "openai-key" if key == "OPENAI_API_KEY" else None
        )

        agent = create_litellm_agent(model="gpt-3.5-turbo")

        assert isinstance(agent, LiteLLMAgent)
        assert agent.model == "gpt-3.5-turbo"
        assert agent.api_key == "openai-key"

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_litellm_agent_claude_model(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test creating LiteLLM agent with Claude model inference."""
        mock_getenv.side_effect = (
            lambda key: "anthropic-key" if key == "ANTHROPIC_API_KEY" else None
        )

        agent = create_litellm_agent(model="claude-3-haiku-20240307")

        assert isinstance(agent, LiteLLMAgent)
        assert agent.model == "claude-3-haiku-20240307"
        assert agent.api_key == "anthropic-key"

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_litellm_agent_explicit_key(self, mock_load_dotenv: Any) -> None:
        """Test creating LiteLLM agent with explicit API key."""
        agent = create_litellm_agent(model="gpt-4", api_key="explicit-key")

        assert isinstance(agent, LiteLLMAgent)
        assert agent.model == "gpt-4"
        assert agent.api_key == "explicit-key"

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_create_litellm_agent_no_key_raises_error(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test that missing API key for LiteLLM agent raises ValueError."""
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="API key not found for model"):
            create_litellm_agent(model="gpt-3.5-turbo")

    @pytest.mark.unit
    @patch("cellsem_llm_client.utils.config.os.getenv")
    @patch("cellsem_llm_client.utils.config.load_dotenv")
    def test_get_available_providers(
        self, mock_load_dotenv: Any, mock_getenv: Any
    ) -> None:
        """Test checking available providers."""
        mock_getenv.side_effect = lambda key: {
            "OPENAI_API_KEY": "openai-key",
            "ANTHROPIC_API_KEY": None,
        }.get(key)

        providers = get_available_providers()

        assert providers == {
            "openai": True,
            "anthropic": False,
        }

    @pytest.mark.unit
    def test_get_default_models(self) -> None:
        """Test getting default models."""
        models = get_default_models()

        assert models == {
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-haiku-20240307",
        }
