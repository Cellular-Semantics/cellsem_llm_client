"""Unit tests for agent connection classes."""

from typing import Any
from unittest.mock import patch

import pytest

# Import will fail initially - that's expected for TDD
from cellsem_llm_client.agents.agent_connection import (
    AgentConnection,
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)


class TestAgentConnection:
    """Test the abstract base AgentConnection class."""

    def test_agent_connection_is_abstract(self) -> None:
        """Test that AgentConnection cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentConnection()  # type: ignore[abstract]

    def test_agent_connection_requires_query_implementation(self) -> None:
        """Test that subclasses must implement query method."""

        class IncompleteAgent(AgentConnection):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent()  # type: ignore[abstract]


class TestLiteLLMAgent:
    """Test the LiteLLMAgent implementation."""

    @pytest.mark.unit
    def test_litellm_agent_initialization(self) -> None:
        """Test basic LiteLLM agent initialization."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        assert agent.model == "gpt-3.5-turbo"
        assert agent.api_key == "test-key"

    @pytest.mark.unit
    def test_litellm_agent_requires_model(self) -> None:
        """Test that model is required during initialization."""
        with pytest.raises(ValueError, match="Model is required"):
            LiteLLMAgent(model=None)  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_litellm_agent_requires_api_key(self) -> None:
        """Test that API key is required during initialization."""
        with pytest.raises(ValueError, match="API key is required"):
            LiteLLMAgent(model="gpt-3.5-turbo", api_key=None)

    @pytest.mark.unit
    def test_litellm_agent_default_max_tokens(self) -> None:
        """Test default max_tokens setting."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        assert agent.max_tokens == 1000

    @pytest.mark.unit
    def test_litellm_agent_custom_max_tokens(self) -> None:
        """Test custom max_tokens setting."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key", max_tokens=2000)
        assert agent.max_tokens == 2000

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_success(
        self, mock_completion: Any, mock_api_response: Any
    ) -> None:
        """Test successful query execution."""
        mock_completion.return_value = mock_api_response

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        response = agent.query("Hello world")

        assert response == "Test response"
        mock_completion.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            max_tokens=1000,
        )

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_system_message(
        self, mock_completion: Any, mock_api_response: Any
    ) -> None:
        """Test query with system message."""
        mock_completion.return_value = mock_api_response

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        response = agent.query("Hello", system_message="You are a helpful assistant")

        assert response == "Test response"
        mock_completion.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=1000,
        )

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_error_handling(self, mock_completion: Any) -> None:
        """Test error handling during query."""
        mock_completion.side_effect = Exception("API Error")

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        with pytest.raises(Exception, match="API Error"):
            agent.query("Hello world")


class TestOpenAIAgent:
    """Test the OpenAI-specific agent implementation."""

    @pytest.mark.unit
    def test_openai_agent_default_model(self) -> None:
        """Test OpenAI agent uses default model."""
        agent = OpenAIAgent(api_key="test-key")
        assert agent.model == "gpt-3.5-turbo"

    @pytest.mark.unit
    def test_openai_agent_custom_model(self) -> None:
        """Test OpenAI agent with custom model."""
        agent = OpenAIAgent(model="gpt-4", api_key="test-key")
        assert agent.model == "gpt-4"

    @pytest.mark.unit
    def test_openai_agent_inherits_from_litellm(self) -> None:
        """Test that OpenAIAgent is a subclass of LiteLLMAgent."""
        agent = OpenAIAgent(api_key="test-key")
        assert isinstance(agent, LiteLLMAgent)


class TestAnthropicAgent:
    """Test the Anthropic-specific agent implementation."""

    @pytest.mark.unit
    def test_anthropic_agent_default_model(self) -> None:
        """Test Anthropic agent uses default model."""
        agent = AnthropicAgent(api_key="test-key")
        assert agent.model == "claude-3-haiku-20240307"

    @pytest.mark.unit
    def test_anthropic_agent_custom_model(self) -> None:
        """Test Anthropic agent with custom model."""
        agent = AnthropicAgent(model="claude-3-sonnet-20240229", api_key="test-key")
        assert agent.model == "claude-3-sonnet-20240229"

    @pytest.mark.unit
    def test_anthropic_agent_inherits_from_litellm(self) -> None:
        """Test that AnthropicAgent is a subclass of LiteLLMAgent."""
        agent = AnthropicAgent(api_key="test-key")
        assert isinstance(agent, LiteLLMAgent)
