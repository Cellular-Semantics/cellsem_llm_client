"""Unit tests for agent connection classes."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

# Import will fail initially - that's expected for TDD
from cellsem_llm_client.agents.agent_connection import (
    AgentConnection,
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics


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

    def test_agent_connection_requires_query_with_tracking_implementation(self) -> None:
        """Test that subclasses must implement query_with_tracking method."""

        class IncompleteAgent(AgentConnection):
            def query(self, message: str, system_message: str | None = None) -> str:
                return "test"

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

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tracking_openai(
        self, mock_completion: Any
    ) -> None:
        """Test query_with_tracking with OpenAI model returns usage metrics."""
        # Mock LiteLLM response with usage data
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        # Ensure no cached tokens details
        mock_response.usage.prompt_tokens_details = None
        mock_completion.return_value = mock_response

        agent = LiteLLMAgent(model="gpt-4", api_key="test-key")
        response, usage = agent.query_with_tracking("Hello world")

        assert response == "Test response"
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.provider == "openai"
        assert usage.model == "gpt-4"
        assert usage.cost_source == "estimated"
        assert usage.cached_tokens is None
        assert usage.thinking_tokens is None

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tracking_anthropic(
        self, mock_completion: Any
    ) -> None:
        """Test query_with_tracking with Anthropic model handles thinking tokens."""
        # Mock LiteLLM response with Anthropic usage data including thinking tokens
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 75
        mock_response.usage.total_tokens = 325
        # Ensure no cached tokens details (Anthropic doesn't use this)
        mock_response.usage.prompt_tokens_details = None
        mock_completion.return_value = mock_response

        agent = LiteLLMAgent(model="claude-3-sonnet-20240229", api_key="test-key")
        response, usage = agent.query_with_tracking("Hello world")

        assert response == "Test response"
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 75
        assert usage.provider == "anthropic"
        assert usage.model == "claude-3-sonnet-20240229"
        assert usage.cost_source == "estimated"

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tracking_with_cached_tokens(
        self, mock_completion: Any
    ) -> None:
        """Test query_with_tracking handles OpenAI cached tokens."""
        # Mock LiteLLM response with OpenAI cached tokens
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        # OpenAI cache tokens
        mock_response.usage.prompt_tokens_details = Mock()
        mock_response.usage.prompt_tokens_details.cached_tokens = 30
        mock_completion.return_value = mock_response

        agent = LiteLLMAgent(model="gpt-4", api_key="test-key")
        response, usage = agent.query_with_tracking("Hello world")

        assert response == "Test response"
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 30
        assert usage.provider == "openai"
        assert usage.model == "gpt-4"

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tracking_with_cost_calculator(
        self, mock_completion: Any
    ) -> None:
        """Test query_with_tracking with cost calculator integration."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 500
        mock_response.usage.total_tokens = 1500
        # Ensure no cached tokens details
        mock_response.usage.prompt_tokens_details = None
        mock_completion.return_value = mock_response

        # Mock cost calculator
        mock_calculator = Mock()
        mock_calculator.calculate_cost.return_value = 0.06

        agent = LiteLLMAgent(model="gpt-4", api_key="test-key")
        response, usage = agent.query_with_tracking(
            "Hello world", cost_calculator=mock_calculator
        )

        assert response == "Test response"
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.estimated_cost_usd == 0.06
        assert usage.cost_source == "estimated"

        # Verify cost calculator was called
        mock_calculator.calculate_cost.assert_called_once()
        call_args = mock_calculator.calculate_cost.call_args[0][0]
        assert isinstance(call_args, UsageMetrics)
        assert call_args.input_tokens == 1000
        assert call_args.output_tokens == 500

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tracking_system_message(
        self, mock_completion: Any
    ) -> None:
        """Test query_with_tracking with system message."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 120
        mock_response.usage.completion_tokens = 60
        mock_response.usage.total_tokens = 180
        # Ensure no cached tokens details
        mock_response.usage.prompt_tokens_details = None
        mock_completion.return_value = mock_response

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        response, usage = agent.query_with_tracking(
            "Hello", system_message="You are a helpful assistant"
        )

        assert response == "Test response"
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens == 120
        assert usage.output_tokens == 60

        # Verify correct messages were sent
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
    def test_litellm_agent_query_with_tracking_error_handling(
        self, mock_completion: Any
    ) -> None:
        """Test query_with_tracking error handling."""
        mock_completion.side_effect = Exception("API Error")

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        with pytest.raises(Exception, match="API Error"):
            agent.query_with_tracking("Hello world")

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tools_executes_tool_call(
        self, mock_completion: Any
    ) -> None:
        """Test tool-call loop executes handlers and returns final content."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the time for a timezone",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                },
            }
        ]

        tool_call = Mock()
        tool_call.id = "call_1"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "get_time"
        tool_call.function.arguments = '{"timezone": "UTC"}'

        first_response = Mock()
        first_response.choices = [Mock()]
        first_response.choices[0].message.content = None
        first_response.choices[0].message.tool_calls = [tool_call]

        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message.content = "It's noon somewhere."
        final_response.choices[0].message.tool_calls = []

        mock_completion.side_effect = [first_response, final_response]

        executed_args: dict[str, Any] = {}

        def get_time(args: dict[str, Any]) -> str:
            executed_args.update(args)
            return "12:00 UTC"

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")
        result = agent.query_with_tools(
            message="What time is it?",
            tools=tools,
            tool_handlers={"get_time": get_time},
            system_message="You are a timekeeper.",
        )

        assert result == "It's noon somewhere."
        assert executed_args == {"timezone": "UTC"}

        first_call_messages = mock_completion.call_args_list[0].kwargs["messages"]
        assert first_call_messages == [
            {"role": "system", "content": "You are a timekeeper."},
            {"role": "user", "content": "What time is it?"},
        ]
        assert mock_completion.call_args_list[0].kwargs["tools"] == tools

        second_call_messages = mock_completion.call_args_list[1].kwargs["messages"]
        assert second_call_messages[-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "12:00 UTC",
        }
        assert (
            second_call_messages[-2]["tool_calls"][0]["function"]["name"] == "get_time"
        )

    @pytest.mark.unit
    @patch("cellsem_llm_client.agents.agent_connection.completion")
    def test_litellm_agent_query_with_tools_missing_handler_raises(
        self, mock_completion: Any
    ) -> None:
        """Test missing tool handler raises an explicit error."""
        tool_call = Mock()
        tool_call.id = "call_1"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "missing_tool"
        tool_call.function.arguments = "{}"

        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = None
        response.choices[0].message.tool_calls = [tool_call]
        mock_completion.return_value = response

        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")

        with pytest.raises(ValueError, match="missing_tool"):
            agent.query_with_tools(
                message="Trigger a tool call",
                tools=[],
                tool_handlers={},
            )


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
