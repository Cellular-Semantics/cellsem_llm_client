"""Unit tests for schema integration with agent connections."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from cellsem_llm_client.agents.agent_connection import LiteLLMAgent


class TaskResponse(BaseModel):
    """Sample model for testing schema integration."""

    task_result: str
    confidence: float
    metadata: dict[str, Any] | None = None


@pytest.mark.unit
class TestSchemaIntegration:
    """Test cases for schema integration with agent connections."""

    def test_query_with_schema_method_exists(self) -> None:
        """Test that query_with_schema method exists on agent."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")

        assert hasattr(agent, "query_with_schema")

    @patch("litellm.completion")
    def test_query_with_schema_openai_native(self, mock_completion: Mock) -> None:
        """Test schema-enabled query with OpenAI native support."""
        agent = LiteLLMAgent(model="gpt-4", api_key="test-key")

        # Mock OpenAI response with structured output
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"task_result": "completed", "confidence": 0.95}'
                    )
                )
            ],
            usage=Mock(
                prompt_tokens=10, completion_tokens=20, prompt_tokens_details=None
            ),
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["task_result", "confidence"],
        }

        result = agent.query_with_schema(
            message="Complete this task", schema=schema_dict
        )

        # Should return validated Pydantic model instance
        assert hasattr(result, "task_result")
        assert hasattr(result, "confidence")
        assert result.task_result == "completed"
        assert result.confidence == 0.95

        # Verify OpenAI's response_format was used
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert "response_format" in call_args[1]
        assert call_args[1]["response_format"]["type"] == "json_schema"

    @patch("litellm.completion")
    def test_query_with_schema_anthropic_tools(self, mock_completion: Mock) -> None:
        """Test schema-enabled query with Anthropic tool choice."""
        agent = LiteLLMAgent(model="claude-3-sonnet", api_key="test-key")

        # Mock Anthropic response with tool call
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        tool_calls=[
                            Mock(
                                function=Mock(
                                    arguments='{"task_result": "completed", "confidence": 0.90}'
                                )
                            )
                        ]
                    )
                )
            ],
            usage=Mock(
                prompt_tokens=15, completion_tokens=25, prompt_tokens_details=None
            ),
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["task_result"],
        }

        result = agent.query_with_schema(
            message="Complete this task",
            schema=schema_dict,
            max_retries=0,  # Disable retries to test the mock response directly
        )

        # Should return validated Pydantic model instance
        assert hasattr(result, "task_result")
        assert hasattr(result, "confidence")
        assert result.task_result == "completed"
        assert result.confidence == 0.90

        # Verify Anthropic's tool choice was used
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert "tools" in call_args[1]
        assert "tool_choice" in call_args[1]

    @patch("litellm.completion")
    def test_query_with_schema_fallback_provider(self, mock_completion: Mock) -> None:
        """Test schema-enabled query with fallback provider."""
        agent = LiteLLMAgent(model="unknown-model", api_key="test-key")

        # Mock response from unknown provider
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"task_result": "completed", "confidence": 0.85}'
                    )
                )
            ],
            usage=Mock(
                prompt_tokens=12, completion_tokens=18, prompt_tokens_details=None
            ),
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

        result = agent.query_with_schema(
            message="Complete this task", schema=schema_dict
        )

        # Should return validated Pydantic model instance
        assert hasattr(result, "task_result")
        assert hasattr(result, "confidence")
        assert result.task_result == "completed"

        # Verify fallback adapter enhanced messages with schema hints
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        messages = call_args[1]["messages"]
        assert len(messages) > 1  # Should have system message with schema hint

    def test_query_with_schema_with_pydantic_model(self) -> None:
        """Test using Pydantic model directly as schema."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"task_result": "success", "confidence": 0.88}'
                        )
                    )
                ],
                usage=Mock(
                    prompt_tokens=10, completion_tokens=15, prompt_tokens_details=None
                ),
            )

            result = agent.query_with_schema(
                message="Complete task", schema=TaskResponse
            )

            assert isinstance(result, TaskResponse)
            assert result.task_result == "success"
            assert result.confidence == 0.88

    @patch("litellm.completion")
    def test_query_with_schema_validation_retry(self, mock_completion: Mock) -> None:
        """Test schema validation with automatic retry logic."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")

        # Mock a valid response directly for this test
        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"task_result": "completed", "confidence": 0.92}'
                    )
                )
            ],
            usage=Mock(
                prompt_tokens=12, completion_tokens=18, prompt_tokens_details=None
            ),
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["task_result", "confidence"],
        }

        result = agent.query_with_schema(
            message="Complete this task", schema=schema_dict, max_retries=1
        )

        assert hasattr(result, "task_result")
        assert hasattr(result, "confidence")
        assert result.task_result == "completed"
        assert result.confidence == 0.92

        # Should have made 1 call since response was valid
        assert mock_completion.call_count == 1

    def test_query_with_schema_with_tracking(self) -> None:
        """Test schema query integration with usage tracking."""
        agent = LiteLLMAgent(model="gpt-4", api_key="test-key")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"task_result": "done", "confidence": 0.95}'
                        )
                    )
                ],
                usage=Mock(
                    prompt_tokens=20, completion_tokens=25, prompt_tokens_details=None
                ),
            )

            result, usage_metrics = agent.query_with_schema_and_tracking(
                message="Complete task",
                schema={
                    "type": "object",
                    "properties": {
                        "task_result": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            )

            assert hasattr(result, "task_result")
            assert usage_metrics.input_tokens == 20
            assert usage_metrics.output_tokens == 25
            assert usage_metrics.provider == "openai"

    def test_query_with_schema_error_handling(self) -> None:
        """Test error handling in schema-enabled queries."""
        agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key="test-key")

        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API Error")

            with pytest.raises((ValueError, RuntimeError)):
                agent.query_with_schema(
                    message="Test message",
                    schema={
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    },
                )

    def test_query_with_schema_different_schema_sources(self) -> None:
        """Test schema query with different schema input types."""
        agent = LiteLLMAgent(model="gpt-4", api_key="test-key")

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content='{"task_result": "success", "confidence": 0.88}'
                        )
                    )
                ],
                usage=Mock(
                    prompt_tokens=10, completion_tokens=15, prompt_tokens_details=None
                ),
            )

            # Test with schema dict
            result1 = agent.query_with_schema(
                message="Test",
                schema={
                    "type": "object",
                    "properties": {
                        "task_result": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            )
            assert hasattr(result1, "task_result")

            # Test with Pydantic model class
            result2 = agent.query_with_schema(message="Test", schema=TaskResponse)
            assert isinstance(result2, TaskResponse)
