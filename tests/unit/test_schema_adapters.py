"""Unit tests for provider-specific schema adapters."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from cellsem_llm_client.schema.adapters import (
    AdapterCapability,
    AnthropicSchemaAdapter,
    FallbackSchemaAdapter,
    OpenAISchemaAdapter,
    SchemaAdapterFactory,
)


class SampleResponseModel(BaseModel):
    """Sample model for adapter tests."""

    task_result: str
    confidence: float
    metadata: dict[str, Any] | None = None


@pytest.mark.unit
class TestSchemaAdapterFactory:
    """Test cases for the SchemaAdapterFactory."""

    def test_factory_creation(self) -> None:
        """Test creating SchemaAdapterFactory."""
        factory = SchemaAdapterFactory()

        assert factory is not None
        assert hasattr(factory, "get_adapter")
        assert hasattr(factory, "get_capabilities")

    def test_openai_adapter_selection(self) -> None:
        """Test factory selects OpenAI adapter for OpenAI models."""
        factory = SchemaAdapterFactory()

        adapter = factory.get_adapter("openai", "gpt-4")

        assert isinstance(adapter, OpenAISchemaAdapter)

    def test_anthropic_adapter_selection(self) -> None:
        """Test factory selects Anthropic adapter for Anthropic models."""
        factory = SchemaAdapterFactory()

        adapter = factory.get_adapter("anthropic", "claude-3-sonnet-20240229")

        assert isinstance(adapter, AnthropicSchemaAdapter)

    def test_fallback_adapter_selection(self) -> None:
        """Test factory selects fallback adapter for unknown providers."""
        factory = SchemaAdapterFactory()

        adapter = factory.get_adapter("unknown-provider", "some-model")

        assert isinstance(adapter, FallbackSchemaAdapter)

    def test_adapter_capabilities(self) -> None:
        """Test getting adapter capabilities for different providers."""
        factory = SchemaAdapterFactory()

        openai_caps = factory.get_capabilities("openai", "gpt-4")
        anthropic_caps = factory.get_capabilities("anthropic", "claude-3-sonnet")
        fallback_caps = factory.get_capabilities("unknown", "model")

        assert AdapterCapability.NATIVE_SCHEMA in openai_caps
        assert AdapterCapability.NATIVE_SCHEMA in anthropic_caps
        assert AdapterCapability.NATIVE_SCHEMA not in fallback_caps
        assert AdapterCapability.VALIDATION_RETRY in fallback_caps


@pytest.mark.unit
class TestOpenAISchemaAdapter:
    """Test cases for the OpenAI schema adapter."""

    def test_adapter_creation(self) -> None:
        """Test creating OpenAI schema adapter."""
        adapter = OpenAISchemaAdapter()

        assert adapter is not None
        assert hasattr(adapter, "apply_schema")
        assert hasattr(adapter, "supports_native_schema")

    def test_supports_native_schema(self) -> None:
        """Test that OpenAI adapter supports native schema."""
        adapter = OpenAISchemaAdapter()

        assert adapter.supports_native_schema() is True

    @patch("litellm.completion")
    def test_apply_schema_with_structured_output(self, mock_completion: Mock) -> None:
        """Test applying schema using OpenAI's structured output format."""
        adapter = OpenAISchemaAdapter()

        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"task_result": "completed", "confidence": 0.95}'
                    )
                )
            ]
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["task_result", "confidence"],
        }

        messages = [{"role": "user", "content": "Complete this task"}]

        adapter.apply_schema(messages, schema_dict, model="gpt-4")

        # Verify the completion was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args

        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["messages"] == messages

        # Verify response_format is set correctly
        response_format = call_args[1]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == schema_dict

    def test_schema_name_generation(self) -> None:
        """Test that adapter generates appropriate schema names."""
        adapter = OpenAISchemaAdapter()

        # Test with simple schema
        simple_schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        name1 = adapter._generate_schema_name(simple_schema)

        assert isinstance(name1, str)
        assert len(name1) > 0

        # Test with complex schema
        complex_schema = {
            "type": "object",
            "title": "ComplexResponse",
            "properties": {"data": {"type": "array"}},
        }
        name2 = adapter._generate_schema_name(complex_schema)

        # Should use title if available
        assert name2 == "ComplexResponse"

    def test_error_handling_in_apply_schema(self) -> None:
        """Test error handling when schema application fails."""
        adapter = OpenAISchemaAdapter()

        with patch("litellm.completion", side_effect=Exception("API Error")):
            schema_dict = {"type": "object"}
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises((ValueError, RuntimeError)):
                adapter.apply_schema(messages, schema_dict, model="gpt-4")


@pytest.mark.unit
class TestAnthropicSchemaAdapter:
    """Test cases for the Anthropic schema adapter."""

    def test_adapter_creation(self) -> None:
        """Test creating Anthropic schema adapter."""
        adapter = AnthropicSchemaAdapter()

        assert adapter is not None
        assert hasattr(adapter, "apply_schema")
        assert hasattr(adapter, "supports_native_schema")

    def test_supports_native_schema(self) -> None:
        """Test that Anthropic adapter supports native schema via tools."""
        adapter = AnthropicSchemaAdapter()

        assert adapter.supports_native_schema() is True

    @patch("litellm.completion")
    def test_apply_schema_with_tool_choice(self, mock_completion: Mock) -> None:
        """Test applying schema using Anthropic's tool choice pattern."""
        adapter = AnthropicSchemaAdapter()

        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        tool_calls=[
                            Mock(
                                function=Mock(
                                    arguments='{"task_result": "completed", "confidence": 0.95}'
                                )
                            )
                        ]
                    )
                )
            ]
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["task_result", "confidence"],
        }

        messages = [{"role": "user", "content": "Complete this task"}]

        adapter.apply_schema(messages, schema_dict, model="claude-3-sonnet")

        # Verify the completion was called with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args

        assert call_args[1]["model"] == "claude-3-sonnet"
        assert call_args[1]["messages"] == messages

        # Verify tools are set correctly (using OpenAI function format)
        tools = call_args[1]["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "structured_response"
        assert tools[0]["function"]["parameters"] == schema_dict

        # Verify tool_choice forces the tool usage (using OpenAI function format)
        tool_choice = call_args[1]["tool_choice"]
        assert tool_choice["type"] == "function"
        assert tool_choice["function"]["name"] == "structured_response"

    def test_tool_definition_creation(self) -> None:
        """Test creating tool definition from schema."""
        adapter = AnthropicSchemaAdapter()

        schema_dict = {
            "type": "object",
            "description": "A response with structured data",
            "properties": {"result": {"type": "string"}},
        }

        tool_def = adapter._create_tool_definition(schema_dict)

        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "structured_response"
        assert tool_def["function"]["parameters"] == schema_dict
        assert "description" in tool_def["function"]

    def test_extract_tool_response(self) -> None:
        """Test extracting structured response from tool call."""
        adapter = AnthropicSchemaAdapter()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.arguments = (
            '{"task_result": "success", "confidence": 0.9}'
        )
        mock_response.choices = [Mock(message=Mock(tool_calls=[mock_tool_call]))]

        extracted = adapter._extract_tool_response(mock_response)

        assert extracted == {"task_result": "success", "confidence": 0.9}


@pytest.mark.unit
class TestFallbackSchemaAdapter:
    """Test cases for the fallback schema adapter."""

    def test_adapter_creation(self) -> None:
        """Test creating fallback schema adapter."""
        adapter = FallbackSchemaAdapter()

        assert adapter is not None
        assert hasattr(adapter, "apply_schema")
        assert hasattr(adapter, "supports_native_schema")

    def test_supports_native_schema(self) -> None:
        """Test that fallback adapter does not support native schema."""
        adapter = FallbackSchemaAdapter()

        assert adapter.supports_native_schema() is False

    @patch("litellm.completion")
    def test_apply_schema_without_native_support(self, mock_completion: Mock) -> None:
        """Test applying schema without native support (post-processing only)."""
        adapter = FallbackSchemaAdapter()

        mock_completion.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content='{"task_result": "completed", "confidence": 0.95}'
                    )
                )
            ]
        )

        schema_dict = {
            "type": "object",
            "properties": {
                "task_result": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

        messages = [{"role": "user", "content": "Complete this task"}]

        adapter.apply_schema(messages, schema_dict, model="some-model")

        # Verify completion called without schema modifications
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args

        assert call_args[1]["model"] == "some-model"

        # Messages should be enhanced with schema hints
        enhanced_messages = call_args[1]["messages"]
        assert len(enhanced_messages) > len(messages)
        assert any("schema" in str(msg).lower() for msg in enhanced_messages)

        # Should NOT have response_format or tools
        assert "response_format" not in call_args[1]
        assert "tools" not in call_args[1]

    def test_post_processing_indication(self) -> None:
        """Test that fallback adapter indicates post-processing is needed."""
        adapter = FallbackSchemaAdapter()

        assert adapter.requires_post_processing() is True

    def test_schema_hint_injection(self) -> None:
        """Test that adapter can inject schema hints into messages."""
        adapter = FallbackSchemaAdapter()

        schema_dict = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        original_messages = [{"role": "user", "content": "Complete task"}]

        enhanced_messages = adapter._enhance_messages_with_schema_hint(
            original_messages, schema_dict
        )

        # Should have added system message with schema information
        assert len(enhanced_messages) > len(original_messages)

        # Check for schema hint in messages
        message_content = " ".join(msg["content"] for msg in enhanced_messages)
        assert "json" in message_content.lower() or "schema" in message_content.lower()

    def test_different_providers_handling(self) -> None:
        """Test that fallback adapter handles different providers consistently."""
        adapter = FallbackSchemaAdapter()

        schema_dict = {"type": "object", "properties": {"value": {"type": "string"}}}
        messages = [{"role": "user", "content": "test"}]

        # Should work the same regardless of model/provider
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[Mock(message=Mock(content='{"value": "test"}'))]
            )

            # Test different models
            models = ["gpt-3.5-turbo", "claude-3-haiku", "unknown-model"]

            for model in models:
                adapter.apply_schema(messages, schema_dict, model=model)
                assert mock_completion.called
