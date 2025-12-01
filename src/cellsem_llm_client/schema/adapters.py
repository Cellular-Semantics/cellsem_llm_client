"""Provider-specific schema adapters for native schema support."""

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from uuid import uuid4

import litellm


class AdapterCapability(Enum):
    """Capabilities that schema adapters can support."""

    NATIVE_SCHEMA = "native_schema"
    VALIDATION_RETRY = "validation_retry"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_USE = "tool_use"


class BaseSchemaAdapter(ABC):
    """Base class for provider-specific schema adapters."""

    @abstractmethod
    def apply_schema(
        self,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Apply schema to LLM request.

        Args:
            messages: Chat messages
            schema_dict: JSON schema dictionary
            model: Model name
            **kwargs: Additional parameters

        Returns:
            LLM response with schema applied
        """
        pass

    @abstractmethod
    def supports_native_schema(self) -> bool:
        """Check if this adapter supports native schema enforcement."""
        pass

    def requires_post_processing(self) -> bool:
        """Check if this adapter requires post-processing validation."""
        return not self.supports_native_schema()

    def get_capabilities(self) -> set[AdapterCapability]:
        """Get set of capabilities this adapter supports."""
        return set()


class OpenAISchemaAdapter(BaseSchemaAdapter):
    """Schema adapter for OpenAI models using structured outputs."""

    def supports_native_schema(self) -> bool:
        """OpenAI supports native schema via response_format."""
        return True

    def get_capabilities(self) -> set[AdapterCapability]:
        """Get OpenAI adapter capabilities."""
        return {AdapterCapability.NATIVE_SCHEMA, AdapterCapability.STRUCTURED_OUTPUT}

    def apply_schema(
        self,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Apply schema using OpenAI's structured output format.

        Args:
            messages: Chat messages
            schema_dict: JSON schema dictionary
            model: OpenAI model name
            **kwargs: Additional litellm parameters

        Returns:
            LiteLLM completion response
        """
        # Generate schema name
        schema_name = self._generate_schema_name(schema_dict)

        # Use OpenAI's response_format with strict schema enforcement
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema_dict, "strict": True},
        }

        # Call LiteLLM with structured output format
        # Configure allowed parameters for OpenAI
        return litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format,
            allowed_openai_params=["response_format"],
            **kwargs,
        )

    def _generate_schema_name(self, schema_dict: dict[str, Any]) -> str:
        """Generate a name for the schema.

        Args:
            schema_dict: JSON schema dictionary

        Returns:
            Schema name for OpenAI API
        """
        # Use title if available
        if "title" in schema_dict:
            return schema_dict["title"]

        # Use description if available
        if "description" in schema_dict:
            # Clean up description to make it a valid name
            name = schema_dict["description"].replace(" ", "_").replace("-", "_")
            # Keep only alphanumeric and underscores
            name = "".join(c for c in name if c.isalnum() or c == "_")
            if name:
                return name

        # Generate a unique name
        return f"schema_{uuid4().hex[:8]}"


class AnthropicSchemaAdapter(BaseSchemaAdapter):
    """Schema adapter for Anthropic models using tool choice pattern."""

    def supports_native_schema(self) -> bool:
        """Anthropic supports native schema via tool use."""
        return True

    def get_capabilities(self) -> set[AdapterCapability]:
        """Get Anthropic adapter capabilities."""
        return {AdapterCapability.NATIVE_SCHEMA, AdapterCapability.TOOL_USE}

    def apply_schema(
        self,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Apply schema using Anthropic's tool choice pattern.

        Args:
            messages: Chat messages
            schema_dict: JSON schema dictionary
            model: Anthropic model name
            **kwargs: Additional litellm parameters

        Returns:
            LiteLLM completion response with tool use
        """
        # Create tool definition from schema
        tool_definition = self._create_tool_definition(schema_dict)

        # Force tool usage with tool_choice (OpenAI compatible format)
        tool_choice = {"type": "function", "function": {"name": "structured_response"}}

        # Call LiteLLM with tool enforcement
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=[tool_definition],
            tool_choice=tool_choice,
            **kwargs,
        )

        # Extract structured response from tool call
        return self._extract_tool_response(response)

    def _create_tool_definition(self, schema_dict: dict[str, Any]) -> dict[str, Any]:
        """Create tool definition from JSON schema.

        Args:
            schema_dict: JSON schema dictionary

        Returns:
            Tool definition for Anthropic API
        """
        description = schema_dict.get(
            "description", "Provide structured response according to schema"
        )

        return {
            "type": "function",
            "function": {
                "name": "structured_response",
                "description": description,
                "parameters": schema_dict,
            },
        }

    def _extract_tool_response(self, response: Any) -> dict[str, Any]:
        """Extract structured response from tool call.

        Args:
            response: LiteLLM response with tool calls

        Returns:
            Parsed JSON response from tool call
        """
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                tool_calls = choice.message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    # Get the first (and should be only) tool call
                    tool_call = tool_calls[0]
                    if hasattr(tool_call, "function") and hasattr(
                        tool_call.function, "arguments"
                    ):
                        # Parse the tool call arguments as JSON
                        return json.loads(tool_call.function.arguments)

        # Fallback: return empty dict if no tool call found
        return {}


class FallbackSchemaAdapter(BaseSchemaAdapter):
    """Fallback adapter for providers without native schema support."""

    def supports_native_schema(self) -> bool:
        """Fallback adapter does not support native schema."""
        return False

    def get_capabilities(self) -> set[AdapterCapability]:
        """Get fallback adapter capabilities."""
        return {AdapterCapability.VALIDATION_RETRY}

    def apply_schema(
        self,
        messages: list[dict[str, Any]],
        schema_dict: dict[str, Any],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Apply schema by enhancing prompt and relying on post-processing.

        Args:
            messages: Chat messages
            schema_dict: JSON schema dictionary
            model: Model name
            **kwargs: Additional litellm parameters

        Returns:
            LiteLLM completion response
        """
        # Enhance messages with schema hint
        enhanced_messages = self._enhance_messages_with_schema_hint(
            messages, schema_dict
        )

        # Call LiteLLM without schema enforcement
        return litellm.completion(model=model, messages=enhanced_messages, **kwargs)

    def _enhance_messages_with_schema_hint(
        self, messages: list[dict[str, Any]], schema_dict: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Add schema hints to messages for better compliance.

        Args:
            messages: Original chat messages
            schema_dict: JSON schema dictionary

        Returns:
            Enhanced messages with schema hints
        """
        # Create schema description
        schema_description = self._describe_schema(schema_dict)

        # Add system message with schema instruction
        schema_instruction = {
            "role": "system",
            "content": (
                f"Please respond with valid JSON that matches this schema: {schema_description}. "
                "Ensure your response is properly formatted JSON with no additional text."
            ),
        }

        # Insert at the beginning or after existing system message
        enhanced_messages = messages.copy()

        # Find if there's already a system message
        system_message_index = None
        for i, msg in enumerate(enhanced_messages):
            if msg.get("role") == "system":
                system_message_index = i
                break

        if system_message_index is not None:
            # Append to existing system message
            existing_content = enhanced_messages[system_message_index]["content"]
            enhanced_messages[system_message_index]["content"] = (
                f"{existing_content}\n\n{schema_instruction['content']}"
            )
        else:
            # Insert new system message at the beginning
            enhanced_messages.insert(0, schema_instruction)

        return enhanced_messages

    def _describe_schema(self, schema_dict: dict[str, Any]) -> str:
        """Create a human-readable description of the schema.

        Args:
            schema_dict: JSON schema dictionary

        Returns:
            Human-readable schema description
        """
        if schema_dict.get("type") == "object":
            properties = schema_dict.get("properties", {})
            required = schema_dict.get("required", [])

            if not properties:
                return "an empty object {}"

            field_descriptions = []
            for field, field_schema in properties.items():
                field_type = field_schema.get("type", "any")
                is_required = field in required
                req_text = "required" if is_required else "optional"

                description = f'"{field}": {field_type} ({req_text})'

                if "description" in field_schema:
                    description += f" - {field_schema['description']}"

                field_descriptions.append(description)

            return "object with fields: " + ", ".join(field_descriptions)

        elif schema_dict.get("type") == "array":
            items_schema = schema_dict.get("items", {})
            item_type = (
                items_schema.get("type", "any")
                if isinstance(items_schema, dict)
                else "mixed"
            )
            return f"array of {item_type} items"

        else:
            schema_type = schema_dict.get("type", "any")
            return f"value of type {schema_type}"


class SchemaAdapterFactory:
    """Factory for creating appropriate schema adapters based on provider."""

    def get_adapter(self, provider: str, model: str) -> BaseSchemaAdapter:
        """Get appropriate schema adapter for provider and model.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model name

        Returns:
            Appropriate schema adapter instance
        """
        provider_lower = provider.lower()

        if provider_lower == "openai" or "gpt" in model.lower():
            return OpenAISchemaAdapter()
        elif provider_lower == "anthropic" or "claude" in model.lower():
            return AnthropicSchemaAdapter()
        else:
            return FallbackSchemaAdapter()

    def get_capabilities(self, provider: str, model: str) -> set[AdapterCapability]:
        """Get capabilities for a provider and model.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Set of capabilities
        """
        adapter = self.get_adapter(provider, model)
        return adapter.get_capabilities()
