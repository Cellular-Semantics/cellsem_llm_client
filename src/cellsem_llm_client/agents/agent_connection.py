"""Agent connection classes for LLM interactions."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from litellm import completion
from pydantic import BaseModel

from cellsem_llm_client.schema import (
    SchemaAdapterFactory,
    SchemaManager,
    SchemaValidator,
)
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics

if TYPE_CHECKING:
    from cellsem_llm_client.tracking.cost_calculator import FallbackCostCalculator


class AgentConnection(ABC):
    """Abstract base class for LLM agent connections."""

    @abstractmethod
    def query(self, message: str, system_message: str | None = None) -> str:
        """Send a query to the LLM and return the response.

        Args:
            message: The user message to send
            system_message: Optional system message to set context

        Returns:
            The LLM's response as a string
        """
        pass

    @abstractmethod
    def query_with_tracking(
        self,
        message: str,
        system_message: str | None = None,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
    ) -> tuple[str, UsageMetrics]:
        """Send a query to the LLM and return response with usage tracking.

        Args:
            message: The user message to send
            system_message: Optional system message to set context
            cost_calculator: Optional cost calculator for estimating costs

        Returns:
            Tuple of (response, usage_metrics)
        """
        pass

    @abstractmethod
    def query_with_schema(
        self,
        message: str,
        schema: dict[str, Any] | type[BaseModel] | str,
        system_message: str | None = None,
        max_retries: int = 2,
    ) -> BaseModel:
        """Send a query to the LLM with schema enforcement.

        Args:
            message: The user message to send
            schema: JSON schema dict, Pydantic model class, or schema name
            system_message: Optional system message to set context
            max_retries: Maximum validation retry attempts

        Returns:
            Validated Pydantic model instance matching the schema
        """
        pass

    @abstractmethod
    def query_with_schema_and_tracking(
        self,
        message: str,
        schema: dict[str, Any] | type[BaseModel] | str,
        system_message: str | None = None,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
        max_retries: int = 2,
    ) -> tuple[BaseModel, UsageMetrics]:
        """Send a query to the LLM with schema enforcement and usage tracking.

        Args:
            message: The user message to send
            schema: JSON schema dict, Pydantic model class, or schema name
            system_message: Optional system message to set context
            cost_calculator: Optional cost calculator for estimating costs
            max_retries: Maximum validation retry attempts

        Returns:
            Tuple of (validated_model_instance, usage_metrics)
        """
        pass


class LiteLLMAgent(AgentConnection):
    """LLM agent using LiteLLM for multi-provider support."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize the LiteLLM agent.

        Args:
            model: The model name (e.g., 'gpt-3.5-turbo', 'claude-3-haiku-20240307')
            api_key: The API key for authentication
            max_tokens: Maximum tokens for response (default: 1000)

        Raises:
            ValueError: If model or api_key is None
        """
        if model is None:
            raise ValueError("Model is required")
        if api_key is None:
            raise ValueError("API key is required")

        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

        # Initialize schema components
        self._schema_manager = SchemaManager()
        self._schema_validator = SchemaValidator()
        self._adapter_factory = SchemaAdapterFactory()

    def query(self, message: str, system_message: str | None = None) -> str:
        """Send a query to the LLM using LiteLLM.

        Args:
            message: The user message to send
            system_message: Optional system message to set context

        Returns:
            The LLM's response as a string
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": message})

        response = completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        return str(response.choices[0].message.content)

    def query_with_tracking(
        self,
        message: str,
        system_message: str | None = None,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
    ) -> tuple[str, UsageMetrics]:
        """Send a query to the LLM with usage tracking using LiteLLM.

        Args:
            message: The user message to send
            system_message: Optional system message to set context
            cost_calculator: Optional cost calculator for estimating costs

        Returns:
            Tuple of (response, usage_metrics)
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": message})

        response = completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        # Extract usage information from LiteLLM response
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Extract cached tokens (OpenAI specific)
        cached_tokens = None
        if (
            hasattr(usage, "prompt_tokens_details")
            and usage.prompt_tokens_details
            and hasattr(usage.prompt_tokens_details, "cached_tokens")
        ):
            cached_tokens = usage.prompt_tokens_details.cached_tokens

        # Extract thinking tokens (Anthropic specific)
        # Note: LiteLLM may not expose thinking tokens directly yet
        thinking_tokens = None

        # Determine provider from model name
        provider = self._get_provider_from_model(self.model)

        # Calculate cost if calculator provided
        estimated_cost_usd = None
        if cost_calculator:
            try:
                # Create temporary metrics for cost calculation
                temp_usage_metrics = UsageMetrics(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    thinking_tokens=thinking_tokens,
                    provider=provider,
                    model=self.model,
                    timestamp=datetime.now(),
                    cost_source="estimated",
                )
                estimated_cost_usd = cost_calculator.calculate_cost(temp_usage_metrics)
            except Exception:
                # If cost calculation fails, continue without cost estimation
                pass

        # Create final usage metrics with cost
        usage_metrics = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
            estimated_cost_usd=estimated_cost_usd,
            provider=provider,
            model=self.model,
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        response_text = str(response.choices[0].message.content)
        return response_text, usage_metrics

    def query_with_schema(
        self,
        message: str,
        schema: dict[str, Any] | type[BaseModel] | str,
        system_message: str | None = None,
        max_retries: int = 2,
    ) -> BaseModel:
        """Send a query to the LLM with schema enforcement.

        Args:
            message: The user message to send
            schema: JSON schema dict, Pydantic model class, or schema name
            system_message: Optional system message to set context
            max_retries: Maximum validation retry attempts

        Returns:
            Validated Pydantic model instance matching the schema

        Raises:
            Exception: If schema validation fails after max retries
        """
        # Get Pydantic model from schema input
        pydantic_model = self._schema_manager.get_pydantic_model(schema)

        # Convert Pydantic model to JSON schema for adapter
        schema_dict = self._pydantic_model_to_schema(pydantic_model)

        # Get appropriate adapter for this provider/model
        provider = self._get_provider_from_model(self.model)
        adapter = self._adapter_factory.get_adapter(provider, self.model)

        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})

        # Apply schema enforcement using adapter
        response = adapter.apply_schema(
            messages=messages,
            schema_dict=schema_dict,
            model=self.model,
            max_tokens=self.max_tokens,
        )

        # Extract response content based on adapter type
        from cellsem_llm_client.schema.adapters import AnthropicSchemaAdapter

        if isinstance(adapter, AnthropicSchemaAdapter):
            # Anthropic adapter returns dict directly from _extract_tool_response
            response_content = json.dumps(response)
        elif adapter.supports_native_schema():
            # For other native schema adapters (OpenAI), get content from message
            response_content = str(response.choices[0].message.content)
        else:
            # For fallback adapters, content is in message
            response_content = str(response.choices[0].message.content)

        # Validate response against schema with retry logic
        validation_result = self._schema_validator.validate_with_retry(
            response_text=response_content,
            target_model=pydantic_model,
            max_retries=max_retries,
        )

        if not validation_result.success:
            raise Exception(
                f"Schema validation failed after {max_retries} retries: {validation_result.error}"
            )

        assert validation_result.model_instance is not None
        return validation_result.model_instance

    def query_with_schema_and_tracking(
        self,
        message: str,
        schema: dict[str, Any] | type[BaseModel] | str,
        system_message: str | None = None,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
        max_retries: int = 2,
    ) -> tuple[BaseModel, UsageMetrics]:
        """Send a query to the LLM with schema enforcement and usage tracking.

        Args:
            message: The user message to send
            schema: JSON schema dict, Pydantic model class, or schema name
            system_message: Optional system message to set context
            cost_calculator: Optional cost calculator for estimating costs
            max_retries: Maximum validation retry attempts

        Returns:
            Tuple of (validated_model_instance, usage_metrics)

        Raises:
            Exception: If schema validation fails after max retries
        """
        # Get Pydantic model from schema input
        pydantic_model = self._schema_manager.get_pydantic_model(schema)

        # Convert Pydantic model to JSON schema for adapter
        schema_dict = self._pydantic_model_to_schema(pydantic_model)

        # Get appropriate adapter for this provider/model
        provider = self._get_provider_from_model(self.model)
        adapter = self._adapter_factory.get_adapter(provider, self.model)

        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})

        # For Anthropic, we need to handle differently to preserve usage info
        from cellsem_llm_client.schema.adapters import AnthropicSchemaAdapter

        if isinstance(adapter, AnthropicSchemaAdapter):
            # Call completion directly to get both response and usage info
            raw_response = completion(
                model=self.model,
                messages=messages,
                tools=[adapter._create_tool_definition(schema_dict)],
                tool_choice={"type": "tool", "name": "structured_response"},
                max_tokens=self.max_tokens,
            )

            # Extract structured response using adapter
            response = adapter._extract_tool_response(raw_response)
            response_for_usage = raw_response
        else:
            # Apply schema enforcement using adapter
            response = adapter.apply_schema(
                messages=messages,
                schema_dict=schema_dict,
                model=self.model,
                max_tokens=self.max_tokens,
            )
            response_for_usage = response

        # Extract usage information from response
        usage = response_for_usage.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Extract cached tokens (OpenAI specific)
        cached_tokens = None
        if (
            hasattr(usage, "prompt_tokens_details")
            and usage.prompt_tokens_details
            and hasattr(usage.prompt_tokens_details, "cached_tokens")
        ):
            cached_tokens = usage.prompt_tokens_details.cached_tokens

        # Extract thinking tokens (future enhancement)
        thinking_tokens = None

        # Calculate cost if calculator provided
        estimated_cost_usd = None
        if cost_calculator:
            try:
                temp_usage_metrics = UsageMetrics(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    thinking_tokens=thinking_tokens,
                    provider=provider,
                    model=self.model,
                    timestamp=datetime.now(),
                    cost_source="estimated",
                )
                estimated_cost_usd = cost_calculator.calculate_cost(temp_usage_metrics)
            except Exception:
                pass

        # Create usage metrics
        usage_metrics = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
            estimated_cost_usd=estimated_cost_usd,
            provider=provider,
            model=self.model,
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        # Extract response content based on adapter type
        if isinstance(adapter, AnthropicSchemaAdapter):
            # Anthropic: response is already the extracted dict
            response_content = json.dumps(response)
        elif adapter.supports_native_schema():
            # For other native schema adapters (OpenAI), get content from message
            response_content = str(response.choices[0].message.content)  # type: ignore
        else:
            # For fallback adapters, content is in message
            response_content = str(response.choices[0].message.content)  # type: ignore

        # Validate response against schema with retry logic
        validation_result = self._schema_validator.validate_with_retry(
            response_text=response_content,
            target_model=pydantic_model,
            max_retries=max_retries,
        )

        if not validation_result.success:
            raise Exception(
                f"Schema validation failed after {max_retries} retries: {validation_result.error}"
            )

        assert validation_result.model_instance is not None
        return validation_result.model_instance, usage_metrics

    def _pydantic_model_to_schema(self, model_class: type[BaseModel]) -> dict[str, Any]:
        """Convert Pydantic model to JSON schema dict.

        Args:
            model_class: Pydantic model class

        Returns:
            JSON schema dictionary
        """
        schema = model_class.model_json_schema()

        # Ensure additionalProperties: false for OpenAI structured output compatibility
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        # Fix required fields for OpenAI strict mode
        self._fix_openai_strict_schema(schema)

        # Recursively set additionalProperties: false for nested objects
        self._ensure_no_additional_properties(schema)

        return schema

    def _fix_openai_strict_schema(self, schema_dict: dict[str, Any]) -> None:
        """Fix schema for OpenAI strict mode requirements.

        OpenAI's strict mode requires:
        1. All properties must be explicitly required or explicitly optional
        2. Cannot use anyOf for nullable fields

        Args:
            schema_dict: Schema dictionary to modify in-place
        """
        if isinstance(schema_dict, dict) and schema_dict.get("type") == "object":
            properties = schema_dict.get("properties", {})
            required = set(schema_dict.get("required", []))

            # Handle nullable fields (anyOf with null type)
            props_to_remove = []
            for prop_name, prop_schema in properties.items():
                if isinstance(prop_schema, dict) and "anyOf" in prop_schema:
                    any_of = prop_schema["anyOf"]
                    # Check if it's a nullable field (has null type)
                    null_type = None
                    other_type = None

                    for option in any_of:
                        if isinstance(option, dict):
                            if option.get("type") == "null":
                                null_type = option
                            else:
                                other_type = option

                    # If it's a nullable field, simplify for OpenAI strict mode
                    if null_type and other_type:
                        # For OpenAI strict mode, remove optional fields entirely
                        # This is the safest approach for strict schema validation
                        if prop_name not in required:
                            props_to_remove.append(prop_name)

            # Remove optional nullable properties
            for prop_name in props_to_remove:
                del properties[prop_name]

            # Update required fields
            schema_dict["required"] = list(required)

        # Recursively process nested schemas
        if isinstance(schema_dict, dict):
            for key, value in schema_dict.items():
                if key == "properties" and isinstance(value, dict):
                    for prop_value in value.values():
                        self._fix_openai_strict_schema(prop_value)
                elif key == "$defs" and isinstance(value, dict):
                    # Handle Pydantic $defs for nested models
                    for def_value in value.values():
                        self._fix_openai_strict_schema(def_value)
                elif isinstance(value, dict):
                    self._fix_openai_strict_schema(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._fix_openai_strict_schema(item)

    def _ensure_no_additional_properties(self, schema_dict: dict[str, Any]) -> None:
        """Recursively ensure no additional properties in schema for OpenAI compatibility.

        Args:
            schema_dict: Schema dictionary to modify in-place
        """
        if isinstance(schema_dict, dict):
            # Set additionalProperties: false for object types
            if (
                schema_dict.get("type") == "object"
                and "additionalProperties" not in schema_dict
            ):
                schema_dict["additionalProperties"] = False

            # Recursively process nested schemas
            for key, value in schema_dict.items():
                if key == "properties" and isinstance(value, dict):
                    for prop_value in value.values():
                        self._ensure_no_additional_properties(prop_value)
                elif key == "$defs" and isinstance(value, dict):
                    # Handle Pydantic $defs for nested models
                    for def_value in value.values():
                        self._ensure_no_additional_properties(def_value)
                elif key in ("items", "allOf", "oneOf", "anyOf") and isinstance(
                    value, (dict, list)
                ):
                    if isinstance(value, dict):
                        self._ensure_no_additional_properties(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                self._ensure_no_additional_properties(item)

    def _extract_response_content(self, response: Any, adapter: Any) -> str:
        """Extract response content based on adapter type.

        Args:
            response: LLM response object
            adapter: Schema adapter used

        Returns:
            Response content as string
        """
        import json

        from cellsem_llm_client.schema.adapters import AnthropicSchemaAdapter

        # For Anthropic adapter, extract from tool call and convert back to JSON string
        if isinstance(adapter, AnthropicSchemaAdapter):
            tool_response = adapter._extract_tool_response(response)
            return json.dumps(tool_response)

        # For OpenAI and others, use standard message content
        return str(response.choices[0].message.content)

    def _get_provider_from_model(self, model: str) -> str:
        """Determine provider from model name.

        Args:
            model: The model name

        Returns:
            Provider name ('openai', 'anthropic', etc.)
        """
        model_lower = model.lower()

        if any(
            prefix in model_lower
            for prefix in ["gpt", "davinci", "curie", "babbage", "ada"]
        ):
            return "openai"
        elif any(prefix in model_lower for prefix in ["claude"]):
            return "anthropic"
        elif any(prefix in model_lower for prefix in ["gemini", "palm", "bison"]):
            return "google"
        elif any(prefix in model_lower for prefix in ["llama", "code-llama"]):
            return "meta"
        else:
            # Default fallback - could also raise an exception
            return "unknown"


class OpenAIAgent(LiteLLMAgent):
    """Convenience class for OpenAI models with sensible defaults."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize OpenAI agent with default model.

        Args:
            model: OpenAI model name (default: 'gpt-3.5-turbo')
            api_key: OpenAI API key
            max_tokens: Maximum tokens for response
        """
        super().__init__(model=model, api_key=api_key, max_tokens=max_tokens)


class AnthropicAgent(LiteLLMAgent):
    """Convenience class for Anthropic models with sensible defaults."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        max_tokens: int = 1000,
    ):
        """Initialize Anthropic agent with default model.

        Args:
            model: Anthropic model name (default: 'claude-3-haiku-20240307')
            api_key: Anthropic API key
            max_tokens: Maximum tokens for response
        """
        super().__init__(model=model, api_key=api_key, max_tokens=max_tokens)
