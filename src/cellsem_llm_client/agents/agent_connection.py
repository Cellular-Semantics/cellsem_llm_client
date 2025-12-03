"""Agent connection classes for LLM interactions."""

import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from litellm import completion
from pydantic import BaseModel

from cellsem_llm_client.exceptions import SchemaValidationException
from cellsem_llm_client.schema import (
    SchemaAdapterFactory,
    SchemaManager,
    SchemaValidator,
)
from cellsem_llm_client.tracking.cost_calculator import FallbackCostCalculator
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics


@dataclass
class QueryResult:
    """Structured result returned by unified query interface."""

    text: str | None
    model: BaseModel | None = None
    usage: UsageMetrics | None = None
    raw_response: Any | None = None


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

    def query_unified(
        self,
        message: str,
        system_message: str | None = None,
        schema: dict[str, Any] | type[BaseModel] | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable[[dict[str, Any]], str | None]] | None = None,
        max_turns: int = 5,
        track_usage: bool = False,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
        max_retries: int = 2,
        auto_cost: bool = True,
    ) -> QueryResult:
        """Unified query interface with optional tools, schema enforcement, and tracking.

        This method consolidates the previous `query*` variants. Use feature flags/args
        instead of separate methods:

        Args:
            message: User message.
            system_message: Optional system prompt.
            schema: JSON Schema dict, Pydantic model class, or schema name for
                enforcement + validation. If provided with tools, validation runs
                on the final assistant message after tool calls finish.
            tools: LiteLLM tool definitions. Enables tool-call loop.
            tool_handlers: Mapping of tool names to callables for execution.
            max_turns: Max tool-call iterations before giving up.
            track_usage: Whether to return usage metrics.
            cost_calculator: Optional cost calculator for estimated cost.
            max_retries: Validation retry limit when `schema` is provided.
            auto_cost: When True, auto-create a fallback cost calculator if none is provided
                and tracking is enabled.

        Returns:
            QueryResult containing final text, optional validated Pydantic model,
            optional usage metrics, and the raw LiteLLM response.

        Raises:
            SchemaValidationException: If schema validation fails after retries.
            ValueError: For missing tool handlers or argument parsing failures.
            RuntimeError: If tool loop exceeds `max_turns`.
        """
        messages: list[dict[str, Any]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": message})

        provider = self._get_provider_from_model(self.model)
        pydantic_model: type[BaseModel] | None = None
        schema_dict: dict[str, Any] | None = None

        if schema is not None:
            pydantic_model = self._schema_manager.get_pydantic_model(schema)
            schema_dict = self._pydantic_model_to_schema(pydantic_model)

        raw_response: Any | None = None
        response_content: str | None = None
        all_tool_responses: list[Any] | None = None

        if tools:
            response_content, raw_response, all_tool_responses = self._run_tool_loop(
                messages=messages,
                tools=tools,
                tool_handlers=tool_handlers or {},
                max_turns=max_turns,
            )
        elif schema_dict is not None:
            adapter = self._adapter_factory.get_adapter(provider, self.model)
            from cellsem_llm_client.schema.adapters import AnthropicSchemaAdapter

            if isinstance(adapter, AnthropicSchemaAdapter):
                raw_response = completion(
                    model=self.model,
                    messages=messages,
                    tools=[adapter._create_tool_definition(schema_dict)],
                    tool_choice={
                        "type": "function",
                        "function": {"name": "structured_response"},
                    },
                    max_tokens=self.max_tokens,
                )
                extracted = adapter._extract_tool_response(raw_response)
                response_content = json.dumps(extracted)
            else:
                raw_response = adapter.apply_schema(
                    messages=messages,
                    schema_dict=schema_dict,
                    model=self.model,
                    max_tokens=self.max_tokens,
                )
                response_content = str(raw_response.choices[0].message.content)
        else:
            raw_response = completion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            response_content = str(raw_response.choices[0].message.content)

        validated_model: BaseModel | None = None
        if pydantic_model is not None and response_content is not None:
            validation_result = self._schema_validator.validate_with_retry(
                response_text=response_content,
                target_model=pydantic_model,
                max_retries=max_retries,
            )
            if not validation_result.success:
                raise SchemaValidationException(
                    f"Schema validation failed after {max_retries} retries: {validation_result.error}",
                    schema=str(schema),
                    response_text=response_content,
                    validation_errors=[str(validation_result.error)]
                    if validation_result.error
                    else [],
                )
            validated_model = validation_result.model_instance

        usage_metrics: UsageMetrics | None = None
        if track_usage and raw_response is not None and hasattr(raw_response, "usage"):
            calc = cost_calculator
            if calc is None and auto_cost:
                calc = self._build_default_calculator()
            # When tools were used, accumulate usage from all API calls
            if all_tool_responses is not None:
                usage_metrics = self._accumulate_usage_metrics(
                    responses=all_tool_responses,
                    provider=provider,
                    cost_calculator=calc,
                )
            else:
                # Single API call without tools
                usage_metrics = self._build_usage_metrics(
                    raw_response=raw_response,
                    provider=provider,
                    cost_calculator=calc,
                )

        return QueryResult(
            text=response_content,
            model=validated_model,
            usage=usage_metrics,
            raw_response=raw_response,
        )

    def query(self, message: str, system_message: str | None = None) -> str:
        """Send a query to the LLM using LiteLLM.

        Args:
            message: The user message to send
            system_message: Optional system message to set context

        Returns:
            The LLM's response as a string
        """
        result = self.query_unified(
            message=message,
            system_message=system_message,
        )
        return result.text or ""

    def query_with_tools(
        self,
        message: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[[dict[str, Any]], str | None]] | None = None,
        system_message: str | None = None,
        max_turns: int = 5,
    ) -> str:
        """Send a query to the LLM with tool-calling support.

        Args:
            message: The user message to send.
            tools: Tool definitions to forward to LiteLLM.
            tool_handlers: Mapping of tool names to callables that execute the tool.
            system_message: Optional system message to set context.
            max_turns: Maximum number of tool-call iterations before giving up.

        Returns:
            The assistant's final message content after executing tools.

        Raises:
            ValueError: If a tool call is returned without a matching handler or
                tool arguments cannot be parsed.
            RuntimeError: If the conversation does not terminate within
                ``max_turns`` iterations.
        """
        warnings.warn(
            "query_with_tools is deprecated; use query_unified with tools/tool_handlers.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        result = self.query_unified(
            message=message,
            system_message=system_message,
            tools=tools,
            tool_handlers=tool_handlers,
            max_turns=max_turns,
        )
        return result.text or ""

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
        warnings.warn(
            "query_with_tracking is deprecated; use query_unified with track_usage=True.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        result = self.query_unified(
            message=message,
            system_message=system_message,
            track_usage=True,
            cost_calculator=cost_calculator,
        )
        if result.usage is None:
            raise RuntimeError("Expected usage metrics but none were populated")
        return result.text or "", result.usage

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
            SchemaValidationException: If schema validation fails after max retries
        """
        warnings.warn(
            "query_with_schema is deprecated; use query_unified with schema=...",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        result = self.query_unified(
            message=message,
            system_message=system_message,
            schema=schema,
            max_retries=max_retries,
        )
        if result.model is None:
            raise RuntimeError("Expected model instance but none was populated")
        return result.model

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
            SchemaValidationException: If schema validation fails after max retries
        """
        warnings.warn(
            "query_with_schema_and_tracking is deprecated; use query_unified with schema=... and track_usage=True.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        result = self.query_unified(
            message=message,
            schema=schema,
            system_message=system_message,
            track_usage=True,
            cost_calculator=cost_calculator,
            max_retries=max_retries,
        )
        if result.model is None or result.usage is None:
            raise RuntimeError(
                "Expected model instance and usage metrics but one or both were not populated"
            )
        return result.model, result.usage

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

    def _build_default_calculator(self) -> "FallbackCostCalculator":
        """Create a fallback calculator with default rates loaded."""
        calculator = FallbackCostCalculator()
        calculator.load_default_rates()
        return calculator

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[[dict[str, Any]], str | None]],
        max_turns: int,
    ) -> tuple[str, Any, list[Any]]:
        """Execute tool calls until completion.

        Returns:
            A tuple of (final_content, final_response, all_responses) where
            all_responses contains every API response from all turns for usage tracking.
        """
        working_messages = list(messages)
        all_responses: list[Any] = []

        for _turn in range(max_turns):
            response = completion(
                model=self.model,
                messages=[*working_messages],
                tools=tools,
                max_tokens=self.max_tokens,
            )
            all_responses.append(response)

            response_message = response.choices[0].message
            tool_calls = getattr(response_message, "tool_calls", None)

            if tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [],
                }
                working_messages.append(assistant_message)

                for tool_call in tool_calls:
                    function_call = getattr(tool_call, "function", None)
                    tool_name = getattr(function_call, "name", None)
                    tool_arguments = getattr(function_call, "arguments", {})

                    assistant_message["tool_calls"].append(
                        {
                            "id": getattr(tool_call, "id", ""),
                            "type": getattr(tool_call, "type", "function"),
                            "function": {
                                "name": tool_name,
                                "arguments": tool_arguments,
                            },
                        }
                    )

                    if not tool_name or tool_name not in tool_handlers:
                        raise ValueError(f"No handler found for tool '{tool_name}'.")

                    try:
                        parsed_args = (
                            json.loads(tool_arguments)
                            if isinstance(tool_arguments, str)
                            else tool_arguments
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"Failed to parse arguments for tool '{tool_name}'."
                        ) from exc

                    tool_result = tool_handlers[tool_name](parsed_args)
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": getattr(tool_call, "id", tool_name),
                            "content": tool_result if tool_result is not None else "",
                        }
                    )
                continue

            return str(response_message.content), response, all_responses

        raise RuntimeError("Max tool-call turns reached without a final response.")

    def _build_usage_metrics(
        self,
        raw_response: Any,
        provider: str,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
    ) -> UsageMetrics:
        """Construct UsageMetrics from a LiteLLM response."""
        usage = raw_response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        cached_tokens = None
        if (
            hasattr(usage, "prompt_tokens_details")
            and usage.prompt_tokens_details
            and hasattr(usage.prompt_tokens_details, "cached_tokens")
        ):
            cached_tokens = usage.prompt_tokens_details.cached_tokens

        thinking_tokens = None

        estimated_cost_usd = None
        rate_last_updated = None
        if cost_calculator:
            try:
                get_rates = getattr(cost_calculator, "get_model_rates", None)
                rate_data = (
                    get_rates(provider, self.model) if callable(get_rates) else None
                )
                if rate_data and hasattr(rate_data, "source"):
                    access_date = getattr(rate_data.source, "access_date", None)
                    rate_last_updated = (
                        access_date if isinstance(access_date, datetime) else None
                    )
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
            except Exception as e:
                logging.warning(
                    f"Cost calculation failed for {provider}/{self.model}: {e}"
                )

        return UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
            estimated_cost_usd=estimated_cost_usd,
            rate_last_updated=rate_last_updated,
            provider=provider,
            model=self.model,
            timestamp=datetime.now(),
            cost_source="estimated",
        )

    def _accumulate_usage_metrics(
        self,
        responses: list[Any],
        provider: str,
        cost_calculator: Optional["FallbackCostCalculator"] = None,
    ) -> UsageMetrics:
        """Accumulate usage metrics from multiple API responses.

        When tools are used, multiple API calls are made across iterations.
        This method sums up the token usage from all calls to provide accurate
        cumulative metrics.

        Args:
            responses: List of LiteLLM response objects from all API calls
            provider: The LLM provider name
            cost_calculator: Optional cost calculator for estimating costs

        Returns:
            Accumulated UsageMetrics with total token counts and costs
        """
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        total_thinking_tokens = 0
        has_cached = False
        has_thinking = False

        for response in responses:
            if not hasattr(response, "usage"):
                continue

            usage = response.usage
            total_input_tokens += usage.prompt_tokens
            total_output_tokens += usage.completion_tokens

            # Accumulate cached tokens if present
            if (
                hasattr(usage, "prompt_tokens_details")
                and usage.prompt_tokens_details
                and hasattr(usage.prompt_tokens_details, "cached_tokens")
                and usage.prompt_tokens_details.cached_tokens is not None
            ):
                total_cached_tokens += usage.prompt_tokens_details.cached_tokens
                has_cached = True

        cached_tokens = total_cached_tokens if has_cached else None
        thinking_tokens = total_thinking_tokens if has_thinking else None

        # Calculate cost based on accumulated tokens
        estimated_cost_usd = None
        rate_last_updated = None
        if cost_calculator:
            try:
                get_rates = getattr(cost_calculator, "get_model_rates", None)
                rate_data = (
                    get_rates(provider, self.model) if callable(get_rates) else None
                )
                if rate_data and hasattr(rate_data, "source"):
                    access_date = getattr(rate_data.source, "access_date", None)
                    rate_last_updated = (
                        access_date if isinstance(access_date, datetime) else None
                    )
                temp_usage_metrics = UsageMetrics(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cached_tokens=cached_tokens,
                    thinking_tokens=thinking_tokens,
                    provider=provider,
                    model=self.model,
                    timestamp=datetime.now(),
                    cost_source="estimated",
                )
                estimated_cost_usd = cost_calculator.calculate_cost(temp_usage_metrics)
            except Exception as e:
                logging.warning(
                    f"Cost calculation failed for {provider}/{self.model}: {e}"
                )

        return UsageMetrics(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
            estimated_cost_usd=estimated_cost_usd,
            rate_last_updated=rate_last_updated,
            provider=provider,
            model=self.model,
            timestamp=datetime.now(),
            cost_source="estimated",
        )

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
