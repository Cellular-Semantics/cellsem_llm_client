"""Agent connection classes for LLM interactions."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from litellm import completion

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
