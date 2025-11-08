"""Enhanced usage metrics for tracking token consumption and costs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    """Enhanced usage metrics for LLM token consumption and cost tracking.

    Supports all major token types across providers:
    - Standard input/output tokens (all providers)
    - Cached tokens (OpenAI cache hits)
    - Thinking tokens (Anthropic Claude 4.1+ series)

    Provides both actual API-reported costs and estimated fallback costs
    with source attribution for transparency.

    Attributes:
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        cached_tokens: Number of cached tokens (OpenAI feature)
        thinking_tokens: Number of thinking tokens (Anthropic Claude 4.1+)
        actual_cost_usd: Actual cost from provider API in USD
        estimated_cost_usd: Estimated cost from rate calculation in USD
        cost_source: Whether cost comes from API or estimation
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Specific model used (e.g., 'gpt-4', 'claude-3-sonnet')
        timestamp: When this usage occurred

    Example:
        ```python
        from datetime import datetime
        from cellsem_llm_client.tracking.usage_metrics import UsageMetrics

        # Basic usage tracking
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=datetime.now(),
            cost_source="api",
            actual_cost_usd=0.003
        )

        print(f"Total tokens: {metrics.total_tokens}")
        print(f"Cost: ${metrics.cost}")
        ```
    """

    input_tokens: int = Field(ge=0, description="Number of input tokens consumed")
    output_tokens: int = Field(ge=0, description="Number of output tokens generated")
    cached_tokens: int | None = Field(
        default=None, ge=0, description="Number of cached tokens (OpenAI feature)"
    )
    thinking_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Number of thinking tokens (Anthropic Claude 4.1+)",
    )
    actual_cost_usd: float | None = Field(
        default=None, ge=0, description="Actual cost from provider API in USD"
    )
    estimated_cost_usd: float | None = Field(
        default=None, ge=0, description="Estimated cost from rate calculation in USD"
    )
    cost_source: Literal["api", "estimated"] = Field(
        description="Whether cost comes from API or estimation"
    )
    provider: str = Field(description="LLM provider (e.g., 'openai', 'anthropic')")
    model: str = Field(
        description="Specific model used (e.g., 'gpt-4', 'claude-3-sonnet')"
    )
    timestamp: datetime = Field(description="When this usage occurred")

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens including all optional token types."""
        total = self.input_tokens + self.output_tokens
        if self.cached_tokens is not None:
            total += self.cached_tokens
        if self.thinking_tokens is not None:
            total += self.thinking_tokens
        return total

    @property
    def cost(self) -> float | None:
        """Return the best available cost (actual if available, otherwise estimated)."""
        if self.actual_cost_usd is not None:
            return self.actual_cost_usd
        return self.estimated_cost_usd
