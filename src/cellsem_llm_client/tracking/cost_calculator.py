"""Fallback cost calculation with rate database and source tracking."""

from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator

from cellsem_llm_client.exceptions import CostCalculationException
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics

# Keep old exception name for backward compatibility
CostCalculationError = CostCalculationException


class RateSource(BaseModel):
    """Source information for rate data with freshness tracking.

    Tracks where rate information came from and when it was accessed
    to enable freshness validation and update scheduling.

    Attributes:
        name: Human-readable name of the rate source
        url: URL where the rate information was obtained
        access_date: When the rate information was last accessed
    """

    name: str = Field(description="Human-readable name of the rate source")
    url: str = Field(description="URL where the rate information was obtained")
    access_date: datetime = Field(
        description="When the rate information was last accessed"
    )

    @property
    def age(self) -> timedelta:
        """Calculate the age of this rate source."""
        return datetime.now() - self.access_date

    @property
    def is_stale(self) -> bool:
        """Check if this rate source is considered stale (>30 days)."""
        return self.age.days > 30


class ModelCostData(BaseModel):
    """Cost data for a specific model with comprehensive token support.

    Supports all major token types across providers:
    - Standard input/output tokens (all providers)
    - Cached tokens (OpenAI cache hits)
    - Thinking tokens (Anthropic Claude 3.7+ series)

    Attributes:
        provider: Provider name (e.g., 'openai', 'anthropic')
        model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet')
        input_cost_per_1k_tokens: Cost per 1000 input tokens in USD
        output_cost_per_1k_tokens: Cost per 1000 output tokens in USD
        cached_cost_per_1k_tokens: Cost per 1000 cached tokens in USD (OpenAI)
        thinking_cost_per_1k_tokens: Cost per 1000 thinking tokens in USD (Anthropic)
        source: Source information for this rate data
    """

    provider: str = Field(description="Provider name (e.g., 'openai', 'anthropic')")
    model: str = Field(
        description="Model identifier (e.g., 'gpt-4', 'claude-3-sonnet')"
    )
    input_cost_per_1k_tokens: float = Field(
        description="Cost per 1000 input tokens in USD"
    )
    output_cost_per_1k_tokens: float = Field(
        description="Cost per 1000 output tokens in USD"
    )
    cached_cost_per_1k_tokens: float | None = Field(
        default=None, description="Cost per 1000 cached tokens in USD (OpenAI)"
    )
    thinking_cost_per_1k_tokens: float | None = Field(
        default=None,
        description="Cost per 1000 thinking tokens in USD (Anthropic)",
    )
    source: RateSource = Field(description="Source information for this rate data")

    @field_validator("input_cost_per_1k_tokens", "output_cost_per_1k_tokens")  # type: ignore[misc]
    @classmethod
    def validate_required_costs(cls, v: float) -> float:
        """Ensure required costs are non-negative."""
        if v < 0:
            raise ValueError("Cost must be non-negative")
        return v

    @field_validator("cached_cost_per_1k_tokens", "thinking_cost_per_1k_tokens")  # type: ignore[misc]
    @classmethod
    def validate_optional_costs(cls, v: float | None) -> float | None:
        """Ensure optional costs are non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("Cost must be non-negative")
        return v


class FallbackCostCalculator:
    """Fallback cost calculator using rate database when API tracking unavailable.

    Maintains a database of model pricing rates with source attribution
    and freshness tracking. Provides cost estimation when real-time API
    tracking is not available.

    Attributes:
        _rate_database: Internal database of model cost rates
    """

    def __init__(self) -> None:
        """Initialize FallbackCostCalculator with empty rate database."""
        self._rate_database: dict[tuple[str, str], ModelCostData] = {}

    def load_default_rates(self) -> None:
        """Load default rate data for major providers and models from bundled JSON.

        Falls back to embedded defaults if the data file cannot be read.
        """
        self._rate_database = {}
        try:
            import json
            from importlib import resources

            with (
                resources.files("cellsem_llm_client.tracking")
                .joinpath("rates.json")
                .open("r", encoding="utf-8") as f
            ):
                data = json.load(f)

            for entry in data:
                source_data = entry.pop("source")
                source = RateSource(
                    name=source_data["name"],
                    url=source_data["url"],
                    access_date=datetime.fromisoformat(source_data["access_date"]),
                )
                rate = ModelCostData(source=source, **entry)
                self._rate_database[(rate.provider, rate.model)] = rate
            return
        except Exception:
            # Fall back to embedded defaults if reading bundled data fails
            default_source = RateSource(
                name="Provider Documentation",
                url="https://openai.com/pricing | https://anthropic.com/pricing",
                access_date=datetime.now(),
            )

            fallback_rates = [
                ModelCostData(
                    provider="openai",
                    model="gpt-4",
                    input_cost_per_1k_tokens=0.03,
                    output_cost_per_1k_tokens=0.06,
                    source=default_source,
                ),
                ModelCostData(
                    provider="openai",
                    model="gpt-3.5-turbo",
                    input_cost_per_1k_tokens=0.0015,
                    output_cost_per_1k_tokens=0.002,
                    source=default_source,
                ),
                ModelCostData(
                    provider="openai",
                    model="gpt-4o-mini",
                    input_cost_per_1k_tokens=0.00015,
                    output_cost_per_1k_tokens=0.0006,
                    cached_cost_per_1k_tokens=0.000075,
                    source=default_source,
                ),
                ModelCostData(
                    provider="anthropic",
                    model="claude-3-sonnet",
                    input_cost_per_1k_tokens=0.003,
                    output_cost_per_1k_tokens=0.015,
                    thinking_cost_per_1k_tokens=0.006,
                    source=default_source,
                ),
                ModelCostData(
                    provider="anthropic",
                    model="claude-3-haiku-20240307",
                    input_cost_per_1k_tokens=0.00025,
                    output_cost_per_1k_tokens=0.00125,
                    thinking_cost_per_1k_tokens=0.0005,
                    source=default_source,
                ),
            ]

            for rate in fallback_rates:
                self._rate_database[(rate.provider, rate.model)] = rate

    def get_model_rates(self, provider: str, model: str) -> ModelCostData | None:
        """Get rate data for a specific provider and model.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet')

        Returns:
            ModelCostData if found, None otherwise
        """
        return self._rate_database.get((provider, model))

    def calculate_cost(self, usage: UsageMetrics) -> float:
        """Calculate cost for given usage metrics using rate database.

        Args:
            usage: Usage metrics containing token counts and model information

        Returns:
            Calculated cost in USD

        Raises:
            CostCalculationError: If rate data not found for the model
        """
        rate_data = self.get_model_rates(usage.provider, usage.model)
        if not rate_data:
            raise CostCalculationException(
                f"No rate data found for {usage.provider}/{usage.model}",
                provider=usage.provider,
                model=usage.model,
                usage_metrics=usage.model_dump(),
            )

        # Calculate base cost (input + output tokens)
        input_cost = usage.input_tokens * rate_data.input_cost_per_1k_tokens / 1000
        output_cost = usage.output_tokens * rate_data.output_cost_per_1k_tokens / 1000
        total_cost = input_cost + output_cost

        # Add cached tokens cost if applicable
        if (
            usage.cached_tokens is not None
            and rate_data.cached_cost_per_1k_tokens is not None
        ):
            cached_cost = (
                usage.cached_tokens * rate_data.cached_cost_per_1k_tokens / 1000
            )
            total_cost += cached_cost

        # Add thinking tokens cost if applicable
        if (
            usage.thinking_tokens is not None
            and rate_data.thinking_cost_per_1k_tokens is not None
        ):
            thinking_cost = (
                usage.thinking_tokens * rate_data.thinking_cost_per_1k_tokens / 1000
            )
            total_cost += thinking_cost

        return total_cost

    def add_model_rates(self, rate_data: ModelCostData) -> None:
        """Add or update rate data for a model.

        Args:
            rate_data: Model cost data to add or update
        """
        key = (rate_data.provider, rate_data.model)
        self._rate_database[key] = rate_data

    def update_rates_from_source(self) -> int:
        """Update rates from external source.

        Returns:
            Number of rates updated

        Note:
            This is a placeholder for future implementation of automated
            rate updates from provider documentation or APIs.
        """
        updated_rates = self._fetch_latest_rates()
        count = 0

        for key, rate_data in updated_rates.items():
            self._rate_database[key] = rate_data
            count += 1

        return count

    def get_stale_rates(self) -> list[ModelCostData]:
        """Get list of rate data that is considered stale.

        Returns:
            List of ModelCostData with stale source information
        """
        return [
            rate_data
            for rate_data in self._rate_database.values()
            if rate_data.source.is_stale
        ]

    def _fetch_latest_rates(self) -> dict[tuple[str, str], ModelCostData]:
        """Fetch latest rates from external sources.

        Returns:
            Dictionary mapping (provider, model) to updated rate data

        Note:
            This is a placeholder for future implementation.
            In practice, this would scrape provider documentation
            or call provider pricing APIs.
        """
        # Placeholder implementation for testing
        return {}
