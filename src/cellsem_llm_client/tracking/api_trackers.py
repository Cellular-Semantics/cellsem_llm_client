"""Real-time API cost tracking for OpenAI and Anthropic."""

from datetime import datetime, timedelta
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field


class UsageReportError(Exception):
    """Exception raised when usage report retrieval fails."""

    pass


class OpenAIUsageReport(BaseModel):
    """Usage report from OpenAI Usage API.

    Represents aggregated usage data from OpenAI's /v1/organization/usage endpoint.
    Data is typically available within 5 minutes of API calls.

    Attributes:
        data: List of usage records with timestamps, token counts, and request metrics
    """

    data: list[dict[str, Any]] = Field(
        description="Raw usage data from OpenAI Usage API"
    )

    @property
    def total_requests(self) -> int:
        """Total number of API requests across all records."""
        return sum(record.get("n_requests", 0) for record in self.data)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens (context tokens) across all records."""
        return sum(record.get("n_context_tokens_total", 0) for record in self.data)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens (generated tokens) across all records."""
        return sum(record.get("n_generated_tokens_total", 0) for record in self.data)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all records."""
        return self.total_input_tokens + self.total_output_tokens


class AnthropicUsageReport(BaseModel):
    """Usage report from Anthropic Usage API.

    Represents aggregated usage data from Anthropic's usage_report/messages endpoint.
    Includes support for thinking tokens (Claude 4.1+ series).

    Attributes:
        data: List of usage records with dates, models, token counts, and costs
    """

    data: list[dict[str, Any]] = Field(
        description="Raw usage data from Anthropic Usage API"
    )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all records."""
        return sum(record.get("input_tokens", 0) for record in self.data)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all records."""
        return sum(record.get("output_tokens", 0) for record in self.data)

    @property
    def total_thinking_tokens(self) -> int:
        """Total thinking tokens across all records (Claude 4.1+ feature)."""
        return sum(record.get("thinking_tokens", 0) for record in self.data)

    @property
    def total_cost_usd(self) -> float:
        """Total cost in USD across all records."""
        return sum(record.get("cost_usd", 0.0) for record in self.data)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output + thinking) across all records."""
        return (
            self.total_input_tokens
            + self.total_output_tokens
            + self.total_thinking_tokens
        )


class ApiCostTracker:
    """Real-time API cost tracking for OpenAI and Anthropic.

    Fetches actual usage data from provider APIs for accurate cost monitoring.
    Both providers offer real-time usage APIs with ~5 minute data availability.

    Attributes:
        openai_api_key: OpenAI API key for usage tracking
        anthropic_api_key: Anthropic API key for usage tracking

    Example:
        ```python
        from datetime import datetime, timedelta
        from cellsem_llm_client.tracking.api_trackers import ApiCostTracker

        # Initialize with API keys
        tracker = ApiCostTracker(
            openai_api_key="sk-...",
            anthropic_api_key="sk-ant-..."
        )

        # Get recent usage (last 24 hours)
        recent_usage = tracker.get_recent_usage("openai", hours=24)
        print(f"Total tokens: {recent_usage.total_tokens}")
        print(f"Total requests: {recent_usage.total_requests}")

        # Get usage for specific date range
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        usage = tracker.get_openai_usage(start, end)
        ```
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        """Initialize ApiCostTracker with optional provider API keys.

        Args:
            openai_api_key: OpenAI API key for usage tracking
            anthropic_api_key: Anthropic API key for usage tracking

        Note:
            At least one API key should be provided for meaningful functionality.
            Keys can be None for testing with mocks.
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

    def get_openai_usage(
        self, start_date: datetime, end_date: datetime
    ) -> OpenAIUsageReport:
        """Get OpenAI usage data for specified date range.

        Calls OpenAI's /v1/organization/usage endpoint to fetch real usage metrics.
        Data includes token counts, request counts, and operation types.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (exclusive)

        Returns:
            OpenAIUsageReport with aggregated usage data

        Raises:
            UsageReportError: If API key is missing, API call fails, or dates are invalid

        Note:
            - Maximum date range is 90 days
            - Data is typically available within 5 minutes of API calls
            - Times are in UTC
        """
        if not self.openai_api_key:
            raise UsageReportError("OpenAI API key not provided")

        self._validate_date_range(start_date, end_date)

        url = "https://api.openai.com/v1/organization/usage"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        params = {
            "start_time": start_date.isoformat(),
            "end_time": end_date.isoformat(),
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                raise UsageReportError(
                    f"OpenAI API error: {response.status_code} - {response.text}"
                )

            data = response.json()
            return OpenAIUsageReport(data=data.get("data", []))

        except requests.RequestException as e:
            raise UsageReportError(f"Failed to fetch OpenAI usage: {e}") from e

    def get_anthropic_usage(
        self, start_date: datetime, end_date: datetime
    ) -> AnthropicUsageReport:
        """Get Anthropic usage data for specified date range.

        Calls Anthropic's /v1/organizations/usage_report/messages endpoint to fetch
        real usage metrics including support for thinking tokens (Claude 4.1+).

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (exclusive)

        Returns:
            AnthropicUsageReport with aggregated usage data

        Raises:
            UsageReportError: If API key is missing, API call fails, or dates are invalid

        Note:
            - Maximum date range is 90 days
            - Data includes thinking tokens for Claude 4.1+ models
            - Costs are provided in USD
        """
        if not self.anthropic_api_key:
            raise UsageReportError("Anthropic API key not provided")

        self._validate_date_range(start_date, end_date)

        url = "https://api.anthropic.com/v1/organizations/usage_report/messages"
        headers = {"x-api-key": self.anthropic_api_key}
        params = {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                raise UsageReportError(
                    f"Anthropic API error: {response.status_code} - {response.text}"
                )

            data = response.json()
            return AnthropicUsageReport(data=data.get("data", []))

        except requests.RequestException as e:
            raise UsageReportError(f"Failed to fetch Anthropic usage: {e}") from e

    def get_recent_usage(
        self,
        provider: Literal["openai", "anthropic"],
        hours: int = 24,
    ) -> OpenAIUsageReport | AnthropicUsageReport:
        """Get recent usage data for specified provider.

        Convenience method to fetch usage for the last N hours.

        Args:
            provider: Provider to fetch usage from ("openai" or "anthropic")
            hours: Number of hours to look back (default: 24)

        Returns:
            Usage report for the specified provider

        Raises:
            UsageReportError: If provider is unsupported or API call fails
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        if provider == "openai":
            return self.get_openai_usage(start_time, end_time)
        elif provider == "anthropic":
            return self.get_anthropic_usage(start_time, end_time)
        else:
            raise UsageReportError(f"Unsupported provider: {provider}")

    def _validate_date_range(self, start_date: datetime, end_date: datetime) -> None:
        """Validate date range parameters.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Raises:
            UsageReportError: If date range is invalid
        """
        if start_date >= end_date:
            raise UsageReportError("Start date must be before end date")

        max_range = timedelta(days=90)
        if end_date - start_date > max_range:
            raise UsageReportError("Date range cannot exceed 90 days")
