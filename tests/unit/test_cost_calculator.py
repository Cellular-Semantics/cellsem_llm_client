"""Unit tests for FallbackCostCalculator class."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from cellsem_llm_client.tracking.cost_calculator import (
    CostCalculationError,
    FallbackCostCalculator,
    ModelCostData,
    RateSource,
)
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics


@pytest.mark.unit
class TestModelCostData:
    """Test cases for the ModelCostData data class."""

    def test_model_cost_data_creation(self) -> None:
        """Test creating ModelCostData with required fields."""
        source = RateSource(
            name="OpenAI Pricing Page",
            url="https://openai.com/pricing",
            access_date=datetime(2024, 1, 15),
        )

        cost_data = ModelCostData(
            provider="openai",
            model="gpt-4",
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
            source=source,
        )

        assert cost_data.provider == "openai"
        assert cost_data.model == "gpt-4"
        assert cost_data.input_cost_per_1k_tokens == 0.03
        assert cost_data.output_cost_per_1k_tokens == 0.06
        assert cost_data.source == source
        assert cost_data.cached_cost_per_1k_tokens is None
        assert cost_data.thinking_cost_per_1k_tokens is None

    def test_model_cost_data_with_all_fields(self) -> None:
        """Test creating ModelCostData with all optional fields."""
        source = RateSource(
            name="Anthropic Pricing",
            url="https://anthropic.com/pricing",
            access_date=datetime(2024, 1, 15),
        )

        cost_data = ModelCostData(
            provider="anthropic",
            model="claude-3-sonnet",
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            cached_cost_per_1k_tokens=0.0003,
            thinking_cost_per_1k_tokens=0.006,
            source=source,
        )

        assert cost_data.provider == "anthropic"
        assert cost_data.cached_cost_per_1k_tokens == 0.0003
        assert cost_data.thinking_cost_per_1k_tokens == 0.006

    def test_model_cost_data_validation(self) -> None:
        """Test that costs must be non-negative."""
        source = RateSource(
            name="Test Source",
            url="https://example.com",
            access_date=datetime.now(),
        )

        # Negative input cost should raise ValidationError
        with pytest.raises(ValueError):
            ModelCostData(
                provider="openai",
                model="gpt-4",
                input_cost_per_1k_tokens=-0.03,
                output_cost_per_1k_tokens=0.06,
                source=source,
            )


@pytest.mark.unit
class TestRateSource:
    """Test cases for the RateSource data class."""

    def test_rate_source_creation(self) -> None:
        """Test creating RateSource with all fields."""
        access_date = datetime(2024, 1, 15, 10, 30, 45)
        source = RateSource(
            name="OpenAI Pricing Page",
            url="https://openai.com/pricing",
            access_date=access_date,
        )

        assert source.name == "OpenAI Pricing Page"
        assert source.url == "https://openai.com/pricing"
        assert source.access_date == access_date

    def test_rate_source_age_property(self) -> None:
        """Test the age property calculation."""
        # Source from 5 days ago
        old_date = datetime.now() - timedelta(days=5)
        source = RateSource(
            name="Old Source",
            url="https://example.com",
            access_date=old_date,
        )

        assert source.age.days == 5

    def test_rate_source_is_stale_property(self) -> None:
        """Test the is_stale property for outdated sources."""
        # Fresh source (1 day old)
        fresh_source = RateSource(
            name="Fresh Source",
            url="https://example.com",
            access_date=datetime.now() - timedelta(days=1),
        )
        assert not fresh_source.is_stale

        # Stale source (40 days old, > 30 day threshold)
        stale_source = RateSource(
            name="Stale Source",
            url="https://example.com",
            access_date=datetime.now() - timedelta(days=40),
        )
        assert stale_source.is_stale


@pytest.mark.unit
class TestFallbackCostCalculator:
    """Test cases for the FallbackCostCalculator class."""

    def test_calculator_initialization(self) -> None:
        """Test basic initialization of FallbackCostCalculator."""
        calculator = FallbackCostCalculator()

        # Should have empty rate database initially
        assert len(calculator._rate_database) == 0

    def test_load_default_rates(self) -> None:
        """Test loading default rate data."""
        calculator = FallbackCostCalculator()
        calculator.load_default_rates()

        # Should have rates for major models
        assert len(calculator._rate_database) > 0

        # Should have OpenAI GPT-4 rates
        gpt4_rate = calculator.get_model_rates("openai", "gpt-4")
        assert gpt4_rate is not None
        assert gpt4_rate.input_cost_per_1k_tokens > 0
        assert gpt4_rate.output_cost_per_1k_tokens > 0

        # Should have Anthropic Claude rates
        claude_rate = calculator.get_model_rates("anthropic", "claude-3-sonnet")
        assert claude_rate is not None
        assert claude_rate.thinking_cost_per_1k_tokens is not None

    def test_get_model_rates_success(self) -> None:
        """Test successful model rate retrieval."""
        calculator = FallbackCostCalculator()
        source = RateSource(
            name="Test Source",
            url="https://example.com",
            access_date=datetime.now(),
        )

        # Add test rate data
        test_rate = ModelCostData(
            provider="test_provider",
            model="test_model",
            input_cost_per_1k_tokens=0.001,
            output_cost_per_1k_tokens=0.002,
            source=source,
        )
        calculator._rate_database[("test_provider", "test_model")] = test_rate

        result = calculator.get_model_rates("test_provider", "test_model")
        assert result == test_rate

    def test_get_model_rates_not_found(self) -> None:
        """Test model rate retrieval when model not found."""
        calculator = FallbackCostCalculator()

        result = calculator.get_model_rates("nonexistent_provider", "nonexistent_model")
        assert result is None

    def test_calculate_cost_success(self) -> None:
        """Test successful cost calculation."""
        calculator = FallbackCostCalculator()
        source = RateSource(
            name="Test Source",
            url="https://example.com",
            access_date=datetime.now(),
        )

        # Add test rate data
        test_rate = ModelCostData(
            provider="openai",
            model="gpt-4",
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
            source=source,
        )
        calculator._rate_database[("openai", "gpt-4")] = test_rate

        # Create usage metrics
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            provider="openai",
            model="gpt-4",
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        # Calculate cost
        cost = calculator.calculate_cost(usage)

        # Expected: 1000 * 0.03/1000 + 500 * 0.06/1000 = 0.03 + 0.03 = 0.06
        assert cost == pytest.approx(0.06)

    def test_calculate_cost_with_cached_tokens(self) -> None:
        """Test cost calculation with cached tokens."""
        calculator = FallbackCostCalculator()
        source = RateSource(
            name="Test Source",
            url="https://example.com",
            access_date=datetime.now(),
        )

        # Add test rate data with cached token pricing
        test_rate = ModelCostData(
            provider="openai",
            model="gpt-4",
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
            cached_cost_per_1k_tokens=0.0015,  # 50% discount
            source=source,
        )
        calculator._rate_database[("openai", "gpt-4")] = test_rate

        # Create usage metrics with cached tokens
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=200,
            provider="openai",
            model="gpt-4",
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        cost = calculator.calculate_cost(usage)

        # Expected: 1000 * 0.03/1000 + 500 * 0.06/1000 + 200 * 0.0015/1000
        # = 0.03 + 0.03 + 0.0003 = 0.0603
        assert cost == pytest.approx(0.0603)

    def test_calculate_cost_with_thinking_tokens(self) -> None:
        """Test cost calculation with thinking tokens (Anthropic)."""
        calculator = FallbackCostCalculator()
        source = RateSource(
            name="Test Source",
            url="https://example.com",
            access_date=datetime.now(),
        )

        # Add test rate data with thinking token pricing
        test_rate = ModelCostData(
            provider="anthropic",
            model="claude-3-sonnet",
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            thinking_cost_per_1k_tokens=0.006,
            source=source,
        )
        calculator._rate_database[("anthropic", "claude-3-sonnet")] = test_rate

        # Create usage metrics with thinking tokens
        usage = UsageMetrics(
            input_tokens=2000,
            output_tokens=1000,
            thinking_tokens=500,
            provider="anthropic",
            model="claude-3-sonnet",
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        cost = calculator.calculate_cost(usage)

        # Expected: 2000 * 0.003/1000 + 1000 * 0.015/1000 + 500 * 0.006/1000
        # = 0.006 + 0.015 + 0.003 = 0.024
        assert cost == pytest.approx(0.024)

    def test_calculate_cost_model_not_found(self) -> None:
        """Test cost calculation when model rates not found."""
        calculator = FallbackCostCalculator()

        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            provider="unknown_provider",
            model="unknown_model",
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        with pytest.raises(CostCalculationError, match="No rate data found"):
            calculator.calculate_cost(usage)

    def test_add_model_rates(self) -> None:
        """Test adding custom model rates."""
        calculator = FallbackCostCalculator()
        source = RateSource(
            name="Custom Source",
            url="https://example.com",
            access_date=datetime.now(),
        )

        rate_data = ModelCostData(
            provider="custom_provider",
            model="custom_model",
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.01,
            source=source,
        )

        calculator.add_model_rates(rate_data)

        # Should be able to retrieve the added rates
        retrieved = calculator.get_model_rates("custom_provider", "custom_model")
        assert retrieved == rate_data

    def test_update_rates_from_source(self) -> None:
        """Test updating rates from external source."""
        calculator = FallbackCostCalculator()

        # Mock the rate fetching
        with patch.object(calculator, "_fetch_latest_rates") as mock_fetch:
            mock_source = RateSource(
                name="Updated Source",
                url="https://example.com",
                access_date=datetime.now(),
            )

            mock_rates = {
                ("openai", "gpt-4"): ModelCostData(
                    provider="openai",
                    model="gpt-4",
                    input_cost_per_1k_tokens=0.025,  # Updated price
                    output_cost_per_1k_tokens=0.055,  # Updated price
                    source=mock_source,
                )
            }
            mock_fetch.return_value = mock_rates

            # Update rates
            updated_count = calculator.update_rates_from_source()

            assert updated_count == 1
            mock_fetch.assert_called_once()

            # Verify rates were updated
            gpt4_rates = calculator.get_model_rates("openai", "gpt-4")
            assert gpt4_rates is not None
            assert gpt4_rates.input_cost_per_1k_tokens == 0.025

    def test_get_stale_rates(self) -> None:
        """Test identification of stale rate data."""
        calculator = FallbackCostCalculator()

        # Add fresh and stale rate data
        fresh_source = RateSource(
            name="Fresh Source",
            url="https://example.com",
            access_date=datetime.now() - timedelta(days=5),
        )

        stale_source = RateSource(
            name="Stale Source",
            url="https://example.com",
            access_date=datetime.now() - timedelta(days=40),
        )

        fresh_rate = ModelCostData(
            provider="openai",
            model="gpt-4",
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
            source=fresh_source,
        )

        stale_rate = ModelCostData(
            provider="anthropic",
            model="claude-3-sonnet",
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            source=stale_source,
        )

        calculator.add_model_rates(fresh_rate)
        calculator.add_model_rates(stale_rate)

        stale_rates = calculator.get_stale_rates()

        assert len(stale_rates) == 1
        assert stale_rates[0] == stale_rate
