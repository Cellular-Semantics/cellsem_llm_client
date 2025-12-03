"""Unit tests for UsageMetrics data class."""

from datetime import datetime
from typing import Literal

import pytest
from pydantic import ValidationError

from cellsem_llm_client.tracking.usage_metrics import UsageMetrics


@pytest.mark.unit
class TestUsageMetrics:
    """Test cases for the UsageMetrics data class."""

    def test_basic_usage_metrics_creation(self) -> None:
        """Test creating basic UsageMetrics with required fields."""
        timestamp = datetime.now()
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
        )

        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.provider == "openai"
        assert metrics.model == "gpt-4"
        assert metrics.timestamp == timestamp
        assert metrics.cost_source == "api"
        assert metrics.cached_tokens is None
        assert metrics.thinking_tokens is None
        assert metrics.actual_cost_usd is None
        assert metrics.estimated_cost_usd is None
        assert metrics.rate_last_updated is None

    def test_usage_metrics_with_all_fields(self) -> None:
        """Test creating UsageMetrics with all optional fields populated."""
        timestamp = datetime.now()
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=25,
            thinking_tokens=10,
            actual_cost_usd=0.0035,
            estimated_cost_usd=0.004,
            cost_source="estimated",
            provider="anthropic",
            model="claude-3-sonnet",
            timestamp=timestamp,
        )

        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.cached_tokens == 25
        assert metrics.thinking_tokens == 10
        assert metrics.actual_cost_usd == 0.0035
        assert metrics.estimated_cost_usd == 0.004
        assert metrics.cost_source == "estimated"
        assert metrics.provider == "anthropic"
        assert metrics.model == "claude-3-sonnet"
        assert metrics.timestamp == timestamp

    def test_usage_metrics_validation_positive_tokens(self) -> None:
        """Test that token counts must be non-negative."""
        timestamp = datetime.now()

        # Negative input tokens should raise ValidationError
        with pytest.raises(ValidationError):
            UsageMetrics(
                input_tokens=-1,
                output_tokens=50,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source="api",
            )

        # Negative output tokens should raise ValidationError
        with pytest.raises(ValidationError):
            UsageMetrics(
                input_tokens=100,
                output_tokens=-1,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source="api",
            )

        # Negative cached tokens should raise ValidationError
        with pytest.raises(ValidationError):
            UsageMetrics(
                input_tokens=100,
                output_tokens=50,
                cached_tokens=-5,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source="api",
            )

    def test_usage_metrics_validation_cost_source(self) -> None:
        """Test that cost_source must be either 'api' or 'estimated'."""
        timestamp = datetime.now()

        # Valid cost sources should work
        valid_sources: tuple[Literal["api"], Literal["estimated"]] = (
            "api",
            "estimated",
        )
        for source in valid_sources:
            metrics = UsageMetrics(
                input_tokens=100,
                output_tokens=50,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source=source,
            )
            assert metrics.cost_source == source

        # Invalid cost source should raise ValidationError
        with pytest.raises(ValidationError):
            UsageMetrics(
                input_tokens=100,
                output_tokens=50,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source="invalid",  # type: ignore
            )

    def test_usage_metrics_validation_positive_costs(self) -> None:
        """Test that cost amounts must be non-negative."""
        timestamp = datetime.now()

        # Negative actual cost should raise ValidationError
        with pytest.raises(ValidationError):
            UsageMetrics(
                input_tokens=100,
                output_tokens=50,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source="api",
                actual_cost_usd=-0.001,
            )

        # Negative estimated cost should raise ValidationError
        with pytest.raises(ValidationError):
            UsageMetrics(
                input_tokens=100,
                output_tokens=50,
                provider="openai",
                model="gpt-4",
                timestamp=timestamp,
                cost_source="estimated",
                estimated_cost_usd=-0.001,
            )

    def test_usage_metrics_total_tokens_property(self) -> None:
        """Test the total_tokens property calculation."""
        timestamp = datetime.now()

        # Basic total without optional tokens
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
        )
        assert metrics.total_tokens == 150

        # Total with cached tokens
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=25,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
        )
        assert metrics.total_tokens == 175

        # Total with thinking tokens
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            thinking_tokens=10,
            provider="anthropic",
            model="claude-3-sonnet",
            timestamp=timestamp,
            cost_source="api",
        )
        assert metrics.total_tokens == 160

        # Total with all token types
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=25,
            thinking_tokens=10,
            provider="anthropic",
            model="claude-3-sonnet",
            timestamp=timestamp,
            cost_source="api",
        )
        assert metrics.total_tokens == 185

    def test_usage_metrics_cost_property(self) -> None:
        """Test the cost property that returns actual or estimated cost."""
        timestamp = datetime.now()

        # When actual cost is available, return it
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
            actual_cost_usd=0.003,
            estimated_cost_usd=0.0035,
        )
        assert metrics.cost == 0.003

        # When only estimated cost is available, return it
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="estimated",
            estimated_cost_usd=0.0035,
        )
        assert metrics.cost == 0.0035

        # When no cost is available, return None
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
        )
        assert metrics.cost is None

    def test_usage_metrics_serialization(self) -> None:
        """Test that UsageMetrics can be properly serialized and deserialized."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45)
        original_metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=25,
            thinking_tokens=10,
            actual_cost_usd=0.003,
            estimated_cost_usd=0.0035,
            cost_source="api",
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
        )

        # Serialize to dict
        metrics_dict = original_metrics.model_dump()
        expected_dict = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 25,
            "thinking_tokens": 10,
            "actual_cost_usd": 0.003,
            "estimated_cost_usd": 0.0035,
            "rate_last_updated": None,
            "cost_source": "api",
            "provider": "openai",
            "model": "gpt-4",
            "timestamp": timestamp,
        }
        assert metrics_dict == expected_dict

        # Deserialize back to object
        restored_metrics = UsageMetrics.model_validate(metrics_dict)
        assert restored_metrics == original_metrics

    def test_usage_metrics_json_serialization(self) -> None:
        """Test JSON serialization and deserialization."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45)
        original_metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
            actual_cost_usd=0.003,
        )

        # Serialize to JSON
        json_str = original_metrics.model_dump_json()
        assert isinstance(json_str, str)
        assert "input_tokens" in json_str
        assert "100" in json_str

        # Deserialize from JSON string
        import json

        json_dict = json.loads(json_str)
        restored_metrics = UsageMetrics.model_validate(json_dict)
        assert restored_metrics.input_tokens == original_metrics.input_tokens
        assert restored_metrics.output_tokens == original_metrics.output_tokens
        assert restored_metrics.provider == original_metrics.provider

    def test_usage_metrics_anthropic_thinking_tokens(self) -> None:
        """Test specific support for Anthropic thinking tokens (Claude 4.1+ feature)."""
        timestamp = datetime.now()
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            thinking_tokens=25,  # Anthropic-specific
            provider="anthropic",
            model="claude-3.5-sonnet",
            timestamp=timestamp,
            cost_source="api",
        )

        assert metrics.thinking_tokens == 25
        assert metrics.total_tokens == 175  # 100 + 50 + 25
        assert metrics.provider == "anthropic"

    def test_usage_metrics_openai_cached_tokens(self) -> None:
        """Test specific support for OpenAI cached tokens feature."""
        timestamp = datetime.now()
        metrics = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=30,  # OpenAI cache hit
            provider="openai",
            model="gpt-4",
            timestamp=timestamp,
            cost_source="api",
        )

        assert metrics.cached_tokens == 30
        assert metrics.total_tokens == 180  # 100 + 50 + 30
        assert metrics.provider == "openai"
