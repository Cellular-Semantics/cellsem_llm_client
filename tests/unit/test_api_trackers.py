"""Unit tests for ApiCostTracker class."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import requests

from cellsem_llm_client.tracking.api_trackers import (
    AnthropicUsageReport,
    ApiCostTracker,
    OpenAIUsageReport,
    UsageReportError,
)


@pytest.mark.unit
class TestApiCostTracker:
    """Test cases for the ApiCostTracker class."""

    def test_api_cost_tracker_initialization(self) -> None:
        """Test basic initialization of ApiCostTracker."""
        tracker = ApiCostTracker(
            openai_api_key="test_openai_key", anthropic_api_key="test_anthropic_key"
        )

        assert tracker.openai_api_key == "test_openai_key"
        assert tracker.anthropic_api_key == "test_anthropic_key"

    def test_api_cost_tracker_optional_keys(self) -> None:
        """Test ApiCostTracker with optional API keys."""
        # Only OpenAI key
        tracker1 = ApiCostTracker(openai_api_key="test_key")
        assert tracker1.openai_api_key == "test_key"
        assert tracker1.anthropic_api_key is None

        # Only Anthropic key
        tracker2 = ApiCostTracker(anthropic_api_key="test_key")
        assert tracker2.openai_api_key is None
        assert tracker2.anthropic_api_key == "test_key"

        # No keys (should work for mock testing)
        tracker3 = ApiCostTracker()
        assert tracker3.openai_api_key is None
        assert tracker3.anthropic_api_key is None

    @patch("requests.get")
    def test_get_openai_usage_success(self, mock_get: Mock) -> None:
        """Test successful OpenAI usage retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "organization.usage",
            "data": [
                {
                    "aggregation_timestamp": 1640995200,  # 2022-01-01
                    "n_requests": 10,
                    "operation": "chat.completions",
                    "snapshot_id": "snapshot_123",
                    "n_context_tokens_total": 5000,
                    "n_generated_tokens_total": 1500,
                }
            ],
        }
        mock_get.return_value = mock_response

        tracker = ApiCostTracker(openai_api_key="test_key")
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        usage_report = tracker.get_openai_usage(start_date, end_date)

        assert isinstance(usage_report, OpenAIUsageReport)
        assert len(usage_report.data) == 1
        assert usage_report.data[0]["n_requests"] == 10
        assert usage_report.data[0]["n_context_tokens_total"] == 5000
        assert usage_report.data[0]["n_generated_tokens_total"] == 1500

        # Verify API call was made correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "v1/organization/usage" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"

    @patch("requests.get")
    def test_get_openai_usage_no_key(self, mock_get: Mock) -> None:
        """Test OpenAI usage retrieval without API key."""
        tracker = ApiCostTracker()
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        with pytest.raises(UsageReportError, match="OpenAI API key not provided"):
            tracker.get_openai_usage(start_date, end_date)

        # Should not make any API calls
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_get_openai_usage_api_error(self, mock_get: Mock) -> None:
        """Test OpenAI usage retrieval with API error."""
        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_get.return_value = mock_response

        tracker = ApiCostTracker(openai_api_key="invalid_key")
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        with pytest.raises(UsageReportError, match="OpenAI API error"):
            tracker.get_openai_usage(start_date, end_date)

    @patch("requests.get")
    def test_get_openai_usage_request_exception(self, mock_get: Mock) -> None:
        """Test OpenAI usage retrieval with request exception."""
        # Mock request exception
        mock_get.side_effect = requests.RequestException("Network error")

        tracker = ApiCostTracker(openai_api_key="test_key")
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        with pytest.raises(UsageReportError, match="Failed to fetch OpenAI usage"):
            tracker.get_openai_usage(start_date, end_date)

    @patch("requests.get")
    def test_get_anthropic_usage_success(self, mock_get: Mock) -> None:
        """Test successful Anthropic usage retrieval."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "date": "2022-01-01",
                    "model": "claude-3-sonnet",
                    "input_tokens": 5000,
                    "output_tokens": 1500,
                    "thinking_tokens": 100,
                    "cost_usd": 0.045,
                }
            ]
        }
        mock_get.return_value = mock_response

        tracker = ApiCostTracker(anthropic_api_key="test_key")
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        usage_report = tracker.get_anthropic_usage(start_date, end_date)

        assert isinstance(usage_report, AnthropicUsageReport)
        assert len(usage_report.data) == 1
        assert usage_report.data[0]["model"] == "claude-3-sonnet"
        assert usage_report.data[0]["input_tokens"] == 5000
        assert usage_report.data[0]["output_tokens"] == 1500
        assert usage_report.data[0]["thinking_tokens"] == 100
        assert usage_report.data[0]["cost_usd"] == 0.045

        # Verify API call was made correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "v1/organizations/usage_report/messages" in call_args[0][0]
        assert call_args[1]["headers"]["x-api-key"] == "test_key"

    @patch("requests.get")
    def test_get_anthropic_usage_no_key(self, mock_get: Mock) -> None:
        """Test Anthropic usage retrieval without API key."""
        tracker = ApiCostTracker()
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)

        with pytest.raises(UsageReportError, match="Anthropic API key not provided"):
            tracker.get_anthropic_usage(start_date, end_date)

        # Should not make any API calls
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_date_validation(self, mock_get: Mock) -> None:
        """Test date range validation."""
        tracker = ApiCostTracker(openai_api_key="test_key")
        start_date = datetime(2022, 1, 2)
        end_date = datetime(2022, 1, 1)  # End before start

        with pytest.raises(
            UsageReportError, match="Start date must be before end date"
        ):
            tracker.get_openai_usage(start_date, end_date)

        # Should not make any API calls due to validation
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_date_range_too_large(self, mock_get: Mock) -> None:
        """Test validation of date range size."""
        tracker = ApiCostTracker(openai_api_key="test_key")
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 4, 2)  # 91 days, more than 90 days

        with pytest.raises(UsageReportError, match="Date range cannot exceed 90 days"):
            tracker.get_openai_usage(start_date, end_date)

        # Should not make any API calls due to validation
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_get_recent_usage_openai(self, mock_get: Mock) -> None:
        """Test getting recent usage for OpenAI (convenience method)."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "organization.usage",
            "data": [
                {
                    "aggregation_timestamp": int(datetime.now().timestamp()),
                    "n_requests": 5,
                    "operation": "chat.completions",
                    "snapshot_id": "snapshot_456",
                    "n_context_tokens_total": 2500,
                    "n_generated_tokens_total": 750,
                }
            ],
        }
        mock_get.return_value = mock_response

        tracker = ApiCostTracker(openai_api_key="test_key")
        usage_report = tracker.get_recent_usage("openai", hours=24)

        assert isinstance(usage_report, OpenAIUsageReport)
        assert len(usage_report.data) == 1

        # Verify the date range is approximately last 24 hours
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        start_time = datetime.fromisoformat(params["start_time"])
        end_time = datetime.fromisoformat(params["end_time"])
        time_diff = end_time - start_time
        assert abs(time_diff.total_seconds() - 24 * 3600) < 300  # Within 5 minutes

    @patch("requests.get")
    def test_get_recent_usage_anthropic(self, mock_get: Mock) -> None:
        """Test getting recent usage for Anthropic (convenience method)."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "date": datetime.now().date().isoformat(),
                    "model": "claude-3-haiku",
                    "input_tokens": 2500,
                    "output_tokens": 750,
                    "cost_usd": 0.025,
                }
            ]
        }
        mock_get.return_value = mock_response

        tracker = ApiCostTracker(anthropic_api_key="test_key")
        usage_report = tracker.get_recent_usage("anthropic", hours=24)

        assert isinstance(usage_report, AnthropicUsageReport)
        assert len(usage_report.data) == 1

    def test_get_recent_usage_unsupported_provider(self) -> None:
        """Test getting recent usage for unsupported provider."""
        tracker = ApiCostTracker()

        with pytest.raises(UsageReportError, match="Unsupported provider"):
            tracker.get_recent_usage("unsupported_provider")  # type: ignore[arg-type]


@pytest.mark.unit
class TestOpenAIUsageReport:
    """Test cases for OpenAIUsageReport data class."""

    def test_openai_usage_report_creation(self) -> None:
        """Test creation of OpenAIUsageReport."""
        data = [
            {
                "aggregation_timestamp": 1640995200,
                "n_requests": 10,
                "operation": "chat.completions",
                "snapshot_id": "snapshot_123",
                "n_context_tokens_total": 5000,
                "n_generated_tokens_total": 1500,
            }
        ]
        report = OpenAIUsageReport(data=data)

        assert len(report.data) == 1
        assert report.data[0]["n_requests"] == 10
        assert report.total_requests == 10
        assert report.total_input_tokens == 5000
        assert report.total_output_tokens == 1500

    def test_openai_usage_report_aggregation(self) -> None:
        """Test aggregation methods in OpenAIUsageReport."""
        data = [
            {
                "aggregation_timestamp": 1640995200,
                "n_requests": 10,
                "operation": "chat.completions",
                "snapshot_id": "snapshot_123",
                "n_context_tokens_total": 3000,
                "n_generated_tokens_total": 1000,
            },
            {
                "aggregation_timestamp": 1641081600,
                "n_requests": 15,
                "operation": "chat.completions",
                "snapshot_id": "snapshot_456",
                "n_context_tokens_total": 2000,
                "n_generated_tokens_total": 500,
            },
        ]
        report = OpenAIUsageReport(data=data)

        assert report.total_requests == 25
        assert report.total_input_tokens == 5000
        assert report.total_output_tokens == 1500
        assert report.total_tokens == 6500

    def test_openai_usage_report_empty_data(self) -> None:
        """Test OpenAIUsageReport with empty data."""
        report = OpenAIUsageReport(data=[])

        assert len(report.data) == 0
        assert report.total_requests == 0
        assert report.total_input_tokens == 0
        assert report.total_output_tokens == 0
        assert report.total_tokens == 0


@pytest.mark.unit
class TestAnthropicUsageReport:
    """Test cases for AnthropicUsageReport data class."""

    def test_anthropic_usage_report_creation(self) -> None:
        """Test creation of AnthropicUsageReport."""
        data = [
            {
                "date": "2022-01-01",
                "model": "claude-3-sonnet",
                "input_tokens": 5000,
                "output_tokens": 1500,
                "thinking_tokens": 100,
                "cost_usd": 0.045,
            }
        ]
        report = AnthropicUsageReport(data=data)

        assert len(report.data) == 1
        assert report.data[0]["model"] == "claude-3-sonnet"
        assert report.total_input_tokens == 5000
        assert report.total_output_tokens == 1500
        assert report.total_thinking_tokens == 100
        assert report.total_cost_usd == 0.045

    def test_anthropic_usage_report_aggregation(self) -> None:
        """Test aggregation methods in AnthropicUsageReport."""
        data = [
            {
                "date": "2022-01-01",
                "model": "claude-3-sonnet",
                "input_tokens": 3000,
                "output_tokens": 1000,
                "thinking_tokens": 50,
                "cost_usd": 0.025,
            },
            {
                "date": "2022-01-02",
                "model": "claude-3-haiku",
                "input_tokens": 2000,
                "output_tokens": 500,
                "thinking_tokens": 25,
                "cost_usd": 0.015,
            },
        ]
        report = AnthropicUsageReport(data=data)

        assert report.total_input_tokens == 5000
        assert report.total_output_tokens == 1500
        assert report.total_thinking_tokens == 75
        assert report.total_cost_usd == 0.04
        assert report.total_tokens == 6575  # 5000 + 1500 + 75

    def test_anthropic_usage_report_without_thinking_tokens(self) -> None:
        """Test AnthropicUsageReport handling missing thinking tokens."""
        data = [
            {
                "date": "2022-01-01",
                "model": "claude-3-sonnet",
                "input_tokens": 5000,
                "output_tokens": 1500,
                "cost_usd": 0.045,
                # No thinking_tokens field
            }
        ]
        report = AnthropicUsageReport(data=data)

        assert report.total_input_tokens == 5000
        assert report.total_output_tokens == 1500
        assert report.total_thinking_tokens == 0  # Should handle missing gracefully
        assert report.total_tokens == 6500
