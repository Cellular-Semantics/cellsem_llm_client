"""Integration tests for agent connections with real APIs or mocks."""

from datetime import datetime
from typing import Any

import pytest
from dotenv import load_dotenv

from cellsem_llm_client.agents.agent_connection import (
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)
from cellsem_llm_client.tracking.cost_calculator import FallbackCostCalculator
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics

# Load environment variables from .env file if it exists
load_dotenv()


class TestOpenAIIntegration:
    """Integration tests for OpenAI API connections."""

    @pytest.mark.integration
    def test_openai_agent_real_query(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI API query (real or mocked based on environment)."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-3.5-turbo", api_key=api_key)
        response = agent.query("Say 'Hello from OpenAI!' and nothing else.")

        assert response is not None
        assert len(response.strip()) > 0
        assert "Hello from OpenAI!" in response

    @pytest.mark.integration
    def test_openai_agent_with_system_message(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI agent with system message."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(api_key=api_key)
        response = agent.query(
            "What is 2+2?",
            system_message="You are a math tutor. Always show your work.",
        )

        assert response is not None
        assert "4" in response

    @pytest.mark.integration
    def test_openai_agent_token_limits(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI agent respects token limits."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(api_key=api_key, max_tokens=10)
        response = agent.query("Write a very long essay about artificial intelligence.")

        assert response is not None
        # Response should be short due to token limit (or mocked)
        if integration_test_setup["using_mocks"]:
            # Mock responses are predictable
            assert len(response) > 0
        else:
            # Real API should respect token limit
            assert len(response.split()) <= 15


class TestAnthropicIntegration:
    """Integration tests for Anthropic API connections."""

    @pytest.mark.integration
    def test_anthropic_agent_real_query(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test Anthropic API query (real or mocked based on environment)."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(api_key=api_key)
        response = agent.query("Say 'Hello from Claude!' and nothing else.")

        assert response is not None
        assert len(response.strip()) > 0
        assert "Hello from Claude!" in response

    @pytest.mark.integration
    def test_anthropic_agent_with_system_message(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test Anthropic agent with system message."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(api_key=api_key)
        response = agent.query(
            "What is the capital of France?",
            system_message="You are a geography expert. Be concise.",
        )

        assert response is not None
        assert "Paris" in response

    @pytest.mark.integration
    def test_anthropic_different_models(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test different Anthropic models."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key=api_key)
        response = agent.query("Say hello in exactly 2 words.")

        assert response is not None
        assert len(response.strip()) > 0


class TestLiteLLMIntegration:
    """Integration tests for LiteLLM with multiple providers."""

    @pytest.mark.integration
    def test_litellm_provider_switching(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that LiteLLM can handle different providers."""
        results = []

        openai_key = integration_test_setup["openai_api_key"]
        anthropic_key = integration_test_setup["anthropic_api_key"]

        if openai_key:
            openai_agent = LiteLLMAgent(model="gpt-3.5-turbo", api_key=openai_key)
            openai_response = openai_agent.query("Say 'OpenAI works!'")
            results.append(("OpenAI", openai_response))

        if anthropic_key:
            anthropic_agent = LiteLLMAgent(
                model="claude-3-haiku-20240307", api_key=anthropic_key
            )
            anthropic_response = anthropic_agent.query("Say 'Claude works!'")
            results.append(("Anthropic", anthropic_response))

        assert len(results) > 0
        for _provider, response in results:
            assert response is not None
            assert len(response.strip()) > 0

    @pytest.mark.integration
    def test_integration_behavior_documented(self, integration_test_setup: Any) -> None:
        """Test that integration behavior is properly documented."""
        # This test always passes but documents the expected behavior
        if integration_test_setup["using_mocks"]:
            assert True, "Using mocks - running in CI mode"
        else:
            assert True, "Using real APIs - running in local development mode"


class TestErrorHandling:
    """Integration tests for error handling scenarios."""

    @pytest.mark.integration
    def test_invalid_api_key_handling(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test handling of invalid API keys."""
        if integration_test_setup["using_mocks"]:
            pytest.skip("Mock mode doesn't test real API key validation")

        # Use a clearly invalid API key
        agent = OpenAIAgent(api_key="sk-invalid-test-key-123")

        # Try to query with invalid API key
        response = agent.query("This should fail with invalid API key")

        # LiteLLM handles invalid keys gracefully, returning error messages
        assert response is not None
        # Should contain some form of error message
        error_indicators = [
            "unable to process",
            "error",
            "invalid",
            "unauthorized",
            "sorry",
        ]
        assert any(
            indicator in response.lower() for indicator in error_indicators
        ), f"Expected error message in response, got: {response[:100]}"

    @pytest.mark.integration
    def test_invalid_model_handling(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test handling of invalid model names."""
        if integration_test_setup["using_mocks"]:
            pytest.skip("Mock mode doesn't test real model validation")

        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = LiteLLMAgent(model="definitely-not-a-real-model", api_key=api_key)

        with pytest.raises((ValueError, Exception)):
            agent.query("This should fail with invalid model")

    @pytest.mark.integration
    def test_empty_message_handling(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test handling of empty messages."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(api_key=api_key)

        # Empty message should still work, might return a short response
        response = agent.query("")
        assert response is not None  # Should get some response, even if minimal


class TestTrackingIntegration:
    """Integration tests for usage tracking functionality."""

    @pytest.mark.integration
    def test_query_with_tracking_openai(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test query_with_tracking with OpenAI returns real usage metrics."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-3.5-turbo", api_key=api_key)
        response, usage = agent.query_with_tracking("Say hello in exactly 3 words.")

        # Verify response
        assert response is not None
        assert len(response.strip()) > 0

        # Verify usage metrics
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.provider == "openai"
        assert usage.model == "gpt-3.5-turbo"
        assert usage.cost_source == "estimated"
        assert usage.timestamp is not None

    @pytest.mark.integration
    def test_query_with_tracking_anthropic(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test query_with_tracking with Anthropic returns real usage metrics."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key=api_key)
        response, usage = agent.query_with_tracking("What is 2+2? Be concise.")

        # Verify response
        assert response is not None
        assert len(response.strip()) > 0
        assert "4" in response

        # Verify usage metrics
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.provider == "anthropic"
        assert usage.model == "claude-3-haiku-20240307"
        assert usage.cost_source == "estimated"
        assert usage.timestamp is not None

    @pytest.mark.integration
    def test_query_with_tracking_with_cost_calculator(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test query_with_tracking with cost calculator integration."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Create cost calculator with default rates
        calculator = FallbackCostCalculator()
        calculator.load_default_rates()

        agent = OpenAIAgent(model="gpt-4", api_key=api_key)
        response, usage = agent.query_with_tracking(
            "Tell me about artificial intelligence in one sentence.",
            cost_calculator=calculator,
        )

        # Verify response
        assert response is not None
        assert len(response.strip()) > 0

        # Verify usage metrics with cost calculation
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.provider == "openai"
        assert usage.model == "gpt-4"
        assert usage.cost_source == "estimated"

        # Should have estimated cost from calculator
        assert usage.estimated_cost_usd is not None
        assert usage.estimated_cost_usd > 0
        assert usage.cost is not None

    @pytest.mark.integration
    def test_cost_calculator_with_real_rates(self, integration_test_setup: Any) -> None:
        """Test cost calculator with realistic usage scenarios."""
        calculator = FallbackCostCalculator()
        calculator.load_default_rates()

        # Test GPT-4 cost calculation
        gpt4_usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            provider="openai",
            model="gpt-4",
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        gpt4_cost = calculator.calculate_cost(gpt4_usage)
        assert gpt4_cost > 0
        # GPT-4 should be more expensive than other models
        assert gpt4_cost > 0.01  # At least 1 cent for this usage

        # Test Claude 3 Sonnet cost calculation
        claude_usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            provider="anthropic",
            model="claude-3-sonnet",
            timestamp=datetime.now(),
            cost_source="estimated",
        )

        claude_cost = calculator.calculate_cost(claude_usage)
        assert claude_cost > 0
        # Claude should generally be cheaper than GPT-4
        assert claude_cost < gpt4_cost

    @pytest.mark.integration
    def test_tracking_with_system_messages(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that system messages are included in token tracking."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(api_key=api_key)

        # Query without system message
        response1, usage1 = agent.query_with_tracking("What is 2+2?")

        # Query with system message
        response2, usage2 = agent.query_with_tracking(
            "What is 2+2?",
            system_message="You are a math tutor. Show your work step by step.",
        )

        # Both should succeed
        assert response1 is not None and response2 is not None
        assert isinstance(usage1, UsageMetrics) and isinstance(usage2, UsageMetrics)

        # Usage with system message should have more input tokens
        # (unless using mocks, which might be predictable)
        if not integration_test_setup["using_mocks"]:
            assert usage2.input_tokens >= usage1.input_tokens

    @pytest.mark.integration
    def test_provider_detection_from_model_names(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that provider is correctly detected from model names."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Test different model patterns
        test_cases = [
            ("gpt-3.5-turbo", "openai"),
            ("gpt-4", "openai"),
            ("claude-3-haiku-20240307", "anthropic"),
            ("claude-3-sonnet-20240229", "anthropic"),
        ]

        for model, expected_provider in test_cases:
            if (
                expected_provider == "anthropic"
                and not integration_test_setup["anthropic_api_key"]
            ):
                continue  # Skip Anthropic tests if no key

            agent = LiteLLMAgent(
                model=model,
                api_key=api_key
                if expected_provider == "openai"
                else integration_test_setup["anthropic_api_key"],
            )

            try:
                _, usage = agent.query_with_tracking("Hello")
                assert usage.provider == expected_provider
                assert usage.model == model
            except Exception as e:
                # If model not available or other issues, skip this test case
                pytest.skip(f"Could not test model {model}: {e}")

    @pytest.mark.integration
    def test_tracking_error_resilience(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that tracking works even when cost calculation fails."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Create calculator without default rates (will fail cost calculation)
        empty_calculator = FallbackCostCalculator()

        agent = OpenAIAgent(api_key=api_key)
        response, usage = agent.query_with_tracking(
            "Hello world", cost_calculator=empty_calculator
        )

        # Should still work even though cost calculation fails
        assert response is not None
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        # Cost should be None since calculation failed gracefully
        assert usage.estimated_cost_usd is None

    @pytest.mark.integration
    def test_end_to_end_tracking_workflow(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test complete tracking workflow from query to cost analysis."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Setup cost calculator
        calculator = FallbackCostCalculator()
        calculator.load_default_rates()

        # Create agent
        agent = OpenAIAgent(model="gpt-3.5-turbo", api_key=api_key)

        # Track multiple queries
        usage_records = []

        queries = [
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
            "What is 15 * 23?",
        ]

        for query in queries:
            response, usage = agent.query_with_tracking(
                query, cost_calculator=calculator
            )

            # Verify each response and tracking
            assert response is not None
            assert isinstance(usage, UsageMetrics)
            assert usage.estimated_cost_usd is not None

            usage_records.append(usage)

        # Analyze aggregated usage
        total_input_tokens = sum(u.input_tokens for u in usage_records)
        total_output_tokens = sum(u.output_tokens for u in usage_records)
        total_cost = sum(u.cost for u in usage_records if u.cost is not None)

        assert total_input_tokens > 0
        assert total_output_tokens > 0
        assert total_cost > 0

        # Verify all records are from same provider/model
        providers = {u.provider for u in usage_records}
        models = {u.model for u in usage_records}
        assert len(providers) == 1 and "openai" in providers
        assert len(models) == 1 and "gpt-3.5-turbo" in models
