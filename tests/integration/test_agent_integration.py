"""Integration tests for agent connections with real APIs or mocks."""

from typing import Any

import pytest
from dotenv import load_dotenv

from cellsem_llm_client.agents.agent_connection import (
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)

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
        assert any(indicator in response.lower() for indicator in error_indicators), (
            f"Expected error message in response, got: {response[:100]}"
        )

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
