"""Shared pytest configuration and fixtures for the test suite."""

import os
from typing import Any
from unittest.mock import Mock

import pytest

# Environment detection for integration testing strategy
USE_MOCKS = os.getenv("USE_MOCKS", "false").lower() == "true"
IS_CI = os.getenv("CI", "false").lower() == "true"


@pytest.fixture
def mock_api_response() -> Mock:
    """Mock API response for testing."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    return mock_response


@pytest.fixture
def sample_api_key() -> str:
    """Sample API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def mock_litellm_completion(monkeypatch: Any) -> Any:
    """Mock litellm.completion for integration tests when USE_MOCKS=true."""
    if USE_MOCKS:

        def mock_completion(
            model: str | None = None,
            messages: list[dict[str, str]] | None = None,
            max_tokens: int | None = None,
            **kwargs: Any,
        ) -> Mock:
            """Realistic mock of litellm.completion function."""
            mock_response = Mock()
            mock_response.choices = [Mock()]

            # Generate realistic responses based on model and input
            if messages and len(messages) > 0:
                user_message = messages[-1].get("content", "")
                if "Hello from OpenAI" in user_message:
                    content = "Hello from OpenAI!"
                elif "Hello from Claude" in user_message:
                    content = "Hello from Claude!"
                elif "OpenAI works" in user_message:
                    content = "OpenAI works!"
                elif "Claude works" in user_message:
                    content = "Claude works!"
                elif "2+2" in user_message or "2 + 2" in user_message:
                    content = "2 + 2 = 4"
                elif "capital of France" in user_message:
                    content = "Paris"
                elif "hello" in user_message.lower():
                    content = "Hello there!"
                elif user_message == "":
                    content = "Hello! How can I help you?"
                else:
                    content = f"Mock response to: {user_message[:50]}"
            else:
                content = "Mock response"

            mock_response.choices[0].message.content = content
            mock_response.model = model or "mocked-model"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 15
            mock_response.usage.total_tokens = 25
            # Ensure prompt_tokens_details is None to avoid mock attribute errors
            mock_response.usage.prompt_tokens_details = None

            return mock_response

        # Apply the mock to litellm.completion
        monkeypatch.setattr(
            "cellsem_llm_client.agents.agent_connection.completion", mock_completion
        )
        return mock_completion

    # If not using mocks, return None (no mocking applied)
    return None


class SecureTestConfig:
    """Test configuration that doesn't expose API keys in repr."""

    def __init__(
        self, using_mocks: bool, openai_key: str | None, anthropic_key: str | None
    ) -> None:
        self.using_mocks = using_mocks
        self._openai_key = openai_key
        self._anthropic_key = anthropic_key

    def __getitem__(self, key: str) -> Any:
        if key == "using_mocks":
            return self.using_mocks
        elif key == "openai_api_key":
            return self._openai_key
        elif key == "anthropic_api_key":
            return self._anthropic_key
        else:
            raise KeyError(key)

    def __repr__(self) -> str:
        return f"SecureTestConfig(using_mocks={self.using_mocks}, keys_available=True)"


@pytest.fixture
def integration_test_setup() -> SecureTestConfig:
    """Setup fixture for integration tests that handles API keys and mocking."""
    if USE_MOCKS:
        # When using mocks, we don't need real API keys
        return SecureTestConfig(
            using_mocks=True,
            openai_key="mock-openai-key",
            anthropic_key="mock-anthropic-key",
        )
    else:
        # When not using mocks, require real API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not openai_key and not anthropic_key:
            pytest.fail(
                "Integration tests require real API keys. Set OPENAI_API_KEY or "
                "ANTHROPIC_API_KEY environment variables, or run with USE_MOCKS=true"
            )

        return SecureTestConfig(
            using_mocks=False, openai_key=openai_key, anthropic_key=anthropic_key
        )


# Pytest configuration
pytest_plugins: list[str] = []
