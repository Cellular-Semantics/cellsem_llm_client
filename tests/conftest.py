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
    """Legacy mock fixture - kept for unit test compatibility only."""
    # Integration tests now use real APIs only
    # This fixture is only used by unit tests that may need simple mocking
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
    """Setup fixture for integration tests - requires real API keys."""
    # Integration tests always require real API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        pytest.fail(
            "Integration tests require real API keys. Set OPENAI_API_KEY or "
            "ANTHROPIC_API_KEY environment variables."
        )

    return SecureTestConfig(
        using_mocks=False, openai_key=openai_key, anthropic_key=anthropic_key
    )


# Pytest configuration
pytest_plugins: list[str] = []
