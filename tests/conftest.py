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
            import json

            mock_response = Mock()
            mock_response.choices = [Mock()]

            # Generate realistic responses based on model and input
            if messages and len(messages) > 0:
                user_message = messages[-1].get("content", "")

                # Handle schema-enhanced queries
                has_response_format = "response_format" in kwargs
                has_tools = "tools" in kwargs

                if has_response_format or has_tools or "schema" in user_message.lower():
                    # Generate structured JSON responses for schema queries
                    if (
                        "task" in user_message.lower()
                        and "confidence" in user_message.lower()
                    ):
                        # SimpleTask schema response
                        if "analyze customer feedback" in user_message.lower():
                            content = '{"task": "analyze customer feedback", "confidence": 0.9, "status": "completed"}'
                        elif "review documentation" in user_message.lower():
                            content = '{"task": "review documentation", "confidence": 0.8, "status": "completed"}'
                        elif "data processing" in user_message.lower():
                            content = '{"task": "data processing", "confidence": 0.85, "status": "completed"}'
                        elif "cross-provider test" in user_message.lower():
                            content = '{"task": "cross-provider test", "confidence": 0.8, "status": "completed"}'
                        elif "validate user input" in user_message.lower():
                            content = '{"task": "validate user input", "confidence": 0.95, "status": "completed"}'
                        elif "research ML algorithms" in user_message.lower():
                            content = '{"task": "research ML algorithms", "confidence": 0.7, "status": "completed"}'
                        elif "greeting" in user_message.lower():
                            content = '{"task": "greeting", "confidence": 0.5, "status": "completed"}'
                        elif "math calculation" in user_message.lower():
                            content = '{"task": "math calculation", "confidence": 0.95, "status": "completed"}'
                        elif "test fallback" in user_message.lower():
                            content = '{"task": "test fallback", "confidence": 0.6, "status": "completed"}'
                        else:
                            # Generic task response
                            content = '{"task": "generic task", "confidence": 0.8, "status": "completed"}'

                    elif (
                        "user profile" in user_message.lower()
                        or "john doe" in user_message.lower()
                        or "sarah smith" in user_message.lower()
                    ):
                        # UserProfile schema response
                        if "john doe" in user_message.lower():
                            content = json.dumps(
                                {
                                    "user": {
                                        "name": "John Doe",
                                        "age": 30,
                                        "email": "john@example.com",
                                    },
                                    "address": {
                                        "street": "123 Main St",
                                        "city": "New York",
                                        "country": "USA",
                                        "postal_code": "10001",
                                    },
                                    "preferences": {
                                        "notifications": True,
                                        "theme": "dark",
                                        "language": "en",
                                    },
                                    "tags": ["customer", "premium"],
                                }
                            )
                        elif "sarah smith" in user_message.lower():
                            content = json.dumps(
                                {
                                    "user": {
                                        "name": "Sarah Smith",
                                        "age": 28,
                                        "email": "sarah@company.com",
                                    },
                                    "address": {
                                        "street": "456 Oak Ave",
                                        "city": "London",
                                        "country": "UK",
                                        "postal_code": "SW1A 1AA",
                                    },
                                    "preferences": {
                                        "notifications": False,
                                        "theme": "light",
                                        "language": "en",
                                    },
                                    "tags": ["employee", "developer"],
                                }
                            )
                        else:
                            content = json.dumps(
                                {
                                    "user": {
                                        "name": "Test User",
                                        "age": 25,
                                        "email": "test@example.com",
                                    },
                                    "address": {
                                        "city": "Test City",
                                        "country": "Test Country",
                                    },
                                }
                            )

                    elif (
                        "analysis" in user_message.lower()
                        and "engagement" in user_message.lower()
                    ):
                        # Analysis result schema response
                        content = json.dumps(
                            {
                                "summary": "User engagement analysis shows positive trends after UI changes",
                                "findings": [
                                    {
                                        "category": "positive",
                                        "description": "25% increase in user engagement metrics",
                                        "severity": 8,
                                        "details": {
                                            "metric": "engagement",
                                            "change": 0.25,
                                        },
                                    },
                                    {
                                        "category": "neutral",
                                        "description": "UI changes were well received by users",
                                        "severity": 5,
                                        "details": {"feedback_score": 4.2},
                                    },
                                ],
                                "recommendations": [
                                    "Continue monitoring engagement metrics",
                                    "Consider A/B testing additional UI improvements",
                                ],
                                "metadata": {
                                    "analysis_type": "engagement_analysis",
                                    "confidence_score": 0.85,
                                },
                            }
                        )
                    else:
                        # Default structured response
                        content = (
                            '{"task": "mock structured response", "confidence": 0.8}'
                        )

                # Handle Anthropic tool calls
                if has_tools:
                    # Mock tool call response for Anthropic
                    tool_call = Mock()
                    tool_call.function = Mock()
                    tool_call.function.arguments = content
                    mock_response.choices[0].message.tool_calls = [tool_call]
                    mock_response.choices[0].message.content = None
                else:
                    # Regular responses
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
                    elif not (
                        has_response_format
                        or has_tools
                        or "schema" in user_message.lower()
                    ):
                        content = f"Mock response to: {user_message[:50]}"

                mock_response.choices[0].message.content = content

            else:
                mock_response.choices[0].message.content = "Mock response"

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
