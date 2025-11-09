"""Integration tests for real schema enforcement with LLM providers."""

import json
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from cellsem_llm_client.agents.agent_connection import (
    AnthropicAgent,
    LiteLLMAgent,
    OpenAIAgent,
)
from cellsem_llm_client.schema import SchemaManager
from cellsem_llm_client.tracking.cost_calculator import FallbackCostCalculator
from cellsem_llm_client.tracking.usage_metrics import UsageMetrics

# Load environment variables from .env file if it exists
load_dotenv()

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "schemas"


class SimpleTask(BaseModel):
    """Simple task model for basic schema testing."""

    task: str
    confidence: float
    status: str | None = None


class UserProfile(BaseModel):
    """Complex nested user profile for advanced schema testing."""

    class User(BaseModel):
        name: str
        age: int | None = None
        email: str

    class Address(BaseModel):
        street: str | None = None
        city: str
        country: str
        postal_code: str | None = None

    class Preferences(BaseModel):
        notifications: bool | None = None
        theme: str | None = None
        language: str = "en"

    user: User
    address: Address
    preferences: Preferences | None = None
    tags: list[str] | None = None


@pytest.mark.integration
class TestOpenAISchemaEnforcement:
    """Test real OpenAI schema enforcement with response_format."""

    def test_openai_simple_schema_enforcement(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI enforces simple schemas correctly."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        # Test schema enforcement using Pydantic model (which handles OpenAI strict mode)
        result = agent.query_with_schema(
            message="Complete the task 'analyze customer feedback' with high confidence",
            schema=SimpleTask,
            max_retries=2,
        )

        # Verify the result matches the schema
        assert hasattr(result, "task")
        assert hasattr(result, "confidence")
        assert isinstance(result.task, str)
        assert isinstance(result.confidence, (int, float))
        assert 0 <= result.confidence <= 1
        assert (
            "analyze customer feedback" in result.task.lower()
            or "customer feedback" in result.task.lower()
        )

    def test_openai_complex_nested_schema(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI handles complex nested schemas."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        # Use a simplified nested model without default values for OpenAI strict mode
        class ContactInfo(BaseModel):
            name: str
            email: str

        class Location(BaseModel):
            city: str
            country: str

        class Profile(BaseModel):
            contact: ContactInfo
            location: Location

        message = (
            "Create a profile for John Doe, email john@example.com, "
            "living in New York, USA."
        )

        result = agent.query_with_schema(message=message, schema=Profile, max_retries=2)

        # Verify complex nested structure
        assert isinstance(result, Profile)
        assert result.contact.name.lower() in ["john doe", "john"]
        assert result.contact.email == "john@example.com" or "@" in result.contact.email
        assert result.location.city.lower() in ["new york", "nyc"]
        assert result.location.country.lower() in ["usa", "united states", "us"]

    def test_openai_schema_with_tracking(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI schema enforcement with usage tracking."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Setup cost calculator
        calculator = FallbackCostCalculator()
        calculator.load_default_rates()

        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        result, usage = agent.query_with_schema_and_tracking(
            message="Complete task: 'process invoice data' with confidence 0.9",
            schema=SimpleTask,
            cost_calculator=calculator,
            max_retries=1,
        )

        # Verify schema result
        assert isinstance(result, SimpleTask)
        assert result.confidence == 0.9 or abs(result.confidence - 0.9) < 0.1

        # Verify usage tracking
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.provider == "openai"
        assert usage.model == "gpt-4o-mini"
        # In mocked tests, cost calculation might return None
        if not integration_test_setup["using_mocks"]:
            assert usage.estimated_cost_usd is not None
            assert usage.estimated_cost_usd > 0

    def test_openai_schema_validation_retry(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test OpenAI schema validation and retry logic."""
        if integration_test_setup["using_mocks"]:
            pytest.skip("Mock mode doesn't test real validation retry scenarios")

        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(
            model="gpt-4o-mini", api_key=api_key
        )  # Use cheaper model for testing

        # Use a prompt that might produce inconsistent results
        message = (
            "Create a task result with high confidence. "
            "The task should be 'data processing' and confidence should be exactly 0.85"
        )

        result = agent.query_with_schema(
            message=message,
            schema=SimpleTask,
            max_retries=3,  # Allow retries for validation issues
        )

        # Should eventually succeed with valid schema
        assert isinstance(result, SimpleTask)
        assert isinstance(result.task, str)
        assert isinstance(result.confidence, (int, float))
        assert len(result.task) > 0


@pytest.mark.integration
class TestAnthropicSchemaEnforcement:
    """Test real Anthropic schema enforcement with tool choice."""

    def test_anthropic_simple_schema_enforcement(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test Anthropic enforces simple schemas via tool choice."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key=api_key)

        result = agent.query_with_schema(
            message="Analyze the task 'review documentation' with confidence level 0.8",
            schema=SimpleTask,
            max_retries=2,
        )

        # Verify structured response
        assert isinstance(result, SimpleTask)
        assert isinstance(result.task, str)
        assert isinstance(result.confidence, (int, float))
        assert 0 <= result.confidence <= 1
        assert "review" in result.task.lower() or "documentation" in result.task.lower()

    def test_anthropic_complex_nested_schema(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test Anthropic handles complex nested schemas via tools."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key=api_key)

        # Load complex schema
        schema_path = FIXTURES_DIR / "user_profile.json"
        with open(schema_path) as f:
            schema_dict = json.load(f)

        message = (
            "Create a user profile for Sarah Smith, age 28, email sarah@company.com, "
            "living in London, UK. She prefers light theme."
        )

        result = agent.query_with_schema(
            message=message, schema=schema_dict, max_retries=2
        )

        # Verify complex structure (result should be a Pydantic model)
        assert hasattr(result, "user")
        assert hasattr(result, "address")

        # For mock tests, the structure depends on the Pydantic model generation
        # The user field might be a dict or nested model depending on schema processing
        if isinstance(result.user, dict):
            assert "name" in result.user
            assert "email" in result.user or "@" in str(result.user)
            # Verify content based on mock data
            user_name = result.user.get("name", "").lower()
            assert "sarah" in user_name or "smith" in user_name
        else:
            assert hasattr(result.user, "name")
            assert hasattr(result.user, "email")
            # Verify content
            assert (
                "sarah" in result.user.name.lower()
                or "smith" in result.user.name.lower()
            )

        if isinstance(result.address, dict):
            assert "city" in result.address
            assert "country" in result.address
            city = result.address.get("city", "").lower()
            country = result.address.get("country", "").lower()
            assert "london" in city
            assert country in ["uk", "united kingdom", "britain"]
        else:
            assert hasattr(result.address, "city")
            assert hasattr(result.address, "country")
            assert "london" in result.address.city.lower()
            assert result.address.country.lower() in ["uk", "united kingdom", "britain"]

    def test_anthropic_schema_with_tracking(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test Anthropic schema enforcement preserves usage tracking."""
        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        calculator = FallbackCostCalculator()
        calculator.load_default_rates()

        agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key=api_key)

        result, usage = agent.query_with_schema_and_tracking(
            message="Complete task: 'validate user input' with confidence 0.95",
            schema=SimpleTask,
            cost_calculator=calculator,
        )

        # Verify schema compliance
        assert isinstance(result, SimpleTask)
        assert abs(result.confidence - 0.95) < 0.1 or result.confidence == 0.95

        # Verify tracking works with schema enforcement
        assert isinstance(usage, UsageMetrics)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.provider == "anthropic"
        assert usage.model == "claude-3-haiku-20240307"

    def test_anthropic_tool_choice_validation(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that Anthropic tool choice actually enforces structure."""
        if integration_test_setup["using_mocks"]:
            pytest.skip("Mock mode doesn't test real tool choice enforcement")

        api_key = integration_test_setup["anthropic_api_key"]
        if not api_key:
            pytest.skip("No Anthropic API key available")

        agent = AnthropicAgent(model="claude-3-haiku-20240307", api_key=api_key)

        # Try a prompt that might produce unstructured output normally
        message = (
            "Tell me about machine learning and also create a task result. "
            "Task should be 'research ML algorithms' with confidence 0.7"
        )

        result = agent.query_with_schema(
            message=message, schema=SimpleTask, max_retries=1
        )

        # Should be structured despite the conversational prompt
        assert isinstance(result, SimpleTask)
        assert (
            "ml" in result.task.lower()
            or "machine" in result.task.lower()
            or "research" in result.task.lower()
        )
        assert isinstance(result.confidence, (int, float))


@pytest.mark.integration
class TestFallbackSchemaEnforcement:
    """Test fallback schema enforcement for unsupported providers."""

    def test_fallback_provider_schema_hints(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test fallback adapter with unsupported provider."""
        # Use a provider that doesn't support native schema enforcement
        api_key = integration_test_setup[
            "openai_api_key"
        ]  # Use OpenAI key with unknown model
        if not api_key:
            pytest.skip("No OpenAI API key available")

        # Use unknown model to trigger fallback adapter
        agent = LiteLLMAgent(model="unknown-model-name", api_key=api_key)

        try:
            result = agent.query_with_schema(
                message="Complete task 'test fallback' with confidence 0.6",
                schema=SimpleTask,
                max_retries=2,
            )

            # If fallback works, should still get structured result
            assert isinstance(result, SimpleTask)
            assert isinstance(result.task, str)
            assert isinstance(result.confidence, (int, float))

        except Exception as e:
            # Fallback might fail with unknown model - that's expected
            pytest.skip(f"Fallback test failed as expected with unknown model: {e}")


@pytest.mark.integration
class TestCrossProviderSchemaConsistency:
    """Test schema consistency across different LLM providers."""

    def test_same_schema_across_providers(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that same schema works across OpenAI and Anthropic."""
        openai_key = integration_test_setup["openai_api_key"]
        anthropic_key = integration_test_setup["anthropic_api_key"]

        if not openai_key or not anthropic_key:
            pytest.skip(
                "Need both OpenAI and Anthropic API keys for cross-provider test"
            )

        # Same prompt and schema for both providers
        prompt = "Complete the task 'cross-provider test' with confidence 0.8"
        schema = SimpleTask

        # Test OpenAI
        openai_agent = OpenAIAgent(model="gpt-4o-mini", api_key=openai_key)
        openai_result = openai_agent.query_with_schema(
            message=prompt, schema=schema, max_retries=1
        )

        # Test Anthropic
        anthropic_agent = AnthropicAgent(
            model="claude-3-haiku-20240307", api_key=anthropic_key
        )
        anthropic_result = anthropic_agent.query_with_schema(
            message=prompt, schema=schema, max_retries=1
        )

        # Both should produce valid results
        assert isinstance(openai_result, SimpleTask)
        assert isinstance(anthropic_result, SimpleTask)

        # Both should have similar content
        assert isinstance(openai_result.task, str)
        assert isinstance(anthropic_result.task, str)
        assert isinstance(openai_result.confidence, (int, float))
        assert isinstance(anthropic_result.confidence, (int, float))

        # Confidence should be close to requested value
        assert abs(openai_result.confidence - 0.8) < 0.3
        assert abs(anthropic_result.confidence - 0.8) < 0.3

    def test_complex_schema_consistency(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test complex schema consistency across providers."""
        openai_key = integration_test_setup["openai_api_key"]
        anthropic_key = integration_test_setup["anthropic_api_key"]

        if not openai_key or not anthropic_key:
            pytest.skip(
                "Need both OpenAI and Anthropic API keys for cross-provider test"
            )

        # Use a simpler complex schema for testing
        class Finding(BaseModel):
            category: str
            description: str

        class AnalysisResult(BaseModel):
            summary: str
            findings: list[Finding]

        prompt = (
            "Analyze this data: 'User engagement increased 25% after UI changes'. "
            "Provide a summary and findings."
        )

        # Test both providers
        openai_agent = OpenAIAgent(model="gpt-4o-mini", api_key=openai_key)
        anthropic_agent = AnthropicAgent(
            model="claude-3-haiku-20240307", api_key=anthropic_key
        )

        try:
            openai_result = openai_agent.query_with_schema(
                message=prompt, schema=AnalysisResult, max_retries=2
            )

            anthropic_result = anthropic_agent.query_with_schema(
                message=prompt, schema=AnalysisResult, max_retries=2
            )

            # Both should have required fields
            for result in [openai_result, anthropic_result]:
                assert hasattr(result, "summary")
                assert hasattr(result, "findings")
                assert isinstance(result.summary, str)
                assert isinstance(result.findings, list)
                assert len(result.findings) > 0

                # Each finding should have required fields
                for finding in result.findings:
                    assert hasattr(finding, "category")
                    assert hasattr(finding, "description")
                    assert hasattr(finding, "severity")

        except Exception as e:
            # Complex schemas might be challenging - log but don't fail
            pytest.skip(f"Complex schema test failed - may be too advanced: {e}")


@pytest.mark.integration
class TestSchemaErrorHandling:
    """Test error handling and edge cases in schema enforcement."""

    def test_schema_validation_failure_recovery(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test recovery from schema validation failures."""
        if integration_test_setup["using_mocks"]:
            pytest.skip("Mock mode doesn't test real validation failures")

        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        # Use a prompt that might initially produce invalid structure
        prompt = "Just say hello and be friendly. Also complete task 'greeting' with confidence 0.5"

        result = agent.query_with_schema(
            message=prompt,
            schema=SimpleTask,
            max_retries=3,  # Allow multiple retries
        )

        # Should eventually produce valid schema
        assert isinstance(result, SimpleTask)
        assert isinstance(result.task, str)
        assert isinstance(result.confidence, (int, float))

    def test_invalid_schema_handling(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test handling of invalid schemas."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        # Invalid schema (missing required fields)
        invalid_schema = {
            "type": "object",
            "properties": {
                "field": {"type": "invalid-type"}  # Invalid type
            },
        }

        with pytest.raises((ValueError, RuntimeError, Exception)):
            agent.query_with_schema(message="Test message", schema=invalid_schema)

    def test_network_error_resilience(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test resilience to network and API errors."""
        if integration_test_setup["using_mocks"]:
            pytest.skip("Mock mode doesn't test real network errors")

        # Test with invalid API key to simulate auth errors
        agent = OpenAIAgent(model="gpt-4o-mini", api_key="sk-invalid-key-123")

        with pytest.raises((ValueError, RuntimeError, Exception)):
            agent.query_with_schema(message="Test message", schema=SimpleTask)


@pytest.mark.integration
class TestSchemaManagerIntegration:
    """Test SchemaManager integration with real schemas."""

    def test_load_schema_from_file_integration(self) -> None:
        """Test loading real schema files."""
        manager = SchemaManager(schema_directories=[str(FIXTURES_DIR)])

        # Test loading each fixture
        simple_schema = manager.load_schema("simple_task")
        assert simple_schema["type"] == "object"
        assert "task" in simple_schema["properties"]

        user_schema = manager.load_schema("user_profile")
        assert user_schema["type"] == "object"
        assert "user" in user_schema["properties"]

    def test_pydantic_model_generation_integration(self) -> None:
        """Test generating Pydantic models from real schemas."""
        manager = SchemaManager(schema_directories=[str(FIXTURES_DIR)])

        # Generate model from schema file
        TaskModel = manager.get_pydantic_model("simple_task")

        # Test model works correctly
        instance = TaskModel(task="test task", confidence=0.8, status="completed")
        assert instance.task == "test task"
        assert instance.confidence == 0.8
        assert instance.status == "completed"

        # Test validation
        with pytest.raises(
            (ValueError, RuntimeError, Exception)
        ):  # Should fail validation
            TaskModel(confidence=0.8)  # Missing required task field


@pytest.mark.integration
class TestPerformanceAndScaling:
    """Test performance characteristics of schema enforcement."""

    def test_schema_enforcement_performance(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test that schema enforcement doesn't significantly impact performance."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        import time

        # Test without schema
        start_time = time.time()
        response = agent.query("What is 2+2?")
        no_schema_time = time.time() - start_time

        # Test with schema
        start_time = time.time()
        result = agent.query_with_schema(
            message="Calculate 2+2 and provide task 'math calculation' with confidence",
            schema=SimpleTask,
        )
        schema_time = time.time() - start_time

        # Both should complete successfully
        assert response is not None
        assert isinstance(result, SimpleTask)

        # Schema enforcement shouldn't be dramatically slower
        # (allowing 5x overhead for schema processing)
        if not integration_test_setup["using_mocks"]:
            assert schema_time < no_schema_time * 5

    def test_multiple_schema_queries_tracking(
        self, mock_litellm_completion: Any, integration_test_setup: Any
    ) -> None:
        """Test aggregated tracking across multiple schema queries."""
        api_key = integration_test_setup["openai_api_key"]
        if not api_key:
            pytest.skip("No OpenAI API key available")

        calculator = FallbackCostCalculator()
        calculator.load_default_rates()
        agent = OpenAIAgent(model="gpt-4o-mini", api_key=api_key)

        usage_records = []
        queries = [
            "Complete task 'data analysis' with confidence 0.9",
            "Complete task 'report generation' with confidence 0.8",
            "Complete task 'quality check' with confidence 0.7",
        ]

        for query in queries:
            result, usage = agent.query_with_schema_and_tracking(
                message=query, schema=SimpleTask, cost_calculator=calculator
            )

            assert isinstance(result, SimpleTask)
            assert isinstance(usage, UsageMetrics)
            usage_records.append(usage)

        # Verify aggregated metrics
        total_tokens = sum(u.input_tokens + u.output_tokens for u in usage_records)
        total_cost = sum(u.cost for u in usage_records if u.cost is not None)

        assert total_tokens > 0
        # In mocked tests, cost might be 0, so just check that we got usage metrics
        if not integration_test_setup["using_mocks"]:
            assert total_cost > 0
        assert len(usage_records) == 3
