"""Unit tests for SchemaValidator - validation with retry logic."""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from cellsem_llm_client.schema.validators import (
    SchemaValidationResult,
    SchemaValidator,
)


class SampleModel(BaseModel):
    """Sample Pydantic model for validation tests."""

    name: str
    age: int
    email: str


@pytest.mark.unit
class TestSchemaValidator:
    """Test cases for the SchemaValidator class."""

    def test_validator_creation(self) -> None:
        """Test creating SchemaValidator with default configuration."""
        validator = SchemaValidator()

        assert validator is not None
        assert hasattr(validator, "validate_response")
        assert hasattr(validator, "validate_with_retry")

    def test_successful_validation(self) -> None:
        """Test successful validation of valid JSON response."""
        validator = SchemaValidator()
        response_text = '{"name": "John", "age": 30, "email": "john@example.com"}'

        result = validator.validate_response(response_text, SampleModel)

        assert isinstance(result, SchemaValidationResult)
        assert result.success is True
        assert result.model_instance.name == "John"  # type: ignore
        assert result.model_instance.age == 30  # type: ignore
        assert result.error is None
        assert result.retry_count == 0

    def test_validation_failure_invalid_json(self) -> None:
        """Test validation failure with invalid JSON."""
        validator = SchemaValidator()
        response_text = '{"name": "John", "age": "thirty"'  # Invalid JSON

        result = validator.validate_response(response_text, SampleModel)

        assert result.success is False
        assert result.model_instance is None
        assert result.error is not None
        assert result.error_category == "json_parse_error"

    def test_validation_failure_schema_mismatch(self) -> None:
        """Test validation failure with valid JSON but schema mismatch."""
        validator = SchemaValidator()
        response_text = '{"name": "John", "age": "thirty", "email": "john@example.com"}'

        result = validator.validate_response(response_text, SampleModel)

        assert result.success is False
        assert result.model_instance is None
        assert isinstance(result.error, ValidationError)

    def test_retry_logic_with_correctable_error(self) -> None:
        """Test retry logic that can correct validation errors."""
        validator = SchemaValidator()

        # Mock the retry strategy to fix the error
        def mock_retry_strategy(
            error: Exception, original_text: str, attempt: int
        ) -> str:
            if "age" in str(error):
                # Fix the age field
                data = json.loads(original_text)
                data["age"] = (
                    int(data["age"])
                    if isinstance(data["age"], str) and data["age"].isdigit()
                    else 25
                )
                return json.dumps(data)
            return original_text

        with patch.object(
            validator, "_apply_retry_strategy", side_effect=mock_retry_strategy
        ):
            response_text = '{"name": "John", "age": "not-a-number", "email": "john@example.com"}'  # Invalid age

            result = validator.validate_with_retry(
                response_text, SampleModel, max_retries=2
            )

            assert result.success is True
            assert result.model_instance is not None
            assert result.model_instance.age == 25
            assert result.retry_count == 1

    def test_retry_exhaustion(self) -> None:
        """Test that retry logic eventually gives up after max retries."""
        validator = SchemaValidator()

        # Mock retry strategy that doesn't fix the error
        def mock_failing_retry(
            error: Exception, original_text: str, attempt: int
        ) -> str:
            return original_text  # Don't fix anything

        with patch.object(
            validator, "_apply_retry_strategy", side_effect=mock_failing_retry
        ):
            response_text = (
                '{"name": "John", "age": "not-a-number", "email": "john@example.com"}'
            )

            result = validator.validate_with_retry(
                response_text, SampleModel, max_retries=3
            )

            assert result.success is False
            assert result.retry_count == 3
            assert result.model_instance is None

    def test_different_validation_strategies(self) -> None:
        """Test different validation strategies for different error types."""
        validator = SchemaValidator()

        # Test missing field strategy - create a ValidationError that would trigger missing field handling
        from pydantic import ValidationError

        try:
            SampleModel(name="John", age=30)  # type: ignore  # Missing email intentionally
        except ValidationError as e:
            result = validator._apply_retry_strategy(
                e,
                '{"name": "John", "age": 30}',  # Missing email
                attempt=1,
            )

            # The retry strategy should attempt to add the missing field
            assert "email" in result.lower() or len(result) > len(
                '{"name": "John", "age": 30}'
            )

    def test_validation_with_custom_error_handler(self) -> None:
        """Test validation with custom error handling strategy."""
        validator = SchemaValidator()

        def custom_handler(error: Exception, text: str, attempt: int) -> str:
            # Custom logic to fix specific error patterns
            if "email" in str(error):
                data = json.loads(text)
                if "email" not in data:
                    data["email"] = "default@example.com"
                return json.dumps(data)
            return text

        validator.set_custom_retry_handler(custom_handler)

        response_text = '{"name": "John", "age": 30}'  # Missing email

        result = validator.validate_with_retry(
            response_text, SampleModel, max_retries=1
        )

        assert result.success is True
        assert result.model_instance is not None
        assert result.model_instance.email == "default@example.com"

    def test_validation_performance_tracking(self) -> None:
        """Test that validation tracks performance metrics."""
        validator = SchemaValidator()
        response_text = '{"name": "John", "age": 30, "email": "john@example.com"}'

        result = validator.validate_response(response_text, SampleModel)

        assert hasattr(result, "validation_time_ms")
        assert result.validation_time_ms >= 0

    def test_complex_nested_schema_validation(self) -> None:
        """Test validation of complex nested schema."""

        class Address(BaseModel):
            street: str
            city: str
            zip_code: str

        class ComplexModel(BaseModel):
            user: SampleModel
            address: Address
            tags: list[str]

        validator = SchemaValidator()
        complex_response = {
            "user": {"name": "John", "age": 30, "email": "john@example.com"},
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "zip_code": "12345",
            },
            "tags": ["important", "customer"],
        }

        result = validator.validate_response(json.dumps(complex_response), ComplexModel)

        assert result.success is True
        assert result.model_instance is not None
        assert result.model_instance.user.name == "John"
        assert result.model_instance.address.city == "Anytown"
        assert "important" in result.model_instance.tags

    def test_validation_error_categorization(self) -> None:
        """Test that validation errors are properly categorized."""
        validator = SchemaValidator()

        # Test different error types
        test_cases = [
            ('{"invalid": json}', "json_parse_error"),
            ('{"name": 123, "age": 30, "email": "john@example.com"}', "type_error"),
            ('{"age": 30, "email": "john@example.com"}', "missing_field"),
        ]

        for response_text, expected_category in test_cases:
            result = validator.validate_response(response_text, SampleModel)

            assert result.success is False
            assert hasattr(result, "error_category")
            assert result.error_category == expected_category

    def test_validator_with_different_models(self) -> None:
        """Test validator works with different Pydantic model types."""

        class SimpleModel(BaseModel):
            value: str

        class ListModel(BaseModel):
            items: list[int]

        validator = SchemaValidator()

        # Test simple model
        simple_result = validator.validate_response('{"value": "test"}', SimpleModel)
        assert simple_result.success is True

        # Test list model
        list_result = validator.validate_response('{"items": [1, 2, 3]}', ListModel)
        assert list_result.success is True

    def test_validation_with_optional_fields(self) -> None:
        """Test validation with models that have optional fields."""

        class OptionalModel(BaseModel):
            required_field: str
            optional_field: str | None = None

        validator = SchemaValidator()

        # Test with optional field present
        result1 = validator.validate_response(
            '{"required_field": "test", "optional_field": "optional"}', OptionalModel
        )
        assert result1.success is True
        assert result1.model_instance is not None
        assert result1.model_instance.optional_field == "optional"

        # Test with optional field missing
        result2 = validator.validate_response(
            '{"required_field": "test"}', OptionalModel
        )
        assert result2.success is True
        assert result2.model_instance is not None
        assert result2.model_instance.optional_field is None
