"""Schema validation with intelligent retry logic."""

import json
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError


class ValidationStrategy(Enum):
    """Validation retry strategies for different error types."""

    MISSING_FIELD = "missing_field"
    TYPE_ERROR = "type_error"
    FORMAT_ERROR = "format_error"
    JSON_PARSE_ERROR = "json_parse_error"


class SchemaValidationResult:
    """Result of schema validation with detailed information."""

    def __init__(
        self,
        success: bool,
        model_instance: BaseModel | None = None,
        error: Exception | None = None,
        retry_count: int = 0,
        validation_time_ms: float = 0,
        error_category: str | None = None,
    ) -> None:
        """Initialize validation result.

        Args:
            success: Whether validation succeeded
            model_instance: Successfully validated Pydantic model instance
            error: Validation error if validation failed
            retry_count: Number of retry attempts made
            validation_time_ms: Time taken for validation in milliseconds
            error_category: Category of validation error
        """
        self.success = success
        self.model_instance = model_instance
        self.error = error
        self.retry_count = retry_count
        self.validation_time_ms = validation_time_ms
        self.error_category = error_category


class SchemaValidator:
    """Validates LLM responses against Pydantic schemas with retry logic."""

    def __init__(self) -> None:
        """Initialize SchemaValidator."""
        self._custom_retry_handler: Callable[[Exception, str, int], str] | None = None

    def validate_response(
        self, response_text: str, target_model: type[BaseModel]
    ) -> SchemaValidationResult:
        """Validate a response against a Pydantic model.

        Args:
            response_text: JSON response text to validate
            target_model: Pydantic model to validate against

        Returns:
            Validation result with success status and details
        """
        start_time = time.time()

        try:
            # First try to parse as JSON
            response_data = json.loads(response_text)

            # Then validate against Pydantic model
            model_instance = target_model.model_validate(response_data)

            validation_time = (time.time() - start_time) * 1000

            return SchemaValidationResult(
                success=True,
                model_instance=model_instance,
                validation_time_ms=validation_time,
            )

        except json.JSONDecodeError as e:
            validation_time = (time.time() - start_time) * 1000
            return SchemaValidationResult(
                success=False,
                error=e,
                validation_time_ms=validation_time,
                error_category="json_parse_error",
            )

        except ValidationError as e:
            validation_time = (time.time() - start_time) * 1000
            error_category = self._categorize_validation_error(e)

            return SchemaValidationResult(
                success=False,
                error=e,
                validation_time_ms=validation_time,
                error_category=error_category,
            )

        except Exception as e:
            validation_time = (time.time() - start_time) * 1000
            return SchemaValidationResult(
                success=False,
                error=e,
                validation_time_ms=validation_time,
                error_category="unknown_error",
            )

    def validate_with_retry(
        self, response_text: str, target_model: type[BaseModel], max_retries: int = 3
    ) -> SchemaValidationResult:
        """Validate with intelligent retry on failure.

        Args:
            response_text: JSON response text to validate
            target_model: Pydantic model to validate against
            max_retries: Maximum number of retry attempts

        Returns:
            Final validation result after retries
        """
        current_text = response_text
        last_result = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            result = self.validate_response(current_text, target_model)

            if result.success:
                result.retry_count = attempt
                return result

            # Store the result for potential return
            last_result = result

            # If this was the last attempt, return the failed result
            if attempt >= max_retries:
                result.retry_count = attempt
                return result

            # Try to fix the error for next attempt
            try:
                if result.error:
                    current_text = self._apply_retry_strategy(
                        result.error, current_text, attempt + 1
                    )
            except Exception:
                # If retry strategy fails, return the original error
                result.retry_count = attempt + 1
                return result

        # Fallback return (should not reach here)
        if last_result:
            last_result.retry_count = max_retries
            return last_result

        return SchemaValidationResult(
            success=False,
            error=Exception("Unknown error in retry logic"),
            retry_count=max_retries,
        )

    def set_custom_retry_handler(
        self, handler: Callable[[Exception, str, int], str]
    ) -> None:
        """Set a custom retry handler function.

        Args:
            handler: Function that takes (error, text, attempt) and returns fixed text
        """
        self._custom_retry_handler = handler

    def _apply_retry_strategy(
        self, error: Exception, original_text: str, attempt: int
    ) -> str:
        """Apply retry strategy based on error type.

        Args:
            error: The validation error that occurred
            original_text: Original response text
            attempt: Current attempt number

        Returns:
            Modified text for retry attempt
        """
        # Use custom handler if available
        if self._custom_retry_handler:
            return self._custom_retry_handler(error, original_text, attempt)

        # Default retry strategies
        if isinstance(error, json.JSONDecodeError):
            return self._fix_json_error(original_text, error)

        if isinstance(error, ValidationError):
            return self._fix_validation_error(original_text, error)

        # For unknown errors, return original text
        return original_text

    def _fix_json_error(self, text: str, error: json.JSONDecodeError) -> str:
        """Attempt to fix JSON parsing errors.

        Args:
            text: Original text with JSON error
            error: JSON decode error

        Returns:
            Potentially fixed text
        """
        # Try common JSON fixes
        fixed_text = text.strip()

        # Remove common prefixes/suffixes
        if fixed_text.startswith("```json"):
            fixed_text = fixed_text[7:]
        if fixed_text.endswith("```"):
            fixed_text = fixed_text[:-3]

        # Try to fix missing quotes
        if '"' not in fixed_text and ":" in fixed_text:
            # Very basic attempt to add quotes around keys
            parts = fixed_text.split(":")
            if len(parts) == 2:
                key, value = parts
                key = key.strip().strip("{").strip()
                value = value.strip().strip("}").strip()

                if not key.startswith('"'):
                    key = f'"{key}"'
                if not value.startswith('"') and not value.replace(".", "").isdigit():
                    value = f'"{value}"'

                fixed_text = f"{{{key}: {value}}}"

        return fixed_text

    def _fix_validation_error(self, text: str, error: ValidationError) -> str:
        """Attempt to fix Pydantic validation errors.

        Args:
            text: Original JSON text
            error: Pydantic validation error

        Returns:
            Potentially fixed text
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text

        # Try to fix missing required fields
        error_details = error.errors()

        for error_detail in error_details:
            if error_detail.get("type") == "missing":
                # Add missing field with a default value
                missing_field = (
                    str(error_detail["loc"][0])
                    if error_detail["loc"]
                    else "missing_field"
                )
                if missing_field not in data:
                    # Add with a reasonable default
                    data[missing_field] = self._get_default_value_for_field(
                        missing_field
                    )

            elif error_detail.get("type") == "string_type":
                # Convert non-strings to strings
                field_path = error_detail["loc"]
                if field_path and field_path[0] in data:
                    data[field_path[0]] = str(data[field_path[0]])

        return json.dumps(data)

    def _get_default_value_for_field(self, field_name: str) -> Any:
        """Get a reasonable default value for a missing field.

        Args:
            field_name: Name of the missing field

        Returns:
            Default value based on field name heuristics
        """
        field_lower = field_name.lower()

        if "email" in field_lower:
            return "default@example.com"
        elif "name" in field_lower:
            return "default_name"
        elif "count" in field_lower or "number" in field_lower:
            return 0
        elif "id" in field_lower:
            return "default_id"
        elif "result" in field_lower:
            return "default_result"
        else:
            return "default_value"

    def _categorize_validation_error(self, error: ValidationError) -> str:
        """Categorize a Pydantic validation error.

        Args:
            error: Pydantic validation error

        Returns:
            Error category string
        """
        error_details = error.errors()

        if not error_details:
            return "unknown_error"

        first_error = error_details[0]
        error_type = first_error.get("type", "unknown")

        if error_type == "missing":
            return "missing_field"
        elif error_type in ["string_type", "int_parsing", "float_parsing"]:
            return "type_error"
        elif "format" in error_type:
            return "format_error"
        else:
            return "validation_error"
