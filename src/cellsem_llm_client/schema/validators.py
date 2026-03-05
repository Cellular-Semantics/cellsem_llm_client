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
            raw_path = error_detail.get("loc", ())
            field_path = [part for part in raw_path if isinstance(part, (str, int))]
            error_type = error_detail.get("type")

            if error_type == "missing":
                if field_path:
                    default_field_name = str(field_path[-1])
                    self._set_nested_value(
                        data,
                        field_path,
                        self._get_default_value_for_field(default_field_name),
                    )
                continue

            if not field_path:
                continue

            current_value = self._get_nested_value(data, field_path)
            if current_value is None:
                continue

            if error_type == "string_type" and not isinstance(current_value, str):
                self._set_nested_value(data, field_path, str(current_value))
            elif error_type in ("int_type", "int_parsing"):
                try:
                    if isinstance(current_value, str):
                        self._set_nested_value(data, field_path, int(current_value))
                except ValueError:
                    pass
            elif error_type in ("float_type", "float_parsing"):
                try:
                    if isinstance(current_value, str):
                        self._set_nested_value(data, field_path, float(current_value))
                except ValueError:
                    pass
            elif error_type in ("model_type", "dict_type"):
                if isinstance(current_value, str):
                    stripped = current_value.strip()
                    if stripped.startswith("{") or stripped.startswith("["):
                        try:
                            parsed = json.loads(stripped)
                            self._set_nested_value(data, field_path, parsed)
                        except json.JSONDecodeError:
                            pass

        return json.dumps(data)

    def _get_nested_value(self, data: Any, path: list[str | int]) -> Any:
        """Read a nested value by path from dict/list structures."""
        current = data
        for part in path:
            if isinstance(part, int):
                if isinstance(current, list) and 0 <= part < len(current):
                    current = current[part]
                else:
                    return None
            else:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        return current

    def _set_nested_value(self, data: Any, path: list[str | int], value: Any) -> None:
        """Set a nested value by path, creating missing dict keys when needed."""
        if not path:
            return

        current = data
        for part in path[:-1]:
            if isinstance(part, int):
                if not isinstance(current, list) or not (0 <= part < len(current)):
                    return
                current = current[part]
            else:
                if not isinstance(current, dict):
                    return
                if part not in current or current[part] is None:
                    current[part] = {}
                current = current[part]

        final_part = path[-1]
        if isinstance(final_part, int):
            if isinstance(current, list) and 0 <= final_part < len(current):
                current[final_part] = value
        elif isinstance(current, dict):
            current[final_part] = value

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
