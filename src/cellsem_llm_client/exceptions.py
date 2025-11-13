"""Custom exceptions for CellSem LLM client."""

from typing import Any


class CellSemLLMException(Exception):
    """Base exception for CellSem LLM client.

    All custom exceptions in this package should inherit from this base class.
    """

    pass


class SchemaValidationException(CellSemLLMException):
    """Raised when schema validation fails.

    This exception is raised when:
    - Pydantic model validation fails
    - JSON schema validation fails
    - Response format doesn't match expected schema
    - Retry attempts for schema validation are exhausted

    Attributes:
        schema: The schema that failed validation
        response_text: The text that failed to validate
        validation_errors: List of validation error details
    """

    def __init__(
        self,
        message: str,
        schema: str | None = None,
        response_text: str | None = None,
        validation_errors: list[str] | None = None,
    ):
        super().__init__(message)
        self.schema = schema
        self.response_text = response_text
        self.validation_errors = validation_errors or []


class ProviderException(CellSemLLMException):
    """Raised when provider-specific errors occur.

    This exception is raised when:
    - API authentication fails
    - Rate limits are exceeded
    - Provider-specific API errors occur
    - Invalid model names or configurations

    Attributes:
        provider: The provider that caused the error
        model: The model that was being used
        original_error: The original exception from the provider
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.original_error = original_error


class ConfigurationException(CellSemLLMException):
    """Raised when configuration errors occur.

    This exception is raised when:
    - Required configuration is missing
    - Invalid configuration values are provided
    - Environment setup is incorrect

    Attributes:
        config_key: The configuration key that caused the error
        config_value: The invalid configuration value
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: str | None = None,
    ):
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value


class CostCalculationException(CellSemLLMException):
    """Raised when cost calculation fails.

    This exception is raised when:
    - Rate data is missing for a model
    - Cost calculation logic fails
    - Invalid usage metrics are provided

    Attributes:
        provider: The provider for which cost calculation failed
        model: The model for which cost calculation failed
        usage_metrics: The usage metrics that caused the error
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        usage_metrics: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.usage_metrics = usage_metrics
