"""Schema management system for structured LLM responses with JSON Schema compliance.

This module provides comprehensive schema management capabilities including:
- JSON Schema loading from files and URLs
- Automatic Pydantic model generation from JSON schemas
- Provider-specific schema adapters for native schema enforcement
- Validation with intelligent retry logic
- Multi-source schema resolution with caching
"""

from .adapters import (
    AdapterCapability,
    AnthropicSchemaAdapter,
    BaseSchemaAdapter,
    FallbackSchemaAdapter,
    OpenAISchemaAdapter,
    SchemaAdapterFactory,
)
from .manager import (
    SchemaManager,
    SchemaNotFoundError,
    SchemaValidationError,
)
from .validators import (
    SchemaValidationResult,
    SchemaValidator,
    ValidationStrategy,
)

__all__ = [
    # Adapters
    "AdapterCapability",
    "BaseSchemaAdapter",
    "OpenAISchemaAdapter",
    "AnthropicSchemaAdapter",
    "FallbackSchemaAdapter",
    "SchemaAdapterFactory",
    # Manager
    "SchemaManager",
    "SchemaNotFoundError",
    "SchemaValidationError",
    # Validators
    "SchemaValidator",
    "SchemaValidationResult",
    "ValidationStrategy",
]
