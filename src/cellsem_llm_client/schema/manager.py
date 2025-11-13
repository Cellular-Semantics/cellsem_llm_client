"""Schema management system with loading, caching, and Pydantic model generation."""

import json
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, create_model


class SchemaNotFoundError(Exception):
    """Exception raised when a requested schema cannot be found."""

    pass


class SchemaValidationError(Exception):
    """Exception raised when a schema fails validation."""

    pass


class SchemaManager:
    """Manages JSON schemas with loading, caching, and Pydantic model generation.

    Supports loading schemas from:
    - Local files in configured directories
    - URLs with caching
    - Runtime registration

    Automatically generates Pydantic models from JSON schemas.
    """

    def __init__(self, schema_directories: list[str] | None = None) -> None:
        """Initialize SchemaManager.

        Args:
            schema_directories: Directories to search for schema files.
                               Defaults to ['schemas/'] if None.
        """
        self.schema_directories = schema_directories or ["schemas/"]
        self._schema_cache: dict[str, dict[str, Any]] = {}
        self._model_cache: dict[str, type[BaseModel]] = {}
        self._url_cache: dict[str, dict[str, Any]] = {}

    def load_schema(self, schema_name: str) -> dict[str, Any]:
        """Load a JSON schema by name.

        Args:
            schema_name: Name of the schema (without .json extension)

        Returns:
            JSON schema as dictionary

        Raises:
            SchemaNotFoundError: If schema cannot be found
        """
        # Check cache first
        if schema_name in self._schema_cache:
            return self._schema_cache[schema_name]

        # Try to find schema file
        for directory in self.schema_directories:
            schema_path = Path(directory) / f"{schema_name}.json"
            if schema_path.exists():
                try:
                    with open(schema_path, encoding="utf-8") as f:
                        schema_dict = json.load(f)

                    # Validate schema before caching
                    self._validate_schema(schema_dict)

                    # Cache and return
                    self._schema_cache[schema_name] = schema_dict
                    return schema_dict

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    raise SchemaValidationError(
                        f"Invalid schema file {schema_path}: {e}"
                    ) from e

        # Schema not found
        raise SchemaNotFoundError(
            f"Schema '{schema_name}' not found in directories: {self.schema_directories}"
        )

    def load_schema_from_url(self, url: str) -> dict[str, Any]:
        """Load a JSON schema from a URL.

        Args:
            url: URL to fetch schema from

        Returns:
            JSON schema as dictionary

        Raises:
            SchemaValidationError: If URL fetch or schema validation fails
        """
        # Check URL cache first
        if url in self._url_cache:
            return self._url_cache[url]

        try:
            response = requests.get(url)
            response.raise_for_status()
            schema_dict = response.json()

            # Validate schema
            self._validate_schema(schema_dict)

            # Cache and return
            self._url_cache[url] = schema_dict
            return schema_dict

        except requests.RequestException as e:
            raise SchemaValidationError(
                f"Failed to fetch schema from {url}: {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"Invalid JSON schema from {url}: {e}") from e

    def register_schema(self, name: str, schema_dict: dict[str, Any]) -> None:
        """Register a schema at runtime.

        Args:
            name: Name to register schema under
            schema_dict: JSON schema as dictionary

        Raises:
            SchemaValidationError: If schema is invalid
        """
        # Validate schema before registration
        self._validate_schema(schema_dict)

        # Register in cache
        self._schema_cache[name] = schema_dict

        # Clear model cache for this name if it exists
        if name in self._model_cache:
            del self._model_cache[name]

    def get_pydantic_model(
        self, schema_input: str | dict[str, Any] | type[BaseModel]
    ) -> type[BaseModel]:
        """Get or generate a Pydantic model from schema.

        Args:
            schema_input: Schema name, schema dict, or existing Pydantic model

        Returns:
            Pydantic model class
        """
        # If already a Pydantic model, return as-is
        if isinstance(schema_input, type) and issubclass(schema_input, BaseModel):
            return schema_input

        # If schema dict, generate model directly
        if isinstance(schema_input, dict):
            return self._generate_model_from_schema(schema_input, "DynamicModel")

        # If string, treat as schema name
        schema_name = str(schema_input)

        # Check model cache first
        if schema_name in self._model_cache:
            return self._model_cache[schema_name]

        # Load schema and generate model
        schema_dict = self.load_schema(schema_name)
        model_class = self._generate_model_from_schema(schema_dict, schema_name)

        # Cache and return
        self._model_cache[schema_name] = model_class
        return model_class

    def list_available_schemas(self) -> list[str]:
        """List all available schemas.

        Returns:
            List of schema names
        """
        schema_names = set()

        # Add schemas from files
        for directory in self.schema_directories:
            dir_path = Path(directory)
            if dir_path.exists():
                for schema_file in dir_path.glob("*.json"):
                    schema_names.add(schema_file.stem)

        # Add runtime-registered schemas
        schema_names.update(self._schema_cache.keys())

        return sorted(schema_names)

    def _validate_schema(self, schema_dict: dict[str, Any]) -> None:
        """Validate that a dictionary is a valid JSON schema.

        Args:
            schema_dict: Dictionary to validate

        Raises:
            SchemaValidationError: If schema is invalid
        """
        # Basic validation - must have type field or be a reference
        if not isinstance(schema_dict, dict):
            raise SchemaValidationError("Schema must be a dictionary")

        # Must have either 'type' or '$ref'
        if "type" not in schema_dict and "$ref" not in schema_dict:
            raise SchemaValidationError("Schema must have 'type' or '$ref' field")

        # If type is object, should have properties
        if schema_dict.get("type") == "object" and "properties" not in schema_dict:
            # This is a warning case, not an error - empty objects are valid
            pass

    def _generate_model_from_schema(
        self, schema_dict: dict[str, Any], model_name: str
    ) -> type[BaseModel]:
        """Generate a Pydantic model from a JSON schema.

        Args:
            schema_dict: JSON schema dictionary
            model_name: Name for the generated model

        Returns:
            Generated Pydantic model class
        """
        # For now, implement a simple version that handles basic cases
        # This would be enhanced to handle complex schemas in production

        if schema_dict.get("type") != "object":
            # For non-object types, create a simple wrapper model
            return create_model(model_name, value=(Any, ...))

        properties = schema_dict.get("properties", {})
        required_fields = schema_dict.get("required", [])

        # Build field definitions for create_model
        field_definitions: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python_type(field_schema)

            if field_name in required_fields:
                field_definitions[field_name] = (field_type, ...)
            else:
                field_definitions[field_name] = (field_type, None)

        # Create the model
        return create_model(model_name, **field_definitions)  # type: ignore[misc]

    def _json_type_to_python_type(self, field_schema: dict[str, Any]) -> type[Any]:
        """Convert JSON schema type to Python type.

        Args:
            field_schema: JSON schema field definition

        Returns:
            Python type
        """
        json_type = field_schema.get("type", "string")

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        python_type = type_mapping.get(json_type, str)

        # Handle optional fields by making them unions with None
        if not field_schema.get("required", True):
            return python_type | type(None)  # type: ignore[return-value]

        return python_type
