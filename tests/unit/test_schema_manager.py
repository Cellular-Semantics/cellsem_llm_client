"""Unit tests for SchemaManager - schema resolution and loading system."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from cellsem_llm_client.schema.manager import (
    SchemaManager,
    SchemaNotFoundError,
    SchemaValidationError,
)


@pytest.mark.unit
class TestSchemaManager:
    """Test cases for the SchemaManager class."""

    def test_schema_manager_creation(self) -> None:
        """Test creating SchemaManager with default configuration."""
        manager = SchemaManager()

        assert manager is not None
        assert hasattr(manager, "load_schema")
        assert hasattr(manager, "get_pydantic_model")
        assert hasattr(manager, "register_schema")

    def test_load_schema_from_file(self) -> None:
        """Test loading a JSON schema from a local file."""
        with TemporaryDirectory() as temp_dir:
            schema_path = Path(temp_dir) / "test_schema.json"
            schema_content = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                },
                "required": ["name"],
            }

            # Write test schema file
            schema_path.write_text(str(schema_content).replace("'", '"'))

            manager = SchemaManager(schema_directories=[temp_dir])

            # Should load schema successfully
            loaded_schema = manager.load_schema("test_schema")

            assert loaded_schema["type"] == "object"
            assert "name" in loaded_schema["properties"]
            assert loaded_schema["properties"]["name"]["type"] == "string"

    def test_load_schema_not_found(self) -> None:
        """Test loading a schema that doesn't exist raises SchemaNotFoundError."""
        manager = SchemaManager()

        with pytest.raises(SchemaNotFoundError) as exc_info:
            manager.load_schema("nonexistent_schema")

        assert "nonexistent_schema" in str(exc_info.value)

    def test_register_runtime_schema(self) -> None:
        """Test registering a schema at runtime."""
        manager = SchemaManager()
        schema_dict = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        manager.register_schema("runtime_schema", schema_dict)

        # Should be able to load the registered schema
        loaded = manager.load_schema("runtime_schema")
        assert loaded == schema_dict

    def test_get_pydantic_model_from_schema(self) -> None:
        """Test generating Pydantic model from JSON schema."""
        manager = SchemaManager()
        schema_dict = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["name"],
        }

        manager.register_schema("test_model", schema_dict)

        # Should generate a Pydantic model
        model_class = manager.get_pydantic_model("test_model")

        assert issubclass(model_class, BaseModel)

        # Should be able to create instances
        instance = model_class(name="test", count=42)
        assert instance.name == "test"
        assert instance.count == 42

    def test_get_pydantic_model_validation(self) -> None:
        """Test that generated Pydantic model validates according to schema."""
        manager = SchemaManager()
        schema_dict = {
            "type": "object",
            "properties": {"email": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["email"],
        }

        manager.register_schema("user", schema_dict)
        user_model = manager.get_pydantic_model("user")

        # Valid data should work
        valid_user = user_model(email="test@example.com", age=25)
        assert valid_user.email == "test@example.com"

        # Missing required field should raise validation error
        with pytest.raises(ValueError):  # Pydantic ValidationError
            user_model(age=25)  # Missing required email field

    def test_list_available_schemas(self) -> None:
        """Test listing all available schemas."""
        with TemporaryDirectory() as temp_dir:
            # Create test schema files
            (Path(temp_dir) / "schema1.json").write_text('{"type": "object"}')
            (Path(temp_dir) / "schema2.json").write_text('{"type": "array"}')

            manager = SchemaManager(schema_directories=[temp_dir])
            manager.register_schema("runtime_schema", {"type": "string"})

            available = manager.list_available_schemas()

            assert "schema1" in available
            assert "schema2" in available
            assert "runtime_schema" in available
            assert len(available) >= 3

    def test_schema_caching(self) -> None:
        """Test that schemas are cached after first load."""
        manager = SchemaManager()
        schema_dict = {"type": "object", "properties": {"id": {"type": "string"}}}

        manager.register_schema("cached_schema", schema_dict)

        # First load
        first_load = manager.load_schema("cached_schema")

        # Second load should return same object (cached)
        second_load = manager.load_schema("cached_schema")

        assert first_load is second_load  # Same object reference

    def test_invalid_json_schema_raises_error(self) -> None:
        """Test that invalid JSON schema raises SchemaValidationError."""
        manager = SchemaManager()

        # Invalid schema (missing type)
        invalid_schema = {"properties": {"name": "not-a-type-definition"}}

        with pytest.raises(SchemaValidationError):
            manager.register_schema("invalid", invalid_schema)

    def test_multiple_schema_directories(self) -> None:
        """Test SchemaManager with multiple schema directories."""
        with TemporaryDirectory() as temp_dir1, TemporaryDirectory() as temp_dir2:
            # Create schemas in different directories
            (Path(temp_dir1) / "schema1.json").write_text('{"type": "object"}')
            (Path(temp_dir2) / "schema2.json").write_text('{"type": "array"}')

            manager = SchemaManager(schema_directories=[temp_dir1, temp_dir2])

            # Should find schemas from both directories
            schema1 = manager.load_schema("schema1")
            schema2 = manager.load_schema("schema2")

            assert schema1["type"] == "object"
            assert schema2["type"] == "array"

    @patch("requests.get")
    def test_load_schema_from_url(self, mock_get: Mock) -> None:
        """Test loading a schema from a URL."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "object",
            "properties": {"remote_field": {"type": "string"}},
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        manager = SchemaManager()

        # Should load schema from URL
        schema_url = "https://example.com/schema.json"
        loaded_schema = manager.load_schema_from_url(schema_url)

        assert loaded_schema["type"] == "object"
        assert "remote_field" in loaded_schema["properties"]
        mock_get.assert_called_once_with(schema_url)

    @patch("requests.get")
    def test_url_schema_caching(self, mock_get: Mock) -> None:
        """Test that URL schemas are cached to avoid repeated requests."""
        mock_response = Mock()
        mock_response.json.return_value = {"type": "object"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        manager = SchemaManager()
        schema_url = "https://example.com/schema.json"

        # First load
        manager.load_schema_from_url(schema_url)

        # Second load should not make another request
        manager.load_schema_from_url(schema_url)

        # Should only call requests.get once due to caching
        assert mock_get.call_count == 1

    def test_get_model_by_different_inputs(self) -> None:
        """Test getting Pydantic model with different input types."""
        manager = SchemaManager()
        schema_dict = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

        manager.register_schema("flexible", schema_dict)

        # By string name
        model1 = manager.get_pydantic_model("flexible")

        # By schema dict
        model2 = manager.get_pydantic_model(schema_dict)

        # Both should create valid models
        instance1 = model1(value="test1")
        instance2 = model2(value="test2")

        assert instance1.value == "test1"
        assert instance2.value == "test2"
