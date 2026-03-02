"""Unit tests for nested schema handling in SchemaManager._generate_model_from_schema."""

import pytest

from cellsem_llm_client.schema.manager import SchemaManager


@pytest.mark.unit
class TestNestedSchemaGeneration:
    """Test cases for nested schema support in _generate_model_from_schema."""

    def test_array_of_strings_produces_typed_items(self) -> None:
        """Test that array of strings generates list[str] with proper items schema."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

        # The regenerated JSON schema must have items.type, not empty items
        json_schema = model.model_json_schema()
        assert json_schema["properties"]["tags"]["items"].get("type") == "string"

    def test_array_of_integers_produces_typed_items(self) -> None:
        """Test that array of integers generates list[int]."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "scores": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["scores"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(scores=[1, 2, 3])
        assert instance.scores == [1, 2, 3]

        json_schema = model.model_json_schema()
        assert json_schema["properties"]["scores"]["items"].get("type") == "integer"

    def test_nested_object_preserves_properties(self) -> None:
        """Test that nested objects generate sub-models with properties."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["street", "city"],
                },
            },
            "required": ["address"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(address={"street": "123 Main St", "city": "Springfield"})
        assert instance.address.street == "123 Main St"
        assert instance.address.city == "Springfield"

        # Regenerated schema should reference a sub-model via $defs
        json_schema = model.model_json_schema()
        address_schema = json_schema["properties"]["address"]
        # Pydantic generates a $ref to a $defs entry for the sub-model
        assert "$ref" in address_schema, (
            f"Expected address to reference a sub-model, got: {address_schema}"
        )

    def test_array_of_objects_produces_typed_items(self) -> None:
        """Test array of objects generates list[SubModel] with proper items schema."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "score": {"type": "number"},
                        },
                        "required": ["name", "score"],
                    },
                },
            },
            "required": ["themes"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(themes=[{"name": "theme1", "score": 0.8}])
        assert instance.themes[0].name == "theme1"
        assert instance.themes[0].score == 0.8

        # Verify JSON schema has proper items with type info
        json_schema = model.model_json_schema()
        themes_schema = json_schema["properties"]["themes"]
        assert themes_schema["type"] == "array"
        # Items should NOT be empty — must reference a sub-model or have inline type
        items = themes_schema.get("items", {})
        # Items either has $ref (sub-model) or inline type/properties
        has_ref = "$ref" in items
        has_type = "type" in items
        assert has_ref or has_type, f"items schema missing type info: {items}"

    def test_ref_defs_resolution(self) -> None:
        """Test that $ref and $defs are resolved correctly."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "$defs": {
                "Theme": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["name", "score"],
                },
            },
            "properties": {
                "themes": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Theme"},
                },
            },
            "required": ["themes"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(themes=[{"name": "resolved", "score": 1.0}])
        assert instance.themes[0].name == "resolved"
        assert instance.themes[0].score == 1.0

    def test_deeply_nested_structures(self) -> None:
        """Test deeply nested: object -> array of objects -> nested field."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "paragraphs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["title", "paragraphs"],
                    },
                },
            },
            "required": ["sections"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(
            sections=[{"title": "Intro", "paragraphs": ["Hello", "World"]}]
        )
        assert instance.sections[0].title == "Intro"
        assert instance.sections[0].paragraphs == ["Hello", "World"]

    def test_ref_in_nested_object_property(self) -> None:
        """Test $ref used inside a nested object property."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["street", "city"],
                },
            },
            "properties": {
                "home": {"$ref": "#/$defs/Address"},
                "work": {"$ref": "#/$defs/Address"},
            },
            "required": ["home"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(
            home={"street": "123 Main St", "city": "Springfield"},
            work={"street": "456 Oak Ave", "city": "Shelbyville"},
        )
        assert instance.home.street == "123 Main St"
        assert instance.work.city == "Shelbyville"

    def test_optional_nested_object(self) -> None:
        """Test optional nested objects default to None."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                    },
                },
            },
            "required": ["name"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(name="test")
        assert instance.name == "test"
        assert instance.metadata is None

    def test_array_of_booleans(self) -> None:
        """Test array of booleans generates list[bool]."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "flags": {"type": "array", "items": {"type": "boolean"}},
            },
            "required": ["flags"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(flags=[True, False, True])
        assert instance.flags == [True, False, True]

    def test_array_without_items_still_works(self) -> None:
        """Test that an array without items specification still works (bare list)."""
        manager = SchemaManager()
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "array"},
            },
            "required": ["data"],
        }

        model = manager.get_pydantic_model(schema)
        instance = model(data=[1, "two", True])
        assert instance.data == [1, "two", True]
