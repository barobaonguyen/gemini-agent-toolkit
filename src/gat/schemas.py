"""Pydantic and Python type helpers for Gemini schemas."""

from __future__ import annotations

from collections.abc import Mapping
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel


def to_gemini_schema(pydantic_model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to Gemini's OpenAPI-style schema subset."""

    raw_schema = pydantic_model.model_json_schema()
    definitions = raw_schema.get("$defs", {})
    converted = _convert_json_schema(raw_schema, definitions)
    typed = cast(dict[str, Any], converted)
    typed.pop("$defs", None)
    return typed


def python_type_to_schema(annotation: Any) -> dict[str, Any]:
    """Return a JSON schema fragment for a Python type annotation."""

    if annotation is Any:
        return {"type": "object"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            schema = python_type_to_schema(non_none[0])
            schema["nullable"] = True
            return schema
        return {"anyOf": [python_type_to_schema(arg) for arg in non_none]}

    if origin is Literal:
        values = list(args)
        if not values:
            return {"type": "string"}
        first = values[0]
        schema_type = _primitive_type(type(first))
        return {"type": schema_type, "enum": values}

    if origin in (list, tuple, set, frozenset):
        item_type = args[0] if args else Any
        return {"type": "array", "items": python_type_to_schema(item_type)}

    if origin in (dict, Mapping):
        value_type = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": python_type_to_schema(value_type)}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return to_gemini_schema(annotation)

    return {"type": _primitive_type(annotation)}


def _primitive_type(annotation: Any) -> str:
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }
    return mapping.get(annotation, "string")


def _convert_json_schema(schema: Any, definitions: Mapping[str, Any]) -> Any:
    if isinstance(schema, list):
        return [_convert_json_schema(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        ref_name = str(schema["$ref"]).rsplit("/", 1)[-1]
        resolved = definitions.get(ref_name, {})
        merged = {**resolved, **{key: value for key, value in schema.items() if key != "$ref"}}
        return _convert_json_schema(merged, definitions)

    if "anyOf" in schema:
        options = schema["anyOf"]
        nullable = any(
            option.get("type") == "null" for option in options if isinstance(option, dict)
        )
        non_null = [
            option
            for option in options
            if not (isinstance(option, dict) and option.get("type") == "null")
        ]
        if len(non_null) == 1:
            nullable_schema = _convert_json_schema(non_null[0], definitions)
            if isinstance(nullable_schema, dict) and nullable:
                nullable_schema["nullable"] = True
            return nullable_schema

    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"$defs", "$schema", "title", "examples"}:
            continue
        if key == "type" and value == "null":
            continue
        if key == "properties" and isinstance(value, dict):
            converted[key] = {
                prop_name: _convert_json_schema(prop_schema, definitions)
                for prop_name, prop_schema in value.items()
            }
        elif key in {"items", "additionalProperties", "anyOf", "allOf", "oneOf"}:
            converted[key] = _convert_json_schema(value, definitions)
        else:
            converted[key] = value

    if converted.get("type") == "object" and "properties" not in converted:
        converted["properties"] = {}
    return converted
