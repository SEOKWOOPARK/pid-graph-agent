import json
import math
from typing import Any
from jsonschema import Draft202012Validator


GRAPH_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["nodes", "edges"],
    "$defs": {
        "text": {"type": "string", "minLength": 1, "pattern": "\\S"},
        "properties": {
            "type": "object",
            "propertyNames": {"$ref": "#/$defs/text"},
            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
        },
        "node": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "type", "name", "description", "properties"],
            "properties": {
                "id": {"$ref": "#/$defs/text"},
                "type": {"$ref": "#/$defs/text"},
                "name": {"$ref": "#/$defs/text"},
                "description": {"$ref": "#/$defs/text"},
                "properties": {"$ref": "#/$defs/properties"},
            },
        },
        "edge": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "source", "target", "type", "properties"],
            "properties": {
                "id": {"$ref": "#/$defs/text"},
                "source": {"$ref": "#/$defs/text"},
                "target": {"$ref": "#/$defs/text"},
                "type": {"$ref": "#/$defs/text"},
                "description": {"$ref": "#/$defs/text"},
                "properties": {"$ref": "#/$defs/properties"},
            },
        },
    },
    "properties": {
        "nodes": {"type": "array", "items": {"$ref": "#/$defs/node"}},
        "edges": {"type": "array", "items": {"$ref": "#/$defs/edge"}},
    },
}

GRAPH_VALIDATOR = Draft202012Validator(GRAPH_SCHEMA)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value

    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite JSON number is not allowed: {value}")
    
    return result


def loads_json(text: str) -> Any:
    """Decode JSON, rejecting duplicate keys and non-finite numeric values."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc
