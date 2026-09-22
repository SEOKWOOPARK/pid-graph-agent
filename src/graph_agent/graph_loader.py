import math
from pathlib import Path
from typing import Any
from .errors import GraphValidationError
from .schema import GRAPH_VALIDATOR, loads_json


def load_graph(path: str | Path) -> dict[str, Any]:
    try:
        document = loads_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise GraphValidationError(f"Cannot load graph from {path}: {exc}") from exc
    return validate_graph(document)


def validate_graph(document: Any) -> dict[str, Any]:
    error = next(GRAPH_VALIDATOR.iter_errors(document), None)

    if error is not None:
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise GraphValidationError(f"Invalid graph at {location}: {error.message}")

    for records in (document["nodes"], document["edges"]):
        for record in records:
            for key, value in record["properties"].items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise GraphValidationError(f"Non-finite property {key!r} on {record['id']!r}")

    node_ids: set[str] = set()

    for node in document["nodes"]:
        node_id = node["id"]

        if node_id in node_ids:
            raise GraphValidationError(f"Duplicate node ID: {node_id}")
        
        node_ids.add(node_id)

    edge_ids: set[str] = set()

    for edge in document["edges"]:
        edge_id, source, target = edge["id"], edge["source"], edge["target"]

        if edge_id in edge_ids:
            raise GraphValidationError(f"Duplicate edge ID: {edge_id}")
        
        if edge_id in node_ids:
            raise GraphValidationError(f"Edge ID collides with a node ID: {edge_id}")
        
        if source not in node_ids or target not in node_ids:
            raise GraphValidationError(
                f"Edge {edge_id} references nonexistent endpoint: {source} -> {target}"
            )
        
        edge_ids.add(edge_id)
        
    return document
