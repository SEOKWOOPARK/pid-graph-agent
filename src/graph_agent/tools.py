"""Graph primitives, exposed to the SDK with strict runtime validation."""
from dataclasses import dataclass, field
from typing import Any
from agents import FunctionTool
from jsonschema import Draft202012Validator
from .errors import GraphAgentError, InvalidInputError
from .graph_access import GraphAccess
from .schema import loads_json
import json
import logging


log = logging.getLogger(__name__)
STRING = {"type": "string", "minLength": 1, "pattern": r"\S"}
NULLABLE_STRING = {"anyOf": [STRING, {"type": "null"}], "default": None}
DIRECTION = {"type": "string", "enum": ["upstream", "downstream", "both"], "default": "both"}
RESOLVE = {"type": "boolean", "default": False, "description": "Set true when resolving one indirectly described target before another operation; never pick arbitrarily."}


def _schema(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOL_SCHEMAS = {
    "get_node": _schema({"node_id": STRING}, ["node_id"]),
    "get_entity": _schema({"entity_id": STRING}, ["entity_id"]),
    "search_nodes": _schema({"query": STRING, "node_type": NULLABLE_STRING, "resolve": RESOLVE}, ["query"]),
    "get_neighbors": _schema({"node_id": STRING, "direction": DIRECTION, "node_type": NULLABLE_STRING, "resolve": RESOLVE}, ["node_id"]),
    "get_connected_edges": _schema({"node_id": STRING, "direction": DIRECTION}, ["node_id"]),
    "get_edge": _schema({"source": STRING, "target": STRING, "edge_id": NULLABLE_STRING}, ["source", "target"]),
    "find_path": _schema({"source": STRING, "target": STRING}, ["source", "target"]),
    "get_connected_subgraph": _schema({"node_id": STRING, "max_hops": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1}, "direction": DIRECTION, "node_type": NULLABLE_STRING}, ["node_id"]),
}
DESCRIPTIONS = {
    "get_node": "Retrieve full metadata and properties for an exact, case-sensitive node ID.",
    "get_entity": "Retrieve a node or edge by exact, case-sensitive ID, including all metadata and properties. Use this when the ID's kind is unknown or only a pipe ID is supplied. No endpoints are needed.",
    "search_nodes": "Search all candidate nodes by literal words in ID, name, description or type (case-insensitive). Use resolve=true to resolve a singular described target.",
    "get_neighbors": "Retrieve directly connected equipment. Upstream means incoming flow, downstream outgoing flow. Optionally filter equipment type. Use resolve=true to identify a singular target before a dependent query.",
    "get_connected_edges": "Retrieve incoming/outgoing/both connection records including their exact source, target and properties.",
    "get_edge": "Retrieve directed source-to-target connections, optionally filtered by edge_id. Both source and target are required, even with edge_id. Use get_entity for an edge ID alone. Reverse arguments to query opposite flow.",
    "find_path": "Find a shortest directed process-flow path; return ordered nodes and all parallel connection alternatives for its hops.",
    "get_connected_subgraph": "Traverse 1 to 10 hops across all equipment types. Return context nodes/edges and matched_node_ids filtered by node_type, excluding the root.",
}


@dataclass
class QueryContext:
    graph: GraphAccess
    calls: list[dict[str, Any]] = field(default_factory=list)
    resolution: dict[str, Any] | None = None
    allow_resolution: bool = True


def validate_arguments(name: str, arguments: Any) -> dict:
    if name not in TOOL_SCHEMAS:
        raise InvalidInputError(f"Unknown tool: {name}")
    
    errors = list(Draft202012Validator(TOOL_SCHEMAS[name]).iter_errors(arguments))

    if errors:
        raise InvalidInputError(f"Invalid arguments for {name}: {errors[0].message}")
    
    if "max_hops" in arguments and type(arguments["max_hops"]) is not int:
        raise InvalidInputError("max_hops must be an integer, without coercion.")
    
    return arguments


def execute_tool(context: QueryContext, name: str, arguments: dict) -> dict:
    try:
        validate_arguments(name, arguments)
        if arguments.get("resolve") and not context.allow_resolution:
            raise InvalidInputError(
                "The question already specifies exact target IDs. Do not use resolve=true. "
                "Use the requested graph operation, including its direction and hop limit."
            )
        if context.resolution is not None:
            result = context.resolution
        else:
            graph_args = {k: v for k, v in arguments.items() if k != "resolve"}
            data = getattr(context.graph, name)(**graph_args)
            result = {"ok": True, "data": data, "error": None}
            if arguments.get("resolve") and len(data) != 1:
                code = "ambiguous_resolution" if data else "no_matching_node"
                result = {"ok": False, "data": data, "error": {"code": code, "message": "Multiple candidate nodes; specify an exact ID." if data else "No matching node was found."}}
                context.resolution = result
    except GraphAgentError as exc:
        log.info("Tool %s rejected input: %s", name, exc)
        result = {"ok": False, "data": None, "error": {"code": exc.code, "message": str(exc)}}
    context.calls.append({"name": name, "arguments": arguments, "result": result})
    
    return result


def build_tools() -> list[FunctionTool]:
    result = []
    for name, schema in TOOL_SCHEMAS.items():
        async def invoke(wrapper, raw, tool_name=name):
            try:
                arguments = loads_json(raw)
            except (ValueError, TypeError) as exc:
                response = {"ok": False, "data": None, "error": {"code": "invalid_input", "message": f"Malformed tool arguments: {exc}"}}
                wrapper.context.calls.append({"name": tool_name, "arguments": raw, "result": response})
            else:
                response = execute_tool(wrapper.context, tool_name, arguments)
            return json.dumps(response, ensure_ascii=False, allow_nan=False)
        
        result.append(FunctionTool(
            name=name, description=DESCRIPTIONS[name], params_json_schema=schema,
            on_invoke_tool=invoke,
            strict_json_schema=False,
        ))
    return result
