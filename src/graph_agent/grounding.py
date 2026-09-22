import json
from jsonschema import Draft202012Validator, ValidationError
from .errors import FinalSelectionError, GroundingError
from .schema import loads_json
from .tools import QueryContext, STRING

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"enum": ["answer", "ambiguous", "not_found", "unavailable"]},
        "node_ids": {"type": "array", "items": STRING, "uniqueItems": True},
        "edge_ids": {"type": "array", "items": STRING, "uniqueItems": True},
        "properties": {"type": "array", "items": {
            "type": "object", "properties": {"entity_id": STRING, "key": STRING},
            "required": ["entity_id", "key"], "additionalProperties": False,
        }},
    },
    "required": ["status", "node_ids", "edge_ids", "properties"],
    "additionalProperties": False,
}

NODE_METADATA_KEYS = frozenset({"id", "name", "type", "description"})
NOT_FOUND_ERROR_CODES = frozenset({"node_not_found", "edge_not_found", "path_not_found", "entity_not_found"})


def evidence_records(calls: list[dict]) -> tuple[dict, dict]:
    nodes, edges = {}, {}
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if {"id", "type", "properties"} <= value.keys():
                (edges if "source" in value and "target" in value else nodes)[value["id"]] = value
            else:
                for item in value.values():
                    visit(item)
    for call in calls:
        if call["result"]["ok"] or call["result"]["error"]["code"] == "ambiguous_resolution":
            visit(call["result"]["data"])
            
    return nodes, edges


def parse_selection(raw: str) -> dict:
    try:
        selection = loads_json(raw)
        Draft202012Validator(FINAL_SCHEMA).validate(selection)
        return selection
    except ValidationError as exc:
        raise FinalSelectionError(f"The model did not produce a valid grounded answer selection: {exc.message}") from exc
    except (ValueError, TypeError, RecursionError) as exc:
        raise FinalSelectionError(f"The model did not produce a valid grounded answer selection: {exc}") from exc


def _is_empty_result(data) -> bool:
    return data == [] or isinstance(data, dict) and data.get("matched_node_ids") == []


def recover_tool_outcome(context: QueryContext) -> tuple[str, dict] | None:
    if not context.calls:
        return None

    if context.resolution is None:
        for call in context.calls:
            result = call["result"]
            if result["ok"]:
                if not _is_empty_result(result["data"]):
                    return None
            elif result["error"]["code"] not in NOT_FOUND_ERROR_CODES:
                return None

    return render_answer(json.dumps({
        "status": "not_found", "node_ids": [], "edge_ids": [], "properties": [],
    }), context)


def render_answer(raw: str, context: QueryContext) -> tuple[str, dict]:
    selection = parse_selection(raw)

    if not context.calls:
        raise GroundingError("Retrieve graph evidence before producing a final answer.")
    
    if context.resolution is not None:
        candidates = context.resolution["data"]

        if candidates:
            selection = {"status": "ambiguous", "node_ids": [n["id"] for n in candidates], "edge_ids": [], "properties": []}
            return "Ambiguous target; specify an exact node ID: " + ", ".join(f'{n["id"]} ({n["name"]})' for n in candidates) + ".", selection
        
        return "No matching node was found.", {"status": "not_found", "node_ids": [], "edge_ids": [], "properties": []}
    
    nodes, edges = evidence_records(context.calls)

    for entity_id in selection["node_ids"]:
        if entity_id not in nodes:
            raise GroundingError(f"Node {entity_id} was not retrieved in this query.")
        
    for entity_id in selection["edge_ids"]:
        if entity_id not in edges:
            raise GroundingError(f"Edge {entity_id} was not retrieved in this query.")
        
    for prop in selection["properties"]:
        if prop["entity_id"] not in nodes and prop["entity_id"] not in edges:
            raise GroundingError(f'Entity {prop["entity_id"]} was not retrieved in this query.')
        
    status = selection["status"]

    if status == "ambiguous":
        raise GroundingError("Ambiguity must be established by a resolution tool call.")
    
    if status in {"not_found", "unavailable"}:
        if selection["node_ids"] or selection["edge_ids"] or selection["properties"]:
            raise GroundingError("Non-answer selections must not include factual claims.")
        
        failures = [c["result"]["error"] for c in context.calls if not c["result"]["ok"]]

        if failures:
            return "Graph query could not be completed: " + "; ".join(dict.fromkeys(e["message"] for e in failures)), selection
        
        empty = any(c["result"]["ok"] and _is_empty_result(c["result"]["data"]) for c in context.calls)

        if status == "not_found":
            if not empty:
                raise GroundingError("No empty result supports a no-match answer.")
            return "No matching results were returned by the graph query.", selection
        
        return "The requested information is unavailable from the retrieved graph data.", selection

    
    properties = []
    
    for prop in selection["properties"]:
        entity_id, key = prop["entity_id"], prop["key"]
        node = nodes.get(entity_id)
        if node is not None and key in NODE_METADATA_KEYS and key not in node["properties"]:
            if entity_id not in selection["node_ids"]:
                selection["node_ids"].append(entity_id)
        else:
            properties.append(prop)
    selection["properties"] = properties

    lines = []

    for entity_id in selection["node_ids"]:
        node = nodes[entity_id]
        lines.append(f'{entity_id} ({node["type"]}): {node["name"]}. {node["description"]}')

    for entity_id in selection["edge_ids"]:
        edge = edges[entity_id]
        line = f'{entity_id}: {edge["source"]} → {edge["target"]} ({edge["type"]}).'
        if edge.get("description"):
            line += " " + edge["description"]
        lines.append(line)

    for prop in selection["properties"]:
        entity_id, key = prop["entity_id"], prop["key"]
        entity = nodes.get(entity_id, edges.get(entity_id))

        if key not in entity["properties"]:
            lines.append(f'{entity_id}: property {key!r} is not available in the graph.')
        else:
            value = json.dumps(entity["properties"][key], ensure_ascii=False, allow_nan=False)
            lines.append(f"{entity_id}: {key} = {value}.")

    if not lines:
        raise GroundingError("An answer must select at least one retrieved entity or property.")
    
    return "\n".join(lines), selection
