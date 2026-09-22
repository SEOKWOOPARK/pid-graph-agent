"""Exact-ID descriptions backed by the same recorded tool evidence as agent answers."""
import json
import re

from .errors import GroundingError
from .grounding import render_answer
from .tools import QueryContext, execute_tool


_ID = r"[A-Za-z0-9](?:[A-Za-z0-9_.:-]*[A-Za-z0-9_])?"
_LOOKUP = re.compile(
    rf"(?:(?:describe|show)(?: me)?\s+)?"
    rf"(?:(?:the\s+)?(?:node|equipment|entity|pipe|edge)\s+)?"
    rf"(?P<id>{_ID})[.?]?",
    re.IGNORECASE,
)


def entity_lookup_id(query: str) -> str | None:
    """Recognize a complete ID description, never a partial natural-language query."""
    match = _LOOKUP.fullmatch(query.strip())
    if match is None:
        return None
    entity_id = match["id"]
    # A single descriptive word such as 'pump' stays in the Qwen resolution flow.
    if not any(char.isdigit() for char in entity_id) and not entity_id.isupper():
        return None
    return entity_id


def describe_entity(context: QueryContext, entity_id: str) -> tuple[str, dict]:
    result = execute_tool(context, "get_entity", {"entity_id": entity_id})
    selection = {"status": "answer", "node_ids": [], "edge_ids": [], "properties": []}
    if not result["ok"]:
        if result["error"]["code"] != "entity_not_found":
            raise GroundingError(result["error"]["message"])
        selection["status"] = "not_found"
    else:
        record = result["data"]
        kind = "edge_ids" if "source" in record and "target" in record else "node_ids"
        selection[kind] = [record["id"]]
        selection["properties"] = [
            {"entity_id": record["id"], "key": key} for key in sorted(record["properties"])
        ]
    return render_answer(json.dumps(selection), context)
