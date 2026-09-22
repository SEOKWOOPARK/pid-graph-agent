import logging
from typing import Any
from jsonschema import Draft202012Validator
from .errors import GraphAgentError
from .grounding import FINAL_SCHEMA, evidence_records
from .schema import loads_json
from .tools import TOOL_SCHEMAS, validate_arguments

log = logging.getLogger(__name__)

METRICS = (
    "retrieval_exact_match", "node_resolution_accuracy", "tool_call_accuracy",
    "required_argument_accuracy", "argument_value_accuracy", "valid_schema_rate",
    "entity_property_presence", "successful_completion_rate",
)


def _arguments(call: dict) -> dict:
    supplied = call.get("arguments", {})

    if not isinstance(supplied, dict):
        return {}
    
    properties = TOOL_SCHEMAS.get(call.get("name"), {}).get("properties", {})
    defaults = {key: value["default"] for key, value in properties.items() if "default" in value}
    return defaults | supplied


def _matches(call: dict, expected: dict) -> bool:
    actual = _arguments(call)
    
    return call.get("name") == expected["name"] and all(
        key in actual and type(actual[key]) is type(value) and actual[key] == value
        for key, value in expected.get("arguments", {}).items()
    )


def _find_call(calls: list[dict], selector: dict | None) -> dict | None:
    if selector is None:
        return None
    return next((call for call in reversed(calls) if _matches(call, selector)), None)


def _result_expectation(case: dict) -> dict | None:
    if "expected_nodes" not in case and "expected_edges" not in case:
        return None
    expected = case.get("expected_calls")

    if not isinstance(expected, list) or not expected:
        raise ValueError("Result checks require a nonempty expected_calls list.")
    
    if len(expected) > 1 and "result_call_index" not in case:
        raise ValueError("Result checks with multiple expected_calls require result_call_index.")
    
    index = case.get("result_call_index", 0)

    if type(index) is not int or not 0 <= index < len(expected):
        raise ValueError("result_call_index must be an integer indexing expected_calls (starting at 0).")
    
    selected = expected[index]

    if (not isinstance(selected, dict)
            or not isinstance(selected.get("name"), str)
            or selected["name"] not in TOOL_SCHEMAS
            or not isinstance(selected.get("arguments"), dict)):
        raise ValueError("The selected expected_calls entry must have a known tool name and arguments object.")
    
    return selected


def _ids(call: dict | None) -> tuple[set[str], set[str]]:
    if call is None:
        return set(), set()
    
    result = call["result"]

    if not result["ok"] and (result.get("error") or {}).get("code") != "ambiguous_resolution":
        return set(), set()
    
    data = result["data"]
    nodes, edges = evidence_records([call])
    
    if isinstance(data, dict) and "matched_node_ids" in data:
        return set(data["matched_node_ids"]), set(edges)
    
    return set(nodes), set(edges)


def _has_sequence(calls: list[dict], expected: list[dict], condition=None) -> bool:
    position = 0

    for wanted in expected:
        index = next((i for i in range(position, len(calls))
                      if calls[i].get("name") == wanted["name"]
                      and (condition is None or condition(calls[i], wanted))), None)
        
        if index is None:
            return False
        
        position = index + 1

    return True


def _has_required_arguments(call: dict, expected: dict) -> bool:
    return isinstance(call.get("arguments"), dict) and (
        set(TOOL_SCHEMAS[expected["name"]]["required"]) <= call["arguments"].keys()
    )


def _allowed(value: Any, expected: Any) -> bool:
    return value in expected if isinstance(expected, list) else value == expected


def _score(case: dict, answer: Any, calls: list[dict], error: Exception | None,
           result_expectation: dict | None, final_output: str | None = None,
           final_outputs: list[str] | None = None) -> dict:
    checks = dict.fromkeys(METRICS)
    selection = answer.selection if answer is not None else {}
    direct_lookup = getattr(answer, "execution_mode", "agent") == "direct_lookup"
    checks["successful_completion_rate"] = (
        _allowed(selection.get("status"), case.get("expected_status", [] if "expected_error" in case else "answer"))
        if error is None else isinstance(error, GraphAgentError)
        and _allowed(error.code, case.get("expected_error", []))
    )
    
    try:
        for call in calls:
            validate_arguments(call["name"], call["arguments"])

        recovery = getattr(answer, "recovery", None)
        raw_outputs = final_outputs or ([final_output] if final_output is not None else [])
        if getattr(error, "code", None) == "model_output_error" or (
            recovery is not None and recovery["code"] == "model_output_error"
        ):
            checks["valid_schema_rate"] = False
        elif raw_outputs:
            for raw in raw_outputs:
                Draft202012Validator(FINAL_SCHEMA).validate(loads_json(raw))
            checks["valid_schema_rate"] = True
        elif error is None:
            Draft202012Validator(FINAL_SCHEMA).validate(selection)
            checks["valid_schema_rate"] = True
    except Exception:
        checks["valid_schema_rate"] = False
    if direct_lookup:
        # No model produced a schema or chose these tool arguments.
        checks["valid_schema_rate"] = None

    expected_calls = case.get("expected_calls")

    if expected_calls and not direct_lookup:
        checks["tool_call_accuracy"] = _has_sequence(calls, expected_calls)
        checks["required_argument_accuracy"] = _has_sequence(calls, expected_calls, _has_required_arguments)
        checks["argument_value_accuracy"] = _has_sequence(calls, expected_calls, _matches)

    selected_call = _find_call(calls, result_expectation)

    if "expected_nodes" in case or "expected_edges" in case:
        nodes, edges = _ids(selected_call)
        usable = selected_call is not None and (selected_call["result"]["ok"] or (
            selected_call["result"].get("error") or {}
        ).get("code") in {"ambiguous_resolution", "no_matching_node"})
        checks["retrieval_exact_match"] = usable and (
            "expected_nodes" not in case or nodes == set(case["expected_nodes"])
        ) and ("expected_edges" not in case or edges == set(case["expected_edges"]))

    if "expected_resolved_node" in case and not direct_lookup:
        resolved = _find_call(calls, case.get("resolution_call"))
        checks["node_resolution_accuracy"] = resolved is not None and resolved["result"]["ok"] and (
            _ids(resolved)[0] == {case["expected_resolved_node"]}
        )

    expected_nodes = case.get("expected_selected_nodes", case.get("expected_nodes", []))
    expected_edges = case.get("expected_selected_edges", case.get("expected_edges", []))

    if expected_nodes or expected_edges or case.get("expected_properties"):
        nodes, edges = evidence_records(calls)
        selected_properties = selection.get("properties", [])
        property_ids = {prop["entity_id"] for prop in selected_properties}
        selected_nodes = set(selection.get("node_ids", [])) | (property_ids & nodes.keys())
        selected_edges = set(selection.get("edge_ids", [])) | (property_ids & edges.keys())

        for edge_id in selection.get("edge_ids", []):
            if edge_id in edges:
                selected_nodes.update((edges[edge_id]["source"], edges[edge_id]["target"]))

        present = answer is not None
        present = present and set(expected_nodes) <= selected_nodes
        present = present and set(expected_edges) <= selected_edges

        for prop in case.get("expected_properties", []):
            reference = {"entity_id": prop["entity_id"], "key": prop["key"]}
            record = (nodes | edges).get(prop["entity_id"])
            present = present and reference in selection.get("properties", []) and record is not None

            if record is not None:
                properties = record["properties"]
                present = present and (
                    prop["key"] not in properties if prop.get("missing", False)
                    else prop["key"] in properties and type(properties[prop["key"]]) is type(prop["value"])
                    and properties[prop["key"]] == prop["value"]
                )
        checks["entity_property_presence"] = bool(present)
    return checks


async def evaluate_cases(session: Any, cases: list[dict]) -> dict:
    """Run live or scripted AgentSession queries, keeping controlled errors explicit."""
    rows = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("query"), str):
            raise ValueError("Each evaluation case must have a string query.")
        result_expectation = _result_expectation(case)
        case_id = case.get("id", str(index + 1))
        log.info("Evaluation case %d/%d: %s", index + 1, len(cases), case_id)
        answer, error, calls = None, None, []

        try:
            answer = await session.ask(case["query"])
            calls = answer.calls
        except Exception as exc:
            error = exc
            calls = getattr(exc, "calls", getattr(session, "last_calls", []))

        final_outputs = list(getattr(session, "last_final_outputs", []))
        checks = _score(
            case, answer, calls, error, result_expectation,
            getattr(session, "last_final_output", None), final_outputs,
        )
        failed = [name for name, value in checks.items() if value is False]
        log.info("Case %s: %s", case_id, "passed applicable checks" if not failed else "failed " + ", ".join(failed))
        rows.append({
            "id": case_id, "query": case["query"],
            "checks": checks, "calls": calls,
            "selection": answer.selection if answer is not None else None,
            "answer": answer.text if answer is not None else None,
            "execution_mode": getattr(answer, "execution_mode", getattr(session, "last_execution_mode", "agent")),
            "retry_count": getattr(session, "last_retry_count", 0),
            "retry_reason": getattr(session, "last_retry_reason", None),
            "final_outputs": final_outputs,
            "recovery": getattr(answer, "recovery", None),
            "error": {"code": getattr(error, "code", type(error).__name__), "message": str(error)} if error else None,
        })

    metrics = {}

    for name in METRICS:
        values = [row["checks"][name] for row in rows if row["checks"][name] is not None]
        numerator = sum(bool(value) for value in values)
        metrics[name] = {"numerator": numerator, "denominator": len(values), "rate": numerator / len(values) if values else None}
    log.info("Completed %d evaluation cases: %s", len(rows), ", ".join(
        f"{name}={metric['numerator']}/{metric['denominator']}" for name, metric in metrics.items()
    ))
    return {"case_count": len(rows), "metrics": metrics, "cases": rows}
