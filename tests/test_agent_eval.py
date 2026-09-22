import asyncio
import json
import runpy
import pytest
from copy import deepcopy
from pathlib import Path
from agents import ModelProvider
from graph_agent.agent import AgentSession, Answer
from graph_agent.config import Config
from graph_agent.errors import GroundingError, InvalidInputError, ModelError
from graph_agent.evaluation import evaluate_cases
from graph_agent.local_qwen import Generation
from graph_agent.model_provider import LocalQwenModel

HERE = Path(__file__).parent
GOLDEN = json.loads((HERE / "eval_cases.json").read_text())
PARAPHRASES = json.loads((HERE / "paraphrase_cases.json").read_text())
FUZZ = json.loads((HERE / "fuzz_cases.json").read_text())


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def generate(self, messages, tools):
        self.requests.append((deepcopy(messages), deepcopy(tools)))
        assert len(tools) == 8
        assert self.outputs, "The SDK requested an unexpected extra generation."
        return Generation(self.outputs.pop(0), input_tokens=10, output_tokens=10)


class ScriptedProvider(ModelProvider):
    def __init__(self, outputs):
        self.backend = ScriptedBackend(outputs)
        self.model = LocalQwenModel(self.backend)

    def get_model(self, model_name):
        return self.model


def tool_output(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def final_output(case):
    status = case.get("expected_status", "answer")
    return json.dumps({
        "status": status[0] if isinstance(status, list) else status,
        "node_ids": case.get("expected_selected_nodes", case.get("expected_nodes", [])),
        "edge_ids": case.get("expected_selected_edges", case.get("expected_edges", [])),
        "properties": [{"entity_id": p["entity_id"], "key": p["key"]} for p in case.get("expected_properties", [])],
    })


def scripted_outputs(case):
    if case["id"] == "empty_input":
        return []
    if case["id"] == "excessive_hops":
        return [tool_output("get_connected_subgraph", node_id="H1007", max_hops=999999)]
    if case["id"] == "invalid_direction":
        return [tool_output("get_neighbors", node_id="P4711", direction="sideways")]
    calls = case.get("expected_calls", [])
    if case["id"] == "nonexistent_node":
        calls = [{"name": "get_node", "arguments": {"node_id": "P999999"}}]
    if case["id"] == "nonexistent_path_target":
        calls = [{"name": "find_path", "arguments": {"source": "P4711", "target": "NOTHING"}}]
    return [tool_output(call["name"], **call["arguments"]) for call in calls] + [final_output(case)]


@pytest.mark.parametrize("case", GOLDEN + PARAPHRASES, ids=lambda case: case["id"])
def test_reviewed_cases_through_real_sdk_runner(access, monkeypatch, case):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = ScriptedProvider(scripted_outputs(case))
    session = AgentSession(access, provider=provider)
    report = asyncio.run(evaluate_cases(session, [case]))
    assert all(value is None or value for value in report["cases"][0]["checks"].values()), report
    assert not provider.backend.outputs
    assert any(message["role"] == "tool" for message in provider.backend.requests[-1][0])
    assert session.run_config.tracing_disabled
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("case", FUZZ, ids=lambda case: case["id"])
def test_reviewed_fuzz_cases_finish_or_fail_explicitly(access, case):
    session = AgentSession(access, provider=ScriptedProvider(scripted_outputs(case)))
    report = asyncio.run(evaluate_cases(session, [case]))
    assert report["metrics"]["successful_completion_rate"]["rate"] == 1, report
    if case["id"] in {"excessive_hops", "invalid_direction"}:
        assert report["metrics"]["valid_schema_rate"]["rate"] == 0
        assert report["cases"][0]["error"]["code"] == "model_output_error"


def test_omitted_defaults_and_extra_calls_are_accepted(access):
    case = GOLDEN[0]
    provider = ScriptedProvider([
        tool_output("get_node", node_id="P4711"),
        tool_output("get_neighbors", node_id="P4711"),
        final_output(case),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 1


def test_node_metadata_normalization_reaches_report_but_preserves_raw_model_output(access):
    case = {
        "query": "Describe P4711, including its name, type, and description.",
        "expected_calls": [{"name": "get_node", "arguments": {"node_id": "P4711"}}],
        "expected_nodes": ["P4711"],
    }
    raw = json.dumps({
        "status": "answer", "node_ids": [], "edge_ids": [],
        "properties": [{"entity_id": "P4711", "key": key} for key in ["name", "type", "description"]],
    })
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("get_node", node_id="P4711"), raw,
    ]))

    report = asyncio.run(evaluate_cases(session, [case]))

    row = report["cases"][0]
    assert row["answer"] == 'P4711 (Pump): Feed Pump. Feed pump supplying heat exchanger H1007'
    assert row["selection"] == {
        "status": "answer", "node_ids": ["P4711"], "edge_ids": [], "properties": [],
    }
    assert all(value is None or value for value in row["checks"].values()), report
    assert session.last_final_output == raw


def test_single_expected_call_supplies_the_retrieval_result(access):
    case = {
        "query": "What is connected to P4711?",
        "expected_calls": [{"name": "get_neighbors", "arguments": {"node_id": "P4711", "direction": "both"}}],
        "expected_nodes": ["H1007", "V1005"],
    }
    provider = ScriptedProvider([
        tool_output("get_neighbors", node_id="P4711", direction="both"),
        final_output(case),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 1
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1


def test_result_index_uses_middle_expected_call_despite_later_context(access):
    case = next(case for case in GOLDEN if case["id"] == "indirect_target")
    outputs = scripted_outputs(case)
    outputs.insert(-1, tool_output("get_node", node_id="P4711"))
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=ScriptedProvider(outputs)), [case]))
    assert case["result_call_index"] == 1
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1
    assert report["metrics"]["node_resolution_accuracy"]["rate"] == 1


@pytest.mark.parametrize("actual_call", [
    None,
    {"name": "get_connected_edges", "arguments": {"node_id": "T9001"}},
    {"name": "get_neighbors", "arguments": {"node_id": "T9001", "direction": "upstream"}},
], ids=["missing-call", "wrong-tool", "wrong-arguments"])
def test_empty_expected_result_requires_the_matching_call(access, actual_call):
    case = next(case for case in GOLDEN if case["id"] == "disconnected")
    outputs = [] if actual_call is None else [tool_output(actual_call["name"], **actual_call["arguments"])]
    outputs.append(final_output(case))
    if actual_call is None:
        outputs.append(final_output(case))
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=ScriptedProvider(outputs)), [case]))
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 0
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 0


def test_correct_call_does_not_pass_an_incorrect_expected_result(access):
    case = dict(GOLDEN[0], expected_nodes=["H1007"])
    session = AgentSession(access, provider=ScriptedProvider(scripted_outputs(case)))
    report = asyncio.run(evaluate_cases(session, [case]))
    assert report["metrics"]["tool_call_accuracy"]["rate"] == 1
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 1
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 0


def test_resolution_context_is_not_counted_as_target_retrieval(access):
    case = next(case for case in GOLDEN if case["id"] == "indirect_target")
    provider = ScriptedProvider([
        tool_output("get_neighbors", node_id="H1007", direction="upstream", node_type="Pump", resolve=True),
        tool_output("get_neighbors", node_id="H1007"),
        final_output({"expected_nodes": ["P4711"]}),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["node_resolution_accuracy"]["rate"] == 1
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 0
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 0
    assert report["metrics"]["entity_property_presence"]["rate"] == 0


def test_reversed_call_sequence_does_not_pass(access):
    case = next(case for case in GOLDEN if case["id"] == "indirect_target")
    calls = list(reversed(case["expected_calls"]))
    provider = ScriptedProvider([tool_output(call["name"], **call["arguments"]) for call in calls] + [final_output(case)])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["tool_call_accuracy"]["rate"] == 0


def test_property_presence_requires_model_selection_and_actual_value(access):
    case = next(case for case in GOLDEN if case["id"] == "pipe_temperature")
    incomplete = dict(case, expected_properties=[])
    provider = ScriptedProvider([tool_output("get_edge", source="P4711", target="H1007"), final_output(incomplete)])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1
    assert report["metrics"]["entity_property_presence"]["rate"] == 0
    assert report["cases"][0]["selection"]["properties"] == []
    assert json.loads(report["cases"][0]["final_outputs"][0])["properties"] == []
    incorrect = deepcopy(case)
    incorrect["expected_properties"][0]["value"] = 121
    provider = ScriptedProvider(scripted_outputs(incorrect))
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [incorrect]))
    assert report["metrics"]["entity_property_presence"]["rate"] == 0


def test_grounded_entity_selection_is_preserved_without_matching_question_intent(access):
    case = {
        "query": "What is the material of the pipe from H1007 to V2001?",
        "expected_calls": [{"name": "get_entity", "arguments": {"entity_id": "MNb20001"}}],
        "expected_edges": ["MNb20001"],
        "expected_properties": [{"entity_id": "MNb20001", "key": "material", "missing": True}],
    }
    raw = final_output({"expected_properties": [{"entity_id": "MNb20001", "key": "design_temperature"}]})
    provider = ScriptedProvider([tool_output("get_entity", entity_id="MNb20001"), raw])
    session = AgentSession(access, provider=provider)
    report = asyncio.run(evaluate_cases(session, [case]))

    row = report["cases"][0]
    assert row["error"] is None
    assert row["execution_mode"] == "agent"
    assert row["selection"] == json.loads(raw)
    assert row["answer"] == "MNb20001: design_temperature = 100."
    assert row["checks"]["retrieval_exact_match"] is True
    assert row["checks"]["valid_schema_rate"] is True
    assert row["checks"]["entity_property_presence"] is False
    assert row["retry_count"] == 0
    assert row["final_outputs"] == [raw]
    assert session.last_final_output == raw
    assert "adjustment" not in row
    assert not provider.backend.outputs


def test_compound_temperature_answer_accepts_get_entity_node_evidence(access):
    case = {
        "query": "Report the design_temperature of P4711 and of the pipe from P4711 to H1007 separately.",
        "expected_calls": [
            {"name": "get_entity", "arguments": {"entity_id": "P4711"}},
            {"name": "get_edge", "arguments": {"source": "P4711", "target": "H1007"}},
        ],
        "expected_properties": [
            {"entity_id": "P4711", "key": "design_temperature", "value": 80},
            {"entity_id": "MNb47122", "key": "design_temperature", "value": 120},
        ],
    }
    raw = final_output(case)
    provider = ScriptedProvider([
        tool_output("get_entity", entity_id="P4711"),
        tool_output("get_edge", source="P4711", target="H1007"),
        raw,
    ])
    session = AgentSession(access, provider=provider)

    report = asyncio.run(evaluate_cases(session, [case]))

    row = report["cases"][0]
    assert row["error"] is None
    assert row["execution_mode"] == "agent"
    assert row["answer"] == "P4711: design_temperature = 80.\nMNb47122: design_temperature = 120."
    assert row["selection"] == json.loads(raw)
    assert row["retry_count"] == 0
    assert row["retry_reason"] is None
    assert row["final_outputs"] == [raw]
    assert all(check is None or check for check in row["checks"].values()), row
    assert [call["name"] for call in row["calls"]] == ["get_entity", "get_edge"]
    assert len(provider.backend.requests) == 3
    assert not provider.backend.outputs


@pytest.mark.parametrize("entity_kind", ["node", "edge"])
def test_property_reference_counts_as_its_retrieved_entity_kind(access, entity_kind):
    if entity_kind == "edge":
        case = next(case for case in GOLDEN if case["id"] == "pipe_temperature")
        call = tool_output("get_edge", source="P4711", target="H1007")
    else:
        case = {
            "query": "Show the design pressure of P4711.",
            "expected_calls": [{"name": "get_node", "arguments": {"node_id": "P4711"}}],
            "expected_nodes": ["P4711"],
            "expected_properties": [{"entity_id": "P4711", "key": "design_pressure", "value": 6.5}],
        }
        call = tool_output("get_node", node_id="P4711")
    only_property = dict(case, expected_selected_nodes=[], expected_selected_edges=[])
    outputs = [call, final_output(only_property)]
    session = AgentSession(access, provider=ScriptedProvider(outputs))
    report = asyncio.run(evaluate_cases(session, [case]))
    assert report["metrics"]["entity_property_presence"]["rate"] == 1
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1
    wrong_kind = dict(case)
    entity_id = case["expected_properties"][0]["entity_id"]
    wrong_kind["expected_selected_nodes" if entity_kind == "edge" else "expected_selected_edges"] = [entity_id]
    session = AgentSession(access, provider=ScriptedProvider(outputs))
    report = asyncio.run(evaluate_cases(session, [wrong_kind]))
    assert report["metrics"]["entity_property_presence"]["rate"] == 0


def test_empty_expected_entities_do_not_inflate_presence_denominator(access):
    case = next(case for case in GOLDEN if case["id"] == "disconnected")
    session = AgentSession(access, provider=ScriptedProvider(scripted_outputs(case)))
    report = asyncio.run(evaluate_cases(session, [case]))
    assert report["metrics"]["entity_property_presence"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1


def test_rendered_path_edges_cover_endpoints_without_changing_node_selection(access):
    case = next(case for case in GOLDEN if case["id"] == "directed_path")
    provider = ScriptedProvider([
        tool_output("find_path", source="P4711", target="V2001"),
        final_output(dict(case, expected_selected_nodes=[])),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 1
    assert report["metrics"]["entity_property_presence"]["rate"] == 1
    assert report["cases"][0]["selection"]["node_ids"] == []
    assert json.loads(report["cases"][0]["final_outputs"][0])["node_ids"] == []
    assert all(node_id in report["cases"][0]["answer"] for node_id in case["expected_nodes"])


def test_edge_property_sentence_does_not_count_unrendered_endpoints(access):
    case = dict(next(case for case in GOLDEN if case["id"] == "pipe_temperature"), expected_selected_nodes=["P4711"])
    provider = ScriptedProvider([
        tool_output("get_edge", source="P4711", target="H1007"),
        final_output(dict(case, expected_selected_nodes=[], expected_selected_edges=[])),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["entity_property_presence"]["rate"] == 0


def test_equivalent_search_words_preserve_final_entities_but_require_expected_retrieval_arguments(access):
    case = next(case for case in FUZZ if case["id"] == "ambiguous_description")
    provider = ScriptedProvider([
        tool_output("search_nodes", query="auxiliary feed pump", node_type="Pump", resolve=True),
        final_output(case),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 0
    assert report["metrics"]["entity_property_presence"]["rate"] == 1
    assert report["metrics"]["tool_call_accuracy"]["rate"] == 1
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 0


def test_tool_selection_is_scored_independently_from_argument_sequence(access):
    case = {
        "query": "Find neighbors and pipes of P4711.",
        "expected_calls": [
            {"name": "get_neighbors", "arguments": {"node_id": "P4711"}},
            {"name": "get_connected_edges", "arguments": {"node_id": "P4711"}},
        ],
    }
    provider = ScriptedProvider([
        tool_output("get_neighbors", node_id="H1007"),
        tool_output("get_connected_edges", node_id="P4711"),
        tool_output("get_neighbors", node_id="P4711"),
        final_output({"expected_nodes": ["H1007"]}),
    ])
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["tool_call_accuracy"]["rate"] == 1
    assert report["metrics"]["required_argument_accuracy"]["rate"] == 1
    assert report["metrics"]["argument_value_accuracy"]["rate"] == 0


@pytest.mark.parametrize("malformed_recorded_call", [False, True])
def test_model_failure_does_not_hide_observed_invalid_arguments(malformed_recorded_call):
    class FailingSession:
        last_calls = [{
            "name": "get_node", "arguments": {},
            "result": {"ok": False, "data": None, "error": {"code": "invalid_input", "message": "missing node_id"}},
        }] if malformed_recorded_call else []

        async def ask(self, query):
            raise ModelError("model failed")
    report = asyncio.run(evaluate_cases(FailingSession(), [{"query": "Tell me about P4711."}]))
    assert report["metrics"]["valid_schema_rate"] == (
        {"numerator": 0, "denominator": 1, "rate": 0.0} if malformed_recorded_call
        else {"numerator": 0, "denominator": 0, "rate": None}
    )
    assert report["metrics"]["successful_completion_rate"]["rate"] == 0
    assert report["cases"][0]["error"]["code"] == "model_error"
    assert bool(report["cases"][0]["calls"]) is malformed_recorded_call


@pytest.mark.parametrize("selection,expected_schema_rate", [
    ({"status": "answer", "node_ids": ["H1007"], "edge_ids": [], "properties": []}, 1),
    ({"status": "answer", "node_ids": ["H1007"]}, 0),
])
def test_schema_validity_is_distinct_from_grounding(access, selection, expected_schema_rate):
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("get_node", node_id="P4711"), json.dumps(selection),
        *([json.dumps(selection)] if expected_schema_rate == 0 else []),
    ]))
    report = asyncio.run(evaluate_cases(session, [{"query": "Tell me about P4711."}]))
    assert report["cases"][0]["error"]["code"] == "grounding_error"
    assert report["metrics"]["valid_schema_rate"]["rate"] == expected_schema_rate
    assert report["metrics"]["successful_completion_rate"]["rate"] == 0


def test_evaluation_logs_case_progress_and_completion(access, caplog):
    case = GOLDEN[0]
    session = AgentSession(access, provider=ScriptedProvider(scripted_outputs(case)))
    with caplog.at_level("INFO", logger="graph_agent.evaluation"):
        asyncio.run(evaluate_cases(session, [case]))
    messages = [record.message for record in caplog.records if record.name == "graph_agent.evaluation"]
    assert messages[0] == "Evaluation case 1/1: direct_neighbors"
    assert "Completed 1 evaluation cases:" in messages[-1]
    assert "retrieval_exact_match=1/1" in messages[-1]
    assert case["query"] not in "\n".join(messages)


def test_subgraph_context_cannot_satisfy_filtered_retrieval(access):
    case = deepcopy(next(case for case in GOLDEN if case["id"] == "two_hop_valves"))
    case["expected_nodes"].append("T3001")
    provider = ScriptedProvider(scripted_outputs(case))
    report = asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert report["metrics"]["retrieval_exact_match"]["rate"] == 0


def test_unknown_exception_never_counts_as_safe_completion():
    class CrashingSession:
        async def ask(self, query):
            raise RuntimeError("unexpected defect")
    report = asyncio.run(evaluate_cases(CrashingSession(), [{"query": "bad input", "expected_error": "RuntimeError"}]))
    assert report["metrics"]["successful_completion_rate"]["rate"] == 0
    assert report["cases"][0]["error"]["code"] == "RuntimeError"


def test_unspecified_controlled_error_is_not_a_success():
    class FailingSession:
        async def ask(self, query):
            raise InvalidInputError("invalid")
    report = asyncio.run(evaluate_cases(FailingSession(), [{"query": "valid question"}]))
    assert report["metrics"]["successful_completion_rate"]["rate"] == 0


def test_expected_error_does_not_accept_an_ordinary_answer():
    class AnsweringSession:
        async def ask(self, query):
            return Answer("unexpected", {"status": "answer", "node_ids": [], "edge_ids": [], "properties": []}, [])
    report = asyncio.run(evaluate_cases(AnsweringSession(), [{"query": "", "expected_error": "invalid_input"}]))
    assert report["metrics"]["successful_completion_rate"]["rate"] == 0


@pytest.mark.parametrize("expectations", [{}, {"expected_calls": []}])
def test_retrieval_requires_expected_calls_before_running_the_model(access, expectations):
    provider = ScriptedProvider([])
    case = {"query": "What is connected to P4711?", "expected_nodes": [], **expectations}
    with pytest.raises(ValueError, match="expected_calls"):
        asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert provider.backend.requests == []


@pytest.mark.parametrize("index", [True, False, "0", -1, 1, None, 0.5])
def test_invalid_result_index_is_rejected_before_running_the_model(access, index):
    provider = ScriptedProvider([])
    case = dict(GOLDEN[0], result_call_index=index)
    with pytest.raises(ValueError, match="result_call_index"):
        asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert provider.backend.requests == []


def test_multiple_expected_calls_require_a_result_index_before_running_the_model(access):
    case = dict(next(case for case in GOLDEN if case["id"] == "indirect_target"))
    case.pop("result_call_index")
    provider = ScriptedProvider([])
    with pytest.raises(ValueError, match="result_call_index"):
        asyncio.run(evaluate_cases(AgentSession(access, provider=provider), [case]))
    assert provider.backend.requests == []


def test_agent_does_not_reuse_previous_question_evidence(access):
    final = final_output({"expected_nodes": ["P4711"]})
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("get_node", node_id="P4711"), final, final, final,
    ]))
    async def run():
        await session.ask("Tell me about P4711.")
        with pytest.raises(GroundingError):
            await session.ask("Show P4711 again without retrieving it")
    asyncio.run(run())


def test_sdk_turn_exhaustion_is_explicit(access):
    session = AgentSession(access, config=Config(max_turns=1), provider=ScriptedProvider([tool_output("get_node", node_id="P4711")]))
    report = asyncio.run(evaluate_cases(session, [{"query": "Tell me about P4711.", "expected_error": "model_error"}]))
    assert report["metrics"]["successful_completion_rate"]["rate"] == 1
    assert len(report["cases"][0]["calls"]) == 1


def test_empty_suite_metrics_are_explicit():
    report = asyncio.run(evaluate_cases(None, []))
    assert report["case_count"] == 0
    assert all(metric == {"numerator": 0, "denominator": 0, "rate": None} for metric in report["metrics"].values())


def test_live_eval_entrypoint_writes_json_without_loading_for_empty_suite(tmp_path):
    cases = tmp_path / "empty.json"
    cases.write_text("[]")
    output = tmp_path / "report.json"
    main = runpy.run_path(str(HERE.parent / "scripts/run_eval.py"))["main"]
    assert main(["--cases", str(cases), "--output", str(output), "--model", "Qwen/Qwen3-0.6B"]) == 0
    report = json.loads(output.read_text())
    assert report["mode"] == "live_local_qwen"
    assert report["model"] == "Qwen/Qwen3-0.6B"
    assert report["case_count"] == 0
