"""Recover terminal tool outcomes without guessing from malformed model output."""
import asyncio
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from graph_agent.agent import AgentSession
from graph_agent.cli import main
from graph_agent.errors import GroundingError, ModelOutputError
from graph_agent.evaluation import evaluate_cases
from graph_agent.grounding import recover_tool_outcome
from graph_agent.tools import QueryContext, execute_tool
from test_agent_eval import ScriptedProvider, final_output, tool_output


@pytest.mark.parametrize("case_index", range(5))
def test_reviewed_recovery_cases_keep_failed_schema_visible(access, case_index):
    cases = json.loads(Path(__file__).with_name("recovery_cases.json").read_text())
    case = cases[case_index]
    provider = ScriptedProvider([
        *(tool_output(call["name"], **call["arguments"]) for call in case["expected_calls"]),
        "{}",
    ])
    session = AgentSession(access, provider=provider)

    report = asyncio.run(evaluate_cases(session, [case]))

    row = report["cases"][0]
    assert row["selection"]["status"] == "not_found"
    assert row["answer"]
    assert row["error"] is None
    assert row["recovery"]["code"] == "grounding_error"
    assert row["recovery"]["message"]
    assert row["checks"]["successful_completion_rate"] is True
    assert row["checks"]["valid_schema_rate"] is False
    assert row["checks"]["argument_value_accuracy"] is True
    assert session.last_final_output == "{}"
    assert len(provider.backend.requests) == len(case["expected_calls"]) + 1
    assert not provider.backend.outputs
    json.dumps(report, allow_nan=False)


def test_invalid_json_after_failed_path_uses_only_retrieved_failure(access):
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("find_path", source="P5001", target="V2001"),
        '{"status":',
    ]))

    report = asyncio.run(evaluate_cases(session, [{
        "query": "Find a path from P5001 to V2001.", "expected_status": "not_found",
    }]))

    row = report["cases"][0]
    assert row["selection"]["status"] == "not_found"
    assert "P5001" in row["answer"] and "V2001" in row["answer"]
    assert row["calls"][0]["result"]["error"]["code"] == "path_not_found"
    assert row["recovery"]["code"] == "grounding_error"
    assert row["checks"]["valid_schema_rate"] is False
    assert row["checks"]["successful_completion_rate"] is True
    assert session.last_final_output == '{"status":'
    assert session.last_final_outputs == ['{"status":']
    assert session.last_retry_count == 0


def test_ambiguous_resolution_survives_missing_final_schema(access):
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("get_neighbors", node_id="H2000", direction="upstream", node_type="Pump", resolve=True),
        "{}",
    ]))

    answer = asyncio.run(session.ask("Describe the pump upstream of H2000."))

    assert answer.selection == {
        "status": "ambiguous", "node_ids": ["P5001", "P5002"],
        "edge_ids": [], "properties": [],
    }
    assert "P5001" in answer.text and "P5002" in answer.text
    assert answer.recovery["code"] == "grounding_error"


def test_filtered_empty_subgraph_does_not_render_its_context_nodes(access):
    context = QueryContext(access)
    result = execute_tool(context, "get_connected_subgraph", {
        "node_id": "T4001", "max_hops": 1, "direction": "upstream", "node_type": "Pump",
    })
    assert result["data"]["nodes"]
    assert result["data"]["matched_node_ids"] == []

    text, selection = recover_tool_outcome(context)

    assert text
    assert selection == {
        "status": "not_found", "node_ids": [], "edge_ids": [], "properties": [],
    }


@pytest.mark.parametrize("calls", [
    [],
    [("get_node", {"node_id": "P4711"})],
    [("get_node", {"node_id": "P4711"}), ("find_path", {"source": "P5001", "target": "V2001"})],
    [("get_neighbors", {"node_id": "T9001"}), ("get_node", {"node_id": "P4711"})],
    [("get_neighbors", {"node_id": "P4711", "direction": "sideways"})],
    [("get_neighbors", {"node_id": "T9001"}), ("get_connected_subgraph", {"node_id": "P4711", "max_hops": 99})],
], ids=["no-evidence", "nonempty", "success-then-failure", "empty-then-success", "invalid-input", "empty-then-invalid-input"])
def test_recovery_refuses_incomplete_or_invalid_evidence(access, calls):
    context = QueryContext(access)
    for name, arguments in calls:
        execute_tool(context, name, arguments)

    assert recover_tool_outcome(context) is None


@pytest.mark.parametrize("calls", [
    [],
    [("get_node", {"node_id": "P4711"})],
    [("get_node", {"node_id": "P4711"}), ("find_path", {"source": "P5001", "target": "V2001"})],
], ids=["no-evidence", "nonempty", "mixed"])
def test_session_still_rejects_invalid_final_without_conclusive_outcome(access, calls):
    session = AgentSession(access, provider=ScriptedProvider([
        *(tool_output(name, **arguments) for name, arguments in calls), "{}", "{}",
    ]))

    with pytest.raises(GroundingError):
        asyncio.run(session.ask("Tell me about P4711."))


def test_grounding_failure_recovery_does_not_mark_valid_raw_schema_invalid(access):
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("get_node", node_id="V9999"),
        final_output({"expected_nodes": ["P4711"]}),
    ]))

    report = asyncio.run(evaluate_cases(session, [{
        "query": "Tell me about V9999.", "expected_status": "not_found",
    }]))

    row = report["cases"][0]
    assert row["recovery"]["code"] == "grounding_error"
    assert row["checks"]["valid_schema_rate"] is True
    assert row["checks"]["successful_completion_rate"] is True
    assert row["selection"]["node_ids"] == []


def test_recovery_and_evidence_do_not_carry_to_next_question(access):
    raw_answer = final_output({"expected_nodes": ["P4711"]})
    session = AgentSession(access, provider=ScriptedProvider([
        tool_output("get_neighbors", node_id="T9001"), "{}",
        tool_output("get_node", node_id="P4711"), raw_answer,
        "{}", "{}",
    ]))

    async def run():
        recovered = await session.ask("What is connected to T9001?")
        assert recovered.recovery["code"] == "grounding_error"
        normal = await session.ask("Tell me about P4711.")
        assert normal.recovery is None
        assert len(normal.calls) == 1
        assert normal.calls[0]["arguments"]["node_id"] == "P4711"
        assert session.last_final_output == raw_answer
        with pytest.raises(GroundingError):
            await session.ask("Describe P4711 without retrieving it.")
        assert session.last_calls == []

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [GroundingError, ModelOutputError])
def test_one_shot_output_failure_shows_guidance_and_keeps_debug_detail(monkeypatch, capsys, caplog, error_type):
    async def ask(self, query):
        raise error_type("technical output detail")
    monkeypatch.setattr("graph_agent.agent.AgentSession.ask", ask)

    with caplog.at_level("DEBUG", logger="graph_agent.cli"):
        assert main(["--query", "Tell me about P4711."]) == 2

    captured = capsys.readouterr()
    assert f"Error [{error_type.code}]" in captured.err
    assert "Could not produce a verified answer." in captured.err
    assert "Try rephrasing the question or specifying exact node IDs." in captured.err
    assert "technical output detail" not in captured.err
    assert "technical output detail" in caplog.text


def test_interactive_output_failure_allows_next_question(monkeypatch, capsys):
    questions = iter(["bad query", "next query", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(questions))

    async def ask(self, query):
        if query == "bad query":
            raise ModelOutputError("invalid JSON details")
        return SimpleNamespace(text="next grounded answer")
    monkeypatch.setattr("graph_agent.agent.AgentSession.ask", ask)

    assert main([]) == 0
    captured = capsys.readouterr()
    assert "Could not produce a verified answer." in captured.err
    assert "invalid JSON details" not in captured.err
    assert "next grounded answer" in captured.out
