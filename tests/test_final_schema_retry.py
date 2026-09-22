"""Correct completed final selections without trusting model-supplied values."""
import asyncio
import json

import pytest

from graph_agent.agent import AgentSession
from graph_agent.config import Config
from graph_agent.errors import FinalSelectionError, GroundingError, ModelOutputError
from graph_agent.evaluation import evaluate_cases
from test_agent_eval import ScriptedProvider, final_output, tool_output


BAD_FINAL = json.dumps({
    "status": "answer", "node_ids": ["H1007"], "edge_ids": [],
    "properties": {"design_pressure": 60},
})
GOOD_FINAL = json.dumps({
    "status": "answer", "node_ids": ["H1007"], "edge_ids": [],
    "properties": [{"entity_id": "H1007", "key": "design_pressure"}],
})


@pytest.mark.parametrize("value", [60, 650])
def test_node_description_bad_property_dictionary_gets_one_schema_correction(access, value):
    raw = BAD_FINAL.replace('60', str(value))
    provider = ScriptedProvider([tool_output("get_node", node_id="H1007"), raw, GOOD_FINAL])
    session = AgentSession(access, provider=provider)
    report = asyncio.run(evaluate_cases(session, [{
        "query": "Tell me about H1007.",
        "expected_calls": [{"name": "get_node", "arguments": {"node_id": "H1007"}}],
        "expected_nodes": ["H1007"],
    }]))

    row = report["cases"][0]
    assert row["error"] is None
    assert "H1007 (HeatExchanger): Main Heat Exchanger" in row["answer"]
    assert "design_pressure = 60." in row["answer"]
    assert "650" not in row["answer"]
    assert row["checks"]["successful_completion_rate"] is True
    assert row["checks"]["valid_schema_rate"] is False
    assert row["final_outputs"] == [raw, GOOD_FINAL]
    assert row["retry_count"] == 1
    assert row["retry_reason"] == "invalid_final_schema"
    assert row["recovery"] is None
    assert len(row["calls"]) == 1
    assert len(provider.backend.requests) == 3
    history = provider.backend.requests[-1][0]
    assert [m["content"] for m in history if m["role"] == "user"] == ["Tell me about H1007."]
    assert any(m["role"] == "tool" and 'design_pressure' in m["content"] for m in history)
    assert any(m["role"] == "assistant" and m["content"] == raw for m in history)
    assert history[-1]["role"] == "system"
    assert "schema" in history[-1]["content"]


@pytest.mark.parametrize("max_turns", [2, 3])
def test_schema_correction_shares_original_turn_budget(access, max_turns):
    outputs = [tool_output("get_node", node_id="H1007"), BAD_FINAL, GOOD_FINAL][:max_turns]
    provider = ScriptedProvider(outputs)
    session = AgentSession(access, config=Config(max_turns=max_turns), provider=provider)
    if max_turns == 2:
        with pytest.raises(FinalSelectionError):
            asyncio.run(session.ask("Tell me about H1007."))
        assert session.last_retry_count == 0
        assert session.last_final_outputs == [BAD_FINAL]
    else:
        assert asyncio.run(session.ask("Tell me about H1007.")).selection["node_ids"] == ["H1007"]
        assert session.last_retry_count == 1
        assert session.last_final_outputs == [BAD_FINAL, GOOD_FINAL]
    assert len(provider.backend.requests) == max_turns
    assert not provider.backend.outputs


@pytest.mark.parametrize("corrected,error_type", [
    (BAD_FINAL, FinalSelectionError),
    (final_output({"expected_nodes": ["P4711"]}), GroundingError),
    ('{"status":', FinalSelectionError),
    ('<tool_call>{"name":', ModelOutputError),
])
def test_failed_correction_stays_blocked_without_third_attempt(access, corrected, error_type):
    provider = ScriptedProvider([tool_output("get_node", node_id="H1007"), BAD_FINAL, corrected])
    session = AgentSession(access, provider=provider)
    with pytest.raises(error_type):
        asyncio.run(session.ask("Tell me about H1007."))
    assert session.last_retry_count == 1
    assert len(provider.backend.requests) == 3
    assert session.last_final_output == (None if error_type is ModelOutputError else corrected)


def test_no_tool_then_invalid_schema_does_not_get_another_correction(access):
    raw = final_output({"expected_nodes": ["H1007"]})
    provider = ScriptedProvider([raw, tool_output("get_node", node_id="H1007"), BAD_FINAL])
    session = AgentSession(access, provider=provider)
    with pytest.raises(FinalSelectionError):
        asyncio.run(session.ask("Tell me about H1007."))
    assert session.last_retry_count == 1
    assert session.last_retry_reason == "no_tool_evidence"
    assert session.last_final_outputs == [raw, BAD_FINAL]
    assert len(provider.backend.requests) == 3


def test_malformed_tool_call_after_positive_lookup_does_not_start_schema_retry(access):
    provider = ScriptedProvider([tool_output("get_node", node_id="H1007"), '<tool_call>{"name":'])
    session = AgentSession(access, provider=provider)
    with pytest.raises(ModelOutputError):
        asyncio.run(session.ask("Tell me about H1007."))
    assert session.last_retry_count == 0
    assert session.last_final_outputs == []


def test_schema_retry_state_resets_for_next_question(access):
    provider = ScriptedProvider([
        tool_output("get_node", node_id="H1007"), BAD_FINAL, GOOD_FINAL,
        tool_output("get_node", node_id="H1007"), GOOD_FINAL,
    ])
    session = AgentSession(access, provider=provider)

    async def run():
        first = await session.ask("Tell me about H1007.")
        assert session.last_retry_reason == "invalid_final_schema"
        second = await session.ask("Tell me about H1007 again.")
        assert second.calls is not first.calls
        assert len(second.calls) == 1
        assert session.last_retry_count == 0
        assert session.last_retry_reason is None
        assert session.last_final_outputs == [GOOD_FINAL]

    asyncio.run(run())
