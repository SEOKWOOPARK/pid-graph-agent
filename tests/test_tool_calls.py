import asyncio
import json
import pytest
from types import SimpleNamespace
from jsonschema import Draft202012Validator
from graph_agent.tools import QueryContext, TOOL_SCHEMAS, build_tools, execute_tool


@pytest.mark.parametrize('name,args', [
    ('get_node', {'node_id': 'P4711'}),
    ('get_entity', {'entity_id': 'MNb47122'}),
    ('search_nodes', {'query': 'feed', 'node_type': 'Pump'}),
    ('get_neighbors', {'node_id': 'H1007', 'direction': 'upstream', 'node_type': 'Pump'}),
    ('get_connected_edges', {'node_id': 'P4711'}),
    ('get_edge', {'source': 'P4711', 'target': 'H1007'}),
    ('find_path', {'source': 'P4711', 'target': 'V2001'}),
    ('get_connected_subgraph', {'node_id': 'H1007', 'max_hops': 2, 'node_type': 'Valve'}),
])
def test_every_sdk_tool_is_callable(access, name, args):
    context = QueryContext(access)
    tool = next(t for t in build_tools() if t.name == name)
    Draft202012Validator.check_schema(tool.params_json_schema)
    response = asyncio.run(tool.on_invoke_tool(SimpleNamespace(context=context), json.dumps(args)))
    parsed = json.loads(response)
    assert parsed['ok'] is True
    assert parsed['error'] is None
    assert len(context.calls) == 1
    assert context.calls[0]['result'] == parsed


@pytest.mark.parametrize('name,args', [
    ('unknown', {}),
    ('get_node', {}),
    ('get_node', {'node_id': 'P4711', 'extra': True}),
    ('get_node', {'node_id': None}),
    ('get_node', {'node_id': '  '}),
    ('get_neighbors', {'node_id': 'P4711', 'direction': 'sideways'}),
    ('get_neighbors', {'node_id': 'P4711', 'resolve': 'true'}),
    ('get_connected_subgraph', {'node_id': 'P4711', 'max_hops': True}),
    ('get_connected_subgraph', {'node_id': 'P4711', 'max_hops': 1.0}),
    ('get_connected_subgraph', {'node_id': 'P4711', 'max_hops': 999}),
    ('get_node', []),
])
def test_bad_inputs_are_machine_readable_errors(access, name, args):
    result = execute_tool(QueryContext(access), name, args)
    assert result['ok'] is False
    assert result['error']['code'] == 'invalid_input'


@pytest.mark.parametrize('raw', ['{', '{"node_id":"P4711","node_id":"P999"}', '{"node_id": NaN}'])
def test_invalid_json_is_controlled(access, raw):
    context = QueryContext(access)
    tool = next(t for t in build_tools() if t.name == 'get_node')
    response = asyncio.run(tool.on_invoke_tool(SimpleNamespace(context=context), raw))
    assert json.loads(response)['ok'] is False
    assert len(context.calls) == 1


def test_ambiguous_resolution_blocks_dependent_graph_calls(access, monkeypatch):
    context = QueryContext(access)
    first = execute_tool(context, 'get_neighbors', {'node_id': 'H2000', 'direction': 'upstream', 'node_type': 'Pump', 'resolve': True})
    assert first['error']['code'] == 'ambiguous_resolution'
    assert {n['id'] for n in first['data']} == {'P5001', 'P5002'}
    def forbidden(*args, **kwargs):
        pytest.fail('An unresolved candidate must not be traversed')
    monkeypatch.setattr(access, 'get_neighbors', forbidden)
    second = execute_tool(context, 'get_neighbors', {'node_id': 'P5001'})
    assert second == first


def test_listing_multiple_nodes_is_not_ambiguous(access):
    context = QueryContext(access)
    result = execute_tool(context, 'get_neighbors', {'node_id': 'H2000', 'direction': 'upstream', 'node_type': 'Pump'})
    assert result['ok'] and len(result['data']) == 2
    assert context.resolution is None


def test_single_resolution_can_be_followed_by_query(access):
    context = QueryContext(access)
    result = execute_tool(context, 'get_neighbors', {'node_id': 'H1007', 'direction': 'upstream', 'node_type': 'Pump', 'resolve': True})
    assert [n['id'] for n in result['data']] == ['P4711']
    assert execute_tool(context, 'get_neighbors', {'node_id': 'P4711'})['ok']


def test_no_match_resolution_blocks_followup(access):
    context = QueryContext(access)
    result = execute_tool(context, 'search_nodes', {'query': 'unknown equipment', 'resolve': True})
    assert result['error']['code'] == 'no_matching_node'
    assert execute_tool(context, 'get_node', {'node_id': 'P4711'}) == result


def test_all_and_only_eight_primitives():
    assert len(build_tools()) == len(TOOL_SCHEMAS) == 8
