import json
import pytest
from graph_agent.errors import GroundingError
from graph_agent.graph_access import GraphAccess
from graph_agent.grounding import render_answer
from graph_agent.tools import QueryContext, execute_tool


def selection(status='answer', nodes=None, edges=None, properties=None):
    return json.dumps({'status': status, 'node_ids': nodes or [], 'edge_ids': edges or [], 'properties': properties or []})


def test_render_retrieved_property_not_model_value(access):
    context = QueryContext(access)
    execute_tool(context, 'get_edge', {'source': 'P4711', 'target': 'H1007'})
    text, _ = render_answer(selection(properties=[{'entity_id': 'MNb47122', 'key': 'design_temperature'}]), context)
    assert text == 'MNb47122: design_temperature = 120.'
    assert 'Celsius' not in text


def test_missing_property_is_explicit(access):
    context = QueryContext(access)
    execute_tool(context, 'get_node', {'node_id': 'P4711'})
    text, _ = render_answer(selection(properties=[{'entity_id': 'P4711', 'key': 'manufacturer'}]), context)
    assert "property 'manufacturer' is not available" in text


@pytest.mark.parametrize('selected_nodes', [[], ['P4711']], ids=['metadata-only', 'node-already-selected'])
def test_node_metadata_property_refs_render_one_node_summary(access, selected_nodes):
    context = QueryContext(access)
    execute_tool(context, 'get_node', {'node_id': 'P4711'})
    actual_properties = [
        {'entity_id': 'P4711', 'key': 'design_pressure'},
        {'entity_id': 'P4711', 'key': 'design_temperature'},
    ]
    raw = selection(nodes=selected_nodes, properties=[
        {'entity_id': 'P4711', 'key': key} for key in ['id', 'name', 'type', 'description']
    ] + actual_properties)

    text, selected = render_answer(raw, context)

    assert text.splitlines() == [
        'P4711 (Pump): Feed Pump. Feed pump supplying heat exchanger H1007',
        'P4711: design_pressure = 6.5.',
        'P4711: design_temperature = 80.',
    ]
    assert selected['node_ids'] == ['P4711']
    assert selected['properties'] == actual_properties


def test_node_metadata_cannot_expose_an_unretrieved_node(access):
    context = QueryContext(access)
    execute_tool(context, 'get_node', {'node_id': 'P4711'})
    with pytest.raises(GroundingError, match='H1007.*not retrieved'):
        render_answer(selection(properties=[{'entity_id': 'H1007', 'key': 'name'}]), context)


def test_actual_same_named_properties_take_precedence_including_null(graph):
    node = next(node for node in graph['nodes'] if node['id'] == 'P4711')
    node['properties'].update({'name': 'Property-specific name', 'type': None})
    context = QueryContext(GraphAccess(graph))
    execute_tool(context, 'get_node', {'node_id': 'P4711'})
    properties = [{'entity_id': 'P4711', 'key': key} for key in ['name', 'type']]

    text, selected = render_answer(selection(properties=properties), context)

    assert text.splitlines() == [
        'P4711: name = "Property-specific name".',
        'P4711: type = null.',
    ]
    assert selected['node_ids'] == []
    assert selected['properties'] == properties


def test_edge_metadata_is_not_normalized_as_node_metadata(access):
    context = QueryContext(access)
    execute_tool(context, 'get_edge', {'source': 'P4711', 'target': 'H1007'})
    raw = selection(properties=[{'entity_id': 'MNb47122', 'key': 'type'}])

    text, selected = render_answer(raw, context)

    assert text == "MNb47122: property 'type' is not available in the graph."
    assert selected == json.loads(raw)


@pytest.mark.parametrize('raw', [
    selection(nodes=['P999999']),
    selection(nodes=['P4711']),
    selection(edges=['MNb47122']),
    selection(properties=[{'entity_id': 'P4711', 'key': 'design_pressure'}]),
    selection(),
    selection(status='not_found'),
    selection(status='ambiguous'),
    selection(status='unavailable'),
    '{"status":"answer","node_ids":[],"edge_ids":[],"properties":[],"text":"The pipe is stainless steel"}',
    '{"status":"answer","node_ids":[],"edge_ids":[],"properties":[{"entity_id":"MNb47122","key":"material","value":"SS"}]}',
    '{"status":"answer","status":"unavailable","node_ids":[],"edge_ids":[],"properties":[]}',
    'The pipe is stainless steel.',
])
def test_unretrieved_or_unverifiable_claims_rejected(access, raw):
    with pytest.raises(GroundingError):
        render_answer(raw, QueryContext(access))


def test_graph_presence_is_insufficient_without_retrieval(access):
    first = QueryContext(access)
    execute_tool(first, 'get_node', {'node_id': 'P4711'})
    assert render_answer(selection(nodes=['P4711']), first)[0].startswith('P4711')
    with pytest.raises(GroundingError):
        render_answer(selection(nodes=['P4711']), QueryContext(access))


def test_direction_comes_from_returned_edge(access):
    context = QueryContext(access)
    execute_tool(context, 'get_connected_edges', {'node_id': 'P4711'})
    text, _ = render_answer(selection(edges=['MNb47122']), context)
    assert 'P4711 → H1007' in text


@pytest.mark.parametrize('raw', [
    selection(nodes=['P5001']),
    selection(properties=[{'entity_id': 'P5001', 'key': 'description'}]),
], ids=['node', 'metadata-property'])
def test_ambiguity_cannot_be_narrowed_by_final_output(access, raw):
    context = QueryContext(access)
    execute_tool(context, 'get_neighbors', {'node_id': 'H2000', 'direction': 'upstream', 'node_type': 'Pump', 'resolve': True})
    text, selected = render_answer(raw, context)
    assert selected['status'] == 'ambiguous'
    assert 'P5001' in text and 'P5002' in text


def test_empty_and_missing_node_results(access):
    empty = QueryContext(access)
    execute_tool(empty, 'get_neighbors', {'node_id': 'T9001'})
    assert 'No matching results' in render_answer(selection(status='not_found'), empty)[0]
    missing = QueryContext(access)
    execute_tool(missing, 'get_node', {'node_id': 'P999999'})
    assert 'does not exist' in render_answer(selection(status='not_found'), missing)[0]


@pytest.mark.parametrize('raw', [
    selection(nodes=['P4711']),
    selection(properties=[{'entity_id': 'P4711', 'key': 'name'}]),
], ids=['node', 'metadata-property'])
def test_resolved_no_match_cannot_be_overridden(access, raw):
    context = QueryContext(access)
    execute_tool(context, 'search_nodes', {'query': 'nothing', 'resolve': True})
    text, selected = render_answer(raw, context)
    assert selected['status'] == 'not_found'
    assert text == 'No matching node was found.'
