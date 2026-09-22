import json
import pytest
from graph_agent.errors import GraphValidationError
from graph_agent.graph_loader import load_graph
from graph_agent.schema import GRAPH_SCHEMA, GRAPH_VALIDATOR, loads_json


def test_fixture_loads_json_records_with_directed_parallel_edges(graph):
    assert isinstance(graph, dict)
    assert len(graph["nodes"]) == 11
    assert len(graph["edges"]) == 9
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {edge["id"]: edge for edge in graph["edges"]}
    assert (edges["MNb47122"]["source"], edges["MNb47122"]["target"]) == ("P4711", "H1007")
    assert not any(edge["source"] == "H1007" and edge["target"] == "P4711" for edge in graph["edges"])
    assert len([edge for edge in graph["edges"] if edge["source"] == "V2001" and edge["target"] == "T4001"]) == 2
    assert nodes["P4711"]["properties"]["design_pressure"] == 6.5
    assert edges["MNb47122"]["properties"]["design_temperature"] == 120


def test_empty_graph_document_is_valid(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"nodes": [], "edges": []}')
    assert load_graph(path) == {"nodes": [], "edges": []}


def test_schema_itself_is_valid():
    GRAPH_VALIDATOR.check_schema(GRAPH_SCHEMA)


@pytest.mark.parametrize("source", ["{", "[]", "null", '{"nodes": []}', '{"nodes": [], "edges": [], "extra": 1}'])
def test_bad_document_rejected(tmp_path, source):
    path = tmp_path / "graph.json"
    path.write_text(source)
    with pytest.raises(GraphValidationError):
        load_graph(path)


@pytest.mark.parametrize("source", [
    '{"nodes": [], "nodes": [], "edges": []}',
    '{"properties": {"value": 1, "value": 2}}',
    '{"value": NaN}',
    '{"value": Infinity}',
    '{"value": -Infinity}',
    '{"value": 1e999}',
    '{"value": 1e1000}',
])
def test_strict_json_rejects_ambiguous_or_nonfinite_data(source):
    with pytest.raises(ValueError):
        loads_json(source)


def test_deeply_nested_json_is_a_controlled_parser_error():
    with pytest.raises(ValueError, match="nesting"):
        loads_json("[" * 2000 + "]" * 2000)


@pytest.mark.parametrize("source", ["[" * 2000 + "]" * 2000, '{"nodes": [], "edges": [], "value": 1e1000}'])
def test_deep_and_overflowed_json_is_controlled_at_startup(tmp_path, source):
    path = tmp_path / "graph.json"
    path.write_text(source)
    with pytest.raises(GraphValidationError):
        load_graph(path)


@pytest.mark.parametrize("mutation", [
    lambda doc: doc["nodes"].append(doc["nodes"][0].copy()),
    lambda doc: doc["edges"].append(doc["edges"][0].copy()),
    lambda doc: doc["edges"][0].update(source="MISSING"),
    lambda doc: doc["nodes"][0].update(type=" "),
    lambda doc: doc["nodes"][0].update(properties=[]),
    lambda doc: doc["nodes"][0].update(unexpected=True),
    lambda doc: doc["nodes"][0].pop("description"),
    lambda doc: doc["edges"][0].update(properties=None),
    lambda doc: doc["edges"][0].update(id=123),
    lambda doc: doc["edges"][0].update(id=True),
    lambda doc: doc["nodes"][0].update(id=True),
    lambda doc: doc["edges"][0].update(id=doc["nodes"][0]["id"]),
    lambda doc: doc["nodes"][0].update(description=""),
    lambda doc: doc["nodes"][0].update(description=" \n\t"),
    lambda doc: doc["edges"][0].update(description=""),
    lambda doc: doc["edges"][0].update(description=" \n\t"),
    lambda doc: doc["nodes"][0].update(properties={"nested": {"value": 1}}),
    lambda doc: doc["edges"][0].update(properties={"array": [1, 2]}),
    lambda doc: doc["nodes"][0].update(properties={"": 1}),
    lambda doc: doc["edges"][0].update(properties={" \t": 1}),
])
def test_invalid_fixture_records(graph_path, tmp_path, mutation):
    document = json.loads(graph_path.read_text())
    mutation(document)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    with pytest.raises(GraphValidationError):
        load_graph(path)


def test_edge_description_and_all_scalar_property_types_are_preserved(graph_path, tmp_path):
    document = json.loads(graph_path.read_text())
    document["edges"][0]["description"] = "Recorded feed connection"
    properties = {"text": "CS", "number": 6.5, "integer": 80, "boolean": True, "null": None}
    document["nodes"][0]["properties"] = properties
    document["edges"][0]["properties"] = properties
    path = tmp_path / "scalar_properties.json"
    path.write_text(json.dumps(document))
    graph = load_graph(path)
    assert next(node for node in graph["nodes"] if node["id"] == "P4711")["properties"] == properties
    edge = next(edge for edge in graph["edges"] if edge["id"] == "MNb10005")
    assert edge["description"] == "Recorded feed connection"
    assert edge["properties"] == properties


def test_missing_graph_file_is_controlled(tmp_path):
    with pytest.raises(GraphValidationError, match="Cannot load graph"):
        load_graph(tmp_path / "missing.json")


def test_invalid_utf8_is_controlled(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff")
    with pytest.raises(GraphValidationError):
        load_graph(path)
