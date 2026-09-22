import pytest
from graph_agent.errors import EdgeNotFoundError, GraphValidationError, InvalidInputError, NodeNotFoundError, PathNotFoundError
from graph_agent.graph_access import GraphAccess


@pytest.mark.parametrize("method,args", [
    ("get_node", ("MISSING",)),
    ("get_neighbors", ("MISSING",)),
    ("get_connected_edges", ("MISSING",)),
    ("get_connected_subgraph", ("MISSING",)),
    ("get_edge", ("P4711", "MISSING")),
    ("find_path", ("MISSING", "P4711")),
])
def test_missing_node_is_validated_before_graph_operation(access, method, args):
    with pytest.raises(NodeNotFoundError):
        getattr(access, method)(*args)


@pytest.mark.parametrize("value", [0, -1, 11, 100000, True, 1.0, "2", None])
def test_hops_must_be_strict_bounded_positive_integer(access, value):
    with pytest.raises(InvalidInputError):
        access.get_connected_subgraph("P4711", max_hops=value)


@pytest.mark.parametrize("direction", ["in", "out", "UPSTREAM", None, [], 1])
@pytest.mark.parametrize("method", ["get_neighbors", "get_connected_edges", "get_connected_subgraph"])
def test_invalid_direction_is_controlled(access, method, direction):
    with pytest.raises(InvalidInputError):
        getattr(access, method)("P4711", direction=direction)


@pytest.mark.parametrize("value", [None, [], 4, "", " "])
def test_invalid_identifiers_are_controlled(access, value):
    with pytest.raises(InvalidInputError):
        access.get_node(value)


@pytest.mark.parametrize("value", [[], 4, "", " "])
def test_invalid_type_filters_are_controlled(access, value):
    with pytest.raises(InvalidInputError):
        access.get_neighbors("P4711", node_type=value)


@pytest.mark.parametrize("value", [None, [], 4, "", " "])
def test_invalid_search_query_is_controlled(access, value):
    with pytest.raises(InvalidInputError):
        access.search_nodes(value)


def test_edge_lookup_is_directed_and_id_specific(access):
    with pytest.raises(EdgeNotFoundError):
        access.get_edge("H1007", "P4711")
    with pytest.raises(EdgeNotFoundError):
        access.get_edge("P4711", "H1007", "OTHER")


@pytest.mark.parametrize("source,target", [("H1007", "P4711"), ("P4711", "T9001")])
def test_missing_directed_path_is_explicit(access, source, target):
    with pytest.raises(PathNotFoundError):
        access.find_path(source, target)


@pytest.mark.parametrize("document", [None, [], {}, {"nodes": []}, {"nodes": {}, "edges": []}])
def test_invalid_graph_document_cannot_enter_access_layer(document):
    with pytest.raises(GraphValidationError):
        GraphAccess(document)


@pytest.mark.parametrize("mutation", [
    lambda doc: doc["nodes"].append(doc["nodes"][0].copy()),
    lambda doc: doc["edges"].append(doc["edges"][0].copy()),
    lambda doc: doc["nodes"][0].update(id=True),
    lambda doc: doc["edges"][0].update(id=doc["nodes"][0]["id"]),
    lambda doc: doc["edges"][0].update(source="MISSING"),
    lambda doc: doc["edges"][0].update(target="MISSING"),
    lambda doc: doc["nodes"][0].update(properties={"nested": {"value": 1}}),
])
def test_access_validates_direct_document_records(graph, mutation):
    mutation(graph)
    with pytest.raises(GraphValidationError):
        GraphAccess(graph)


@pytest.mark.parametrize("records", ["nodes", "edges"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_access_rejects_nonfinite_properties_in_direct_documents(graph, records, value):
    graph[records][0]["properties"]["invalid_number"] = value
    with pytest.raises(GraphValidationError):
        GraphAccess(graph)


def test_empty_graph_supports_search_and_controlled_missing_nodes():
    access = GraphAccess({"nodes": [], "edges": []})
    assert access.search_nodes("pump") == []
    with pytest.raises(NodeNotFoundError):
        access.get_node("P4711")
