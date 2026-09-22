from graph_agent.graph_access import GraphAccess

def ids(records):
    return [record["id"] for record in records]


def test_get_node_returns_full_record(access):
    record = access.get_node("P4711")
    assert record["name"] == "Feed Pump"
    assert record["type"] == "Pump"
    assert record["properties"] == {"design_pressure": 6.5, "design_temperature": 80}


def test_owned_snapshot_and_returned_values_are_isolated(graph):
    access = GraphAccess(graph)
    next(node for node in graph["nodes"] if node["id"] == "P4711")["properties"]["design_pressure"] = 999
    next(edge for edge in graph["edges"] if edge["id"] == "MNb10005")["properties"]["diameter"] = 999
    graph["nodes"].clear()
    graph["edges"].clear()
    result = access.get_node("P4711")
    result["properties"]["design_pressure"] = -1
    assert access.get_node("P4711")["properties"]["design_pressure"] == 6.5
    edges = access.get_connected_edges("P4711")
    edges[0]["properties"]["diameter"] = -1
    assert access.get_connected_edges("P4711")[0]["properties"]["diameter"] == 80
    assert ids(access.get_neighbors("P4711")) == ["H1007", "V1005"]
    assert access.find_path("P4711", "V2001")["path_node_ids"] == ["P4711", "H1007", "V2001"]


def test_neighbors_direction_and_type(access):
    assert ids(access.get_neighbors("P4711")) == ["H1007", "V1005"]
    assert ids(access.get_neighbors("P4711", "upstream")) == ["V1005"]
    assert ids(access.get_neighbors("P4711", "downstream")) == ["H1007"]
    assert ids(access.get_neighbors("H1007", "upstream", "pump")) == ["P4711"]
    assert access.get_neighbors("H1007", "upstream", "Valve") == []


def test_connected_edge_properties_and_direction(access):
    assert ids(access.get_connected_edges("P4711")) == ["MNb10005", "MNb47122"]
    assert ids(access.get_connected_edges("P4711", "upstream")) == ["MNb10005"]
    edge, = access.get_connected_edges("P4711", "downstream")
    assert edge["source"] == "P4711" and edge["target"] == "H1007"
    assert edge["properties"]["design_temperature"] == 120


def test_parallel_edges_and_neighbor_deduplication(access):
    assert ids(access.get_edge("V2001", "T4001")) == ["MNb40001", "MNb40002"]
    assert ids(access.get_edge("V2001", "T4001", "MNb40002")) == ["MNb40002"]
    assert ids(access.get_neighbors("V2001", "downstream")) == ["T4001"]


def test_directed_path_preserves_records_and_parallel_hops(access):
    path = access.find_path("P4711", "V2001")
    assert path["path_node_ids"] == ["P4711", "H1007", "V2001"]
    assert ids(path["nodes"]) == path["path_node_ids"]
    assert ids(path["edges"]) == ["MNb47122", "MNb20001"]
    parallel = access.find_path("V2001", "T4001")
    assert ids(parallel["edges"]) == ["MNb40001", "MNb40002"]


def test_zero_edge_path_to_self(access):
    result = access.find_path("P4711", "P4711")
    assert result["path_node_ids"] == ["P4711"]
    assert result["edges"] == []


def test_n_hops_keeps_transit_context_and_filters_matches(access):
    result = access.get_connected_subgraph("H1007", 2, "downstream", "Valve")
    assert result["matched_node_ids"] == ["V2001", "V2002"]
    assert ids(result["nodes"]) == ["H1007", "T3001", "T4001", "V2001", "V2002"]
    assert ids(result["edges"]) == ["MNb20001", "MNb20002", "MNb30001", "MNb40001", "MNb40002"]
    upstream = access.get_connected_subgraph("H1007", 2, "upstream")
    assert upstream["matched_node_ids"] == ["P4711", "V1005"]
    assert access.get_connected_subgraph("H1007", 1, "downstream", "Valve")["matched_node_ids"] == ["V2001"]


def test_disconnected_node_is_successful_empty_neighborhood(access):
    assert access.get_neighbors("T9001") == []
    assert access.get_connected_edges("T9001") == []
    result = access.get_connected_subgraph("T9001")
    assert ids(result["nodes"]) == ["T9001"]
    assert result["matched_node_ids"] == []
    assert result["edges"] == []


def test_search_is_case_insensitive_literal_all_tokens(access):
    assert ids(access.search_nodes("p4711", "Pump")) == ["P4711"]
    assert ids(access.search_nodes("auxiliary   feed", "Pump")) == ["P5001", "P5002"]
    assert ids(access.search_nodes("storage")) == ["T9001"]
    assert access.search_nodes(".*") == []
    assert access.search_nodes("unknown") == []


def test_search_prioritizes_exact_ids_over_description_mentions(access):
    assert ids(access.search_nodes("p4711")) == ["P4711"]
    assert ids(access.search_nodes(" P4711 \t")) == ["P4711"]
    assert access.search_nodes("p4711", "HeatExchanger") == []
    assert access.search_nodes("p4711", "UnknownType") == []
    assert ids(access.search_nodes("p4711", "pump")) == ["P4711"]


def test_search_preserves_case_variant_id_ambiguity(graph):
    graph["nodes"].append({
        "id": "p4711", "type": "Valve", "name": "Case Variant Valve",
        "description": "A distinct graph entity whose ID differs only by case", "properties": {},
    })
    access = GraphAccess(graph)
    assert ids(access.search_nodes("p4711")) == ["P4711", "p4711"]
    assert ids(access.search_nodes("P4711", "Valve")) == ["p4711"]
    assert ids(access.search_nodes("p4711", "Pump")) == ["P4711"]


def test_ambiguous_resolution_retains_all_candidates(access):
    assert ids(access.get_neighbors("H2000", "upstream", "Pump")) == ["P5001", "P5002"]


def test_cycle_self_loop_and_both_direction_deduplication(graph):
    graph["edges"].append({"id": "return", "source": "H1007", "target": "P4711", "type": "Pipe", "properties": {}})
    graph["edges"].append({"id": "loop", "source": "P4711", "target": "P4711", "type": "Pipe", "properties": {}})
    access = GraphAccess(graph)
    assert ids(access.get_neighbors("P4711")) == ["H1007", "P4711", "V1005"]
    assert ids(access.get_connected_edges("P4711")).count("loop") == 1
    assert "P4711" not in access.get_connected_subgraph("P4711", 10)["matched_node_ids"]


def test_path_ties_use_sorted_neighbor_ids(graph):
    graph["edges"].append({"id": "direct", "source": "H1007", "target": "T4001", "type": "Pipe", "properties": {}})
    graph["edges"].append({"id": "alternate", "source": "P4711", "target": "V2001", "type": "Pipe", "properties": {}})
    graph["nodes"].reverse()
    graph["edges"].reverse()
    access = GraphAccess(graph)
    assert access.find_path("P4711", "T4001")["path_node_ids"] == ["P4711", "H1007", "T4001"]
