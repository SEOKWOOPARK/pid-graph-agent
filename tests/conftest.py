from pathlib import Path

import pytest

from graph_agent.graph_access import GraphAccess
from graph_agent.graph_loader import load_graph


@pytest.fixture
def graph_path():
    return Path(__file__).resolve().parents[1] / "data" / "graph.json"


@pytest.fixture
def graph(graph_path):
    return load_graph(graph_path)


@pytest.fixture
def access(graph):
    return GraphAccess(graph)
