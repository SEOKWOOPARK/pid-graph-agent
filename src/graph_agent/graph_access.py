from collections import deque
from copy import deepcopy
from typing import Any, Literal
from .errors import (
    EdgeNotFoundError,
    EntityNotFoundError,
    InvalidInputError,
    NodeNotFoundError,
    PathNotFoundError,
)
from .graph_loader import validate_graph

Direction = Literal["upstream", "downstream", "both"]
MAX_HOPS = 10


class GraphAccess:
    def __init__(self, graph: dict[str, Any]):
        document = deepcopy(validate_graph(graph))
        self._nodes = {node["id"]: node for node in document["nodes"]}
        self._edges = {edge["id"]: edge for edge in document["edges"]}
        self._incoming: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        self._outgoing: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        
        for edge_id, edge in self._edges.items():
            self._outgoing[edge["source"]].append(edge_id)
            self._incoming[edge["target"]].append(edge_id)

    @staticmethod
    def _text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError(f"{field} must be a nonempty string")
        return value

    def _node(self, node_id: str) -> str:
        self._text(node_id, "node_id")
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Node {node_id!r} does not exist")
        return node_id

    @staticmethod
    def _direction(direction: str) -> None:
        if direction not in ("upstream", "downstream", "both"):
            raise InvalidInputError("direction must be upstream, downstream, or both")

    def _type(self, node_type: str | None) -> None:
        if node_type is not None:
            self._text(node_type, "node_type")

    def _matches(self, node_id: str, node_type: str | None) -> bool:
        return node_type is None or (
            self._nodes[node_id]["type"].casefold() == node_type.casefold()
        )

    def _adjacent(self, node_id: str, direction: Direction) -> list[str]:
        node_ids = set()

        if direction in ("upstream", "both"):
            node_ids.update(self._edges[edge_id]["source"] for edge_id in self._incoming[node_id])

        if direction in ("downstream", "both"):
            node_ids.update(self._edges[edge_id]["target"] for edge_id in self._outgoing[node_id])
            
        return sorted(node_ids)

    def get_node(self, node_id: str) -> dict[str, Any]:
        return deepcopy(self._nodes[self._node(node_id)])

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        self._text(entity_id, "entity_id")

        if entity_id in self._nodes:
            return deepcopy(self._nodes[entity_id])
        
        if entity_id in self._edges:
            return deepcopy(self._edges[entity_id])
        
        raise EntityNotFoundError(f"Entity {entity_id!r} does not exist")

    def get_neighbors(
        self,
        node_id: str,
        direction: Direction = "both",
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        self._node(node_id)
        self._direction(direction)
        self._type(node_type)
        return [
            self.get_node(candidate)
            for candidate in self._adjacent(node_id, direction)
            if self._matches(candidate, node_type)
        ]

    def get_connected_edges(
        self, node_id: str, direction: Direction = "both"
    ) -> list[dict[str, Any]]:
        self._node(node_id)
        self._direction(direction)
        edge_ids = set()

        if direction in ("upstream", "both"):
            edge_ids.update(self._incoming[node_id])

        if direction in ("downstream", "both"):
            edge_ids.update(self._outgoing[node_id])

        return [deepcopy(self._edges[edge_id]) for edge_id in sorted(edge_ids)]

    def get_edge(
        self, source: str, target: str, edge_id: str | None = None
    ) -> list[dict[str, Any]]:
        self._node(source)
        self._node(target)

        if edge_id is not None:
            self._text(edge_id, "edge_id")

        selected = [
            self._edges[key]
            for key in self._outgoing[source]
            if self._edges[key]["target"] == target and (edge_id is None or key == edge_id)
        ]

        if not selected:
            detail = f" with ID {edge_id!r}" if edge_id is not None else ""
            raise EdgeNotFoundError(f"No edge{detail} from {source} to {target}")
        
        return deepcopy(sorted(selected, key=lambda record: record["id"]))

    def find_path(self, source: str, target: str) -> dict[str, Any]:
        self._node(source)
        self._node(target)
        parents: dict[str, str | None] = {source: None}
        pending = deque([source])
        
        while pending and target not in parents:
            current = pending.popleft()
            for candidate in self._adjacent(current, "downstream"):
                if candidate not in parents:
                    parents[candidate] = current
                    pending.append(candidate)

        if target not in parents:
            raise PathNotFoundError(f"No directed path from {source} to {target}")

        path = []
        current = target

        while current is not None:
            path.append(current)
            current = parents[current]

        path.reverse()
        edges = [edge for start, end in zip(path, path[1:]) for edge in self.get_edge(start, end)]

        return {
            "nodes": [self.get_node(node_id) for node_id in path],
            "edges": edges,
            "path_node_ids": path,
        }

    def get_connected_subgraph(
        self,
        node_id: str,
        max_hops: int = 1,
        direction: Direction = "both",
        node_type: str | None = None,
    ) -> dict[str, Any]:
        self._node(node_id)
        self._direction(direction)
        self._type(node_type)

        if type(max_hops) is not int or not 1 <= max_hops <= MAX_HOPS:
            raise InvalidInputError(f"max_hops must be an integer between 1 and {MAX_HOPS}")
        
        distances = {node_id: 0}
        pending = deque([node_id])

        while pending:
            current = pending.popleft()
            if distances[current] >= max_hops:
                continue

            for candidate in self._adjacent(current, direction):
                if candidate not in distances:
                    distances[candidate] = distances[current] + 1
                    pending.append(candidate)

        node_ids = sorted(distances)
        edges = [
            deepcopy(record)
            for record in self._edges.values()
            if record["source"] in distances and record["target"] in distances
        ]

        return {
            "nodes": [self.get_node(candidate) for candidate in node_ids],
            "edges": sorted(edges, key=lambda record: record["id"]),
            "matched_node_ids": [
                candidate
                for candidate in node_ids
                if candidate != node_id and self._matches(candidate, node_type)
            ],
        }

    def search_nodes(
        self, query: str, node_type: str | None = None
    ) -> list[dict[str, Any]]:
        self._text(query, "query")
        self._type(node_type)
        exact_ids = [
            node_id
            for node_id in sorted(self._nodes)
            if node_id.casefold() == query.strip().casefold()
        ]
        if exact_ids:
            return [
                self.get_node(node_id)
                for node_id in exact_ids
                if self._matches(node_id, node_type)
            ]
        tokens = query.casefold().split()
        results = []

        for node_id in sorted(self._nodes):
            record = self._nodes[node_id]
            haystack = " ".join(
                record[field] for field in ("id", "name", "description", "type")
            ).casefold()

            if self._matches(node_id, node_type) and all(token in haystack for token in tokens):
                results.append(self.get_node(node_id))

        return results
