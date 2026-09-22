"""Controlled errors shared by graph, tool, and local-model boundaries."""


class GraphAgentError(Exception):
    code = "graph_agent_error"


class GraphValidationError(GraphAgentError):
    code = "graph_validation_error"


class InvalidInputError(GraphAgentError):
    code = "invalid_input"


class NodeNotFoundError(GraphAgentError):
    code = "node_not_found"


class EntityNotFoundError(GraphAgentError):
    code = "entity_not_found"


class EdgeNotFoundError(GraphAgentError):
    code = "edge_not_found"


class PathNotFoundError(GraphAgentError):
    code = "path_not_found"


class ModelError(GraphAgentError):
    code = "model_error"


class ModelOutputError(ModelError):
    code = "model_output_error"


class GroundingError(GraphAgentError):
    code = "grounding_error"


class FinalSelectionError(GroundingError):
    """A completed final response has invalid JSON syntax or selection schema."""
