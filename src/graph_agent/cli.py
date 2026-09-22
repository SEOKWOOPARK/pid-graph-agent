"""One-shot and interactive local CLI."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from .config import MAX_QUERY_LENGTH, Config
from .errors import GraphAgentError, GroundingError, ModelOutputError
from .graph_access import GraphAccess
from .graph_loader import load_graph

log = logging.getLogger(__name__)


def _print_error(exc: GraphAgentError) -> None:
    if isinstance(exc, (GroundingError, ModelOutputError)):
        log.debug("Rejected model response: %s", exc)
        message = (
            "Could not produce a verified answer. "
            "Try rephrasing the question or specifying exact node IDs."
        )
    else:
        message = str(exc)
    print(f"Error [{exc.code}]: {message}", file=sys.stderr)


def default_graph_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "graph.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Query a pre-constructed DEXPI-style graph using local Qwen.")
    result.add_argument("--query", help=f"Ask once and exit (up to {MAX_QUERY_LENGTH} characters, including whitespace); omit for an interactive session.",)
    result.add_argument("--model", help="Qwen model ID or local model directory.")
    result.add_argument("--graph", type=Path, default=default_graph_path(), help="Graph JSON path.")
    result.add_argument("--verbose", action="store_true", help="Show diagnostic logging.")

    return result


async def _run(args) -> int:
    from .agent import AgentSession
    config = Config.from_env(args.model)
    graph = GraphAccess(load_graph(args.graph))
    session = AgentSession(graph, config)

    if args.query is not None:
        print((await session.ask(args.query)).text)
        return 0
    
    print(f"Graph Agent\nLocal model: {config.model_id}\nType 'exit' to quit.")
    print("Questions are independent; use exact IDs when clarifying a previous result.")

    while True:
        try:
            question = input("\n> ")
        except EOFError:
            return 0
        
        if question.strip().lower() in {"exit", "quit"}:
            return 0
        
        try:
            print((await session.ask(question)).text)
        except GraphAgentError as exc:
            _print_error(exc)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("graph_agent.local_qwen").setLevel(logging.INFO)
    
    try:
        return asyncio.run(_run(args))
    except GraphAgentError as exc:
        _print_error(exc)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
