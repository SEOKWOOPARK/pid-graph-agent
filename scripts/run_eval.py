import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graph_agent.agent import AgentSession, INSTRUCTIONS
from graph_agent.config import Config
from graph_agent.errors import GraphAgentError
from graph_agent.evaluation import evaluate_cases
from graph_agent.graph_access import GraphAccess
from graph_agent.graph_loader import load_graph


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local Hugging Face model ID (default: QWEN_MODEL_ID or Qwen/Qwen3-1.7B).")
    parser.add_argument("--graph", type=Path, default=root / "data/graph.json")
    parser.add_argument("--cases", type=Path, nargs="+", default=[root / "tests" / name for name in ("eval_cases.json", "paraphrase_cases.json", "fuzz_cases.json")])
    parser.add_argument("--output", type=Path, help="Optional JSON report path; report is otherwise printed to stdout.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        cases = []

        for path in args.cases:
            suite = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(suite, list):
                raise ValueError(f"Expected a list of evaluation cases in {path}.")
            cases.extend(suite)

        config = Config.from_env(args.model)
        session = AgentSession(GraphAccess(load_graph(args.graph)), config=config)
        report = asyncio.run(evaluate_cases(session, cases))
        report["model"] = config.model_id
        report["mode"] = "live_local_qwen"
        report["config"] = asdict(config)
        report["graph_sha256"] = hashlib.sha256(args.graph.read_bytes()).hexdigest()
        report["instructions_sha256"] = hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()
        rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")

        return 0
    except (GraphAgentError, OSError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
