# Repository Instructions

- Read `SPEC.md` completely before making implementation decisions.
- Treat `SPEC.md` as the source of truth for architecture and scope.
- Do not introduce Neo4j, LangGraph, LangSmith, vector databases, hosted LLM APIs, or a UI unless explicitly required by `SPEC.md`.
- Keep the architecture and terminology defined in `SPEC.md`.
- Implement incrementally.
- Run deterministic tests after each major component.
- Do not commit secrets, downloaded model weights, or local caches.
- Before coding, summarize the planned repository structure, interfaces, and implementation sequence.