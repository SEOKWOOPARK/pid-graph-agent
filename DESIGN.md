# Specification review and implementation contract

Read AGENTS.md and SPEC.md in full before implementation. SPEC.md remains the
source of truth. This document records the current interfaces and resolves
unspecified boundary cases; it does not expand the POC into a production system.

## Risks and decisions

- Short ID descriptions use a deterministic `get_entity(entity_id)` lookup across
  both node and edge indexes. No ID prefix or type hint decides which kind it is.
  `entity_lookup.py` recognizes complete bare-ID/Describe/Show/Pipe forms and
  records the real tool call in QueryContext before grounded rendering of all
  metadata and property keys. This branch does not load Qwen or run the SDK loop.
  Its QueryContext disables resolution. Ordinary Qwen-driven queries keep the
  default resolution behavior.
  Natural-language questions continue through the SDK with all eight tools.
  Missing IDs use `EntityNotFoundError` (`entity_not_found`). Node-only and
  source-to-target lookup semantics stay explicit in their existing tools.
- Evaluation distinguishes `execution_mode=direct_lookup` from `agent`. Direct
  lookups retain calls and selections but no model outputs or retries, and are
  excluded from model choice, argument, resolution and schema metrics. Retrieval
  and answer checks still apply. Evidence and mode reset for each question.
- Directed source→target means downstream. Edge lookup and paths are directed.
- Parallel edges retain unique IDs. Edge lookup returns all matching records.
  Paths are shortest directed node paths with sorted-ID tie breaking; every
  parallel edge on each selected hop is returned as an alternative connection.
- N-hop traversal crosses all equipment types. The result retains the root,
  reachable nodes and induced edges, with matched_node_ids separately filtered
  by equipment type and excluding the root. max_hops is an integer from 1 to 10.
- Exact lookups are case-sensitive. Search is case-insensitive literal matching
  of tokens across IDs, types, names and descriptions, with no invented aliases.
  Case-insensitive exact-ID matches take precedence, retaining case variants and applying the type filter without falling back to descriptions.
- Empty searches/neighborhoods are valid empty results. Missing nodes, requested
  edges and paths produce explicit errors. A node has a zero-edge path to itself.
- Listing multiple nodes is normal. For singular indirect resolution, Qwen uses
  resolve=true on search_nodes/get_neighbors. Non-singleton resolution blocks
  dependent calls and returns candidates or a no-match result. Correctly choosing
  resolution intent remains a measured model capability.
- Unrestricted prose cannot be deterministically verified with ID checks. Qwen
  therefore produces a final evidence selection, and the application validates
  references against this run's successful tool results and renders facts using
  fixed sentences. No model-generated property values or prose claims are trusted.
  This deliberately trades prose flexibility for grounding. It cannot prove that
  the model understood the user's intent; evaluations measure that separately.
- A property request names a retrieved entity and key. The renderer checks key
  existence and reports absence. Bare values acquire no inferred units.
- If a retrieved node's `id`, `name`, `type`, or `description` is mistakenly
  selected as a property, normalize it into the node selection and render the
  node summary once. A genuine nested property with that key takes precedence.
  Return the normalized selection for evaluation, retain the raw model output
  on the session, and preserve retrieval, status, and ambiguity validation.
- SDK version is pinned to the locally tested Model/ModelProvider contract.
  Transformers and torch are installed through `requirements.txt`. Deterministic
  tests use fake inference backends and load no weights.
- If an SDK run completes a final response while `QueryContext.calls` is empty,
  `AgentSession` makes at most one corrective SDK continuation. Preserve the
  original question and response history with `result.to_input_list()`, append a
  developer reminder that the previous answer is unverified and a graph tool
  must be called, and reuse the same query context. The second run receives only
  `config.max_turns - len(result.raw_responses)` turns. Do not retry if none remain.
  In particular, `MAX_TURNS=1` permits no correction. A second final response
  without tool evidence remains a grounding failure. Malformed tool calls, empty output,
  and model loading failures do not trigger this retry. Do not execute tools
  manually or infer a graph outcome.
- A completed final response with invalid JSON syntax or selection schema raises
  `FinalSelectionError`, a `GroundingError` subclass with the same error code.
  If evidence recovery cannot already render an established absence or ambiguity,
  request one schema correction through the SDK using the same history and tool
  results. Include the validation diagnostic and request valid JSON using the
  recorded evidence. Non-tool text remains an untrusted SDK message until grounding checks
  its syntax, schema and retrieved references. Do not trim stray quotes or guess
  repaired JSON. This shares the single correction and total budget with the
  no-tool case. Record `invalid_final_schema` as the retry reason. Never normalize
  model-supplied property values into facts. Revalidate the corrected selection,
  and do not treat unsupported claims as schema errors eligible for correction.
- Natural-language questions remain with Qwen for operation, argument and final
  fact selection. Grounding verifies retrieved facts, not question intent or
  completeness. There is no question-pattern layer that requires specific graph
  calls or narrows selected facts to the wording of a question.
- On `GroundingError` or `ModelOutputError`, `AgentSession` may recover a known
  outcome through `recover_tool_outcome(context)`. A blocked resolution preserves
  all ambiguity candidates or its no-match outcome. Otherwise every recorded
  tool result must establish absence: a known missing node, edge, or path error,
  a successful empty list, or an empty `matched_node_ids` result. No recovery is
  inferred from nonempty retrievals, mixed retrievals and failures, invalid
  arguments, or absent evidence. This evidence-based fallback itself does not
  retry the model or repair its interpretation of the question.
- A recovered `Answer` includes an optional `recovery` diagnostic with `code` and
  `message`. Retain tool calls and all completed raw final outputs for inspection.
  Reset `last_retry_count` and `last_final_outputs` for each question. Keep
  `last_final_output` as the latest completed final response, but clear it before
  the correction so a failed retry does not leave the prior answer as the latest
  result. Earlier completed responses remain in `last_final_outputs`.
  Malformed output rejected by the model adapter is available in verbose logs.
  `last_retry_reason` identifies a no-tool or final-format correction. Evaluation
  includes `retry_reason` while retaining the original calls and final outputs. Final-answer
  metrics measure the application output, not the model's first choice.
  Unrecoverable
  grounding/output failures ask for rephrasing with exact IDs in the CLI, expose
  the original diagnostic with `--verbose`, and exit 2 in one-shot mode.
  Context overflow, load errors and turn exhaustion still fail explicitly.
  No hosted fallback.
- User questions must be nonblank and at most 200 characters, measured on the
  original input including spaces. Exactly 200 characters is allowed; 201 or more
  raises `InvalidInputError` before model execution.

## Exact repository structure

```
README.md, DESIGN.md, requirements.txt, pytest.ini, main.py, .gitignore
AGENTS.md, SPEC.md                         # original requirements
 data/graph.json
 src/graph_agent/
   __init__.py, __main__.py, config.py, schema.py, errors.py
   graph_loader.py, graph_access.py, tools.py, grounding.py, entity_lookup.py
   local_qwen.py, model_provider.py, agent.py, cli.py, evaluation.py
 tests/
   conftest.py, test_graph_loader.py, test_graph_access.py, test_validation.py
   test_tool_calls.py, test_local_qwen.py, test_grounding.py
   test_agent_eval.py, test_cli.py, test_recovery.py, test_final_schema_retry.py
   eval_cases.json, paraphrase_cases.json, fuzz_cases.json, recovery_cases.json
 scripts/run_eval.py
```

Install dependencies with `python -m pip install -r requirements.txt` and run the
CLI with `python main.py` from the repository root. The repository itself does not
need to be installed as a Python distribution. Both `main.py` and
`scripts/run_eval.py` add the checkout's `src` directory to the import path before
importing application modules. `pytest.ini` configures `pythonpath = src`,
`testpaths = tests`, and `addopts = -ra`, so `python -m pytest` works from the
checkout after dependencies are installed.

## JSON schema

schema.py exports the executable Draft 2020-12 GRAPH_SCHEMA. Root requires only
nodes and edges arrays. Node records require id, type, name, description (nonblank
strings) and properties. Edge records require id, source, target, type (nonblank
strings) and properties, and may include description. Properties are objects of
JSON scalar values (string, finite number, boolean, null); units can be explicit
properties. Structural extra fields, duplicate JSON keys, duplicate entity IDs,
nonfinite values and dangling endpoints are rejected before graph queries.
This is a small DEXPI-inspired fixture schema, not a DEXPI interchange schema.

## GraphAccess and tools

`load_graph(path)` validates the JSON and returns a plain dictionary with `nodes`
and `edges` lists. `GraphAccess(load_graph(path))` validates and owns deep copies
of those records. It indexes nodes by ID and incoming/outgoing edges by endpoint
using built-in dictionaries and lists. Tools access only its public methods.
Shortest paths and N-hop queries use breadth-first traversal; sorting neighbor IDs
preserves deterministic path tie breaking. Edge IDs preserve parallel connections,
and a self-loop appears once in a combined incoming/outgoing result. It exposes:

```
get_entity(entity_id) -> dict (node or edge)
get_node(node_id) -> dict
search_nodes(query, node_type=None) -> list[dict]
get_neighbors(node_id, direction="both", node_type=None) -> list[dict]
get_connected_edges(node_id, direction="both") -> list[dict]
get_edge(source, target, edge_id=None) -> list[dict]
find_path(source, target) -> {nodes, edges, path_node_ids}
get_connected_subgraph(node_id, max_hops=1, direction="both", node_type=None)
    -> {nodes, edges, matched_node_ids}
```

All returned records are copies with IDs/types/properties preserved. Tool names
match methods. JSON argument schemas prohibit extra properties, require positional
IDs/query, use nonblank strings, allow nullable optional type/edge ID, enumerate
direction as upstream/downstream/both, and bound max_hops to 1–10. Only tools
search_nodes/get_neighbors add optional resolve:boolean=false; it does not enter
GraphAccess. Results are {ok: bool, data: value|null, error: object|null}.

The final response is JSON:

```
{"status":"answer", "node_ids":[], "edge_ids":[],
 "properties":[{"entity_id":"MNb47122", "key":"design_temperature"}]}
```

status is answer, ambiguous, not_found, or unavailable. All four fields are
required; extra fields are rejected. Ambiguity/no-match/error output is checked
against tool evidence. The renderer inserts actual values, never model guesses.

## Local integration and parsing

CLI loads JSON → validated Python dictionary → GraphAccess at startup. Runtime uses a single
OpenAI Agents SDK Agent and Runner, with LocalQwenProvider supplying LocalQwenModel.
The adapter maps SDK message/function-call/function-output items into Qwen chat
history, attaches the eight function schemas to apply_chat_template, disables
thinking, and generates locally through Transformers/PyTorch. It converts native
<tool_call>{"name":...,"arguments":{...}}</tool_call> blocks into SDK function
calls with unique call IDs. It parses strict JSON, validates name and argument
schema, and rejects conflicting prose/calls or malformed markup. Nonempty final
assistant text is preserved unchanged, even when it is not valid JSON, so the SDK
can retain history and count the model turn. Grounding validates JSON syntax,
selection schema and evidence before rendering. Invalid final syntax uses
`FinalSelectionError` (`grounding_error`) and can share the single correction.
Malformed tool calls and empty output still raise `ModelOutputError` immediately.
Streaming and hosted conversations are
outside this CLI POC. Hosted tracing is always disabled.

Qwen/Qwen3-1.7B is default; Qwen/Qwen3-0.6B is a configurable smaller choice.
CPU uses float32 by default; auto prefers CUDA, then MPS, then CPU. Hugging Face
handles first download and cache reuse. No API key, paid service or Ollama is used.

## Stages and verification

1. Schema, fixture, graph loader and GraphAccess: deterministic topology,
   parallel edges, filters, paths, copy isolation, malformed input tests.
2. Eight tools and grounding: exact schemas, errors, ambiguous resolution,
   nonexistent properties, fabricated references and cross-run isolation tests.
3. Local adapter: fake backend tests exercise the installed SDK types, history
   conversion, parser failures and load/generation boundaries without weights.
4. Agent and CLI: scripted adapter responses through the real SDK Runner verify
   multi-step resolution, limits, no credentials, tracing off and CLI errors.
5. Reviewed golden, paraphrase and fuzz cases: compare structured results from
   `expected_calls` and ordered calls, report retrieval exact match, node resolution,
   tool/argument/schema accuracy, property/entity presence and completion.
   Result checks reuse the single expected call automatically; multiple expected
   calls require a zero-based `result_call_index`. Match the last actual call with
   that name and its specified argument values, then compare returned IDs with
   `expected_nodes` / `expected_edges`. Missing calls fail even for empty expected
   lists. Filtered subgraphs use `matched_node_ids`; never combine intermediate
   resolution or traversal context into the target result. Case references are
   validated before inference. Node resolution still uses `resolution_call`.
   Live local Qwen evaluation is separate from deterministic tests. Never report
   scripted-model success as measured Qwen accuracy.
6. Evidence recovery: scripted malformed outputs verify missing-node/edge/path,
   empty-list and empty-filter outcomes, blocked resolution, and rejection of
   unsupported recovery. Evaluation rows record `recovery` when used and assess
   the model's raw final JSON for schema validity, not the recovered selection.
   The five reviewed absence cases in `tests/recovery_cases.json` run explicitly
   with `python scripts/run_eval.py --cases tests/recovery_cases.json`; default
   suites remain unchanged. Live queries do not guarantee model output failure
   or sufficient evidence for recovery.
7. No-tool correction: scripted SDK tests verify exactly one correction, retained
   question/history, the shared turn budget, no retry for an unsupported claim or
   malformed tool call, continued rejection without evidence, and state isolation
   between questions. Evaluation rows expose `retry_count` and `final_outputs`.
   Schema scoring checks all completed raw finals, so an invalid first schema
   still fails after a valid correction. Adapter output parsing failures also
   fail the schema check. Keep the existing eight metrics. Live Qwen may still
   ignore the reminder and remain blocked.
8. Final-schema correction: scripted tests reproduce a correct node lookup
   followed by a property dictionary. Verify valid property references, one
   shared correction, retained raw schema failures, and rejection when correction
   fails.
9. Final JSON syntax: reproduce a path lookup followed by a trailing quote, verify
   the original call and result stay in correction history, and accept only a new
   valid and grounded model selection. Keep one shared correction, count every
   model turn, retain malformed raw finals for schema scoring, and reject a
   second invalid response or unsupported claim.
10. The 18 main cases in `tests/eval_cases.json` include paths, specific parallel
    pipes, typed N-hop lists, compound properties and missing properties. Run them
    with `python scripts/run_eval.py --cases tests/eval_cases.json`. The default
    evaluation includes these cases plus 3 paraphrase and 8 fuzz cases, totaling
    29. They diagnose Qwen interpretation and selection errors without claiming
    that grounding guarantees question intent or complete answers.

References: [OpenAI models/providers](https://developers.openai.com/api/docs/guides/agents/models),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B),
[pyDEXPI domain reference](https://github.com/process-intelligence-research/pyDEXPI).
