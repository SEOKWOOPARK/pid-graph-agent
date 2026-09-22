# pid-graph-agent Development Specification

## Goal

Build a small CLI-based POC that lets a user query an already-constructed DEXPI-style process graph using natural language.

The graph-construction stage itself is **out of scope**. Assume that the underlying graph has already been constructed.

For this POC:

- Represent the pre-constructed graph as JSON.
- Validate and load the JSON into Python dictionaries/lists at application startup.
- Use **OpenAI Agents SDK** as the agent framework.
- Use a **local Qwen model** as the LLM so the project can run without an OpenAI API key or paid hosted inference.
- Run Qwen locally through Python, preferably with Hugging Face Transformers + PyTorch.
- Integrate the local Qwen model with OpenAI Agents SDK through a custom `Model` / `ModelProvider` adapter.
- Disable OpenAI-hosted tracing when no OpenAI API key is present.
- The agent must answer using only facts retrieved from the graph.
- No GUI is required. CLI/script is sufficient.
- pyDEXPI should be treated as the domain/schema reference for constructing the sample graph fixture. It does not need to be imported or executed at runtime unless there is a clear benefit.

The repository should be runnable by an evaluator without creating an OpenAI account or supplying an API key.

---

## 1. Fixed Architecture

There are **two separate flows**.

### A. Application startup / graph loading

```text
graph.json
    ↓
Python Graph Records
    ↓
GraphAccess
```

### B. Runtime user-query flow

```text
User Prompt
    ↓
OpenAI Agents SDK
    ↓
Local Qwen LLM
    ↓
Tool selection + arguments
    ↓
Tool
    ↓
GraphAccess
    ↓
Python Graph Records
    ↓
Tool Result
    ↓
Local Qwen LLM
    ↓
Final JSON selection (status, entity IDs, property keys)
    ↓
render_answer(): validate retrieved evidence and render values
    ↓
Final Answer
```

For a short exact-ID description, use a deterministic branch:

```text
User ID / Describe ID / Pipe ID
    ↓
recorded get_entity tool
    ↓
GraphAccess → Python Graph Records
    ↓
grounded renderer → Final Answer
```

This branch does not invoke Qwen or the SDK agent loop. It is limited to complete
simple ID descriptions, so model guesses cannot change an exact ID into a type
filter or a node-only lookup. Natural-language traversal, properties, paths and
descriptive resolution continue through the SDK agent loop above.

Do not merge or rename these layers unnecessarily.

### Responsibilities

#### `graph.json`

- Static mock/fixture representing an already-constructed DEXPI graph.
- Based loosely on pyDEXPI/DEXPI process-equipment concepts.
- Contains nodes, directed edges, IDs, descriptions, and properties.

#### Python Graph Records

- Runtime in-memory representation of `graph.json`, using built-in Python data structures.
- `load_graph(path)` validates the JSON and returns a plain dictionary with `nodes` and `edges` lists.
- Preserve directed parallel connections as separate edge records with unique IDs.

#### `GraphAccess`

- Abstraction between tools and the graph records.
- Validate and own deep copies of node/edge records, indexed with dictionaries and incoming/outgoing edge lists.
- Implement deterministic shortest paths and N-hop queries with breadth-first traversal.
- Agent/tools must **not** manipulate the underlying dictionaries/lists directly.
- This allows the storage backend to change later without changing the tool/agent interfaces.

#### Tools

- Small reusable graph operations exposed to the OpenAI agent.
- Tools call `GraphAccess` only.

#### OpenAI Agents SDK

- Provides the agent/tool orchestration layer.
- Receives model outputs from the local Qwen provider.
- Executes tool calls and returns tool results to the model.
- Do not use OpenAI-hosted inference for this POC.

#### Local Qwen LLM

- Performs natural-language understanding.
- Selects tools.
- Produces tool arguments.
- Performs natural-language node resolution through tool use.
- Combines tool results.
- Returns a final JSON selection of status, retrieved entity IDs, and property keys.
- Runs locally, without a paid API.

#### Grounded renderer

- `render_answer()` validates the final selection against the current query's tool results.
- Reads property values from retrieved records and generates the answer text.
- Rejects unsupported claims and explicitly reports missing properties.

#### CLI

- Reads user questions.
- Sends them to the agent.
- Prints answers/errors.

---

## 2. Local LLM Strategy

Use a small local Qwen instruct model appropriate for laptop inference.

Recommended default:

```text
Qwen/Qwen3-1.7B
```

The model ID must be configurable rather than hard-coded.

Allow a smaller fallback if necessary for limited hardware, for example:

```text
Qwen/Qwen3-0.6B
```

### Requirements

- No OpenAI API key should be required.
- No external paid LLM API should be required.
- Ollama should **not** be a mandatory dependency.
- Prefer direct Python loading through:
  - `transformers`
  - `torch`
  - `accelerate` when useful
- First run may download the configured model from Hugging Face.
- Reuse the local Hugging Face cache on subsequent runs.
- Support CPU execution.
- If CUDA or Apple Silicon acceleration is available, use it when practical.
- Keep model-loading code isolated from the graph and tool layers.

### OpenAI Agents SDK integration

Implement a local model adapter compatible with OpenAI Agents SDK.

Suggested structure:

```text
OpenAI Agents SDK
        ↓
LocalQwenModel / LocalQwenProvider
        ↓
Transformers + PyTorch
        ↓
Qwen
```

The adapter is responsible for:

- converting Agents SDK model requests into a Qwen-compatible prompt/message format;
- exposing available tool schemas to Qwen;
- parsing Qwen output into:
  - normal assistant messages, or
  - tool calls with structured arguments;
- converting model output back into the format expected by OpenAI Agents SDK;
- reporting local inference errors clearly.

Do not bypass the Agents SDK by manually implementing the entire agent loop unless the SDK cannot support the required behavior.

Because no OpenAI API key is used, disable OpenAI-hosted tracing/export by default.

---

## 3. Sample Graph Format

Create:

```text
data/graph.json
```

Use a schema similar to:

```json
{
  "nodes": [
    {
      "id": "P4711",
      "type": "Pump",
      "name": "Feed Pump",
      "description": "Feed pump supplying heat exchanger H1007",
      "properties": {
        "design_pressure": 6.5,
        "design_temperature": 80
      }
    },
    {
      "id": "H1007",
      "type": "HeatExchanger",
      "name": "Main Heat Exchanger",
      "description": "Process heat exchanger",
      "properties": {
        "design_pressure": 60
      }
    }
  ],
  "edges": [
    {
      "id": "MNb47122",
      "source": "P4711",
      "target": "H1007",
      "type": "Pipe",
      "properties": {
        "diameter": 80,
        "design_temperature": 120,
        "material": "CS"
      }
    }
  ]
}
```

### Graph semantics

- Directed edge `source → target` represents process-flow direction.
- Therefore:
  - `predecessors(node)` = upstream
  - `successors(node)` = downstream

Include enough nodes/edges to demonstrate:

- direct neighbors
- upstream/downstream
- multiple equipment types
- N-hop traversal
- paths
- edge properties
- disconnected nodes
- ambiguous node descriptions

---

## 4. Graph Access Layer

Implement a `GraphAccess` class/interface.

Suggested public methods:

```python
get_entity(entity_id)

get_node(node_id)

get_neighbors(
    node_id,
    direction="both",
    node_type=None
)

get_connected_edges(
    node_id,
    direction="both"
)

get_edge(
    source,
    target,
    edge_id=None
)

find_path(
    source,
    target
)

get_connected_subgraph(
    node_id,
    max_hops=1,
    direction="both",
    node_type=None
)

search_nodes(
    query,
    node_type=None
)
```

### Rules

- All graph operations must go through `GraphAccess`.
- `GraphAccess` validates and owns deep copies of graph records and their lookup indexes.
- Validate node existence before traversal.
- Return structured dictionaries/lists rather than natural-language strings.
- `get_entity(entity_id)` looks up exact IDs across both node and edge indexes,
  returns a deep copy of the full record, and raises `EntityNotFoundError` with
  code `entity_not_found` when neither exists. Do not infer type from prefixes.
  `get_node` remains node-only and `get_edge` still requires both endpoints.
- Preserve node IDs, edge IDs, types, and properties in results.
- Allowed direction values:
  - `upstream`
  - `downstream`
  - `both`

---

## 5. Agent Tools

Expose the following tool capabilities to OpenAI Agents SDK.

Do **not** make one tool for every possible user question. Use reusable graph primitives.

### 1. `get_node`

Retrieve node metadata/properties by exact node ID.

### 2. `search_nodes`

Search candidate nodes using description/type information.

### 3. `get_neighbors`

Retrieve directly connected nodes.

Supports:

- upstream/downstream/both
- optional node type filtering

### 4. `get_connected_edges`

Retrieve edges directly connected to a node, including edge properties.

### 5. `get_edge`

Retrieve connection(s) between two nodes.

### 6. `find_path`

Retrieve a path between two nodes including nodes and edges along it.

### 7. `get_connected_subgraph`

Retrieve nodes/edges reachable within N hops.

### 8. `get_entity`

Retrieve a node or edge by exact `entity_id`, including its type, metadata and
properties, without requiring edge endpoints. This is also available to Qwen for
natural-language requests that identify a pipe by ID without source/target.

### Tool requirements

Tools must:

- call `GraphAccess`;
- validate inputs;
- return machine-readable structured output;
- never invent graph facts.

---

## 6. Natural-Language Node Resolution

Do **not** create a separate LLM specifically for resolution.

Assign this responsibility to the same local Qwen model used by the agent.

### Case A: explicit node ID

User:

```text
Show me everything connected to P4711.
```

Expected behavior:

```text
P4711 is already resolved
    ↓
get_neighbors(...)
get_connected_edges(...)
    ↓
answer
```

### Case B: node described indirectly

Assume:

```text
P4711 → H1007
```

User:

```text
Show me everything connected to the pump upstream of H1007.
```

Expected behavior:

```text
Qwen interprets:

reference node = H1007
direction = upstream
target type = Pump

    ↓

get_neighbors(
    node_id="H1007",
    direction="upstream",
    node_type="Pump"
)

    ↓

P4711

    ↓

get_neighbors(node_id="P4711")
get_connected_edges(node_id="P4711")

    ↓

final answer
```

Important:

- `H1007` is only the reference node used to resolve the target.
- `P4711` becomes the target node for the subsequent query.
- If multiple candidate pumps match, do **not** choose one arbitrarily.
- Return an ambiguity message or ask for clarification.
- If no candidate exists, explicitly report that no matching node was found.

---

## 7. Agent Framework Choice

Use:

```text
OpenAI Agents SDK + local Qwen
```

### Reason for choosing OpenAI Agents SDK

- This is a lightweight, tool-centric POC.
- Startup environment favors fast implementation and iteration.
- Agents SDK reduces orchestration boilerplate.
- Current workflow does not require complex state machines, checkpoints, human-in-the-loop, or sophisticated branching.
- Tool execution should remain easy to inspect.
- The local model provider keeps the application free from hosted-API credentials.

### Reason for choosing local Qwen

- Natural-language processing requirements are narrow:
  - intent understanding;
  - tool selection;
  - structured arguments;
  - structured selection of retrieved facts for the grounded renderer.
- A small local model avoids requiring the evaluator to create an API account or pay for inference.
- Model choice remains configurable so larger/smaller Qwen variants can be tested.

### Architecture must remain decoupled

```text
CLI
 ↓
OpenAI Agents SDK
 ↓
Local Qwen
 ↓
Tools
 ↓
GraphAccess
 ↓
Python dictionaries/lists
 ↓
JSON source
```

This keeps migration possible later.

For example:

```text
JSON / Python dictionaries and lists
    ↓ can later become
Neo4j / remote graph service
```

without changing the public tool interface.

Likewise:

```text
Local Qwen
    ↓ can later become
another local model / hosted model
```

without rewriting graph logic.

Do not introduce LangGraph unless the workflow actually becomes complex enough to justify explicit state-machine orchestration.

---

## 8. Agent Behavior / Grounding

Recognize complete short ID descriptions such as `H2000`, `MNb20002`,
`Describe MNb20002`, `Show me H2000` and `Pipe MNb20002` in `entity_lookup.py`.
IDs are case-sensitive and preserved without inserted spaces or prefix guesses.
Keep recognition conservative: single ID tokens containing a digit or consisting
of uppercase letters/punctuation, optionally preceded by describe/show and a
generic node/equipment/entity/pipe/edge label. Descriptive words and additional
conditions remain with Qwen. A label must not override the actual record kind.

After ordinary question length/blank validation, AgentSession may answer these
short requests without model execution. Invoke `execute_tool` for `get_entity`
and retain the real `{name, arguments, result}` in a new QueryContext. Build a
selection only from that record, including its node or edge ID and all recorded
property keys. Use the existing grounded renderer, displaying optional edge
descriptions as well as endpoints. Never synthesize model output or recovered
facts. Missing IDs yield `not_found` with the explicit entity error. Do not alter
the JSON fixture to accommodate model guesses.
Use `QueryContext.allow_resolution=False` for this direct lookup branch. Ordinary
Qwen-driven queries retain the default resolution behavior.

Set `Answer.execution_mode` and `session.last_execution_mode` to `direct_lookup`
for this branch, otherwise `agent`. Reset evidence, raw finals, retry diagnostics
and mode for every question. Direct lookups have no model raw finals or retries.
Evaluation must expose the mode and exclude model tool-choice, argument,
resolution and schema metrics for direct lookups while retaining retrieval and
answer checks. Do not report a deterministic lookup as successful Qwen inference.

Agent/model instructions should enforce:

- Answer only from graph/tool results.
- Never invent equipment, connections, directions, or properties.
- If information is unavailable, explicitly say so.
- If node resolution is ambiguous, report ambiguity.
- Use exact graph IDs when referring to equipment or edges.
- Prefer graph tools over assumptions.
- Do not infer upstream/downstream without edge-direction evidence.
- Multiple tool calls are allowed when necessary.

When a final selection mistakenly requests a retrieved node's `id`, `name`,
`type`, or `description` as a property, normalize that reference into the node
selection and render its metadata once. A real nested property with the same
key takes precedence. Unknown properties still receive an explicit missing
message. This correction must use only the current query's retrieved evidence
and preserve ambiguity and no-match handling. Return the normalized selection
and retain the model's raw final output on the session.

When the SDK completes a final response without any recorded graph tool call,
`AgentSession` must request one correction if the query has remaining turns.
Continue through the SDK with the same question, response history, and
`QueryContext`. Add a reminder that the previous answer is unverified, must not
be treated as evidence, and must be followed by an appropriate graph tool call
before another final JSON response. Never execute a tool manually for this retry.
The correction shares the original turn limit: its budget is `config.max_turns`
minus the number of raw model responses in the first SDK result. With no turns
remaining, including `MAX_TURNS=1`, do not retry. A second final response without
tool evidence must remain rejected. Do not extend this mechanism to malformed
tool calls, empty model output, or model loading failures.

When a completed final response fails JSON syntax or the selection schema after a
tool call, distinguish that format error as `FinalSelectionError`, a `GroundingError`
subclass retaining the `grounding_error` code. If the existing evidence fallback
already establishes absence or ambiguity, render that outcome directly.
Otherwise request one SDK correction using the original history and tool results.
Include the validation error and ask for valid final JSON using the recorded
results. Do not repeat a completed lookup solely to correct the response format.
Explain the required schema, including the array of property references, without
trusting or copying model-supplied values. This shares the same single correction
and total turn budget with the no-tool case. Record `invalid_final_schema` as the
retry reason. The corrected response must pass schema and evidence validation.
Do not retry mere unsupported claims, malformed tool calls, or model failures.
A second malformed selection remains blocked unless evidence fallback applies.

The adapter must preserve nonempty non-tool output as untrusted assistant text,
even when its JSON is malformed. `grounding.parse_selection` strictly checks JSON
syntax and schema before the application renders anything. This lets the SDK
complete the response and retain its original history and model-turn count for
the existing correction. Do not trim stray quotes, extract a guessed JSON object,
normalize model values, or manually rerun graph operations. The correction asks
Qwen to resubmit one valid JSON object using recorded tool results. A final JSON
syntax failure is a `FinalSelectionError` with code `grounding_error`, and its
raw text remains in `last_final_outputs` even after a successful correction.
Tool-call JSON, markup, names and arguments remain strictly validated by the
adapter, with `ModelOutputError` and no execution on malformed calls.

For natural-language queries, Qwen selects the graph operation, arguments and
facts to include. Do not add a question-pattern parser that narrows answers or
requires particular calls. Grounding verifies the final schema and retrieved
references, but it does not establish that the selected facts answer the user's
intent or cover every part of a compound question. A recorded but unrelated
property can pass grounding. Measure these failures with reviewed live queries.

If model output fails parsing or grounding, the application may render an outcome
already established by the current query's tool evidence. A blocked resolution
must retain the full ambiguity candidate set or the no-match outcome. Otherwise,
every recorded tool result must establish absence: a known missing-node,
missing-edge, or missing-path error, a successful empty list, or an empty
`matched_node_ids` result. Do not infer an answer from arbitrary nonempty results,
mixed retrievals and failures, invalid arguments, or absent evidence. This
evidence-based fallback itself does not retry the model or correct its
interpretation of the question. Every recorded result remains relevant to the
fallback check.

Retain the original diagnostic and all completed raw final outputs separately
from a recovered answer. Log malformed output rejected by the model adapter at
debug level. Evaluation must record recovery and assess the model's raw
output schemas rather than treating the recovered selection as valid model output.
Reset the session's `last_retry_count` and `last_final_outputs` for each question.
Keep `last_final_output` for the latest completed response, clearing it before a
retry so it remains `None` if that retry fails before producing a final response.
Evaluation rows must include `retry_count` and `final_outputs`. Any invalid
completed final schema or adapter output parsing failure must fail the existing
schema metric, even if the question ultimately receives a verified answer.
Do not add another metric or silently count a corrected schema as a valid first
attempt.

### Example queries

```text
What is connected to P4711?
```

```text
What equipment is downstream of P4711?
```

```text
What pump is upstream of H1007?
```

```text
What valves are reachable from H1007 within 2 hops?
```

```text
What pipe connects P4711 and H1007?
```

```text
What is the design temperature of the pipe between P4711 and H1007?
```

```text
Find a path from P4711 to V2001 and list the pipes along the way.
```

---

## 9. Runtime Validation

Runtime validation is part of production execution, not only testing.

User questions must be nonblank and no longer than 200 characters, counting the
original input including spaces. A 200-character question is allowed; 201 or more
characters must raise `InvalidInputError` before model execution.

Validate at minimum:

- requested node exists;
- requested edge exists;
- direction is `upstream`, `downstream`, or `both`;
- `max_hops` is a positive bounded integer;
- path existence;
- expected property existence;
- ambiguous node resolution;
- empty query result;
- malformed graph JSON at startup;
- malformed tool-call arguments produced by the local model;
- model output that cannot be parsed into a valid tool call;
- local model load/inference failures.

Do not silently return incorrect guesses.

Example using the node-ID lookup dictionary:

```python
if node_id not in nodes_by_id:
    raise NodeNotFoundError(node_id)
```

Errors should be converted into useful agent/user-facing messages.

For unrecoverable model-format or grounding errors, the CLI should ask the user
to rephrase with exact IDs. Include the original diagnostic when `--verbose` is
enabled and retain exit code 2 for failed one-shot requests. Evidence-based
recovery returns the established graph outcome as a normal answer.

Use Python logging for diagnostics, but keep validation and logging conceptually separate:

```text
validation = detect/block invalid state
logging    = record what happened
```

---

## 10. Testing and Evaluation

Use multiple evaluation layers.

### A. Deterministic unit tests — pytest

No LLM required.

Test:

- JSON loading
- validated Python graph records and lookup index construction
- `GraphAccess.get_node`
- `get_neighbors`
- upstream/downstream behavior
- type filters
- `get_edge`
- N-hop traversal
- `find_path`
- disconnected nodes
- nonexistent nodes
- malformed/invalid parameters

These should be deterministic and should **not** load Qwen.

---

### B. Local-model tool-call evaluation

Evaluate whether Qwen selects the correct tool and arguments.

Example:

User query:

```text
What pump is upstream of H1007?
```

Expected structured call:

```text
tool = get_neighbors
node_id = H1007
direction = upstream
node_type = Pump
```

Track:

- correct tool selection;
- correct required arguments;
- correct argument values;
- valid schema generation;
- correct multi-tool sequencing when needed.

This is a core metric because small local models may understand the language but still produce incorrect or malformed tool calls.

---

### C. Golden agent evaluation

Create a small manually verified dataset:

```text
tests/eval_cases.json
```

Example:

```json
[
  {
    "query": "What is connected to P4711?",
    "expected_calls": [
      {
        "name": "get_neighbors",
        "arguments": {"node_id": "P4711", "direction": "both"}
      }
    ],
    "expected_nodes": ["H1007", "V1005"]
  },
  {
    "query": "Find the pump upstream of H1007.",
    "expected_calls": [
      {
        "name": "get_neighbors",
        "arguments": {"node_id": "H1007", "direction": "upstream", "node_type": "Pump", "resolve": true}
      }
    ],
    "resolution_call": {
      "name": "get_neighbors",
      "arguments": {"node_id": "H1007", "direction": "upstream", "node_type": "Pump", "resolve": true}
    },
    "expected_resolved_node": "P4711",
    "expected_nodes": ["P4711"]
  },
  {
    "query": "What pipe connects P4711 to H1007?",
    "expected_calls": [
      {
        "name": "get_edge",
        "arguments": {"source": "P4711", "target": "H1007"}
      }
    ],
    "expected_edges": ["MNb47122"]
  },
  {
    "query": "What is the design temperature of the pipe from P4711 to H1007?",
    "expected_calls": [
      {
        "name": "get_edge",
        "arguments": {"source": "P4711", "target": "H1007"}
      }
    ],
    "expected_edges": ["MNb47122"],
    "expected_properties": [
      {"entity_id": "MNb47122", "key": "design_temperature", "value": 120}
    ]
  }
]
```

Use `expected_calls` for tool and argument checks, `expected_nodes` and
`expected_edges` for retrieved IDs, and `expected_properties` for property
references and their values (or `missing: true`). Retrieval checks require a
nonempty `expected_calls` list. When it contains multiple calls, specify the
zero-based `result_call_index` to select which call's result to check. Use
`resolution_call` with `expected_resolved_node` to score node resolution.

Prefer evaluating retrieved structured evidence/tool results rather than doing brittle exact string comparison against the final prose answer.

Report metrics such as:

- retrieval exact-match rate
- node-resolution accuracy
- tool-call accuracy
- expected entity/property presence
- successful completion rate

All agent evaluation runs locally and should require no paid API.

---

### D. Paraphrase tests

Test semantically equivalent questions.

Example:

```text
What pump is upstream of H1007?
Which pump feeds H1007?
Find the pump before H1007.
```

Verify that they resolve to the same graph entity/result when appropriate.

Create a small fixed set of reviewed paraphrases.

LLM-generated paraphrases may be used to expand coverage, but generated golden answers must not be trusted without deterministic/manual validation.

---

### E. Fuzz / robustness tests

Test unusual or malformed user input.

Examples:

```text
what   is connected to   P4711??
CONNECTED EQUIPMENT TO p4711
Show me P999999
Find path between P4711 and NOTHING
```

Also test:

- empty input
- very large hop count
- invalid direction
- ambiguous descriptions
- malformed model-generated tool arguments

Goal:

- application should not crash;
- failures should be explicit and controlled.

---

### F. Grounding checks

Verify that final factual claims originate from retrieved graph data.

Avoid adding another LLM-as-a-judge.

Prefer deterministic checks against retrieved node IDs, edge IDs, and properties.

Example:

Retrieved:

```json
{
  "edge": "MNb47122",
  "design_temperature": 120
}
```

Allowed:

```text
The design temperature is 120.
```

Not allowed unless retrieved:

```text
The pipe is stainless steel.
```

Deterministic tests must also simulate malformed model output after known absent
results and blocked resolution, and verify that unsupported recovery remains
rejected. Keep five reviewed absence queries in `tests/recovery_cases.json`,
covering a missing node, missing direct edge, missing path, disconnected node,
and empty type-filtered neighborhood. Run this suite explicitly with
`python scripts/run_eval.py --cases tests/recovery_cases.json` without changing
the default golden/paraphrase/fuzz suites. Live model results remain a separate
measurement and are not guaranteed by scripted tests.

Scripted SDK regressions in `tests/test_final_schema_retry.py` cover a natural
node-description query whose successful lookup is followed by a property
dictionary instead of a reference array, correction limits, preserved history,
rejected values/claims, and raw schema scoring. Along with `tests/test_agent_eval.py`
and `tests/test_recovery.py`, these tests use predetermined model responses
without loading Qwen. They do not establish live-model reliability.

Keep reviewed diagnostic queries for paths, specific parallel pipes, typed N-hop
lists, compound properties and missing properties in the main
`tests/eval_cases.json` dataset. Its 18 cases run with
`python scripts/run_eval.py --cases tests/eval_cases.json` or as part of the default
29-case suite, together with 3 paraphrase and 8 fuzz cases. These cases measure
live Qwen interpretation and answer selection, without guaranteeing intent
correction or completeness at runtime. Do not add another score. Retain actual
calls and raw final responses so failures can be inspected separately from
application output.

---

## 11. CLI

Implement a simple CLI, launched directly from the repository checkout through
the root `main.py` script.

Example:

```bash
python main.py
```

Example session:

```text
Graph Agent
Local model: Qwen/Qwen3-1.7B
Type 'exit' to quit.

> Show me everything connected to P4711.

P4711 is connected to:
- H1007 via MNb47122
...

> What pump is upstream of H1007?

P4711 ...
```

Optionally support one-shot execution:

```bash
python main.py \
  --query "What is connected to P4711?"
```

Also support a model override:

```bash
python main.py \
  --model Qwen/Qwen3-0.6B \
  --query "What is connected to P4711?"
```

---

## 12. Repository Structure

Use a clean structure similar to:

```text
pid-graph-agent/
├── README.md
├── requirements.txt
├── pytest.ini
├── main.py
├── .gitignore
├── data/
│   └── graph.json
├── src/
│   └── graph_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── graph_loader.py
│       ├── graph_access.py
│       ├── tools.py
│       ├── agent.py
│       ├── local_qwen.py
│       ├── model_provider.py
│       ├── cli.py
│       └── errors.py
├── tests/
│   ├── test_graph_loader.py
│   ├── test_graph_access.py
│   ├── test_validation.py
│   ├── test_tool_calls.py
│   ├── test_final_schema_retry.py
│   ├── eval_cases.json
│   ├── paraphrase_cases.json
│   └── test_agent_eval.py
└── scripts/
    └── run_eval.py
```

---

## 13. Dependencies and Configuration

List dependency version constraints in `requirements.txt`. Install dependencies
with `python -m pip install -r requirements.txt`; do not require installing the
repository itself as a Python distribution.

Keep the application modules under `src/graph_agent`. The root `main.py` and
`scripts/run_eval.py` add the checkout's `src` directory to the import path before
importing application modules. Configure pytest's source path and test discovery
in `pytest.ini`.

Expected dependencies include:

```text
openai-agents
transformers
torch
accelerate
pytest
```

Add other lightweight dependencies only when justified.

### Configuration


```bash
QWEN_MODEL_ID=Qwen/Qwen3-1.7B
QWEN_DEVICE=auto
```

Optional settings may include:

```bash
QWEN_DTYPE=auto
MAX_NEW_TOKENS=512
```

Do not commit downloaded model weights to the GitHub repository.

Allow Hugging Face/Transformers to cache them locally.

If the model is unavailable locally, print a clear message that the first run may download model weights.

---

## 14. Evaluator Setup Experience

Expected flow:

```bash
git clone <repo>
cd pid-graph-agent

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python -m pytest

python main.py
```

The first live-agent run may download the configured Qwen model.

Document approximate model download/storage expectations in the README.

If hardware is limited, provide a smaller configurable Qwen fallback.

The deterministic graph tests must remain runnable even if the evaluator chooses not to download the model.

---

## 15. Design Principles

Prioritize:

1. Correctness over cleverness.
2. Small reusable tools rather than one tool per natural-language query.
3. Explicit separation between agent logic and graph storage.
4. Deterministic graph operations outside the LLM whenever possible.
5. Use Qwen for natural-language interpretation/tool routing, not graph math.
6. Grounded answers only.
7. Simple POC architecture with clear future extension points.
8. Avoid unnecessary infrastructure.
9. Keep hosted-model dependencies optional or absent.
10. Keep model-specific code behind the local model adapter.

---

## 16. Definition of Done

The project is complete when:

- `graph.json` loads successfully into validated Python dictionaries/lists;
- `GraphAccess` works independently of the agent;
- all graph tools work;
- local Qwen loads through the Agents SDK-compatible model/provider adapter;
- no OpenAI API key is required;
- OpenAI agent runtime can select and call graph tools using local Qwen;
- exact-ID queries work;
- natural-language node-description queries work for representative cases;
- upstream/downstream works;
- N-hop traversal works;
- path queries work;
- node/edge properties can be returned;
- malformed tool calls are handled safely;
- invalid/ambiguous requests are handled safely;
- pytest passes;
- golden/paraphrase/fuzz evaluation can be executed;
- tool-call accuracy can be measured;
- CLI works;
- README documents architecture and tradeoffs;
- repository contains no secrets;
- evaluator can run deterministic tests without downloading the model;
- evaluator can run the full local agent after downloading the configured Qwen model.

---

## Codex Execution Instruction

Before implementing, first propose:

1. the exact repository structure;
2. the JSON schema;
3. the `GraphAccess` interface;
4. the agent tool schemas;
5. the local Qwen `Model` / `ModelProvider` integration strategy for OpenAI Agents SDK;
6. the tool-call parsing strategy;
7. the testing/evaluation plan.

Do not start coding until those pieces are internally consistent with this specification.

Then implement incrementally, keeping the architecture and terminology exactly as defined above.

If a specific OpenAI Agents SDK feature does not support the local Qwen integration cleanly, do not silently replace the architecture with a hosted OpenAI model. Instead, keep the local-model requirement and implement the smallest justified custom model/provider adapter.
