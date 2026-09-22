import asyncio
import json
import pytest
from contextlib import nullcontext
from types import SimpleNamespace
from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner
from agents.models.interface import ModelTracing
from graph_agent import local_qwen
from graph_agent.config import Config
from graph_agent.errors import ModelError, ModelOutputError
from graph_agent.local_qwen import Generation, LocalQwen
from graph_agent.model_provider import LocalQwenModel, LocalQwenProvider


class Backend:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate(self, messages, tools):
        self.requests.append((messages, tools))
        return Generation(next(self.outputs), 10, 4)


@pytest.fixture
def node_tool():
    async def invoke(context, arguments):
        return json.dumps({"node": {"id": json.loads(arguments)["node_id"]}})

    return FunctionTool(
        name="get_node", description="Get a graph node by exact ID.",
        params_json_schema={
            "type": "object", "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"], "additionalProperties": False,
        }, on_invoke_tool=invoke,
    )


def respond(backend, tools=(), **kwargs):
    return asyncio.run(LocalQwenModel(backend).get_response(
        system_instructions="Use graph tools.", input=kwargs.pop("input", "Describe P4711"),
        model_settings=kwargs.pop("model_settings", ModelSettings()), tools=list(tools), output_schema=None,
        handoffs=[], tracing=ModelTracing.DISABLED, **kwargs,
    ))


def test_tool_call_schema_and_usage(node_tool):
    backend = Backend('<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>')
    result = respond(backend, [node_tool])
    call = result.output[0]
    assert call.name == "get_node"
    assert json.loads(call.arguments) == {"node_id": "P4711"}
    assert call.call_id
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 14
    messages, tools = backend.requests[0]
    assert messages == [
        {"role": "system", "content": "Use graph tools."},
        {"role": "user", "content": "Describe P4711"},
    ]
    assert tools[0]["function"]["parameters"] == node_tool.params_json_schema


def test_multiple_calls_have_unique_ids(node_tool):
    raw = '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>'
    result = respond(Backend(raw + "\n" + raw), [node_tool])
    assert len(result.output) == 2
    assert result.output[0].call_id != result.output[1].call_id


def test_parallel_tool_call_setting_is_enforced(node_tool):
    raw = '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>'
    with pytest.raises(ModelOutputError, match="parallel_tool_calls=False"):
        respond(Backend(raw + raw), [node_tool], model_settings=ModelSettings(parallel_tool_calls=False))
    assert len(respond(Backend(raw), [node_tool], model_settings=ModelSettings(parallel_tool_calls=False)).output) == 1
    assert len(respond(Backend(raw + raw), [node_tool], model_settings=ModelSettings(parallel_tool_calls=True)).output) == 2


@pytest.mark.parametrize("settings,field", [
    (ModelSettings(tool_choice="none"), "tool_choice"),
    (ModelSettings(tool_choice="required"), "tool_choice"),
    (ModelSettings(tool_choice="get_node"), "tool_choice"),
    (ModelSettings(max_tokens=1), "max_tokens"),
    (ModelSettings(temperature=0), "temperature"),
    (ModelSettings(top_p=0.9), "top_p"),
    (ModelSettings(store=False), "store"),
    (ModelSettings(extra_args={"unexpected": True}), "extra_args"),
])
def test_unsupported_model_settings_rejected_before_inference(settings, field):
    backend = Backend('{"answer":[]}')
    with pytest.raises(ModelError, match=f"ModelSettings fields: {field}"):
        respond(backend, model_settings=settings)
    assert backend.requests == []


def test_auto_tool_choice_is_supported(node_tool):
    raw = '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>'
    result = respond(Backend(raw), [node_tool], model_settings=ModelSettings(tool_choice="auto"))
    assert result.output[0].name == "get_node"


@pytest.mark.parametrize("output", [
    "",
    '<tool_call>{"name":"get_node","arguments":{}}</tool_call>',
    '<tool_call>{"name":"get_node","arguments":{"node_id":123}}</tool_call>',
    '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711","extra":1}}</tool_call>',
    '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
    '<tool_call>{"name":[],"arguments":{}}</tool_call>',
    '<tool_call>{"name":"get_node","arguments":"{}"}</tool_call>',
    '<tool_call>{"name":"get_node","arguments":{},"extra":1}</tool_call>',
    '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}',
    'Here is the call: <tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>',
    '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call> {"answer":[]}',
    '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711","node_id":"H1007"}}</tool_call>',
])
def test_empty_output_and_malformed_tool_calls_are_controlled(output, node_tool):
    with pytest.raises(ModelOutputError):
        respond(Backend(output), [node_tool])


def test_final_json_is_assistant_message():
    result = respond(Backend('{"answer":[]}'))
    assert result.output[0].type == "message"
    assert result.output[0].content[0].text == '{"answer":[]}'


@pytest.mark.parametrize("output", [
    "Unverified prose", "[]", '{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}',
    '{"status":', '{"status":"answer","node_ids":[],"edge_ids":[],"properties":[]}"',
])
def test_invalid_final_text_is_preserved_but_never_passes_grounding(output, access):
    from graph_agent.errors import FinalSelectionError
    from graph_agent.grounding import render_answer
    from graph_agent.tools import QueryContext, execute_tool

    result = respond(Backend(output))
    message = result.output[0]
    assert message.type == "message"
    assert message.content[0].text == output
    context = QueryContext(access)
    execute_tool(context, "get_node", {"node_id": "P4711"})
    with pytest.raises(FinalSelectionError):
        render_answer(message.content[0].text, context)


def test_float_hops_and_batched_resolution_rejected():
    from graph_agent.tools import build_tools

    with pytest.raises(ModelOutputError, match="Invalid arguments"):
        respond(Backend('<tool_call>{"name":"get_connected_subgraph","arguments":{"node_id":"P4711","max_hops":1.0}}</tool_call>'), build_tools())
    with pytest.raises(ModelOutputError, match="resolution must finish"):
        respond(Backend(
            '<tool_call>{"name":"search_nodes","arguments":{"query":"pump","resolve":true}}</tool_call>'
            '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>'
        ), build_tools())


def test_sdk_tool_history_translated(node_tool):
    backend = Backend('{"answer":[]}')
    respond(backend, [node_tool], input=[
        {"role": "user", "content": [{"type": "input_text", "text": "Describe P4711"}]},
        {"type": "function_call", "call_id": "call_1", "name": "get_node", "arguments": '{"node_id":"P4711"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": '{"id":"P4711"}'},
    ])
    messages = backend.requests[0][0]
    assert messages[2]["tool_calls"][0]["function"] == {
        "name": "get_node", "arguments": {"node_id": "P4711"},
    }
    assert messages[3] == {
        "role": "tool", "name": "get_node", "tool_call_id": "call_1", "content": '{"id":"P4711"}',
    }


def test_agents_sdk_executes_local_tool_round_trip(node_tool):
    backend = Backend(
        '<tool_call>{"name":"get_node","arguments":{"node_id":"P4711"}}</tool_call>',
        '{"answer":[]}',
    )
    agent = Agent(name="test", model=LocalQwenModel(backend), tools=[node_tool])
    result = asyncio.run(Runner.run(agent, "Describe P4711", run_config=RunConfig(tracing_disabled=True)))
    assert result.final_output == '{"answer":[]}'
    assert backend.requests[1][0][-1]["role"] == "tool"
    assert json.loads(backend.requests[1][0][-1]["content"]) == {"node": {"id": "P4711"}}


def test_provider_reuses_lazy_model():
    provider = LocalQwenProvider(Config())
    first = provider.get_model(None)
    assert provider.get_model("Qwen/Qwen3-1.7B") is first
    assert first.backend._model is None
    assert provider.get_model("Qwen/Qwen3-0.6B") is not first


def test_unknown_history_and_streaming_fail_explicitly():
    with pytest.raises(ModelError, match="matching invocation"):
        respond(Backend(), input=[{"type": "function_call_output", "call_id": "missing", "output": "{}"}])
    with pytest.raises(ModelError, match="inline history"):
        respond(Backend(), previous_response_id="hosted")

    async def stream():
        async for _ in LocalQwenModel(Backend()).stream_response(None, "query", ModelSettings(), [], None, [], ModelTracing.DISABLED):
            pass
    with pytest.raises(ModelError, match="streaming"):
        asyncio.run(stream())


def test_backend_errors_are_controlled():
    class Broken:
        def generate(self, messages, tools):
            raise RuntimeError("out of memory")

    with pytest.raises(ModelError, match="inference failed: out of memory"):
        respond(Broken())


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def __getitem__(self, index):
        return self.values[index[1]]


@pytest.fixture
def fake_libraries(monkeypatch):
    calls = {"load": 0, "cuda": False, "mps": False}

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = None

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer_load"] = (model_id, kwargs)
            return Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = kwargs
            return "prompt"

        def __call__(self, prompt, **kwargs):
            calls["tokenize"] = kwargs
            return {"input_ids": Tensor([10, 11]), "attention_mask": Tensor([1, 1])}

        def decode(self, tokens, **kwargs):
            calls["decode"] = tokens
            return '{"answer":[]}'

    class Model:
        config = SimpleNamespace(max_position_embeddings=100)

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["load"] += 1
            calls["model_load"] = kwargs
            if calls.get("load_failure"):
                raise OSError("weights unavailable")
            model = Model()
            model.generation_config = SimpleNamespace(eos_token_id=calls.get("eos"))
            return model

        def to(self, device):
            calls["device"] = device

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            if calls.get("inference_failure"):
                raise RuntimeError("device out of memory")
            return Tensor(calls.get("output", [10, 11, 21, 22, 99]))

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: calls["cuda"]),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: calls["mps"])),
        float32="float32", float16="float16", bfloat16="bfloat16",
        inference_mode=nullcontext,
    )
    monkeypatch.setattr(local_qwen, "torch", torch)
    monkeypatch.setattr(local_qwen, "AutoTokenizer", Tokenizer)
    monkeypatch.setattr(local_qwen, "AutoModelForCausalLM", Model)
    return calls


def test_lazy_local_loading_cpu_and_token_accounting(fake_libraries):
    backend = LocalQwen(Config(max_new_tokens=8, local_files_only=True))
    assert fake_libraries["load"] == 0
    result = backend.generate([{"role": "user", "content": "query"}], [])
    backend.generate([{"role": "user", "content": "second query"}], [])
    assert result == Generation('{"answer":[]}', 2, 3)
    assert fake_libraries["load"] == 1
    assert fake_libraries["device"] == "cpu"
    assert fake_libraries["model_load"]["torch_dtype"] == "float32"
    assert fake_libraries["model_load"]["local_files_only"] is True
    assert fake_libraries["model_load"]["trust_remote_code"] is False
    assert fake_libraries["generate"]["do_sample"] is False
    assert fake_libraries["generate"]["max_new_tokens"] == 8
    assert fake_libraries["decode"] == [21, 22, 99]
    assert fake_libraries["template"]["enable_thinking"] is False
    assert fake_libraries["tokenize"]["truncation"] is False


@pytest.mark.parametrize("device,dtype,cuda,mps,expected_device,expected_dtype", [
    ("auto", "auto", False, False, "cpu", "float32"),
    ("auto", "auto", True, True, "cuda", "auto"),
    ("auto", "auto", False, True, "mps", "float16"),
    ("cpu", "auto", True, True, "cpu", "float32"),
    ("cpu", "bfloat16", False, False, "cpu", "bfloat16"),
    ("cuda", "float16", True, False, "cuda", "float16"),
    ("mps", "float32", True, True, "mps", "float32"),
])
def test_device_selection_and_dtype_overrides(
    fake_libraries, device, dtype, cuda, mps, expected_device, expected_dtype,
):
    fake_libraries.update(cuda=cuda, mps=mps)
    LocalQwen(Config(device=device, dtype=dtype, max_new_tokens=8)).generate([], [])
    assert fake_libraries["device"] == expected_device
    assert fake_libraries["model_load"]["torch_dtype"] == expected_dtype


def test_input_limit_does_not_silently_truncate(fake_libraries):
    with pytest.raises(ModelError, match="exceeding the configured limit"):
        LocalQwen(Config(max_input_tokens=1, max_new_tokens=8)).generate([], [])
    assert "generate" not in fake_libraries


def test_context_limit_and_unavailable_device(fake_libraries):
    with pytest.raises(ModelError, match="context window"):
        LocalQwen(Config(max_new_tokens=100)).generate([], [])
    with pytest.raises(ModelError, match="CUDA was requested"):
        LocalQwen(Config(device="cuda")).generate([], [])
    with pytest.raises(ModelError, match="Apple MPS was requested"):
        LocalQwen(Config(device="mps")).generate([], [])


def test_model_load_failure_has_actionable_error(fake_libraries):
    fake_libraries["load_failure"] = True
    with pytest.raises(ModelError, match="Could not load local Qwen model.*weights unavailable.*Try Qwen/Qwen3-0.6B"):
        LocalQwen(Config()).generate([], [])


def test_failed_load_can_retry_then_reuses_loaded_model(fake_libraries):
    fake_libraries["load_failure"] = True
    backend = LocalQwen(Config(max_new_tokens=8))
    with pytest.raises(ModelError, match="weights unavailable"):
        backend.generate([], [])
    assert backend._model is None
    fake_libraries["load_failure"] = False
    backend.generate([], [])
    backend.generate([], [])
    assert fake_libraries["load"] == 2


def test_output_limit_rejects_truncation_but_accepts_terminal_eos(fake_libraries):
    backend = LocalQwen(Config(max_new_tokens=3))
    fake_libraries["output"] = [10, 11, 21, 22, 23]
    with pytest.raises(ModelOutputError, match="MAX_NEW_TOKENS"):
        backend.generate([], [])
    assert "decode" not in fake_libraries
    fake_libraries["output"] = [10, 11, 21, 22, 99]
    assert backend.generate([], []).output_tokens == 3


@pytest.mark.parametrize("eos", [98, [99, 98]])
def test_generation_config_eos_is_respected_at_token_limit(fake_libraries, eos):
    fake_libraries.update(eos=eos, output=[10, 11, 21, 22, 98])
    result = LocalQwen(Config(max_new_tokens=3)).generate([], [])
    assert result.output_tokens == 3


def test_inference_failure_is_controlled_and_model_remains_loaded(fake_libraries):
    fake_libraries["inference_failure"] = True
    backend = LocalQwen(Config(max_new_tokens=8))
    with pytest.raises(ModelError, match="inference failed: device out of memory"):
        backend.generate([], [])
    fake_libraries["inference_failure"] = False
    assert backend.generate([], []).text == '{"answer":[]}'
    assert fake_libraries["load"] == 1
