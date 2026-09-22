import asyncio
import json
import logging
import re
from typing import Any
from uuid import uuid4
from dataclasses import replace
from agents import FunctionTool, Model, ModelProvider, ModelResponse
from agents.usage import Usage
from jsonschema import Draft202012Validator, validators
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from .config import Config
from .errors import ModelError, ModelOutputError
from .local_qwen import LocalQwen
from .schema import loads_json

logger = logging.getLogger(__name__)
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ArgumentValidator = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda checker, value: type(value) is int,
    ),
)


def _json(text: str) -> Any:
    try:
        return loads_json(text)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ModelOutputError(f"Local Qwen returned invalid JSON: {exc}") from exc


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("input_text", "output_text"):
                raise ModelError("Local Qwen supports text messages only.")
            parts.append(part["text"])
        return "\n".join(parts)
    raise ModelError("Local Qwen received unsupported message content.")


def _messages(system_instructions: str | None, items: str | list) -> list[dict[str, Any]]:
    messages = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})

    if isinstance(items, str):
        messages.append({"role": "user", "content": items})
        return messages
    
    calls: dict[str, str] = {}

    for raw in items:
        item = raw.model_dump(exclude_none=True) if hasattr(raw, "model_dump") else raw
        kind = item.get("type", "message")
        if kind == "function_call":
            calls[item["call_id"]] = item["name"]
            call = {
                "id": item["call_id"], "type": "function",
                "function": {"name": item["name"], "arguments": _json(item["arguments"])},
            }
            if messages and messages[-1].get("tool_calls"):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            call_id = item["call_id"]
            if call_id not in calls:
                raise ModelError(f"Tool output has no matching invocation: {call_id}.")
            messages.append({
                "role": "tool", "tool_call_id": call_id, "name": calls[call_id],
                "content": _text_content(item["output"]),
            })
        elif kind == "message":
            role = item["role"]
            if role not in ("user", "assistant", "system", "developer"):
                raise ModelError(f"Unsupported message role: {role}.")
            messages.append({
                "role": "system" if role == "developer" else role,
                "content": _text_content(item["content"]),
            })
        else:
            raise ModelError(f"Unsupported local model input item: {kind}.")
    return messages


def _parse_output(text: str, tools: list[FunctionTool]) -> list:
    text = text.strip()
    if not text:
        raise ModelOutputError("Local Qwen returned an empty response.")
    matches = list(_TOOL_CALL.finditer(text))
    if matches or "<tool_call" in text or "</tool_call" in text:
        if not matches or _TOOL_CALL.sub("", text).strip():
            raise ModelOutputError("Malformed tool-call tags or mixed prose and tool calls.")
        available = {tool.name: tool for tool in tools}
        output = []
        for match in matches:
            call = _json(match.group(1))

            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ModelOutputError("A tool call must contain exactly name and arguments.")
            
            name, arguments = call["name"], call["arguments"]

            if not isinstance(name, str) or name not in available:
                raise ModelOutputError(f"Unknown tool name: {name!r}.")
            if not isinstance(arguments, dict):
                raise ModelOutputError(f"Arguments for {name} must be a JSON object.")
            
            errors = list(_ArgumentValidator(available[name].params_json_schema).iter_errors(arguments))

            if errors:
                raise ModelOutputError(f"Invalid arguments for {name}: {errors[0].message}")
            if arguments.get("resolve") and len(matches) > 1:
                raise ModelOutputError(
                    "Node resolution must finish before another tool call. "
                    "Return the resolve=true call by itself."
                )
            output.append(ResponseFunctionToolCall(
                id=f"fc_{uuid4().hex}", call_id=f"call_{uuid4().hex}",
                name=name, arguments=json.dumps(arguments, allow_nan=False),
                type="function_call", status="completed",
            ))
        return output
    
    # Preserve non-tool text as an untrusted final message. Grounding validates
    # its JSON syntax/schema before rendering. Completing the SDK response keeps
    # history and turn accounting available for the single corrective retry.
    return [ResponseOutputMessage(
        id=f"msg_{uuid4().hex}", role="assistant", type="message", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )]


class LocalQwenModel(Model):
    def __init__(self, backend: Any):
        self.backend = backend

    async def get_response(
        self, system_instructions, input, model_settings, tools, output_schema,
        handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None,
    ) -> ModelResponse:
        unsupported = [
            name for name, value in model_settings.to_json_dict().items()
            if value is not None and name not in {"tool_choice", "parallel_tool_calls"}
        ]
        if model_settings.tool_choice not in (None, "auto"):
            unsupported.append("tool_choice")

        if unsupported:
            raise ModelError(
                "Local Qwen does not support these ModelSettings fields: "
                + ", ".join(sorted(unsupported))
                + ". Configure token limits through Config; local generation is greedy."
            )
        
        if previous_response_id or conversation_id or prompt or handoffs:
            raise ModelError("Local Qwen requires inline history and does not support hosted prompts or handoffs.")
        
        if any(not isinstance(tool, FunctionTool) for tool in tools):
            raise ModelError("Local Qwen supports function tools only.")
        messages = _messages(system_instructions, input)

        if output_schema is not None and not output_schema.is_plain_text():
            messages.insert(0, {
                "role": "system",
                "content": "Return final answers as JSON matching this schema: " + json.dumps(output_schema.json_schema()),
            })

        schemas = [{"type": "function", "function": {
            "name": tool.name, "description": tool.description,
            "parameters": tool.params_json_schema,
        }} for tool in tools]

        try:
            generated = await asyncio.to_thread(self.backend.generate, messages, schemas)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Local Qwen inference failed: {exc}") from exc
        
        logger.debug("Local Qwen generated output: %s", generated.text)
        output = _parse_output(generated.text, tools)

        if model_settings.parallel_tool_calls is False and len(output) > 1:
            raise ModelOutputError("Local Qwen returned multiple tool calls while parallel_tool_calls=False.")
        
        return ModelResponse(
            output=output, response_id=None,
            usage=Usage(
                requests=1, input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                total_tokens=generated.input_tokens + generated.output_tokens,
            ),
        )

    async def stream_response(
        self, system_instructions, input, model_settings, tools, output_schema,
        handoffs, tracing, *, previous_response_id=None, conversation_id=None, prompt=None,
    ):
        raise ModelError("Local Qwen streaming is not supported; use Runner.run().")
        yield  # Make this an async iterator, matching the SDK interface.


class LocalQwenProvider(ModelProvider):
    def __init__(self, config: Config):
        self.config = config
        self._models: dict[str, LocalQwenModel] = {}

    def get_model(self, model_name: str | None) -> LocalQwenModel:
        name = model_name or self.config.model_id

        if name not in self._models:
            self._models[name] = LocalQwenModel(LocalQwen(replace(self.config, model_id=name)))

        return self._models[name]
