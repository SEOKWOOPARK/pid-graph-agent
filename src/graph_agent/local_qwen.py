import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any
from .config import Config
from .errors import ModelError, ModelOutputError
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


class LocalQwen:
    def __init__(self, config: Config):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._lock = Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                raise ModelError("CUDA was requested but is not available.")
            if device == "mps" and not torch.backends.mps.is_available():
                raise ModelError("Apple MPS was requested but is not available.")
            
            if self.config.dtype != "auto":
                dtype = getattr(torch, self.config.dtype)
            elif device == "cpu":
                dtype = torch.float32
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = "auto"

            logger.info(
                "Loading local model %s on %s; the first run may download model weights "
                "from Hugging Face. Subsequent runs reuse the local cache.",
                self.config.model_id, device,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype=dtype,
                local_files_only=self.config.local_files_only, trust_remote_code=False,
            )

            model.to(device)
            model.eval()
            self._tokenizer, self._model = tokenizer, model
            self._torch, self._device = torch, device
        except Exception as exc:
            raise ModelError(
                f"Could not load local Qwen model {self.config.model_id!r}: {exc}. "
                "Install the local model dependencies and make sure weights are cached "
                "or Hugging Face is reachable. Try Qwen/Qwen3-0.6B for limited memory."
            ) from exc

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Generation:
        with self._lock:
            self._load()

            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tools=tools or None, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                encoded = self._tokenizer(prompt, return_tensors="pt", truncation=False)
                input_tokens = encoded["input_ids"].shape[-1]

                if input_tokens > self.config.max_input_tokens:
                    raise ModelError(
                        f"Model input has {input_tokens} tokens, exceeding the configured "
                        f"limit of {self.config.max_input_tokens}. Ask a smaller query."
                    )
                
                context_limit = getattr(self._model.config, "max_position_embeddings", None)

                if isinstance(context_limit, int) and (
                    input_tokens + self.config.max_new_tokens > context_limit
                ):
                    raise ModelError(
                        "The requested input and output exceed the model context window. "
                        "Reduce the query or MAX_NEW_TOKENS."
                    )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                eos_id = self._tokenizer.eos_token_id
                pad_id = self._tokenizer.pad_token_id

                with self._torch.inference_mode():
                    output = self._model.generate(
                        **encoded, max_new_tokens=self.config.max_new_tokens,
                        do_sample=False, pad_token_id=pad_id if pad_id is not None else eos_id,
                    )
                generated = output[0, input_tokens:]
                output_tokens = len(generated)
                generation_config = getattr(self._model, "generation_config", None)
                effective_eos = getattr(generation_config, "eos_token_id", None)
                if effective_eos is None:
                    effective_eos = eos_id
                eos_ids = effective_eos if isinstance(effective_eos, list) else [effective_eos]
                if output_tokens >= self.config.max_new_tokens and int(generated[-1]) not in eos_ids:
                    raise ModelOutputError(
                        "Local Qwen reached MAX_NEW_TOKENS before completing its response. "
                        "Increase the limit or ask a smaller query."
                    )
                return Generation(
                    text=self._tokenizer.decode(generated, skip_special_tokens=True).strip(),
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
            except (ModelError, ModelOutputError):
                raise
            except Exception as exc:
                raise ModelError(f"Local Qwen inference failed: {exc}") from exc
