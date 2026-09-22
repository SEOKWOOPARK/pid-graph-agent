from dataclasses import dataclass
from .errors import InvalidInputError
import os

MAX_QUERY_LENGTH = 200


@dataclass(frozen=True)
class Config:
    model_id: str = "Qwen/Qwen3-1.7B"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 512
    max_input_tokens: int = 8192
    max_turns: int = 12
    local_files_only: bool = False

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise InvalidInputError("Model ID must be nonblank.")
        
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise InvalidInputError("QWEN_DEVICE must be auto, cpu, cuda, or mps.")
        
        if not isinstance(self.dtype, str) or self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise InvalidInputError("QWEN_DTYPE must be auto, float32, float16, or bfloat16.")
        
        for name, upper in (("max_new_tokens", 4096), ("max_input_tokens", 32768), ("max_turns", 30)):
            value = getattr(self, name)

            if type(value) is not int or not 1 <= value <= upper:
                raise InvalidInputError(f"{name} must be an integer from 1 to {upper}.")
            
        if type(self.local_files_only) is not bool:
            raise InvalidInputError("local_files_only must be boolean.")

    @classmethod
    def from_env(cls, model_id: str | None = None):
        try:
            return cls(
                model_id=model_id if model_id is not None else os.getenv("QWEN_MODEL_ID", cls.model_id),
                device=os.getenv("QWEN_DEVICE", cls.device),
                dtype=os.getenv("QWEN_DTYPE", cls.dtype),
                max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", cls.max_new_tokens)),
                max_input_tokens=int(os.getenv("MAX_INPUT_TOKENS", cls.max_input_tokens)),
                max_turns=int(os.getenv("MAX_TURNS", cls.max_turns)),
                local_files_only=os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"},
            )
        except ValueError as exc:
            raise InvalidInputError(f"Invalid local model configuration: {exc}") from exc
