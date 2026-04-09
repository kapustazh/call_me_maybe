from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, Field


# import torch
import numpy as np


class Promptisation(BaseModel):
    prompt: str
    model_config = ConfigDict(extra="forbid")

    # boilerplate code for pydantic
    def execute(self, _prompt: str | None = None) -> str:
        return self.prompt


class Tokenization(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    model: Small_LLM_Model = Field(exclude=True)

    # returns a tensor of shape (1, seq_len), need to convert to list of ints
    def execute(self, prompt: str) -> list[int]:
        return self.model.encode(prompt)[0].tolist()

    def decode(self, tokens: list[int]) -> str:
        return self.model.decode(tokens)


class TokenIndexer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def execute(self, token_ids: list[int]) -> list[int]:
        # Identity stage kept for compatibility with your initial design.
        return token_ids


class LLMProcessing(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    model: Small_LLM_Model = Field(exclude=True)

    def execute(self, tokens: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(tokens)


class TokenSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def execute(self, logits: list[float]) -> int:
        return int(np.argmax(logits))
