from abc import ABC, abstractmethod
from llm_sdk import Small_LLM_Model
import torch
import numpy as np

class Pipeline(ABC):

    @abstractmethod
    def execute(self, prompt: str) -> str:
        ...

class Promptisation(Pipeline):

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def execute(self, prompt: str) -> str:
        return self.prompt

class Tokenization(Pipeline):

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model

    def execute(self, prompt: str) -> list[torch.Tensor]:
        return self.model.encode(prompt)

class TokenIndexer(Pipeline):
    def execute(self, tokens: list[torch.Tensor]) -> list[int]:
        return [token.item() for token in tokens]

class LLMPRocessing(Pipeline):
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model

    def execute(self, tokens: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(tokens)

class TokenSelection(Pipeline):
    def execute(self, logits: list[float]) -> int:
        return np.argmax(logits)