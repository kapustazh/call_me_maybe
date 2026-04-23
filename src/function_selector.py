from __future__ import annotations

from dataclasses import dataclass

from llm_sdk import Small_LLM_Model
from src.math_utils import MathUtils
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
import torch


@dataclass(frozen=True)
class Selection:
    name: str
    confidence: float


class FunctionSelectorError(Exception):
    """Error handling during selection of the function"""


class FunctionSelector:
    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        *,
        confidence_threshold: float = 0.90,
    ) -> None:
        if not functions:
            raise ValueError("No functions provided for selection")
        self._model = model
        self._functions = functions
        self._threshold = confidence_threshold
        self._prompter = BobThePrompter(functions)

    def _to_ids(t: torch.Tensor) -> list[int]:
        """Transforms data from torch.Tensor to list[int]"""
        return t[0].tolist()

    def select(self, user_prompt: str) -> Selection:
        prefix = self._prompter.function_name_prefix()
        prompt = self._prompter.build_selection_prompt(user_prompt)

        base_ids = self._to_ids(self._model.encode(prompt))
        logits = self._model.get_logits_from_input_ids(base_ids)

        prefix_ids = self._to_ids(self._model.encode(prefix)) if prefix else []

        scores: list[float] = []
        for fn in self._functions:
            name_ids = self._to_ids(self._model.encode(fn.name))
            if not name_ids:
                scores.append(-float("inf"))
                continue

            # Prefer first token after common prefix, else fall back to first
            # token.
            tok_id: int
            if (
                prefix_ids
                and len(name_ids) > len(prefix_ids)
                and name_ids[: len(prefix_ids)] == prefix_ids
            ):
                tok_id = name_ids[len(prefix_ids)]
            else:

                tok_id = name_ids[0]
            scores.append(
                float(logits[tok_id])
                if tok_id < len(logits)
                else -float("inf")
            )

        probs = MathUtils.softmax(scores)
        best_ids = max(range(len(probs)), key=lambda i: probs[i])
        best_guess = Selection(
            name=self._functions[best_ids].name,
            confidence=float(probs[best_ids]),
        )

        if best_guess.confidence < self._threshold:
            raise FunctionSelectorError(
                f"Low selection confidence: {best_guess.confidence:.3f} < "
                f"{self._threshold:.3f}"
            )
        return best_guess
