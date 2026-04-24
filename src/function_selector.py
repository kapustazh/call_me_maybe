from __future__ import annotations

from dataclasses import dataclass

from llm_sdk import Small_LLM_Model
from src.math_utils import softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
import torch


@dataclass(frozen=True)
class Selection:
    name: str
    confidence: float
    scores: dict[str, float]


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

    @staticmethod
    def _to_ids(t: torch.Tensor) -> list[int]:
        """Transforms data from torch.Tensor to list[int]"""
        return t[0].tolist()

    def _name_score(
        self, base_ids: list[int], name_ids: list[int]
    ) -> float:
        """Score full candidate from token probabilities.

        Uses geometric mean so shorter function names do not auto-win.
        """
        if not name_ids:
            return 0.0
        ids = list(base_ids)
        score = 1.0
        for tok in name_ids:
            logits = self._model.get_logits_from_input_ids(ids)
            probs = softmax(logits)
            if tok >= len(probs):
                return 0.0
            score *= float(probs[tok])
            ids.append(tok)
        return score ** (1.0 / len(name_ids))

    def select(self, user_prompt: str) -> Selection:
        prefix = self._prompter.function_name_prefix()
        prompt = self._prompter.build_selection_prompt(user_prompt)
        base_ids = self._to_ids(self._model.encode(prompt))

        scores: list[float] = []
        score_by_name: dict[str, float] = {}
        for fn in self._functions:
            name_suffix = fn.name.removeprefix(prefix)
            name_ids = self._to_ids(self._model.encode(name_suffix))
            if not name_ids:
                score = 0.0
                scores.append(score)
                score_by_name[fn.name] = score
                continue
            score = self._name_score(base_ids, name_ids)
            scores.append(score)
            score_by_name[fn.name] = score

        score_sum = sum(scores)
        probs = (
            [score / score_sum for score in scores]
            if score_sum > 0
            else [1.0 / len(scores) for _ in scores]
        )
        best_ids = max(
            range(len(probs)),
            key=lambda i: (probs[i], scores[i]),
        )
        best_guess = Selection(
            name=self._functions[best_ids].name,
            confidence=float(probs[best_ids]),
            scores=score_by_name,
        )

        if best_guess.confidence < self._threshold:
            raise FunctionSelectorError(
                f"Low selection confidence: {best_guess.confidence:.3f} < "
                f"{self._threshold:.3f}"
            )
        return best_guess
