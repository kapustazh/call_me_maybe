from __future__ import annotations

import math

from typing import Any, cast

from src.math_utils import softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter


class FunctionSelectorError(Exception):
    """Error handling during selection of the function"""


class FunctionSelector:
    def __init__(
        self,
        model: Any,
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
    def _to_ids(t: Any) -> list[int]:
        """Convert encoder output to list[int]."""
        return cast(list[int], t[0].tolist())

    def _name_score(self, base_ids: list[int], name_ids: list[int]) -> float:
        """Return average log probability for a candidate function name."""
        if not name_ids:
            return -math.inf

        ids = list(base_ids)
        total_log_prob = 0.0

        for tok in name_ids:
            logits = self._model.get_logits_from_input_ids(ids)
            probs = softmax(logits)
            if tok >= len(probs) or probs[tok] <= 0:
                return -math.inf

            total_log_prob += math.log(float(probs[tok]))
            ids.append(tok)

        return total_log_prob / len(name_ids)

    def select(self, user_prompt: str) -> str:
        prefix = self._prompter.function_name_prefix()
        prompt = self._prompter.build_selection_prompt(user_prompt)
        base_ids = self._to_ids(self._model.encode(prompt))

        scores: list[float] = []
        for fn in self._functions:
            name_suffix = fn.name.removeprefix(prefix)
            name_ids = self._to_ids(self._model.encode(name_suffix))
            if not name_ids:
                score = -math.inf
                scores.append(score)
                continue
            score = self._name_score(base_ids, name_ids)
            scores.append(score)

        probs = softmax(scores)
        best_ids = max(range(len(probs)), key=probs.__getitem__)
        best_name = self._functions[best_ids].name
        # confidence = float(probs[best_ids])

        # if confidence < self._threshold:
        #     return (
        #         f"Low selection confidence: {confidence:.3f} < "
        #         f"{self._threshold:.3f} for candidate {best_name}"
        #     )

        return best_name
