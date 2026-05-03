from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from llm_sdk import Small_LLM_Model  # type: ignore

from src.model_utils import encoded_to_token_ids
from src.math_utils import softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter

# Default selection confidence used across pipeline and selector
DEFAULT_SELECTION_CONFIDENCE = 0.90


class FunctionSelectorError(Exception):
    """Error handling during selection of the function"""


@dataclass(frozen=True)
class _FunctionCandidate:
    name: str
    suffix_ids: tuple[int, ...]
    keywords: frozenset[str]


class FunctionSelector:
    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        *,
        confidence_threshold: float = DEFAULT_SELECTION_CONFIDENCE,
    ) -> None:
        if not functions:
            raise ValueError("No functions provided for selection")
        self._model: Small_LLM_Model = model
        self._functions: list[FunctionDefinition] = functions
        self._threshold: float = confidence_threshold
        self._prompter: BobThePrompter = BobThePrompter(functions)
        self._prefix: str = self._prompter.function_name_prefix()
        self._candidates: list[_FunctionCandidate] = self._build_candidates()

    def _build_candidates(self) -> list[_FunctionCandidate]:
        candidates: list[_FunctionCandidate] = []
        for function_definition in self._functions:
            suffix = function_definition.name.removeprefix(self._prefix)
            suffix_ids = encoded_to_token_ids(self._model.encode(suffix))
            if not suffix_ids:
                raise FunctionSelectorError(
                    f"Function name '{function_definition.name}' has empty "
                    "suffix after prefix split"
                )
            candidates.append(
                _FunctionCandidate(
                    name=function_definition.name,
                    suffix_ids=tuple(suffix_ids),
                    keywords=self._keywords_for(function_definition),
                )
            )
        return candidates

    @staticmethod
    def _keywords_for(
        function_definition: FunctionDefinition,
    ) -> frozenset[str]:
        raw = (
            f"{function_definition.name} " f"{function_definition.description}"
        ).lower()
        tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", raw)
            if token and len(token) > 1
        }
        return frozenset(tokens)

    def _continuation_score(
        self,
        base_ids: list[int],
        continuation_ids: tuple[int, ...],
    ) -> float:
        if not continuation_ids:
            return 0.0
        total_log_prob = 0.0
        history = list(base_ids)
        for token_id in continuation_ids:
            logits = self._model.get_logits_from_input_ids(history)
            probs = softmax(logits)
            if token_id >= len(probs) or probs[token_id] <= 0:
                return -math.inf
            total_log_prob += math.log(float(probs[token_id]))
            history.append(token_id)
        return total_log_prob / len(continuation_ids)

    @staticmethod
    def _best_index(probs: list[float]) -> int:
        if not probs:
            raise FunctionSelectorError("No function candidates")
        best_index = max(range(len(probs)), key=probs.__getitem__)
        return int(best_index)

    def _validate_confidence(
        self,
        probs: list[float],
        best_index: int,
        best_name: str,
    ) -> None:
        confidence = float(probs[best_index])
        if confidence < self._threshold:
            raise FunctionSelectorError(
                f"Low selection confidence: {confidence:.3f} < "
                f"{self._threshold:.3f} for '{best_name}'"
            )

    def _boundary_scores(self, base_ids: list[int]) -> list[float]:
        logits = self._model.get_logits_from_input_ids(base_ids)
        scores: list[float] = []
        groups: dict[int, list[int]] = defaultdict(list)
        for index, candidate in enumerate(self._candidates):
            first_token = candidate.suffix_ids[0]
            groups[first_token].append(index)
            if first_token >= len(logits):
                scores.append(-math.inf)
                continue
            score = float(logits[first_token])
            scores.append(score)

        for first_token, indices in groups.items():
            if len(indices) <= 1:
                continue
            for index in indices:
                continuation = self._candidates[index].suffix_ids[1:]
                if not continuation:
                    continue
                scores[index] += self._continuation_score(
                    base_ids + [first_token],
                    continuation,
                )
        return scores

    def select(self, user_prompt: str) -> str:
        prompt = self._prompter.build_selection_prompt(user_prompt)
        base_ids = encoded_to_token_ids(self._model.encode(prompt))
        scores = self._boundary_scores(base_ids)
        if not scores or all(
            math.isinf(score) and score < 0 for score in scores
        ):
            raise FunctionSelectorError(
                "No valid function candidate from logits"
            )
        model_probs = softmax(scores)
        best_index = self._best_index(model_probs)
        best_name = self._candidates[best_index].name
        # validate using model-only probabilities
        self._validate_confidence(model_probs, best_index, best_name)
        return best_name
