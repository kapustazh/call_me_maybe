from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from src.model_utils import encoded_to_token_ids
from src.math_utils import softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter

_UNIFORM_MULTIPLIER = 3.0
# Sharpen softmax over selection scores until top mass ≥ this (or schedule ends).
_TARGET_TOP_SOFTMAX_PROB = 0.9
_TEMPERATURE_SCHEDULE = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)


class FunctionSelectorError(Exception):
    """Error handling during selection of the function"""


@dataclass(frozen=True)
class _FunctionCandidate:
    """Function candidate for selection"""

    name: str
    distinguishing_ids: tuple[
        int, ...
    ]  # tokens after selection_completion_prefix


def adaptive_threshold(n_candidates: int) -> float:
    """Scale threshold with candidate count.

    Matches 0.90 for 5 candidates, scales inversely for more.
    Capped at 0.90 for small candidate sets.
    """
    if n_candidates <= 0:
        return 1.0
    return min(_UNIFORM_MULTIPLIER / n_candidates, 0.9)


class FunctionSelector:
    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        *,
        confidence_threshold: float | None = None,
        peak_softmax_target: float | None = None,
    ) -> None:
        if not functions:
            raise ValueError("No functions provided for selection")
        self._model: Small_LLM_Model = model
        self._functions: list[FunctionDefinition] = functions
        self._threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else adaptive_threshold(len(functions))
        )
        self._peak_target: float = (
            peak_softmax_target
            if peak_softmax_target is not None
            else _TARGET_TOP_SOFTMAX_PROB
        )
        self._prompter: BobThePrompter = BobThePrompter(functions)
        self._candidates: list[_FunctionCandidate] = self._build_candidates()

    def _build_candidates(self) -> list[_FunctionCandidate]:
        """Tokenize the character-level suffix after the shared name prefix.

        Slicing *name* token ids by *len(prefix token ids)* is wrong: BPE
        boundaries for ``prefix`` need not align with those for ``full name``.
        """
        candidates: list[_FunctionCandidate] = []
        prefix = self._prompter.function_name_prefix()
        prefix_len = len(prefix)
        for function_definition in self._functions:
            name = function_definition.name
            if prefix and name.startswith(prefix):
                suffix = name[prefix_len:]
            else:
                suffix = name
            if not suffix:
                suffix = name
            distinguishing_ids = encoded_to_token_ids(
                self._model.encode(suffix)
            )
            if not distinguishing_ids:
                raise FunctionSelectorError(
                    "Tokenizer produced no token ids for the selection "
                    f"suffix of {name!r} (prefix {prefix!r})"
                )
            candidates.append(
                _FunctionCandidate(
                    name=function_definition.name,
                    distinguishing_ids=tuple(distinguishing_ids),
                )
            )
        return candidates

    @staticmethod
    def _best_index(probs: list[float]) -> int:
        if not probs:
            raise FunctionSelectorError("No function candidates")
        best_index = np.argmax(probs)
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

    @staticmethod
    def _softmax_at_temperature(
        scores: list[float], temperature: float
    ) -> list[float]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scaled = [s / temperature for s in scores]
        return softmax(scaled)

    def _probs_with_peak_target(self, scores: list[float]) -> list[float]:
        """Cool temperature until the top softmax mass reaches the target."""
        if self._peak_target >= 1.0:
            return softmax(scores)
        probs = self._softmax_at_temperature(scores, _TEMPERATURE_SCHEDULE[0])
        peak = max(probs) if probs else 0.0
        for t in _TEMPERATURE_SCHEDULE[1:]:
            if peak >= self._peak_target:
                break
            probs = self._softmax_at_temperature(scores, t)
            peak = max(probs)
        return probs

    def _candidate_scores(self, base_ids: list[int]) -> list[float]:
        logits = self._model.get_logits_from_input_ids(base_ids)
        scores: list[float] = []
        groups: dict[int, list[int]] = defaultdict(list)
        for candidate in self._candidates:
            first_token = candidate.distinguishing_ids[0]
            groups[first_token].append(len(scores))
            if first_token >= len(logits):
                scores.append(-math.inf)
                continue
            scores.append(float(logits[first_token]))

        for first_token, indices in groups.items():
            if len(indices) <= 1:
                continue
            for index in indices:
                continuation = self._candidates[index].distinguishing_ids[1:]
                if not continuation or math.isinf(scores[index]):
                    continue
                continuation_score = self._continuation_tie_break_score(
                    base_ids + [first_token],
                    continuation,
                )
                if math.isinf(continuation_score):
                    scores[index] = -math.inf
                    continue
                scores[index] += continuation_score
        return scores

    def _continuation_tie_break_score(
        self,
        base_ids: list[int],
        continuation_ids: tuple[int, ...],
    ) -> float:
        if not continuation_ids:
            return 0.0
        weighted_score = 0.0
        weight = 1.0
        history = list(base_ids)
        for token_id in continuation_ids:
            logits = self._model.get_logits_from_input_ids(history)
            probs = softmax(logits)
            if token_id >= len(probs) or probs[token_id] <= 0:
                return -math.inf
            weighted_score += math.log(float(probs[token_id])) * weight
            weight *= 0.1
            history.append(token_id)
        return weighted_score

    def select(self, user_prompt: str) -> str:
        prompt = self._prompter.build_selection_prompt(user_prompt)
        base_ids = encoded_to_token_ids(self._model.encode(prompt))
        scores = self._candidate_scores(base_ids)
        if not scores or all(
            math.isinf(score) and score < 0 for score in scores
        ):
            raise FunctionSelectorError(
                "No valid function candidate from logits"
            )
        model_probs = self._probs_with_peak_target(scores)
        best_index = self._best_index(model_probs)
        best_name = self._candidates[best_index].name
        self._validate_confidence(model_probs, best_index, best_name)
        return best_name
