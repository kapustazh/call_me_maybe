from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from src.model_protocol import LLMModelProtocol
from src.model_utils import encoded_to_token_ids
from src.math_utils import softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter


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
        model: LLMModelProtocol,
        functions: list[FunctionDefinition],
        *,
        confidence_threshold: float = 0.90,
    ) -> None:
        if not functions:
            raise ValueError("No functions provided for selection")
        self._model: LLMModelProtocol = model
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
        lexical_scores = self._lexical_scores(user_prompt)
        lexical_override = self._lexical_override_index(lexical_scores)
        if lexical_override is not None:
            return self._candidates[lexical_override].name

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
        lexical_probs = softmax(lexical_scores)
        combined_probs = [
            0.4 * model_prob + 0.6 * lexical_prob
            for model_prob, lexical_prob in zip(model_probs, lexical_probs)
        ]
        best_index = self._best_index(combined_probs)
        best_name = self._candidates[best_index].name
        self._validate_confidence(combined_probs, best_index, best_name)
        return best_name

    def _lexical_scores(self, user_prompt: str) -> list[float]:
        prompt_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", user_prompt.lower())
            if token and len(token) > 1
        }
        if not prompt_tokens:
            return [0.0] * len(self._candidates)

        scores: list[float] = []
        for candidate in self._candidates:
            overlap = prompt_tokens.intersection(candidate.keywords)
            score = float(len(overlap))
            scores.append(score)
        return scores

    @staticmethod
    def _lexical_override_index(
        lexical_scores: list[float],
    ) -> int | None:
        if not lexical_scores:
            return None
        best_index = int(np.argmax(lexical_scores))
        best_score = lexical_scores[best_index]
        if best_score <= 0:
            return None
        sorted_scores = sorted(lexical_scores, reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else -math.inf
        # One extra keyword match is enough to override weak model logits.
        if best_score >= second_best + 1.0:
            return int(best_index)
        return None
