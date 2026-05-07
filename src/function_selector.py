from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from src.model_utils import encoded_to_token_ids
from src.math_utils import softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter

# For N=5 candidates, 4.5/N == 0.90 (matches docstring of adaptive_threshold).
_UNIFORM_MULTIPLIER = 4.5
_TARGET_TOP_SOFTMAX_PROB = 0.9
_TEMPERATURE_SCHEDULE = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
_LEXICAL_BONUS_WEIGHT = 5.0
_NO_LEXICAL_SUPPORT_MIN_CONFIDENCE = 0.90
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "has",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "what",
    "with",
    "you",
}
_TERM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "add": ("sum", "plus"),
    "sum": ("add", "plus"),
    "plus": ("add", "sum"),
    "multiply": ("product", "times"),
    "product": ("multiply", "times"),
    "times": ("multiply", "product"),
    "greeting": ("greet",),
    "hello": ("greet",),
    "hi": ("greet",),
}


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
        """Tokenize token-level continuation after selection prompt prefix."""
        candidates: list[_FunctionCandidate] = []
        prefix = self._prompter.function_name_prefix()
        prefix_ids = (
            encoded_to_token_ids(self._model.encode(prefix)) if prefix else []
        )
        for function_definition in self._functions:
            name = function_definition.name
            name_ids = encoded_to_token_ids(self._model.encode(name))
            if prefix_ids:
                if len(name_ids) <= len(prefix_ids):
                    raise FunctionSelectorError(
                        f"Function name {name!r} has no continuation "
                        f"after prefix {prefix!r}"
                    )
                if name_ids[: len(prefix_ids)] != prefix_ids:
                    raise FunctionSelectorError(
                        "Selection prefix tokenization mismatch for "
                        f"{name!r}: prefix {prefix!r} tokens do not match "
                        "name-token prefix"
                    )
                distinguishing_ids = name_ids[len(prefix_ids) :]
            else:
                distinguishing_ids = name_ids
            if not distinguishing_ids:
                raise FunctionSelectorError(
                    "Tokenizer produced no token ids for the selection "
                    f"continuation of {name!r} (prefix {prefix!r})"
                )
            if not name.startswith(prefix):
                raise FunctionSelectorError(
                    f"Function name {name!r} does not start with "
                    f"selection prefix {prefix!r}"
                )
            candidates.append(
                _FunctionCandidate(
                    name=function_definition.name,
                    distinguishing_ids=tuple(distinguishing_ids),
                )
            )
        return candidates

    @staticmethod
    def _normalize_terms(text: str) -> set[str]:
        terms = set(_WORD_RE.findall(text.lower().replace("_", " ")))
        filtered = {
            term for term in terms if term not in _STOPWORDS and term != "fn"
        }
        expanded = set(filtered)
        for term in filtered:
            expanded.update(_TERM_SYNONYMS.get(term, ()))
        return expanded

    def _lexical_overlap_count(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
    ) -> int:
        prompt_terms = self._normalize_terms(user_prompt)
        if not prompt_terms:
            return 0

        function_text = (
            f"{function_definition.name} {function_definition.description}"
        )
        function_terms = self._normalize_terms(function_text)
        overlap = prompt_terms & function_terms
        return len(overlap)

    def _lexical_bonus(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
    ) -> float:
        overlap_count = self._lexical_overlap_count(
            user_prompt, function_definition
        )
        if overlap_count <= 0:
            return 0.0

        return float(overlap_count) * _LEXICAL_BONUS_WEIGHT

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
        best_overlap_count: int,
    ) -> None:
        confidence = float(probs[best_index])
        if confidence < self._threshold:
            raise FunctionSelectorError(
                f"Low selection confidence: {confidence:.3f} < "
                f"{self._threshold:.3f} for '{best_name}'"
            )
        if (
            best_overlap_count <= 0
            and confidence < _NO_LEXICAL_SUPPORT_MIN_CONFIDENCE
        ):
            raise FunctionSelectorError(
                "Selection has no lexical support: "
                f"{confidence:.3f} < "
                f"{_NO_LEXICAL_SUPPORT_MIN_CONFIDENCE:.3f} for '{best_name}'"
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

    def _candidate_scores(
        self,
        base_ids: list[int],
        user_prompt: str,
    ) -> list[float]:
        logits = self._model.get_logits_from_input_ids(base_ids)
        scores: list[float] = []
        groups: dict[int, list[int]] = defaultdict(list)
        for candidate, function_definition in zip(
            self._candidates,
            self._functions,
        ):
            first_token = candidate.distinguishing_ids[0]
            groups[first_token].append(len(scores))
            if first_token >= len(logits):
                scores.append(-math.inf)
                continue
            scores.append(
                float(logits[first_token])
                + self._lexical_bonus(user_prompt, function_definition)
            )

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
        scores = self._candidate_scores(base_ids, user_prompt)
        if not scores or all(
            math.isinf(score) and score < 0 for score in scores
        ):
            raise FunctionSelectorError(
                "No valid function candidate from logits"
            )

        # Use true softmax probabilities for confidence gating.
        # Temperature cooling (peak target) is only a selection heuristic.
        confidence_probs = softmax(scores)

        selection_probs = self._probs_with_peak_target(scores)
        best_index = self._best_index(selection_probs)
        best_name = self._candidates[best_index].name
        best_overlap_count = self._lexical_overlap_count(
            user_prompt, self._functions[best_index]
        )
        self._validate_confidence(
            confidence_probs,
            best_index,
            best_name,
            best_overlap_count,
        )
        return best_name
