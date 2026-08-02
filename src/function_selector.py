from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from llm_sdk import Small_LLM_Model  # type: ignore
from src.tokenizer_vocab import encoded_to_token_ids
from src.math_utils import log_softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
from src.math_utils import softmax

_UNIFORM_MULTIPLIER = 4.5
_MAX_SELECTION_THRESHOLD = 0.90


class FunctionSelectorError(Exception):
    """Raised when function selection fails or is too low-confidence."""


@dataclass(frozen=True)
class _FunctionCandidate:
    """Tokenized function-name continuation for selection."""

    name: str
    distinguishing_ids: tuple[
        int, ...
    ]  # tokens after selection_completion_prefix


def adaptive_threshold(n_candidates: int) -> float:
    """Minimum softmax probability required for acceptance vs candidate count.

    Caps at ``_MAX_SELECTION_THRESHOLD``. Roughly uniform baseline scaled by
    candidate cardinality.

    Args:
        n_candidates: Number of competing function names.

    Returns:
        Confidence threshold in ``[0, 1]``. Returns ``1.0`` if
        ``n_candidates <= 0`` (effectively impossible pass).
    """
    if n_candidates <= 0:
        return 1.0
    return min(_UNIFORM_MULTIPLIER / n_candidates, _MAX_SELECTION_THRESHOLD)


class FunctionSelector:
    """Route prompts to tools from LLM logits over function-name candidates.

    Builds tokenized continuations after a shared prefix from
    :class:`~src.prompt.BobThePrompter`, then validates confidence on a
    softmax over candidate scores.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
    ) -> None:
        """Create selector for routing prompts to one function.

        Args:
            model: LLM wrapper used to score candidate continuations.
            functions: Available function schemas.

        Raises:
            ValueError: If "functions" is empty.
            FunctionSelectorError: If candidate tokenization is inconsistent.
        """
        if not functions:
            raise ValueError("No functions provided for selection")
        self._model: Small_LLM_Model = model
        self._functions: list[FunctionDefinition] = functions
        self._threshold: float = adaptive_threshold(len(functions))
        self._prompter: BobThePrompter = BobThePrompter(functions)
        self._candidates: list[_FunctionCandidate] = self._build_candidates()

    def _build_candidates(self) -> list[_FunctionCandidate]:
        """Tokenize each function name after the shared selection prefix.

        Returns:
            Parallel list to ``self._functions`` with distinguishing token ids.

        Raises:
            FunctionSelectorError: If names cannot align with prefix tokens.
        """
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
                prefix_len = len(prefix_ids)
                distinguishing_ids = name_ids[prefix_len:]
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

    def _validate_confidence(
        self,
        probs: list[float],
        best_index: int,
        best_name: str,
    ) -> None:
        """Validate selection probability against threshold.

        Raises:
            FunctionSelectorError: If confidence is below threshold.
        """
        confidence = float(probs[best_index])
        if confidence < self._threshold:
            raise FunctionSelectorError(
                f"Low selection confidence: {confidence:.3f} < "
                f"{self._threshold:.3f} for '{best_name}'"
            )

    def _candidate_scores(self, base_ids: list[int]) -> list[float]:
        """Compute per-candidate selection scores from model logits."""
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
        """Score continuation tokens to break first-token ties."""
        if not continuation_ids:
            return 0.0
        weighted_score = 0.0
        weight = 1.0
        history = list(base_ids)
        for token_id in continuation_ids:
            logits = self._model.get_logits_from_input_ids(history)
            log_probs = log_softmax(logits)
            if token_id >= len(log_probs) or math.isinf(log_probs[token_id]):
                return -math.inf
            weighted_score += float(log_probs[token_id]) * weight
            weight *= 0.1
            history.append(token_id)
        return weighted_score

    def select(self, user_prompt: str) -> str:
        """Select best matching function name for a user prompt.

        Args:
            user_prompt: Raw natural-language request.

        Returns:
            Selected function name.

        Raises:
            FunctionSelectorError: If no valid candidate or confidence too low.
        """
        prompt = self._prompter.build_selection_prompt(user_prompt)
        base_ids = encoded_to_token_ids(self._model.encode(prompt))
        scores = self._candidate_scores(base_ids)
        if not scores or all(
            math.isinf(score) and score < 0 for score in scores
        ):
            raise FunctionSelectorError(
                "No valid function candidate from logits"
            )

        probs = softmax(scores)
        best_index = max(range(len(probs)), key=probs.__getitem__)
        best_name = self._candidates[best_index].name
        self._validate_confidence(probs, best_index, best_name)
        return best_name
