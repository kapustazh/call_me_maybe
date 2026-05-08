from __future__ import annotations

import math
from typing import Protocol

from src import prompt_value_extraction as pvex
from src.math_utils import log_softmax
from src.tokenizer_vocab import encoded_to_token_ids


class _ModelLike(Protocol):
    def encode(self, text: str) -> object: ...

    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> list[float]: ...


class RegexValueResolver:
    """Resolve regex-pattern parameter value from prompt text.

    Strategy:
    - If prompt contains an explicit/obvious pattern, return it.
    - Else score a small candidate set using model log-prob and pick best.
    """

    def __init__(self, model: _ModelLike) -> None:
        self._model = model

    def resolve(self, prompt_text: str) -> str:
        pattern = pvex.regex_pattern_from_prompt(prompt_text)
        if pattern is not None:
            return pattern

        candidates = pvex.regex_candidate_patterns()
        if not candidates:
            return r"\d+"

        scoring_text = (
            f'For the request "{prompt_text}", '
            "the correct regex pattern is: "
        )
        scoring_ids = encoded_to_token_ids(self._model.encode(scoring_text))

        best_candidate = candidates[0]
        best_score = -math.inf
        for candidate in candidates:
            token_ids = encoded_to_token_ids(self._model.encode(candidate))
            score = self._score_word(scoring_ids, token_ids)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _score_word(self, base_ids: list[int], word_ids: list[int]) -> float:
        if not word_ids:
            return -math.inf

        history = list(base_ids)
        total = 0.0
        for token_id in word_ids:
            logits = self._model.get_logits_from_input_ids(history)
            log_probs = log_softmax(logits)
            total += (
                float(log_probs[token_id])
                if token_id < len(log_probs)
                else -math.inf
            )
            history.append(token_id)
        return total
