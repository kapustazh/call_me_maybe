from __future__ import annotations

import math
from typing import Protocol

from src import prompt_value_extraction as pvex
from src.math_utils import log_softmax
from src.tokenizer_vocab import encoded_to_token_ids


class _ModelLike(Protocol):
    """Minimal LLM interface for regex scoring (encode + logits).

    Implementations must match :class:`llm_sdk.Small_LLM_Model` usage in this
    package (duck typing via Protocol).
    """

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
        """Store model reference for scoring.

        Args:
            model: Encoder + logits provider used to rank regex candidates.
        """
        self._model = model

    def resolve(self, prompt_text: str) -> str:
        """Pick regex pattern string for the prompt.

        Uses explicit pattern extraction when possible; otherwise scores a
        fixed candidate list with token log-probabilities.

        Args:
            prompt_text: Full user natural-language request.

        Returns:
            Regex pattern string (non-empty; falls back to digits-class pattern
            when candidate list empty).
        """
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
        """Sum log-probabilities of generating ``word_ids`` after ``base_ids``.

        Args:
            base_ids: Token ids of prompt prefix already fed to the model.
            word_ids: Target continuation token ids (e.g. one regex literal).

        Returns:
            Total log-probability along the continuation. Negative infinity
            if ``word_ids`` is empty or any step is impossible.
        """
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
