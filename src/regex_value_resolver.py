from __future__ import annotations

import math

from llm_sdk import Small_LLM_Model  # type: ignore

from src import prompt_value_extraction as pvex
from src.math_utils import cumulative_sequence_logprob
from src.tokenizer_vocab import encoded_to_token_ids

_DEFAULT_NUMBER_REGEX: str = r"\d+"


class RegexValueResolver:
    """Resolve regex-pattern parameter value from prompt text.

    Strategy:
    - If prompt contains an explicit/obvious pattern, return it.
    - Else score a small candidate set using model log-prob and pick best.
    - If that list is empty, use digit-only pattern (numeric literals).
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """Store model reference for scoring.

        Args:
            model: Encoder + logits provider used to rank regex candidates.
        """
        self._model: Small_LLM_Model = model

    def resolve(self, prompt_text: str) -> str:
        """Pick regex pattern string for the prompt.

        Uses explicit pattern extraction when possible; otherwise scores a
        fixed candidate list with token log-probabilities.

        Args:
            prompt_text: Full user natural-language request.

        Returns:
            Regex pattern string (non-empty). If heuristic extraction fails and
            there are no LM-scorable candidates, returns digit-sequence pattern
            (numeric literals only).
        """
        pattern = pvex.regex_pattern_from_prompt(prompt_text)
        if pattern is not None:
            return pattern

        candidates = pvex.regex_candidate_patterns()
        if not candidates:
            return _DEFAULT_NUMBER_REGEX

        scoring_text = (
            f'For the request "{prompt_text}", '
            "the correct regex pattern is: "
        )
        scoring_ids = encoded_to_token_ids(self._model.encode(scoring_text))

        best_candidate = candidates[0]
        best_score = -math.inf
        gl = self._model.get_logits_from_input_ids
        for candidate in candidates:
            token_ids = encoded_to_token_ids(self._model.encode(candidate))
            score = cumulative_sequence_logprob(gl, scoring_ids, token_ids)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate
