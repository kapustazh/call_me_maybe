from __future__ import annotations
from re import Pattern


import math
import re
from collections import defaultdict
from dataclasses import dataclass


from llm_sdk import Small_LLM_Model  # type: ignore
from src.tokenizer_vocab import encoded_to_token_ids
from src.math_utils import log_softmax
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
from src.math_utils import softmax

_UNIFORM_MULTIPLIER = 4.5
_TARGET_TOP_SOFTMAX_PROB = 0.90
_TEMPERATURE_SCHEDULE = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
_LEXICAL_BONUS_WEIGHT = 5.0
_NO_LEXICAL_SUPPORT_MIN_CONFIDENCE = 0.90
_MAX_SELECTION_THRESHOLD = 0.90
_WORD_RE: Pattern[str] = re.compile(r"[a-z0-9]+")
_STOPWORDS: set[str] = {
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
    """Score-first-token logits plus lexical bonus to route prompts to tools.

    Builds tokenized continuations after a shared prefix from
    :class:`~src.prompt.BobThePrompter`, adds overlap-based tie-breakers,
    softmax-temps until peak mass target, then validates confidence.
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
        self._peak_target: float = _TARGET_TOP_SOFTMAX_PROB
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

    @staticmethod
    def _normalize_terms(text: str) -> set[str]:
        """Extract lowercase alphanumeric tokens from text for overlap checks.

        Args:
            text: Arbitrary natural-language or identifier string.

        Returns:
            Set of tokens with underscores treated as spaces and stopwords
            removed (including literal ``fn`` token).
        """
        terms = set(_WORD_RE.findall(text.lower().replace("_", " ")))
        filtered = {
            term for term in terms if term not in _STOPWORDS and term != "fn"
        }
        return set(filtered)

    def _lexical_overlap_count(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
    ) -> int:
        """Count normalized terms shared between prompt and function metadata.

        Args:
            user_prompt: Raw user request.
            function_definition: Candidate tool schema.

        Returns:
            Intersection size (non-negative integer).
        """
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
        """Additive score bump proportional to lexical overlap count.

        Args:
            user_prompt: Raw user request.
            function_definition: Candidate tool schema.

        Returns:
            Weighted bonus (zero when no overlap).
        """
        overlap_count = self._lexical_overlap_count(
            user_prompt, function_definition
        )
        if overlap_count <= 0:
            return 0.0

        return float(overlap_count) * _LEXICAL_BONUS_WEIGHT

    def _validate_confidence(
        self,
        probs: list[float],
        best_index: int,
        best_name: str,
        best_overlap_count: int,
    ) -> None:
        """Validate selection probability against thresholds.

        Args:
            probs: Probability distribution over candidates.
            best_index: Index of chosen candidate.
            best_name: Chosen function name (for error messages).
            best_overlap_count: Lexical overlap count for chosen function.

        Raises:
            FunctionSelectorError: If confidence is below threshold.
        """
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
        """Apply softmax to scores scaled by ``temperature``.

        Args:
            scores: Unnormalized logits or energies.
            temperature:
                Positive temperature; lower yields sharper distribution.

        Returns:
            Probability vector of same length as ``scores``.

        Raises:
            ValueError: If ``temperature`` is not positive.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scaled = [s / temperature for s in scores]
        return softmax(scaled)

    def _probs_with_peak_target(self, scores: list[float]) -> list[float]:
        """Softmax at decreasing temperatures until top mass reaches target.

        Args:
            scores: Candidate logits before normalization.

        Returns:
            Final probability distribution used for winner selection.
        """
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
        """Compute per-candidate selection scores from model logits.

        Args:
            base_ids: Token ids for selection prompt prefix.
            user_prompt: Raw user request (for lexical bonus).

        Returns:
            List of scores aligned to "self._candidates".
        """
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
        """Score continuation tokens to break first-token ties.

        Args:
            base_ids: Prompt ids including the tied first token.
            continuation_ids: Remaining token ids for candidate name.

        Returns:
            Weighted log-prob score for continuation (or -inf if impossible).
        """
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
        scores = self._candidate_scores(base_ids, user_prompt)
        if not scores or all(
            math.isinf(score) and score < 0 for score in scores
        ):
            raise FunctionSelectorError(
                "No valid function candidate from logits"
            )

        probs = softmax(scores)
        best_index = max(range(len(probs)), key=probs.__getitem__)
        best_name = self._candidates[best_index].name
        best_overlap_count = self._lexical_overlap_count(
            user_prompt, self._functions[best_index]
        )

        self._validate_confidence(
            probs,
            best_index,
            best_name,
            best_overlap_count,
        )
        return best_name
