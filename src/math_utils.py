import math
from collections.abc import Callable
from typing import cast

import numpy as np


def log_softmax(x: list[float]) -> list[float]:
    """Compute log-softmax over 1-D scores.

    Args:
        x: Unnormalized scores (logits).

    Returns:
        Log-probabilities with same length as "x".
    """
    if not x:
        return []
    a = np.asarray(x, dtype=np.float64)

    if np.isneginf(a).all():
        n = len(a)
        return cast(list[float], (np.full(n, -np.log(n))).tolist())

    a = a - a.max()
    lse = np.log(np.exp(a).sum())
    return cast(list[float], (a - lse).tolist())


def cumulative_sequence_logprob(
    get_logits: Callable[[list[int]], list[float]],
    base_ids: list[int],
    continuation_ids: list[int],
) -> float:
    """Sum log-probabilities of ``continuation_ids`` greedy-autoregressively.

    At each step, conditions on ``base_ids`` plus all continuation tokens
    generated so far.

    Args:
        get_logits: Returns full-vocabulary logits for current prefix ids.
        base_ids: Token ids already fed before the continuation.
        continuation_ids: Target continuation token ids.

    Returns:
        Total log-probability, or ``-inf`` if ``continuation_ids`` is empty or
        any step references an out-of-range vocab index.
    """
    if not continuation_ids:
        return -math.inf
    history = list(base_ids)
    total = 0.0
    for token_id in continuation_ids:
        logits = get_logits(history)
        log_probs = log_softmax(logits)
        total += (
            float(log_probs[token_id])
            if token_id < len(log_probs)
            else -math.inf
        )
        history.append(token_id)
    return total


def softmax(x: list[float]) -> list[float]:
    """Compute softmax probabilities over 1-D scores.

    Args:
        x: Unnormalized scores (logits).

    Returns:
        Probabilities summing to 1.0 (empty list if "x" empty).
    """
    log_probs = log_softmax(x)
    if not log_probs:
        return []
    probs_arr = np.exp(np.asarray(log_probs, dtype=np.float64))
    return cast(list[float], probs_arr.tolist())
