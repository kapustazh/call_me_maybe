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
