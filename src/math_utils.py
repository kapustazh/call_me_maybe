from typing import cast
import numpy as np


def log_softmax(x: list[float]) -> list[float]:
    """Convert raw scores to log-prob distribution (log-softmax)."""
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
    """Convert log-prob distribution to exp probability."""
    log_probs = log_softmax(x)
    if not log_probs:
        return []
    probs_arr = np.exp(np.asarray(log_probs, dtype=np.float64))
    return cast(list[float], probs_arr.tolist())
