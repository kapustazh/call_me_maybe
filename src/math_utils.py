from collections.abc import Sequence

import numpy as np


def softmax(x: Sequence[float]) -> list[float]:
    """Normalisation function for the vector"""
    if not x:
        return []
    a = np.asarray(x, dtype=np.float64)
    a = a - a.max()  # handles overflow for big exponent values
    # & normalisation | max values after extraction becomes 0
    e = np.exp(a)  # takes exp for each value, makes them positive and
    # nonlinear amplifes distance between small and big values
    z = e.sum()
    # each single exp is being divided to the sum of the exp of the vector
    return (e / z).tolist()


def log_softmax(x: Sequence[float]) -> list[float]:
    """Per-dimension log(exp(x_i) / sum_j exp(x_j)); stable for masking / scoring."""
    if not x:
        return []
    a = np.asarray(x, dtype=np.float64)
    m = float(a.max())
    log_z = m + np.log(np.sum(np.exp(a - m)))
    return (a - log_z).tolist()
