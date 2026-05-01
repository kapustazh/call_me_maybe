from numpy._typing._array_like import NDArray


from numpy import float64


import numpy as np
from typing import cast


def softmax(x: list[float]) -> list[float]:
    """Normalisation function for the vector"""
    if not x:
        return []
    a: NDArray[float64] = np.asarray(x, dtype=np.float64)
    if np.isneginf(a).all():
        n: int = len(a)
        return [1.0 / n] * n  # all equal if all are -inf
    a = a - a.max()  # handles overflow for big exponent values
    # & normalisation | max values after extraction becomes 0
    e = np.exp(a)  # takes exp for each value, makes them positive and
    # nonlinear amplifes distance between small and big values
    z = e.sum()
    # each single exp is being divided to the sum of the exp of the vector
    return cast(list[float], (e / z).tolist())
