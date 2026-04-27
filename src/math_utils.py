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
