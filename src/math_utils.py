from typing import cast

import numpy as np
import numpy.typing as npt


def softmax(x: list[float]) -> list[float]:
    """Convert raw scores to a probability distribution via softmax."""
    if not x:
        return []
    a: npt.NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    if np.isneginf(a).all():
        n: int = len(a)
        return [1.0 / n] * n
    a = a - a.max()
    e = np.exp(a)
    return cast(list[float], (e / e.sum()).tolist())
