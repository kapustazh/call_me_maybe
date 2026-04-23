import numpy as np
from collections.abc import Sequence


class MathUtils:
    @staticmethod
    def softmax(x: Sequence[float]) -> list[float]:
        if not x:
            return []
        a = np.asarray(x, dtype=np.float64)
        a = a - a.max()
        e = np.exp(a)
        z = e.sum()
        return (e / z).tolist() if z != 0.0 else [0.0] * len(a)
