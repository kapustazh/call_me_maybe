from __future__ import annotations

from typing import Any


def encoded_to_token_ids(encoded: object) -> list[int]:
    """Normalize model.encode output to list[int] for one prompt."""
    raw: Any
    tolist = getattr(encoded, "tolist", None)
    if callable(tolist):
        raw = tolist()
    else:
        raw = encoded

    if isinstance(raw, tuple):
        raw = list(raw)

    if not isinstance(raw, list):
        raise TypeError("model.encode() returned unsupported type")

    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise TypeError("model.encode() must return one batch row")
        nested = raw[0]
        if not isinstance(nested, list):
            raise TypeError("model.encode() nested output is not a list")
        return [int(value) for value in nested]

    return [int(value) for value in raw]
