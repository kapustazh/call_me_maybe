from __future__ import annotations

from typing import Any


# TODO: i think is overkill but keep for now
def encoded_to_token_ids(encoded: object) -> list[int]:
    """Normalize model.encode output to list[int]."""
    raw: Any
    if hasattr(encoded, "tolist"):
        raw = getattr(encoded, "tolist")()
    else:
        raw = encoded

    if isinstance(raw, tuple):
        raw = list(raw)

    if not isinstance(raw, list):
        raise TypeError("model.encode() returned unsupported type")

    if raw and isinstance(raw[0], list):
        nested = raw[0]
        if not isinstance(nested, list):
            raise TypeError("model.encode() nested output is not a list")
        return [int(value) for value in nested]

    return [int(value) for value in raw]
