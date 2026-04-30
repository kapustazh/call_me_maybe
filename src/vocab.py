from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


class Vocab:
    def __init__(
        self,
        path_to_vocab: str | Path,
        model: Any | None = None,
    ) -> None:
        p = Path(path_to_vocab)
        raw_vocab = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw_vocab, dict):
            raise ValueError("Vocab file must contain a JSON object")
        self.token_to_id = cast(dict[str, int], raw_vocab)

        self._id_to_token: dict[int, str] = {
            token_id: token for token, token_id in self.token_to_id.items()
        }
        self._model = model

    def get_token_by_id(self, token_id: int) -> str | None:
        """Return raw vocab token, not necessarily decoded text."""
        return self._id_to_token.get(token_id)

    def get_id_by_token(self, token: str) -> int | None:
        return self.token_to_id.get(token)

    def contains_id(self, token_id: int) -> bool:
        return token_id in self._id_to_token

    def id_to_text(self, token_id: int) -> str | None:
        """Return decoded token text for masking.

        Unknown IDs return None. Known special tokens may decode to "".
        If no model was provided, raw vocab token is returned.
        """
        raw_token = self._id_to_token.get(token_id)
        if raw_token is None:
            return None
        if self._model is None:
            return raw_token
        return str(self._model.decode([token_id]))

    def id_to_text_map(self) -> dict[int, str]:
        """Return decoded text for every known vocab ID."""
        out: dict[int, str] = {}
        for token_id in sorted(self._id_to_token):
            text = self.id_to_text(token_id)
            if text is not None:
                out[token_id] = text
        return out

    def token_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._id_to_token))

    # def print_vocab(self, limit: int = 50) -> None:
    #     items = sorted(self.token_to_id.items(), key=lambda kv: kv[1])
    #     print(f"Vocab size: {len(items)}")
    #     for token, token_id in items[: max(0, limit)]:
    #         print(f"{token_id}\t{token!r}")
