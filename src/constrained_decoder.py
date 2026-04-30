from __future__ import annotations
from typing import Any

from src.models import FunctionDefinition
from src.vocab import Vocab


class ConstrainedDecoder:
    def __init__(
        self,
        model: Any,
        vocab: Vocab,
        functions: list[FunctionDefinition],
        max_new_tokens: int = 300,
    ) -> None:
        self._model = model
        self._vocab = vocab
        self._functions = {fn.name for fn in functions}
        self._max_new_tokens = max_new_tokens
        quote_ids = self._model.encode('"')[0].tolist()
        self._quote_id = int(quote_ids[0]) if quote_ids else None

    def get_valid_number_ids(self) -> set[int]:
        allowed = set("0123456789-+.eE")  # +eE
        out = set()
        for token_id in self._vocab.token_ids():
            s = self._vocab.id_to_text(token_id)
            if s and all(ch in allowed for ch in s):
                out.add(token_id)
        return out
