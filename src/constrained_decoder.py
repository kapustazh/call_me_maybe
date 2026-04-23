from __future__ import annotations
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from src.vocab import Vocab


class ConstrainedDecoder:
    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: Vocab,
        functions: list[FunctionDefinition],
        max_new_tokens: int = 300,
    ) -> None:
        self._model = model
        self._vocab = vocab
        self._functions = {fn.name for fn in functions}
        self._quote_id = self._model.encode('"')[0]

    def get_valid_number_ids(self) -> set[int]:
        allowed = set("0123456789-+.eE")  # +eE
        vocab_size = len(self._vocab.token_to_id)
        out = set()
        for token_id in range(vocab_size):
            s = self._model.decode([token_id])
            if s and all(ch in allowed for ch in s):
                out.add(token_id)
        return out
