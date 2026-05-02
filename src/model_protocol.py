from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMModelProtocolAdapter(Protocol):
    def encode(self, text: str) -> object: ...

    def decode(self, ids: object) -> str: ...

    def get_logits_from_input_ids(
        self, input_ids: list[int]
    ) -> list[float]: ...

    def get_path_to_tokenizer_file(self) -> str: ...

    def get_path_to_vocab_file(self) -> str: ...
