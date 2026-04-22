from __future__ import annotations
from src.io_utils import load_json_file


class Vocab:
    def __init__(self, path_to_vocab: str) -> None:
        self.token_to_id: dict[str, int] = load_json_file(path_to_vocab)
        self._id_to_token: dict[int, str] = {
            v: k for k, v in self._token_to_id.items()
        }

    def get_token_by_id(self, id: int) -> str | None:
        return self._id_to_token.get(id)

    def get_id_by_token(self, str_: str) -> str | None:
        return self._token_to_id.get(str_)
