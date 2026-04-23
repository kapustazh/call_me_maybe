from __future__ import annotations

from src.io_utils import load_json_file


class Vocab:
    def __init__(self, path_to_vocab: str) -> None:
        self.token_to_id: dict[str, int] = load_json_file(path_to_vocab)
        self._id_to_token: dict[int, str] = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    def get_token_by_id(self, id: int) -> str | None:
        return self._id_to_token.get(id)

    def get_id_by_token(self, token: str) -> int | None:
        return self.token_to_id.get(token)

    def print_vocab(self, limit: int = 50) -> None:
        items = sorted(self.token_to_id.items(), key=lambda kv: kv[1])
        print(f"Vocab size: {len(items)}")
        for token, token_id in items[: max(0, limit)]:
            print(f"{token_id}\t{token!r}")
