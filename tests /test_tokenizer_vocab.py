import json
from pathlib import Path

from src.tokenizer_vocab import TokenizerVocab


class FakeTokenizerModel:
    def __init__(self, tokenizer_path: Path, vocab_path: Path) -> None:
        self._tokenizer_path = tokenizer_path
        self._vocab_path = vocab_path

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        _ = input_ids
        return [0.0] * 256

    def get_path_to_tokenizer_file(self) -> str:
        return str(self._tokenizer_path)

    def get_path_to_vocab_file(self) -> str:
        return str(self._vocab_path)


class FakeDecodedModel:
    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if not isinstance(ids, list):
            raise TypeError("Expected list[int]")
        pieces = {
            0: "",
            1: '"',
            2: "3",
            3: "true",
            4: "<unk>",
        }
        return "".join(pieces[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        _ = input_ids
        return [0.0] * 256

    def get_path_to_tokenizer_file(self) -> str:
        return "unused"

    def get_path_to_vocab_file(self) -> str:
        return "unused"


def test_loads_token_map_from_tokenizer_file(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_path = tmp_path / "vocab.json"
    tokenizer_path.write_text(
        json.dumps({"model": {"vocab": {"A": 65, "B": 66}}}),
        encoding="utf-8",
    )
    vocab_path.write_text(
        json.dumps({"X": 88}),
        encoding="utf-8",
    )

    model = FakeTokenizerModel(tokenizer_path, vocab_path)
    vocab = TokenizerVocab.from_model(model)

    assert vocab.id_to_text(65) == "A"
    assert vocab.id_to_text(66) == "B"


def test_falls_back_to_vocab_file(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_path = tmp_path / "vocab.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    vocab_path.write_text(
        json.dumps({"C": 67, "D": 68}),
        encoding="utf-8",
    )

    model = FakeTokenizerModel(tokenizer_path, vocab_path)
    vocab = TokenizerVocab.from_model(model)

    assert vocab.id_to_text(67) == "C"
    assert vocab.id_to_text(68) == "D"


def test_raw_and_decoded_id_maps() -> None:
    token_map = {
        "<bos>": 0,
        '"': 1,
        "3": 2,
        "true": 3,
        "<unk>": 4,
    }
    raw_vocab = TokenizerVocab(token_map)
    decoded_vocab = TokenizerVocab(token_map, FakeDecodedModel())

    assert raw_vocab.id_to_text(1) == '"'
    assert decoded_vocab.id_to_text(1) == '"'
    assert decoded_vocab.id_to_text(2) == "3"
    assert decoded_vocab.id_to_text(3) == "true"
    assert decoded_vocab.id_to_text(0) == ""
    assert decoded_vocab.id_to_text(999) is None
    assert decoded_vocab.id_to_text_map() == {
        0: "",
        1: '"',
        2: "3",
        3: "true",
        4: "<unk>",
    }
