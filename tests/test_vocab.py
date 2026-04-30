import json
from pathlib import Path

import pytest

from src.vocab import Vocab


class FakeModel:
    def decode(self, ids: list[int]) -> str:
        pieces = {
            0: "",
            1: '"',
            2: "3",
            3: "true",
            4: "<unk>",
        }
        return "".join(pieces[token_id] for token_id in ids)


def _write_vocab(directory: Path) -> Path:
    path = directory / "vocab.json"
    path.write_text(
        json.dumps(
            {
                "<bos>": 0,
                '"': 1,
                "3": 2,
                "true": 3,
                "<unk>": 4,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_raw_and_decoded_id_maps(tmp_path: Path) -> None:
    vocab_path = _write_vocab(tmp_path)
    vocab = Vocab(vocab_path, FakeModel())

    assert vocab.get_token_by_id(1) == '"'
    assert vocab.get_id_by_token("3") == 2
    assert vocab.id_to_text(1) == '"'
    assert vocab.id_to_text(2) == "3"
    assert vocab.id_to_text(3) == "true"
    assert vocab.id_to_text(0) == ""
    assert vocab.id_to_text(999) is None
    assert vocab.id_to_text_map() == {
        0: "",
        1: '"',
        2: "3",
        3: "true",
        4: "<unk>",
    }


def test_rejects_bad_vocab_shape(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps(["not", "object"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        Vocab(vocab_path)
