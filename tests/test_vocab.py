import json
import tempfile
import unittest
from pathlib import Path

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


class VocabSmokeTest(unittest.TestCase):
    def _write_vocab(self, directory: Path) -> Path:
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

    def test_raw_and_decoded_id_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = self._write_vocab(Path(tmp))
            vocab = Vocab(vocab_path, FakeModel())

            self.assertEqual(vocab.get_token_by_id(1), '"')
            self.assertEqual(vocab.get_id_by_token("3"), 2)
            self.assertEqual(vocab.id_to_text(1), '"')
            self.assertEqual(vocab.id_to_text(2), "3")
            self.assertEqual(vocab.id_to_text(3), "true")
            self.assertEqual(vocab.id_to_text(0), "")
            self.assertIsNone(vocab.id_to_text(999))
            self.assertEqual(
                vocab.id_to_text_map(),
                {
                    0: "",
                    1: '"',
                    2: "3",
                    3: "true",
                    4: "<unk>",
                },
            )

    def test_rejects_bad_vocab_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.json"
            vocab_path.write_text(
                json.dumps(["not", "object"]),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                Vocab(vocab_path)


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.vocab import Vocab


class FakeModel:
    def __init__(self, vocab_path: Path) -> None:
        self._vocab_path = vocab_path

    def decode(self, ids: list[int]) -> str:
        pieces = {
            0: "",
            1: '"',
            2: "3",
            3: "true",
            4: "<unk>",
        }
        return "".join(pieces[token_id] for token_id in ids)


class VocabSmokeTest(unittest.TestCase):
    def _write_vocab(self, directory: Path) -> Path:
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

    def test_raw_and_decoded_id_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = self._write_vocab(Path(tmp))
            model = FakeModel(vocab_path)
            vocab = Vocab(vocab_path, model)

            self.assertEqual(vocab.get_token_by_id(1), '"')
            self.assertEqual(vocab.get_id_by_token("3"), 2)
            self.assertEqual(vocab.id_to_text(1), '"')
            self.assertEqual(vocab.id_to_text(2), "3")
            self.assertEqual(vocab.id_to_text(3), "true")
            self.assertEqual(vocab.id_to_text(0), "")
            self.assertIsNone(vocab.id_to_text(999))
            self.assertEqual(
                vocab.id_to_text_map(),
                {
                    0: "",
                    1: '"',
                    2: "3",
                    3: "true",
                    4: "<unk>",
                },
            )

    def test_rejects_bad_vocab_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path = Path(tmp) / "vocab.json"
            vocab_path.write_text(
                json.dumps(["not", "object"]),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                Vocab(vocab_path)


if __name__ == "__main__":
    unittest.main()
