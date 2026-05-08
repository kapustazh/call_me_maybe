import json
from pathlib import Path
from typing import Any, cast

from src.constrained_decoder import ConstrainedDecoder
from src.models import FunctionDefinition
from src.tokenizer_vocab import TokenizerVocab


class _AsciiModel:
    """Minimal model stub for decoder tests.

    Object parameters short-circuit to '{}' in current implementation, so
    logits are never consulted. We still provide encode/decode for prompt
    building and tokenizer plumbing.
    """

    def __init__(self, tokenizer_path: Path, vocab_path: Path) -> None:
        self._tokenizer_path = tokenizer_path
        self._vocab_path = vocab_path

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, _input_ids: list[int]) -> list[float]:
        raise AssertionError("logits should not be needed for object params")

    def get_path_to_tokenizer_file(self) -> str:
        return str(self._tokenizer_path)

    def get_path_to_vocab_file(self) -> str:
        return str(self._vocab_path)


def _write_ascii_tokenizer_files(tmp_path: Path) -> tuple[Path, Path]:
    token_map = {chr(code): code for code in range(32, 127)}
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_path = tmp_path / "vocab.json"
    tokenizer_path.write_text(
        json.dumps({"model": {"vocab": token_map}}),
        encoding="utf-8",
    )
    vocab_path.write_text(json.dumps(token_map), encoding="utf-8")
    return tokenizer_path, vocab_path


def test_object_param_decodes_to_empty_object(tmp_path: Path) -> None:
    tokenizer_path, vocab_path = _write_ascii_tokenizer_files(tmp_path)
    model = _AsciiModel(tokenizer_path, vocab_path)

    tokenizer_vocab = TokenizerVocab.from_model(cast(Any, model))
    definitions = [
        FunctionDefinition.model_validate(
            {
                "name": "fn_create_user",
                "description": "Create user with nested payload",
                "parameters": {"payload": {"type": "object"}},
                "returns": {"type": "string"},
            }
        )
    ]
    decoder = ConstrainedDecoder(
        model,  # type: ignore[arg-type]
        tokenizer_vocab,
        definitions,
    )

    params = decoder.decode_parameters(
        'Create user payload {"name":"alice","age":30}',
        definitions[0],
    )
    assert params == {"payload": {}}
