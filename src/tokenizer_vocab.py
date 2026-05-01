from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.io_utils import load_json_file
from src.model_protocol import LLMModelProtocol


class TokenizerVocabError(ValueError):
    """Raised when tokenizer/vocab files cannot be loaded."""


class TokenizerVocab:
    """Token ID helpers aligned to model tokenizer assets."""

    def __init__(
        self,
        token_to_id: Mapping[str, int],
        model: LLMModelProtocol | None = None,
    ) -> None:
        if not token_to_id:
            raise TokenizerVocabError("Tokenizer vocab is empty")

        self._token_to_id: dict[str, int] = {
            str(token): int(token_id)
            for token, token_id in token_to_id.items()
        }
        self._id_to_token: dict[int, str] = {
            token_id: token for token, token_id in self._token_to_id.items()
        }
        self._model = model

    @classmethod
    def from_model(cls, model: LLMModelProtocol) -> "TokenizerVocab":
        tokenizer_path = Path(model.get_path_to_tokenizer_file())
        try:
            token_map = _read_token_map(
                tokenizer_path,
                from_tokenizer_file=True,
            )
            return cls(token_map, model)
        except ValueError:
            # Some model packages only ship a flat vocab file.
            pass

        vocab_path = Path(model.get_path_to_vocab_file())
        try:
            token_map = _read_token_map(
                vocab_path,
                from_tokenizer_file=False,
            )
        except ValueError as exc:
            raise TokenizerVocabError(
                "Cannot load tokenizer_file or vocab_file for token map"
            ) from exc
        return cls(token_map, model)

    def id_to_text(self, token_id: int) -> str | None:
        raw_token = self._id_to_token.get(token_id)
        if raw_token is None:
            return None
        if self._model is None:
            return raw_token
        return str(self._model.decode([token_id]))

    def id_to_text_map(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for token_id in sorted(self._id_to_token):
            text = self.id_to_text(token_id)
            if text is not None:
                out[token_id] = text
        return out


# TODO: review tommorow
def _read_token_map(
    path: Path,
    *,
    from_tokenizer_file: bool,
) -> dict[str, int]:
    raw = load_json_file(path)
    if not isinstance(raw, dict):
        source = "tokenizer_file" if from_tokenizer_file else "vocab_file"
        raise TokenizerVocabError(f"{source} JSON must be an object")

    if not from_tokenizer_file:
        return _normalize_map(raw)

    model_data = raw.get("model")
    if not isinstance(model_data, dict):
        raise TokenizerVocabError("tokenizer_file missing 'model' object")

    vocab_data = model_data.get("vocab")
    if not isinstance(vocab_data, dict):
        raise TokenizerVocabError("tokenizer_file missing 'model.vocab'")

    return _normalize_map(vocab_data)


def _normalize_map(raw: Mapping[object, object]) -> dict[str, int]:
    out: dict[str, int] = {}
    for token, token_id in raw.items():
        if not isinstance(token, str):
            raise TokenizerVocabError("tokenizer map keys must be strings")
        if not isinstance(token_id, int):
            raise TokenizerVocabError("tokenizer map values must be integers")
        out[token] = token_id
    if not out:
        raise TokenizerVocabError("tokenizer map is empty")
    return out
