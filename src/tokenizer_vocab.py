from __future__ import annotations

from typing import Mapping

from src.io_utils import load_json_file
from src.model_protocol import LLMModelProtocolAdapter


class TokenizerVocabError(ValueError):
    """Raised when tokenizer/vocab files cannot be loaded."""


class TokenizerVocab:
    """Token ID helpers aligned to model tokenizer assets."""

    def __init__(
        self,
        token_to_id: Mapping[str, int],
        model: LLMModelProtocolAdapter | None = None,
    ) -> None:
        if not token_to_id:
            raise TokenizerVocabError("Tokenizer vocab is empty")

        self._id_to_token: dict[int, str] = {
            int(token_id): str(token) for token, token_id in token_to_id.items()
        }
        self._model: LLMModelProtocolAdapter | None = model

    @classmethod
    def from_model(cls, model: LLMModelProtocolAdapter) -> "TokenizerVocab":
        tokenizer_error: ValueError | None = None
        try:
            token_map: dict[str, int] = _read_token_map(
                path=model.get_path_to_tokenizer_file(),
                from_tokenizer_file=True,
            )
            return cls(token_map, model)
        except ValueError as exc:
            # Some model packages only ship a flat vocab file.
            tokenizer_error = exc

        try:
            token_map = _read_token_map(
                path=model.get_path_to_vocab_file(),
                from_tokenizer_file=False,
            )
        except ValueError as exc:
            raise TokenizerVocabError(
                "Cannot load tokenizer_file or vocab_file for token map"
            ) from (tokenizer_error or exc)
        return cls(token_map, model)

    def id_to_text(self, token_id: int) -> str | None:
        raw_token: str | None = self._id_to_token.get(token_id)
        if raw_token is None:
            return None
        if self._model is None:
            return raw_token
        return str(self._model.decode(ids=[token_id]))

    def id_to_text_map(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for token_id in sorted(self._id_to_token):
            text: str | None = self.id_to_text(token_id)
            if text is not None:
                out[token_id] = text
        return out


def _read_token_map(
    path: str,
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
