"""Vocabulary utilities: parse tokenizer.json into id -> piece map.

Byte-level BPE decoding (GPT-2 style) so the constrained decoder can
reason on real text instead of raw tokenizer strings (e.g. "Ġ" -> " ").
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vocab:
    """Immutable-ish vocabulary map consumed by the constrained decoder."""

    id_to_piece: dict[int, str] = field(default_factory=dict)
    special_ids: set[int] = field(default_factory=set)
    size: int = 0

    @classmethod
    def from_tokenizer_file(cls, path: str) -> "Vocab":
        """Build a Vocab by parsing a Hugging Face tokenizer.json file."""
        ...

    def piece(self, token_id: int) -> str:
        """Return decoded text piece for the given token id."""
        ...

    def is_special(self, token_id: int) -> bool:
        """Return True if the id is a special/control token (EOS/BOS/pad)."""
        ...


def _bytes_to_unicode() -> dict[int, str]:
    """Return the canonical GPT-2 byte -> printable unicode mapping."""
    ...


def _unicode_to_bytes() -> dict[str, int]:
    """Return the inverse of :func:`_bytes_to_unicode`."""
    ...


def _decode_piece(raw: str, u2b: dict[str, int]) -> str:
    """Decode a raw tokenizer piece (byte-level unicode) into real text."""
    ...


def _load_tokenizer_json(
    path: str,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Read tokenizer.json and return (model.vocab, added_tokens)."""
    ...
