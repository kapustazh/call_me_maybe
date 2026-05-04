"""JSON literal prefix checks and typed parse for constrained decoding.

(1) Validators: `is_valid_prefix` / `allows_token_piece` / `is_complete` keep
    the partial text in a JSON literal subset the tokenizer can emit.
(2) `parse_value` is only for *complete* strings that already passed
    `is_complete`; it decodes the same subset to Python values.
"""

from __future__ import annotations

import json
import string
from typing import Protocol


class ConstrainedDecodingError(RuntimeError):
    """Raised when constrained decoding cannot produce a valid value."""


def _expect_json_type(
    text: str,
    py_type: type | tuple[type, ...],
    what: str,
) -> object:
    value = json.loads(text)
    if not isinstance(value, py_type):
        raise ConstrainedDecodingError(
            f"Expected {what} JSON value, got: {text}"
        )
    return value


def _expect_json_number(
    text: str,
    *,
    integer_only: bool,
) -> int | float:
    value = json.loads(text)
    if isinstance(value, bool):
        raise ConstrainedDecodingError(
            f"Expected numeric JSON value, got: {text}"
        )
    if integer_only:
        if not isinstance(value, int):
            raise ConstrainedDecodingError(
                f"Expected integer JSON value, got: {text}"
            )
        return value
    if not isinstance(value, (int, float)):
        raise ConstrainedDecodingError(
            f"Expected number JSON value, got: {text}"
        )
    return value


class LiteralValidator(Protocol):
    def allows_token_piece(self, piece: str) -> bool: ...

    def is_valid_prefix(self, text: str) -> bool: ...

    def is_complete(self, text: str) -> bool: ...

    def parse_value(self, text: str) -> object: ...


class BooleanValidator:
    _literals = ("true", "false")
    _allowed_chars = frozenset("".join(_literals))
    _max_piece_length = max(len(literal) for literal in _literals)

    def allows_token_piece(self, piece: str) -> bool:
        return (
            piece != ""
            and len(piece) <= self._max_piece_length
            and all(char in self._allowed_chars for char in piece)
        )

    def is_valid_prefix(self, text: str) -> bool:
        return text == "" or any(
            literal.startswith(text) for literal in self._literals
        )

    def is_complete(self, text: str) -> bool:
        return text in self._literals

    def parse_value(self, text: str) -> object:
        if text == "true":
            return True
        if text == "false":
            return False
        raise ConstrainedDecodingError(
            f"Expected boolean JSON value, got: {text}"
        )


class EmptyObjectValidator:
    _valid_prefixes = ("", "{", "{}")

    # Only these strings may be emitted as one tokenizer piece for `{}`.
    _ALLOWED_PIECES = frozenset({"{", "}", "{}"})

    def allows_token_piece(self, piece: str) -> bool:
        return piece in self._ALLOWED_PIECES

    def is_valid_prefix(self, text: str) -> bool:
        return text in self._valid_prefixes

    def is_complete(self, text: str) -> bool:
        return text == "{}"

    def parse_value(self, text: str) -> object:
        value = _expect_json_type(text, dict, "object")
        if value:
            raise ConstrainedDecodingError(
                "Object parameter currently supports only empty '{}'"
            )
        return value


class NumberValidator:
    _INT_PIECE_CHARS = frozenset("-0123456789")
    _FLOAT_PIECE_CHARS = frozenset("-+0123456789.eE")
    # Upper bound on one BPE merge piece while scanning a JSON number (avoids
    # pathological long runs; aligned with small vocab slices).
    _MAX_TOKEN_PIECE_LEN = 8

    def __init__(self, *, integer_only: bool) -> None:
        self._integer_only: bool = integer_only
        self._piece_chars: frozenset[str] = (
            self._INT_PIECE_CHARS if integer_only else self._FLOAT_PIECE_CHARS
        )

    def allows_token_piece(self, piece: str) -> bool:
        return (
            piece != ""
            and len(piece) <= self._MAX_TOKEN_PIECE_LEN
            and all(char in self._piece_chars for char in piece)
        )

    def is_valid_prefix(self, text: str) -> bool:
        valid, _ = _scan_number_prefix(
            text,
            allow_fraction=not self._integer_only,
            allow_exponent=not self._integer_only,
        )
        return valid

    def is_complete(self, text: str) -> bool:
        _, complete = _scan_number_prefix(
            text,
            allow_fraction=not self._integer_only,
            allow_exponent=not self._integer_only,
        )
        return complete

    def parse_value(self, text: str) -> object:
        return _expect_json_number(text, integer_only=self._integer_only)


class StringValidator:
    # Single tokenizer merge max length for string body chars (no quote).
    _MAX_TOKEN_PIECE_LEN = 4

    def allows_token_piece(self, piece: str) -> bool:
        if piece == "" or "\n" in piece or "\r" in piece:
            return False
        if len(piece) > self._MAX_TOKEN_PIECE_LEN:
            return False
        if any(ord(char) < 32 for char in piece):
            return False
        return True

    def is_valid_prefix(self, text: str) -> bool:
        valid, _ = _scan_string_literal_prefix(text)
        return valid

    def is_complete(self, text: str) -> bool:
        _, complete = _scan_string_literal_prefix(text)
        return complete

    def parse_value(self, text: str) -> object:
        return _expect_json_type(text, str, "string")


def _scan_number_prefix(
    text: str,
    *,
    allow_fraction: bool,
    allow_exponent: bool,
) -> tuple[bool, bool]:
    if text == "":
        return True, False

    state: str | None = "start"
    for char in text:
        if state is None:
            return False, False
        new_state = _next_number_state(
            state,
            char,
            allow_fraction=allow_fraction,
            allow_exponent=allow_exponent,
        )
        if new_state is None:
            return False, False
        state = new_state

    complete_states = {"int_zero", "int"}
    if allow_fraction:
        complete_states.add("frac")
    if allow_exponent:
        complete_states.add("exp")
    assert state is not None
    return True, state in complete_states


def _next_number_state(
    state: str,
    char: str,
    *,
    allow_fraction: bool,
    allow_exponent: bool,
) -> str | None:
    if state in {"start", "sign"}:
        return _number_start_state(char, allow_sign=state == "start")
    if state in {"int_zero", "int", "frac"}:
        return _number_body_state(
            state,
            char,
            allow_fraction=allow_fraction,
            allow_exponent=allow_exponent,
        )
    if state in {"dot", "exp_sign", "exp"}:
        if state == "dot" and char in string.digits:
            return "frac"
        if state != "dot" and char in string.digits:
            return "exp"
        return None
    if state == "exp_mark":
        if char in "+-":
            return "exp_sign"
        return "exp" if char in string.digits else None
    return None


def _number_start_state(char: str, *, allow_sign: bool) -> str | None:
    if allow_sign and char == "-":
        return "sign"
    if char == "0":
        return "int_zero"
    if char in "123456789":
        return "int"
    return None


def _number_body_state(
    state: str,
    char: str,
    *,
    allow_fraction: bool,
    allow_exponent: bool,
) -> str | None:
    if state != "int_zero" and char in string.digits:
        return state
    if state in {"int_zero", "int"} and allow_fraction and char == ".":
        return "dot"
    if allow_exponent and char in "eE":
        return "exp_mark"
    return None


def _scan_string_literal_prefix(text: str) -> tuple[bool, bool]:
    """Reject strings that look like nested JSON or bad tool-style opens."""
    if text == "":
        return True, False
    if text[0] != '"':
        return False, False

    inner = text[1:]
    if inner:
        stripped = inner.lstrip()
        if stripped.startswith("{"):
            return False, False
        if len(stripped) >= 2 and stripped[0] == ">":
            if stripped[1] == "{" or stripped[1].isspace():
                return False, False

    escaped = False
    unicode_digits_left = 0
    index = 1
    while index < len(text):
        char = text[index]
        if unicode_digits_left > 0:
            if not _is_hex_digit(char):
                return False, False
            unicode_digits_left -= 1
            index += 1
            continue
        if escaped:
            escaped = False
            if _is_simple_escape(char):
                index += 1
                continue
            if char == "u":
                unicode_digits_left = 4
                index += 1
                continue
            return False, False
        if char == "\\":
            escaped = True
            index += 1
            continue
        if _is_closing_quote(char):
            return index == len(text) - 1, True
        if _is_control_char(char):
            return False, False
        index += 1
    return True, False


def _is_hex_digit(char: str) -> bool:
    return char in string.hexdigits


def _is_simple_escape(char: str) -> bool:
    return char in '"\\/bfnrt'


def _is_closing_quote(char: str) -> bool:
    return char == '"'


def _is_control_char(char: str) -> bool:
    return ord(char) < 32
