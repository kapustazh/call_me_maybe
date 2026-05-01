"""JSON literal prefix checks and typed parse for constrained decoding."""

from __future__ import annotations

import json
import string
from typing import Protocol


class ConstrainedDecodingError(RuntimeError):
    """Raised when constrained decoding cannot produce a valid value."""


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
        value = json.loads(text)
        if not isinstance(value, bool):
            raise ConstrainedDecodingError(
                f"Expected boolean JSON value, got: {text}"
            )
        return value


class ObjectValidator:
    _valid_prefixes = ("", "{", "{}")

    def allows_token_piece(self, piece: str) -> bool:
        return piece in {"{", "}", "{}"} or (
            piece != "" and len(piece) <= 2 and set(piece) <= {"{", "}"}
        )

    def is_valid_prefix(self, text: str) -> bool:
        return text in self._valid_prefixes

    def is_complete(self, text: str) -> bool:
        return text == "{}"

    def parse_value(self, text: str) -> object:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ConstrainedDecodingError(
                f"Expected object JSON value, got: {text}"
            )
        if value:
            raise ConstrainedDecodingError(
                "Object parameter currently supports only empty '{}'"
            )
        return value


class NumberValidator:
    def __init__(self, *, integer_only: bool) -> None:
        self._integer_only: bool = integer_only

    def allows_token_piece(self, piece: str) -> bool:
        allowed_chars = (
            frozenset("-0123456789")
            if self._integer_only
            else frozenset("-+0123456789.eE")
        )
        return (
            piece != ""
            and len(piece) <= 8
            and all(char in allowed_chars for char in piece)
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
        value = json.loads(text)
        if isinstance(value, bool):
            raise ConstrainedDecodingError(
                f"Expected numeric JSON value, got: {text}"
            )
        if self._integer_only:
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


class StringValidator:
    def allows_token_piece(self, piece: str) -> bool:
        if piece == "" or "\n" in piece or "\r" in piece:
            return False
        if len(piece) > 4:
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
        value = json.loads(text)
        if not isinstance(value, str):
            raise ConstrainedDecodingError(
                f"Expected string JSON value, got: {text}"
            )
        return value


# TODO: review tommorow
def _scan_number_prefix(
    text: str,
    *,
    allow_fraction: bool,
    allow_exponent: bool,
) -> tuple[bool, bool]:
    if text == "":
        return True, False

    state = "start"
    for char in text:
        if state == "start":
            if char == "-":
                state = "sign"
            elif char == "0":
                state = "int_zero"
            elif char in "123456789":
                state = "int"
            else:
                return False, False
            continue

        if state == "sign":
            if char == "0":
                state = "int_zero"
            elif char in "123456789":
                state = "int"
            else:
                return False, False
            continue

        if state == "int_zero":
            if allow_fraction and char == ".":
                state = "dot"
            elif allow_exponent and char in "eE":
                state = "exp_mark"
            else:
                return False, False
            continue

        if state == "int":
            if char in string.digits:
                state = "int"
            elif allow_fraction and char == ".":
                state = "dot"
            elif allow_exponent and char in "eE":
                state = "exp_mark"
            else:
                return False, False
            continue

        if state == "dot":
            if char in string.digits:
                state = "frac"
            else:
                return False, False
            continue

        if state == "frac":
            if char in string.digits:
                state = "frac"
            elif allow_exponent and char in "eE":
                state = "exp_mark"
            else:
                return False, False
            continue

        if state == "exp_mark":
            if char in "+-":
                state = "exp_sign"
            elif char in string.digits:
                state = "exp"
            else:
                return False, False
            continue

        if state == "exp_sign":
            if char in string.digits:
                state = "exp"
            else:
                return False, False
            continue

        if state == "exp":
            if char in string.digits:
                state = "exp"
            else:
                return False, False
            continue

    complete_states = {"int_zero", "int"}
    if allow_fraction:
        complete_states.add("frac")
    if allow_exponent:
        complete_states.add("exp")
    return True, state in complete_states


def _scan_string_literal_prefix(text: str) -> tuple[bool, bool]:
    if text == "":
        return True, False
    if text[0] != '"':
        return False, False

    escaped = False
    unicode_digits_left = 0
    index = 1
    while index < len(text):
        char = text[index]
        if unicode_digits_left > 0:
            if char not in string.hexdigits:
                return False, False
            unicode_digits_left -= 1
            index += 1
            continue
        if escaped:
            escaped = False
            if char in '"\\/bfnrt':
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
        if char == '"':
            return index == len(text) - 1, True
        if ord(char) < 32:
            return False, False
        index += 1
    return True, False
