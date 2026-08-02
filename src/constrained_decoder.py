from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model  # type: ignore

from src.models import FunctionDefinition
from src.prompt import BobThePrompter
from src.tokenizer_vocab import TokenizerVocab, encoded_to_token_ids
from src.math_utils import cumulative_sequence_logprob

_SPACE_MARK = "Ġ"
_NEWLINE_MARK = "Ċ"
_UNKNOWN_MARK = "Äł"


class ConstrainedDecodingError(RuntimeError):
    """Error when constrained decoding cannot finish a valid JSON literal.

    Typical causes include an unsupported schema type, tokenizer issues, or
    impossible token masks during greedy generation.
    """


def _parse_number_text(
    value_text: str, *, integer_only: bool
) -> int | float | None:
    """Parse digits produced by masked decoding into int/float."""
    if integer_only:
        try:
            return int(float(value_text))
        except ValueError:
            return None
    try:
        return float(value_text)
    except ValueError:
        return None


class ConstrainedDecoder:
    """Fill in a tool-call JSON body under hard token constraints.

    The model only sees allowed next tokens when emitting string bodies,
    numbers, and booleans, so the result stays syntactically valid for the
    chosen :class:`~src.models.FunctionDefinition`. Known structure
    (name, keys, punctuation) is inserted; values are generated via logit
    masking.
    """

    _DEFAULT_MAX_NEW_TOKENS_STRING = 50
    _DEFAULT_MAX_NEW_TOKENS_NUMBER = 15

    @dataclass(frozen=True)
    class _ValueGenCtx:
        """Per-parameter generation context for ``_gen_*`` helpers."""

        current_ids: list[int]
        param_type: str
        integer_only: bool

    _ValueGenerator = Callable[
        ["ConstrainedDecoder._ValueGenCtx"],
        tuple[Any, list[int]],
    ]

    def __init__(
        self,
        model: Small_LLM_Model,
        tokenizer_vocab: TokenizerVocab,
        functions: list[FunctionDefinition],
        max_new_tokens_string: int = _DEFAULT_MAX_NEW_TOKENS_STRING,
        max_new_tokens_number: int = _DEFAULT_MAX_NEW_TOKENS_NUMBER,
    ) -> None:
        """Wire the model, vocab, and precomputed allow-lists for decoding.

        Args:
            model: LLM wrapper providing logits and encoding helpers.
            tokenizer_vocab: Token id to decoded piece, for mask construction.
            functions: Tool schemas used to build the decode prompt template.
            max_new_tokens_string: Cap on greedy steps inside a string literal.
            max_new_tokens_number: Cap on greedy steps for a sampled number.

        Raises:
            ConstrainedDecodingError: If the tokenizer yields no id for a
                double-quote character (string literals cannot be closed).
        """
        self._model: Small_LLM_Model = model
        self._max_new_tokens_string: int = max_new_tokens_string
        self._max_new_tokens_number: int = max_new_tokens_number
        self._prompter: BobThePrompter = BobThePrompter(functions)

        self._piece_by_id: dict[int, str] = tokenizer_vocab.id_to_text_map()

        quote_ids = encoded_to_token_ids(self._model.encode('"'))
        if not quote_ids:
            raise ConstrainedDecodingError(
                "Tokenizer produced no ids for double-quote token"
            )
        self._closing_quote_id: int = quote_ids[0]

        self._true_ids: list[int] = encoded_to_token_ids(
            self._model.encode("true")
        )
        self._false_ids: list[int] = encoded_to_token_ids(
            self._model.encode("false")
        )

        self._valid_number_ids_arr: npt.NDArray[np.int32] = np.array(
            list(self._build_valid_number_ids()),
            dtype=np.int32,
        )
        self._safe_string_ids_arr: npt.NDArray[np.int32] = np.array(
            list(self._build_safe_string_ids()),
            dtype=np.int32,
        )

        self._value_generators: dict[
            str, ConstrainedDecoder._ValueGenerator
        ] = {
            "string": self._gen_string,
            "number": self._gen_number,
            "integer": self._gen_number,
            "boolean": self._gen_boolean,
            "object": self._gen_object,
        }

    def _insert_tokens(self, current_ids: list[int], text: str) -> list[int]:
        """Decode ``text`` with the model encoder and append ids to a copy."""
        text_token_ids = encoded_to_token_ids(self._model.encode(text))
        out = current_ids[:]
        for tid in text_token_ids:
            out.append(tid)
        return out

    def decode_parameters(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
    ) -> dict[str, object]:
        """Decode argument values for a chosen function schema.

        Args:
            user_prompt: Raw user request.
            function_definition: Selected function schema.

        Returns:
            Mapping of parameter name to decoded Python value.

        Raises:
            ConstrainedDecodingError: If decoding cannot complete.
        """
        decode_prompt = self._prompter.build_decode_prompt(
            user_prompt,
            function_definition,
        )
        input_ids = encoded_to_token_ids(self._model.encode(decode_prompt))
        result = self._decode_full_call(input_ids, function_definition)
        return dict(result["parameters"])

    def _build_safe_string_ids(self) -> set[int]:
        """Collect vocab ids allowed inside JSON string quotes before closing."""
        forbidden = {'"', "\n", "\r"}
        ids = {
            token_id
            for token_id, piece in self._piece_by_id.items()
            if piece != "" and not any(c in piece for c in forbidden)
        }
        ids.add(self._closing_quote_id)
        return ids

    def _build_valid_number_ids(self) -> set[int]:
        """Collect vocab ids whose decoded pieces are numeric-character only."""
        valid_chars = set("0123456789.-")
        out: set[int] = set()
        for token_id, piece in self._piece_by_id.items():
            stripped = piece.strip()
            if not stripped:
                continue
            if all(c in valid_chars for c in stripped):
                out.add(token_id)
        return out

    def _decode_full_call(
        self,
        input_ids: list[int],
        chosen_function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Greedy-fill JSON tool-call structure matching ``chosen_function``."""
        current_ids = list(input_ids)
        parameters: dict[str, Any] = {}

        current_ids = self._insert_tokens(current_ids, chosen_function.name)
        current_ids = self._insert_tokens(current_ids, '", "parameters": {')

        param_items = list(chosen_function.parameters.items())
        for i, (param_name, param_def) in enumerate(param_items):
            is_last = i == len(param_items) - 1
            current_ids = self._insert_tokens(current_ids, f'"{param_name}": ')

            value, current_ids = self._generate_value(
                current_ids=current_ids,
                param_type=param_def.type,
                integer_only=(param_def.type == "integer"),
            )
            parameters[param_name] = value

            if not is_last:
                current_ids = self._insert_tokens(current_ids, ", ")

        current_ids = self._insert_tokens(current_ids, "}}")

        return {
            "name": chosen_function.name,
            "parameters": parameters,
        }

    def _generate_value(
        self,
        *,
        current_ids: list[int],
        param_type: str,
        integer_only: bool,
    ) -> tuple[Any, list[int]]:
        """Dispatch to string/number/boolean/object generators."""
        generator = self._value_generators.get(param_type)
        if generator is None:
            raise ConstrainedDecodingError(
                f"Unsupported parameter type: {param_type!r}"
            )
        ctx = ConstrainedDecoder._ValueGenCtx(
            current_ids=current_ids,
            param_type=param_type,
            integer_only=integer_only,
        )
        return generator(ctx)

    def _gen_string(self, ctx: _ValueGenCtx) -> tuple[str, list[int]]:
        """Generate a JSON string literal via masked greedy decoding."""
        return self._generate_string_value(ctx.current_ids)

    def _gen_number(self, ctx: _ValueGenCtx) -> tuple[int | float, list[int]]:
        """Generate a JSON number literal via masked greedy decoding."""
        return self._generate_number_value(
            ctx.current_ids,
            param_type=ctx.param_type,
            integer_only=ctx.integer_only,
        )

    def _gen_boolean(self, ctx: _ValueGenCtx) -> tuple[bool, list[int]]:
        """Emit JSON ``true`` or ``false`` using token log-prob tie-break."""
        return self._generate_boolean_value(ctx.current_ids)

    def _gen_object(
        self, ctx: _ValueGenCtx
    ) -> tuple[dict[str, object], list[int]]:
        """Emit empty JSON object literal ``{}``."""
        next_ids = self._insert_tokens(ctx.current_ids, "{}")
        return {}, next_ids

    @staticmethod
    def _normalize_piece(piece: str) -> str:
        """Strip SentencePiece-style markers from a tokenizer piece."""
        return (
            piece.replace(_SPACE_MARK, " ")
            .replace(_NEWLINE_MARK, "")
            .replace(_UNKNOWN_MARK, "")
        )

    def _generate_string_value(
        self,
        current_ids: list[int],
    ) -> tuple[str, list[int]]:
        """Masked greedy decoding until closing quote.

        Raises:
            ConstrainedDecodingError: If the closing quote is never emitted.
        """
        current_ids = self._insert_tokens(current_ids, '"')
        value_chars = ""
        closed = False
        for _ in range(self._max_new_tokens_string):
            logits = self._model.get_logits_from_input_ids(current_ids)
            masked = self._apply_mask(logits, self._safe_string_ids_arr)
            next_id = int(np.argmax(masked))
            current_ids.append(next_id)
            piece = self._piece_by_id.get(next_id, "")
            piece = self._normalize_piece(piece)
            if next_id == self._closing_quote_id:
                closed = True
                break
            value_chars += piece
        if not closed:
            raise ConstrainedDecodingError(
                "String decode exceeded max tokens without closing quote"
            )
        inner = value_chars.strip().split("\\n")[0].strip()
        return inner, current_ids

    def _generate_number_value(
        self,
        current_ids: list[int],
        param_type: str,
        *,
        integer_only: bool,
    ) -> tuple[int | float, list[int]]:
        """Sample digit tokens under a numeric allow-list.

        Raises:
            ConstrainedDecodingError: If no valid numeric literal is produced.
        """
        valid_chars: set[str] = set("0123456789.-")
        value_str = ""
        for _ in range(self._max_new_tokens_number):
            logits = self._model.get_logits_from_input_ids(current_ids)
            masked = self._apply_mask(logits, self._valid_number_ids_arr)
            finite = masked[np.isfinite(masked)]
            if (
                value_str
                and finite.size > 0
                and self._numeric_literal_complete(value_str, integer_only)
                and float(np.max(finite) - np.min(finite)) < 1e-5
            ):
                break
            next_id = int(np.argmax(masked))
            next_piece = self._piece_by_id.get(next_id, "").strip()
            if not next_piece or not all(c in valid_chars for c in next_piece):
                break
            value_str += next_piece
            current_ids.append(next_id)

        if not value_str:
            raise ConstrainedDecodingError(
                f"Failed to decode {param_type} literal"
            )
        parsed = _parse_number_text(value_str, integer_only=integer_only)
        if parsed is None:
            raise ConstrainedDecodingError(
                f"Invalid {param_type} literal: {value_str!r}"
            )
        return parsed, current_ids

    @staticmethod
    def _numeric_literal_complete(value_str: str, integer_only: bool) -> bool:
        """Return True if ``value_str`` is a complete valid numeric literal."""
        if not value_str:
            return False
        return _parse_number_text(value_str, integer_only=integer_only) is not None

    def _generate_boolean_value(
        self,
        current_ids: list[int],
    ) -> tuple[bool, list[int]]:
        """Choose ``true`` vs ``false`` by cumulative token log-probability."""
        gl = self._model.get_logits_from_input_ids
        true_score = cumulative_sequence_logprob(
            gl, current_ids, self._true_ids
        )
        false_score = cumulative_sequence_logprob(
            gl, current_ids, self._false_ids
        )

        if true_score >= false_score:
            current_ids = self._insert_tokens(current_ids, "true")
            return True, current_ids

        current_ids = self._insert_tokens(current_ids, "false")
        return False, current_ids

    def _apply_mask(
        self,
        logits: list[float],
        valid_ids_arr: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:
        """Restrict logits to allowed vocab ids; mask others as ``-inf``."""
        arr = np.full(len(logits), -np.inf, dtype=np.float32)
        logits_arr = np.asarray(logits, dtype=np.float32)
        valid_ids_arr = valid_ids_arr[valid_ids_arr < len(arr)]
        arr[valid_ids_arr] = logits_arr[valid_ids_arr]
        return arr
