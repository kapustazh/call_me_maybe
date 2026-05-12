from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model  # type: ignore

from src.tokenizer_vocab import encoded_to_token_ids
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
from src import prompt_value_extraction as pvex
from src.tokenizer_vocab import TokenizerVocab
from src.regex_value_resolver import RegexValueResolver
from src.math_utils import log_softmax

_SPACE_MARK = "Ġ"
_NEWLINE_MARK = "Ċ"
_UNKNOWN_MARK = "Äł"


class ConstrainedDecodingError(RuntimeError):
    """Raised when constrained decoding cannot produce a valid value."""


class ConstrainedDecoder:
    """Decode JSON-typed parameters with token-level constraints.

    Given chosen function schema and user prompt, decoder generates JSON object
    '{"name": ..., "parameters": {...}}'. Restricts next-token choices so each
    literal value stays valid for expected type.
    """

    _DEFAULT_MAX_NEW_TOKENS_STRING = 50
    _DEFAULT_MAX_NEW_TOKENS_NUMBER = 15

    @dataclass(frozen=True)
    class _ValueGenCtx:
        """State passed into per-type value generators during JSON fill-in.

        Attributes:
            current_ids: Token ids generated so far for the decode prefix.
            param_type: JSON schema type key (``string``, ``number``, etc.).
            is_regex_string: True when this string parameter uses regex
                resolver.
            prompt_text: Original user prompt (for heuristics / regex).
            string_plain_index: Index among non-regex string parameters (-1 if
                N/A).
            numeric_index: Index among numeric parameters (-1 if N/A).
            integer_only: Whether integer schema forbids fractional literals.
        """

        current_ids: list[int]
        param_type: str
        is_regex_string: bool
        prompt_text: str
        string_plain_index: int
        numeric_index: int
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
        """Initialize decoder and precompute token masks.

        Args:
            model: LLM wrapper providing logits and encoding helpers.
            tokenizer_vocab: Token-to-text mapping for mask construction.
            functions: Function definitions (used to build decode prompt).
            max_new_tokens_string: Max tokens for string literal generation.
            max_new_tokens_number: Max tokens for numeric literal generation.

        Raises:
            ConstrainedDecodingError:
                If tokenizer cannot produce quote token id for string literals.
        """
        self._model: Small_LLM_Model = model
        self._max_new_tokens_string: int = max_new_tokens_string
        self._max_new_tokens_number: int = max_new_tokens_number
        self._prompter: BobThePrompter = BobThePrompter(functions)
        self._regex_resolver = RegexValueResolver(model)

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
        self._token_visual: Callable[[str, int], None] | None = None

    def set_token_visual(
        self, callback: Callable[[str, int], None] | None
    ) -> None:
        """Register optional ``(piece, pair)`` sink for TUI token trace.

        Args:
            callback: Invoked on worker thread after each emitted token
                fragment; ``pair`` is :class:`LogColorPair` value (often
                ``LogColorPair.INFO`` / ``1``). ``None`` disables tracing.
        """
        self._token_visual = callback

    def _emit_token_piece(self, piece: str, *, pair: int = 1) -> None:
        if self._token_visual is not None:
            self._token_visual(piece, pair)

    def _insert_tokens(self, current_ids: list[int], text: str) -> list[int]:
        """Decode ``text`` with the model encoder and append ids to a copy.

        Args:
            current_ids: Existing token-id prefix.
            text: Literal snippet to append (not masked sampling).

        Returns:
            New list: ``current_ids`` followed by encoded ``text``.
        """
        text_token_ids = encoded_to_token_ids(self._model.encode(text))
        out = current_ids[:]
        for tid in text_token_ids:
            out.append(tid)
            self._emit_token_piece(
                self._normalize_piece(self._piece_by_id.get(tid, "")),
                pair=1,
            )
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
        result = self._decode_full_call(
            input_ids,
            function_definition,
            prompt_text=user_prompt,
        )
        return dict(result["parameters"])

    def _build_safe_string_ids(self) -> set[int]:
        """Collect vocab ids allowed inside JSON string quotes before closing.

        Excludes pieces containing quote or newline controls; always includes
        the closing quote token id.

        Returns:
            Set of legal continuation token ids for string body generation.
        """
        forbidden = {'"', "\n", "\r"}
        ids = {
            token_id
            for token_id, piece in self._piece_by_id.items()
            if piece != "" and not any(c in piece for c in forbidden)
        }
        ids.add(self._closing_quote_id)
        return ids

    def _build_valid_number_ids(self) -> set[int]:
        """Collect vocab ids whose decoded pieces are numeric-character only.

        Returns:
            Set of token ids allowed when sampling numeric literals.
        """
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
        *,
        prompt_text: str,
    ) -> dict[str, Any]:
        """Greedy-fill JSON tool-call structure matching ``chosen_function``.

        Inserts function name and walks parameters in schema order, calling
        ``_generate_value`` for each.

        Args:
            input_ids: Token ids for decode conditioning prompt.
            chosen_function: Schema driving literal generation.
            prompt_text: Raw user text (regex routing and heuristics).

        Returns:
            Dict with keys ``name`` and ``parameters`` (typed values).
        """
        self._last_regex_value: str | None = None

        current_ids = list(input_ids)
        parameters: dict[str, Any] = {}

        param_names = list(chosen_function.parameters.keys())
        string_params = [
            p
            for p in param_names
            if chosen_function.parameters[p].type == "string"
        ]
        regex_string_params = {
            p for p in string_params if pvex.is_regex_like_param_name(p)
        }
        if (
            not regex_string_params
            and pvex.prompt_requests_regex(prompt_text)
            and len(string_params) >= 3
        ):
            regex_string_params.add(string_params[1])
        string_params_plain = [
            p for p in string_params if p not in regex_string_params
        ]
        numeric_params = [
            p
            for p in param_names
            if chosen_function.parameters[p].type in ("number", "integer")
        ]

        current_ids = self._insert_tokens(current_ids, chosen_function.name)
        current_ids = self._insert_tokens(current_ids, '", "parameters": {')

        param_items = list(chosen_function.parameters.items())
        for i, (param_name, param_def) in enumerate(param_items):
            is_last = i == len(param_items) - 1
            current_ids = self._insert_tokens(current_ids, f'"{param_name}": ')

            value, current_ids = self._generate_value(
                current_ids=current_ids,
                param_type=param_def.type,
                is_regex_string=(param_name in regex_string_params),
                prompt_text=prompt_text,
                string_plain_index=(
                    string_params_plain.index(param_name)
                    if param_name in string_params_plain
                    else -1
                ),
                numeric_index=(
                    numeric_params.index(param_name)
                    if param_name in numeric_params
                    else -1
                ),
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
        is_regex_string: bool,
        prompt_text: str,
        string_plain_index: int,
        numeric_index: int,
        integer_only: bool,
    ) -> tuple[Any, list[int]]:
        """Dispatch to string/number/boolean/object generators.

        Args:
            current_ids: Token ids before emitting this parameter value.
            param_type: JSON schema type for this parameter.
            is_regex_string: Route string params through regex resolver.
            prompt_text: Original user prompt.
            string_plain_index: Ordinal among plain string params.
            numeric_index: Ordinal among numeric params.
            integer_only: Integer schema coercion flag.

        Returns:
            Parsed Python value and updated token-id suffix.

        Raises:
            ConstrainedDecodingError: If ``param_type`` is unsupported.
        """
        generator = self._value_generators.get(param_type)
        if generator is None:
            raise ConstrainedDecodingError(
                f"Unsupported parameter type: {param_type!r}"
            )
        ctx = ConstrainedDecoder._ValueGenCtx(
            current_ids=current_ids,
            param_type=param_type,
            is_regex_string=is_regex_string,
            prompt_text=prompt_text,
            string_plain_index=string_plain_index,
            numeric_index=numeric_index,
            integer_only=integer_only,
        )
        return generator(ctx)

    def _gen_string(self, ctx: _ValueGenCtx) -> tuple[str, list[int]]:
        """Generate a JSON string literal (plain or regex-specialized).

        Args:
            ctx: Current decoding context.

        Returns:
            Unquoted string value and extended ``current_ids``.
        """
        if ctx.is_regex_string:
            return self._generate_regex_value(ctx.current_ids, ctx.prompt_text)
        return self._generate_string_value(
            ctx.current_ids,
            prompt_text=ctx.prompt_text,
            string_plain_index=ctx.string_plain_index,
        )

    def _gen_number(self, ctx: _ValueGenCtx) -> tuple[int | float, list[int]]:
        """Generate a JSON number literal per schema (int vs float).

        Args:
            ctx: Current decoding context.

        Returns:
            Parsed numeric value and extended ``current_ids``.
        """
        return self._generate_number_value(
            ctx.current_ids,
            param_type=ctx.param_type,
            prompt_text=ctx.prompt_text,
            numeric_index=ctx.numeric_index,
            integer_only=ctx.integer_only,
        )

    def _gen_boolean(self, ctx: _ValueGenCtx) -> tuple[bool, list[int]]:
        """Emit JSON ``true`` or ``false`` using token log-prob tie-break.

        Args:
            ctx: Current decoding context (uses ``current_ids`` only).

        Returns:
            Boolean value and extended ``current_ids``.
        """
        return self._generate_boolean_value(ctx.current_ids)

    def _gen_object(
        self, ctx: _ValueGenCtx
    ) -> tuple[dict[str, object], list[int]]:
        """Emit empty JSON object literal ``{}``.

        Args:
            ctx: Current decoding context.

        Returns:
            Empty dict and ids with ``{}`` tokens appended.
        """
        next_ids = self._insert_tokens(ctx.current_ids, "{}")
        return {}, next_ids

    @staticmethod
    def _normalize_piece(piece: str) -> str:
        """Strip SentencePiece-style markers from a tokenizer piece.

        Args:
            piece: Raw decoded vocabulary fragment.

        Returns:
            Piece suitable for concatenating into JSON string body text.
        """
        return (
            piece.replace(_SPACE_MARK, " ")
            .replace(_NEWLINE_MARK, "")
            .replace(_UNKNOWN_MARK, "")
        )

    def _generate_string_value(
        self,
        current_ids: list[int],
        *,
        prompt_text: str,
        string_plain_index: int,
    ) -> tuple[str, list[int]]:
        """Sample or extract a non-regex string literal.

        Uses :mod:`src.prompt_value_extraction` heuristics first; otherwise
        masked greedy decoding until closing quote.

        Args:
            current_ids: Prefix ids before opening quote.
            prompt_text: Original user prompt.
            string_plain_index: Which plain string parameter this value is.

        Returns:
            Inner string content (without JSON escapes) and extended ids.
        """
        extracted = pvex.try_non_regex_string(
            prompt_text,
            string_plain_index,
            self._last_regex_value,
        )
        if extracted is not None:
            current_ids = self._insert_tokens(current_ids, f'"{extracted}"')
            return extracted, current_ids

        current_ids = self._insert_tokens(current_ids, '"')
        value_chars = ""
        for _ in range(self._max_new_tokens_string):
            logits = self._model.get_logits_from_input_ids(current_ids)
            masked = self._apply_mask(logits, self._safe_string_ids_arr)
            next_id = int(np.argmax(masked))
            current_ids.append(next_id)
            piece = self._piece_by_id.get(next_id, "")
            piece = self._normalize_piece(piece)
            self._emit_token_piece(piece, pair=1)
            if next_id == self._closing_quote_id:
                break
            value_chars += piece
        inner = value_chars.strip().split("\\n")[0].strip()
        return inner, current_ids

    def _generate_number_value(
        self,
        current_ids: list[int],
        param_type: str,
        *,
        prompt_text: str,
        numeric_index: int,
        integer_only: bool,
    ) -> tuple[int | float, list[int]]:
        """Parse numeric literal from prompt index or sample digit tokens.

        Args:
            current_ids: Prefix ids before numeric literal starts.
            param_type: ``number`` or ``integer`` schema discriminator.
            prompt_text: Original user prompt.
            numeric_index: Which numeric slot to read from prompt (-1 skips).
            integer_only: Integer schema flag.

        Returns:
            Parsed Python number and extended ids.
        """
        if numeric_index >= 0:
            parsed = pvex.parse_numeric_at_index(
                prompt_text,
                numeric_index,
                integer_only=integer_only,
            )
            if parsed is not None:
                fragment = (
                    repr(parsed) if isinstance(parsed, float) else str(parsed)
                )
                current_ids = self._insert_tokens(current_ids, fragment)
                return parsed, current_ids

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
            self._emit_token_piece(next_piece, pair=1)
            value_str += next_piece
            current_ids.append(next_id)

        if not value_str:
            return (0 if param_type == "integer" else 0.0), current_ids
        parsed = pvex.parse_number_text(
            value_str,
            integer_only=(param_type == "integer"),
        )
        if parsed is None:
            return 0.0, current_ids
        return parsed, current_ids

    @staticmethod
    def _numeric_literal_complete(value_str: str, integer_only: bool) -> bool:
        """Return True if ``value_str`` is a complete valid numeric literal.

        Args:
            value_str: Accumulated characters from tokenizer pieces.
            integer_only: Whether to validate as integer.

        Returns:
            True when :func:`~src.prompt_value_extraction.parse_number_text`
            succeeds.
        """
        if not value_str:
            return False
        return (
            pvex.parse_number_text(value_str, integer_only=integer_only)
            is not None
        )

    def _generate_boolean_value(
        self,
        current_ids: list[int],
    ) -> tuple[bool, list[int]]:
        """Choose ``true`` vs ``false`` by cumulative token log-probability.

        Args:
            current_ids: Prefix ids immediately before boolean literal.

        Returns:
            Selected bool and ids including emitted literal tokens.
        """
        true_score = self._score_word(current_ids, self._true_ids)
        false_score = self._score_word(current_ids, self._false_ids)

        if true_score >= false_score:
            current_ids = self._insert_tokens(current_ids, "true")
            return True, current_ids

        current_ids = self._insert_tokens(current_ids, "false")
        return False, current_ids

    def _score_word(
        self,
        base_ids: list[int],
        word_ids: list[int],
    ) -> float:
        """Sum log-probs for generating ``word_ids`` after ``base_ids``.

        Args:
            base_ids: Conditioning prefix token ids.
            word_ids: Target token sequence (e.g. encoded ``true``).

        Returns:
            Total log score, or negative infinity if empty or impossible step.
        """
        if not word_ids:
            return -math.inf

        temp_ids = list(base_ids)
        total = 0.0

        for token_id in word_ids:
            logits = self._model.get_logits_from_input_ids(temp_ids)
            log_probs = log_softmax(logits)
            total += (
                float(log_probs[token_id])
                if token_id < len(log_probs)
                else -math.inf
            )
            temp_ids.append(token_id)

        return total

    def _generate_regex_value(
        self,
        current_ids: list[int],
        prompt_text: str,
    ) -> tuple[str, list[int]]:
        """Resolve regex pattern via ``RegexValueResolver``.

        Args:
            current_ids: Prefix ids before string literal.
            prompt_text: User prompt for resolver heuristics.

        Returns:
            Pattern string and ids with quoted literal appended.
        """
        pattern = self._regex_resolver.resolve(prompt_text)
        self._last_regex_value = pattern
        current_ids = self._insert_tokens(current_ids, f'"{pattern}"')
        return pattern, current_ids

    def _apply_mask(
        self,
        logits: list[float],
        valid_ids_arr: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:
        """Restrict logits to allowed vocab ids; mask others as ``-inf``.

        Args:
            logits: Full vocabulary logits from model.
            valid_ids_arr: Row vector of legal token indices.

        Returns:
            Same-length float array suitable for ``argmax`` sampling.
        """
        arr = np.full(len(logits), -np.inf, dtype=np.float32)
        logits_arr = np.array(logits, dtype=np.float32)
        valid_ids_arr = valid_ids_arr[valid_ids_arr < len(arr)]
        arr[valid_ids_arr] = logits_arr[valid_ids_arr]
        return arr
