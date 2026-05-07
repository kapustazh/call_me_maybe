from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model  # type: ignore

from src.tokenizer_vocab import encoded_to_token_ids
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
from src import prompt_value_extraction as pvex
from src.tokenizer_vocab import TokenizerVocab
from src.math_utils import log_softmax


class ConstrainedDecodingError(RuntimeError):
    """Raised when constrained decoding cannot produce a valid value."""


class ConstrainedDecoder:
    _DEFAULT_MAX_NEW_TOKENS_STRING = 50
    _DEFAULT_MAX_NEW_TOKENS_NUMBER = 15

    def __init__(
        self,
        model: Small_LLM_Model,
        tokenizer_vocab: TokenizerVocab,
        functions: list[FunctionDefinition],
        max_new_tokens_string: int = _DEFAULT_MAX_NEW_TOKENS_STRING,
        max_new_tokens_number: int = _DEFAULT_MAX_NEW_TOKENS_NUMBER,
    ) -> None:
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

    def _force(self, current_ids: list[int], text: str) -> list[int]:
        frag = encoded_to_token_ids(self._model.encode(text))
        return current_ids + frag

    def decode_parameters(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
    ) -> dict[str, object]:
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
        forbidden = {'"', "\n", "\r"}
        ids = {
            token_id
            for token_id, piece in self._piece_by_id.items()
            if piece != "" and not any(c in piece for c in forbidden)
        }
        ids.add(self._closing_quote_id)
        return ids

    def _build_valid_number_ids(self) -> set[int]:
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

        current_ids = self._force(current_ids, chosen_function.name)
        current_ids = self._force(current_ids, '", "parameters": {')

        param_items = list(chosen_function.parameters.items())
        for i, (param_name, param_def) in enumerate(param_items):
            is_last = i == len(param_items) - 1
            current_ids = self._force(current_ids, f'"{param_name}": ')

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
                current_ids = self._force(current_ids, ", ")

        current_ids = self._force(current_ids, "}}")

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
        if param_type == "string":
            if is_regex_string:
                return self._generate_regex_value(current_ids, prompt_text)
            return self._generate_string_value(
                current_ids,
                prompt_text=prompt_text,
                string_plain_index=string_plain_index,
            )
        if param_type in ("number", "integer"):
            return self._generate_number_value(
                current_ids,
                param_type=param_type,
                prompt_text=prompt_text,
                numeric_index=numeric_index,
                integer_only=integer_only,
            )
        if param_type == "boolean":
            return self._generate_boolean_value(current_ids)
        if param_type == "object":
            current_ids = self._force(current_ids, "{}")
            return {}, current_ids
        return self._generate_string_value(
            current_ids,
            prompt_text=prompt_text,
            string_plain_index=string_plain_index,
        )

    def _generate_string_value(
        self,
        current_ids: list[int],
        *,
        prompt_text: str,
        string_plain_index: int,
    ) -> tuple[str, list[int]]:
        extracted = pvex.try_non_regex_string(
            prompt_text,
            string_plain_index,
            self._last_regex_value,
        )
        if extracted is not None:
            current_ids = self._force(current_ids, f'"{extracted}"')
            return extracted, current_ids

        current_ids = self._force(current_ids, '"')
        value_chars = ""
        for _ in range(self._max_new_tokens_string):
            logits = self._model.get_logits_from_input_ids(current_ids)
            masked = self._apply_mask(logits, self._safe_string_ids_arr)
            next_id = int(np.argmax(masked))
            current_ids.append(next_id)
            if next_id == self._closing_quote_id:
                break
            piece = self._piece_by_id.get(next_id, "")
            if any(m in piece for m in ("Ġ", "Ċ", "Äł")):
                piece = (
                    piece.replace("Ġ", " ").replace("Ċ", "").replace("Äł", "")
                )
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
                current_ids = self._force(current_ids, fragment)
                return parsed, current_ids

        valid_chars = set("0123456789.-")
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
            return (0 if param_type == "integer" else 0.0), current_ids
        try:
            return (
                int(float(value_str))
                if param_type == "integer"
                else float(value_str)
            ), current_ids
        except ValueError:
            return 0.0, current_ids

    @staticmethod
    def _numeric_literal_complete(value_str: str, integer_only: bool) -> bool:
        if not value_str:
            return False
        try:
            if integer_only:
                int(float(value_str))
            else:
                float(value_str)
        except ValueError:
            return False
        return True

    def _generate_boolean_value(
        self,
        current_ids: list[int],
    ) -> tuple[bool, list[int]]:
        true_score = self._score_word(current_ids, self._true_ids)
        false_score = self._score_word(current_ids, self._false_ids)

        if true_score >= false_score:
            current_ids = self._force(current_ids, "true")
            return True, current_ids

        current_ids = self._force(current_ids, "false")
        return False, current_ids

    def _score_word(
        self,
        base_ids: list[int],
        word_ids: list[int],
    ) -> float:
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
        pattern = pvex.regex_pattern_from_prompt(prompt_text)
        if pattern is not None:
            self._last_regex_value = pattern
            current_ids = self._force(current_ids, f'"{pattern}"')
            return pattern, current_ids

        candidates = pvex.regex_candidate_patterns()
        scoring_text = (
            f'For the request "{prompt_text}", '
            f"the correct regex pattern is: "
        )
        scoring_ids = encoded_to_token_ids(self._model.encode(scoring_text))

        best_candidate = candidates[0]
        best_score = -math.inf
        for candidate in candidates:
            token_ids = encoded_to_token_ids(self._model.encode(candidate))
            score = self._score_word(scoring_ids, token_ids)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        self._last_regex_value = best_candidate
        current_ids = self._force(current_ids, f'"{best_candidate}"')
        return best_candidate, current_ids

    def _apply_mask(
        self,
        logits: list[float],
        valid_ids_arr: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float32]:
        arr = np.full(len(logits), -np.inf, dtype=np.float32)
        logits_arr = np.array(logits, dtype=np.float32)
        valid_ids_arr = valid_ids_arr[valid_ids_arr < len(arr)]
        arr[valid_ids_arr] = logits_arr[valid_ids_arr]
        return arr
