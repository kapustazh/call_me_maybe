from __future__ import annotations

import numpy as np

from src.json_literal_validators import (
    BooleanValidator,
    ConstrainedDecodingError,
    EmptyObjectValidator,
    LiteralValidator,
    NumberValidator,
    StringValidator,
)
from src.model_protocol import LLMModelProtocolAdapter
from src.model_utils import encoded_to_token_ids
from src.models import FunctionDefinition
from src.prompt import BobThePrompter
from src.tokenizer_vocab import TokenizerVocab
import math


class ConstrainedDecoder:
    def __init__(
        self,
        model: LLMModelProtocolAdapter,
        tokenizer_vocab: TokenizerVocab,
        functions: list[FunctionDefinition],
        max_new_tokens: int = 120,
    ) -> None:
        self._model: LLMModelProtocolAdapter = model
        self._max_new_tokens: int = max_new_tokens
        self._max_string_literal_length: int = max_new_tokens
        self._prompter: BobThePrompter = BobThePrompter(functions)
        self._piece_by_id: dict[int, str] = tokenizer_vocab.id_to_text_map()

        self._validators: dict[str, LiteralValidator] = {
            "string": StringValidator(),
            "number": NumberValidator(integer_only=False),
            "integer": NumberValidator(integer_only=True),
            "boolean": BooleanValidator(),
            "object": EmptyObjectValidator(),
        }
        self._candidate_pool_by_type: dict[str, tuple[int, ...]] = {
            value_type: self._collect_pool(validator)
            for value_type, validator in self._validators.items()
        }

    def decode_parameters(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
    ) -> dict[str, object]:
        out: dict[str, object] = {}
        for parameter_name in function_definition.parameters:
            parameter_spec = function_definition.parameters[parameter_name]
            parameter_type = parameter_spec.type
            prompt = self._prompter.build_parameter_prompt(
                user_prompt,
                function_definition,
                parameter_name,
            )
            raw_literal = self._decode_literal(
                prompt=prompt,
                value_type=parameter_type,
                parameter_name=parameter_name,
            )
            validator = self._validators[parameter_type]
            out[parameter_name] = validator.parse_value(raw_literal)
        return out

    def _decode_literal(
        self,
        *,
        prompt: str,
        value_type: str,
        parameter_name: str,
    ) -> str:
        context_ids = encoded_to_token_ids(self._model.encode(prompt))
        generated_ids: list[int] = []
        generated_text = ""

        validator: LiteralValidator = self._validators[value_type]
        candidate_pool = self._candidate_pool_by_type[value_type]

        for _ in range(self._max_new_tokens):
            logits = self._model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            allowed_ids = self._allowed_ids(
                generated_text,
                candidate_pool,
                validator,
            )
            if not allowed_ids:
                raise ConstrainedDecodingError(
                    f"No allowed tokens left for '{parameter_name}' "
                    f"({value_type}) with prefix {generated_text!r}"
                )

            token_id: int = self._select_best_token(logits, allowed_ids)
            generated_ids.append(token_id)
            piece: str = self._piece_by_id[token_id]
            generated_text += piece

            if validator.is_complete(generated_text) and self._should_stop(
                context_ids=context_ids,
                generated_ids=generated_ids,
                generated_text=generated_text,
                candidate_pool=candidate_pool,
                validator=validator,
            ):
                return generated_text

            forced_literal: str | None = self._force_complete_string(
                value_type=value_type,
                generated_text=generated_text,
                validator=validator,
            )
            if forced_literal is not None:
                return forced_literal

        forced_literal = self._force_complete_string(
            value_type=value_type,
            generated_text=generated_text,
            validator=validator,
        )
        if forced_literal is not None:
            return forced_literal
        raise ConstrainedDecodingError(
            "Max tokens reached while decoding "
            f"'{parameter_name}' ({value_type})"
        )

    def _should_stop(
        self,
        *,
        context_ids: list[int],
        generated_ids: list[int],
        generated_text: str,
        candidate_pool: tuple[int, ...],
        validator: LiteralValidator,
    ) -> bool:
        next_logits = self._model.get_logits_from_input_ids(
            context_ids + generated_ids
        )
        next_allowed = self._allowed_ids(
            generated_text,
            candidate_pool,
            validator,
        )
        if not next_allowed:
            return True
        best_next = int(np.argmax(next_logits))
        return best_next not in next_allowed

    def _allowed_ids(
        self,
        current_text: str,
        candidate_pool: tuple[int, ...],
        validator: LiteralValidator,
    ) -> list[int]:
        allowed: list[int] = []
        for token_id in candidate_pool:
            piece = self._piece_by_id[token_id]
            if piece == "":
                continue
            candidate = current_text + piece
            if validator.is_valid_prefix(candidate):
                allowed.append(token_id)
        return allowed

    @staticmethod
    def _select_best_token(logits: list[float], allowed_ids: list[int]) -> int:
        best_id = -1
        best_logit = -math.inf
        for token_id in allowed_ids:
            if token_id >= len(logits):
                continue
            score = logits[token_id]
            if score > best_logit:
                best_logit = score
                best_id = token_id
        if best_id < 0:
            raise ConstrainedDecodingError("Allowed set has no logits overlap")
        return best_id

    def _collect_pool(
        self,
        validator: LiteralValidator,
    ) -> tuple[int, ...]:
        out = [
            token_id
            for token_id, piece in self._piece_by_id.items()
            if validator.allows_token_piece(piece)
        ]
        return tuple(sorted(out))

    def _force_complete_string(
        self,
        *,
        value_type: str,
        generated_text: str,
        validator: LiteralValidator,
    ) -> str | None:
        if value_type != "string":
            return None
        if len(generated_text) < self._max_string_literal_length:
            return None
        candidate = generated_text + '"'
        if validator.is_complete(candidate):
            return candidate
        return None
