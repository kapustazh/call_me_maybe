from typing import cast

import re

from llm_sdk import Small_LLM_Model  # type: ignore

from src.constrained_decoder import ConstrainedDecoder, ConstrainedDecodingError
from src.models import FunctionDefinition, FunctionParameter
from src.tokenizer_vocab import TokenizerVocab


class FakeDecoderModel:
    """Tiny LM: ascii tokens only; drives logits from decoded prefix."""

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(i) for i in input_ids if 0 <= i < 256)

        logits = [-1000.0] * 256

        def emit(ch: str) -> list[float]:
            out = [-1000.0] * 256
            out[ord(ch)] = 1000.0
            return out

        if "fn_demo" in text:
            m = re.search(r'"a":\s*([0-9.]*)$', text)
            if m:
                target = "12.5"
                partial = m.group(1)
                if len(partial) < len(target):
                    return emit(target[len(partial)])
            m = re.search(r'"name": "([^"]*)$', text)
            if m:
                target = "Ada"
                partial = m.group(1)
                if len(partial) < len(target):
                    return emit(target[len(partial)])
                return emit('"')
            if '"enabled": ' in text:
                suf = text.rsplit('"enabled": ', 1)[-1]
                if suf == "":
                    self._n_empty = getattr(self, "_n_empty", 0) + 1
                    return emit("t" if self._n_empty == 1 else "f")
                for word in ("true", "false"):
                    if word.startswith(suf) and suf != word:
                        return emit(word[len(suf)])

        if "fn_add_numbers" in text:
            m = re.search(r'"a":\s*([0-9.]*)$', text)
            if m:
                target = "2"
                partial = m.group(1)
                if len(partial) < len(target):
                    return emit(target[len(partial)])
            m = re.search(r'"b":\s*([0-9.]*)$', text)
            if m:
                target = "3"
                partial = m.group(1)
                if len(partial) < len(target):
                    return emit(target[len(partial)])

        if "fn_reverse_string" in text or "fn_greet" in text:
            m = re.search(r'"(?:s|name)": "([^"]*)$', text)
            if m:
                target = "hello" if "fn_reverse_string" in text else "john"
                partial = m.group(1)
                if len(partial) < len(target):
                    return emit(target[len(partial)])
                return emit('"')

        return logits

    def get_path_to_tokenizer_file(self) -> str:
        return "unused"

    def get_path_to_vocab_file(self) -> str:
        return "unused"


def _vocab(model: FakeDecoderModel) -> TokenizerVocab:
    token_map = {chr(code): code for code in range(32, 127)}
    return TokenizerVocab(token_map, model=cast(Small_LLM_Model, model))


def test_decodes_typed_parameters() -> None:
    model = FakeDecoderModel()
    function_definition = FunctionDefinition(
        name="fn_demo",
        description="Demo fn",
        parameters={
            "a": FunctionParameter(type="number"),
            "name": FunctionParameter(type="string"),
            "enabled": FunctionParameter(type="boolean"),
        },
        returns=FunctionParameter(type="object"),
    )
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, model),
        tokenizer_vocab=_vocab(model),
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        "Use demo with Ada and enable flag", function_definition
    )

    assert parameters == {"a": 12.5, "name": "Ada", "enabled": True}


def test_decodes_numbers_via_mask() -> None:
    model = FakeDecoderModel()
    function_definition = FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={
            "a": FunctionParameter(type="number"),
            "b": FunctionParameter(type="number"),
        },
        returns=FunctionParameter(type="number"),
    )
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, model),
        tokenizer_vocab=_vocab(model),
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        "What is the sum of 2 and 3?",
        function_definition,
    )

    assert parameters == {"a": 2.0, "b": 3.0}


def test_decodes_string_via_mask() -> None:
    model = FakeDecoderModel()
    function_definition = FunctionDefinition(
        name="fn_reverse_string",
        description="Reverse a string and return the reversed result.",
        parameters={"s": FunctionParameter(type="string")},
        returns=FunctionParameter(type="string"),
    )
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, model),
        tokenizer_vocab=_vocab(model),
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        "Reverse the string 'hello'",
        function_definition,
    )

    assert parameters == {"s": "hello"}


def test_decodes_plain_string_via_mask() -> None:
    model = FakeDecoderModel()
    function_definition = FunctionDefinition(
        name="fn_greet",
        description="Generate a greeting message for a person by name.",
        parameters={"name": FunctionParameter(type="string")},
        returns=FunctionParameter(type="string"),
    )
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, model),
        tokenizer_vocab=_vocab(model),
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters("Greet john", function_definition)

    assert parameters == {"name": "john"}


def test_number_decode_failure_raises() -> None:
    model = FakeDecoderModel()
    function_definition = FunctionDefinition(
        name="fn_unknown_numbers",
        description="Unknown",
        parameters={"a": FunctionParameter(type="integer")},
        returns=FunctionParameter(type="integer"),
    )
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, model),
        tokenizer_vocab=_vocab(model),
        functions=[function_definition],
        max_new_tokens_number=3,
    )

    try:
        _ = decoder.decode_parameters("no digits here", function_definition)
    except ConstrainedDecodingError as exc:
        assert "Failed to decode" in str(exc) or "Invalid" in str(exc)
    else:
        raise AssertionError("Expected ConstrainedDecodingError")
