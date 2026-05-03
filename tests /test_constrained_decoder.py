from typing import cast

from llm_sdk import Small_LLM_Model  # type: ignore

from src.constrained_decoder import ConstrainedDecoder
from src.models import FunctionDefinition, FunctionParameter
from src.tokenizer_vocab import TokenizerVocab


class FakeDecoderModel:
    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(token_id) for token_id in input_ids)
        marker = "JSON literal:"
        marker_pos = text.rfind(marker)
        generated = "" if marker_pos < 0 else text[marker_pos + len(marker) :]

        if "Function name: fn_add_numbers" in text:
            if "Parameter: a" in text:
                target = "2"
            elif "Parameter: b" in text:
                target = "3"
            else:
                target = "{}"
        elif (
            "Function name: fn_reverse_string" in text
            and "Parameter: s" in text
        ):
            target = '"hello"'
        elif "Function name: fn_substitute_string_with_regex" in text:
            if "Parameter: source_string" in text:
                target = '"A with B 12"'
            elif "Parameter: regex" in text:
                target = '"\\\\d+"'
            elif "Parameter: replacement" in text:
                target = '"NUMBERS"'
            else:
                target = "{}"
        elif "Parameter: a" in text:
            target = "12.5"
        elif "Parameter: name" in text:
            target = '"Ada"'
        elif "Parameter: enabled" in text:
            target = "true"
        else:
            target = "{}"

        next_char = (
            target[len(generated)] if len(generated) < len(target) else " "
        )
        logits = [-1000.0] * 256
        logits[ord(next_char)] = 1000.0
        return logits

    def get_path_to_tokenizer_file(self) -> str:
        return "unused"

    def get_path_to_vocab_file(self) -> str:
        return "unused"


def test_decodes_typed_parameters() -> None:
    model = cast(Small_LLM_Model, FakeDecoderModel())
    token_map = {chr(code): code for code in range(32, 127)}
    vocab = TokenizerVocab(token_map, model=model)
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
        model=model,
        tokenizer_vocab=vocab,
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        "Use demo with Ada and enable flag", function_definition
    )

    assert parameters == {"a": 12.5, "name": "Ada", "enabled": True}


def test_extracts_sum_numbers_from_prompt() -> None:
    model = cast(Small_LLM_Model, FakeDecoderModel())
    token_map = {chr(code): code for code in range(32, 127)}
    vocab = TokenizerVocab(token_map, model=model)
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
        model=model,
        tokenizer_vocab=vocab,
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        "What is the sum of 2 and 3?",
        function_definition,
    )

    assert parameters == {"a": 2.0, "b": 3.0}


def test_extracts_reverse_string_from_prompt() -> None:
    model = cast(Small_LLM_Model, FakeDecoderModel())
    token_map = {chr(code): code for code in range(32, 127)}
    vocab = TokenizerVocab(token_map, model=model)
    function_definition = FunctionDefinition(
        name="fn_reverse_string",
        description="Reverse a string and return the reversed result.",
        parameters={"s": FunctionParameter(type="string")},
        returns=FunctionParameter(type="string"),
    )
    decoder = ConstrainedDecoder(
        model=model,
        tokenizer_vocab=vocab,
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        "Reverse the string 'hello'",
        function_definition,
    )

    assert parameters == {"s": "hello"}


def test_extracts_regex_replacement_with_inner_with() -> None:
    model = cast(Small_LLM_Model, FakeDecoderModel())
    token_map = {chr(code): code for code in range(32, 127)}
    vocab = TokenizerVocab(token_map, model=model)
    function_definition = FunctionDefinition(
        name="fn_substitute_string_with_regex",
        description="Replace all occurrences matching a regex pattern in a string.",
        parameters={
            "source_string": FunctionParameter(type="string"),
            "regex": FunctionParameter(type="string"),
            "replacement": FunctionParameter(type="string"),
        },
        returns=FunctionParameter(type="string"),
    )
    decoder = ConstrainedDecoder(
        model=model,
        tokenizer_vocab=vocab,
        functions=[function_definition],
    )

    parameters = decoder.decode_parameters(
        'Replace all numbers in "A with B 12" with NUMBERS',
        function_definition,
    )

    assert parameters == {
        "source_string": "A with B 12",
        "regex": r"\d+",
        "replacement": "NUMBERS",
    }
