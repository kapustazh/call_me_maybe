from typing import cast

from llm_sdk import Small_LLM_Model  # type: ignore

from src.function_selector import FunctionSelector, FunctionSelectorError
from src.models import FunctionDefinition, FunctionParameter


class FakeSelectorModel:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(token_id) for token_id in input_ids).lower()
        logits = [-100.0] * 256
        if self._mode == "sum" and "sum" in text:
            logits[ord("a")] = 10.0
            logits[ord("g")] = 1.0
            return logits
        if self._mode == "greet" and "greet" in text:
            logits[ord("a")] = 1.0
            logits[ord("g")] = 10.0
            return logits
        if self._mode == "biased_add":
            logits[ord("a")] = 10.0
            logits[ord("g")] = 1.0
            return logits
        logits[ord("a")] = 1.0
        logits[ord("g")] = 1.0
        return logits

    def get_path_to_tokenizer_file(self) -> str:
        return "unused"

    def get_path_to_vocab_file(self) -> str:
        return "unused"


def _definitions() -> list[FunctionDefinition]:
    number = FunctionParameter(type="number")
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add numbers",
            parameters={"a": number, "b": number},
            returns=number,
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet by name",
            parameters={"name": FunctionParameter(type="string")},
            returns=FunctionParameter(type="string"),
        ),
    ]


def test_selects_best_name() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="sum")),
        _definitions(),
        confidence_threshold=0.60,
    )
    selected = selector.select("What is sum of 2 and 3?")
    assert selected == "fn_add_numbers"


def test_raises_on_low_confidence() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="ambiguous")),
        _definitions(),
        confidence_threshold=0.90,
    )
    try:
        _ = selector.select("do thing")
    except FunctionSelectorError as exc:
        assert "Low selection confidence" in str(exc)
    else:
        raise AssertionError("Expected FunctionSelectorError")


def test_model_selection_when_logits_favor_add() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="biased_add")),
        _definitions(),
        confidence_threshold=0.60,
    )
    selected = selector.select("Greet john")
    assert selected == "fn_add_numbers"


def test_single_function_does_not_crash() -> None:
    number = FunctionParameter(type="number")
    only = [
        FunctionDefinition(
            name="fn_strlen",
            description="String length",
            parameters={"s": FunctionParameter(type="string")},
            returns=number,
        ),
    ]
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="ambiguous")),
        only,
        confidence_threshold=0.01,
    )
    assert selector.select("How long is hello?") == "fn_strlen"
