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
        if self._mode == "prefer_square_small" and "greet" in text:
            logits[ord("a")] = 1.2
            logits[ord("g")] = 1.0
            return logits
        if self._mode == "biased_add":
            logits[ord("a")] = 10.0
            logits[ord("g")] = 1.0
            return logits
        if self._mode == "medium_add":
            logits[ord("a")] = 2.0
            logits[ord("g")] = 1.0
            return logits
        logits[ord("a")] = 1.0
        logits[ord("g")] = 1.0
        return logits

    def get_path_to_tokenizer_file(self) -> str:
        return "unused"

    def get_path_to_vocab_file(self) -> str:
        return "unused"


class FakePrefixTokenModel:
    def encode(self, text: str) -> list[list[int]]:
        token_map: dict[str, list[int]] = {
            "fn": [1],
            "fn_add_numbers": [1, 2],
            "add_numbers": [3],
            "fn_get_square_root": [1, 4],
            "get_square_root": [5],
            "fn_greet": [1, 6],
            "greet": [7],
        }
        if text in token_map:
            return [token_map[text]]
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(token_id) for token_id in input_ids).lower()
        logits = [-100.0] * 256
        if "sum" in text:
            logits[2] = 10.0
            logits[4] = 1.0
            return logits
        logits[2] = 1.0
        logits[4] = 10.0
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


def test_lexical_prior_can_override_small_logit_bias() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="prefer_square_small")),
        _definitions(),
        confidence_threshold=0.60,
    )
    selected = selector.select("Greet john")
    assert selected == "fn_greet"


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


def test_prefix_token_continuation_uses_fn_boundary() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakePrefixTokenModel()),
        _definitions(),
        confidence_threshold=0.60,
    )
    assert selector.select("What is sum of 2 and 3?") == "fn_add_numbers"


def test_rejects_medium_confidence_without_lexical_support() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="medium_add")),
        _definitions(),
        confidence_threshold=0.60,
    )
    try:
        _ = selector.select("What is purpose of live?")
    except FunctionSelectorError as exc:
        assert "no lexical support" in str(exc)
    else:
        raise AssertionError("Expected FunctionSelectorError")


def test_plus_synonym_provides_lexical_support_for_add() -> None:
    selector = FunctionSelector(
        cast(Small_LLM_Model, FakeSelectorModel(mode="medium_add")),
        _definitions(),
        confidence_threshold=0.60,
    )
    selected = selector.select("What is three plus -2 equal to?")
    assert selected == "fn_add_numbers"
