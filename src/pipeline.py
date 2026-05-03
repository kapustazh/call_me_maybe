from __future__ import annotations

import sys
from collections.abc import Callable

from llm_sdk import Small_LLM_Model  # type: ignore

from src.constrained_decoder import ConstrainedDecoder
from src.function_selector import (
    DEFAULT_SELECTION_CONFIDENCE,
    FunctionSelector,
    FunctionSelectorError,
)
from src.io_utils import (
    load_function_definitions,
    load_prompt_items,
    write_function_results,
)
from src.json_literal_validators import ConstrainedDecodingError
from src.models import FunctionResult
from src.tokenizer_vocab import TokenizerVocab


class Pipeline:
    def __init__(
        self,
        functions_path: str,
        input_path: str,
        output_path: str,
        model: str = "",
        selection_confidence_threshold: float = DEFAULT_SELECTION_CONFIDENCE,
        model_factory: Callable[[str], Small_LLM_Model] | None = None,
    ) -> None:
        self.functions_path: str = functions_path
        self.input_path: str = input_path
        self.output_path: str = output_path
        self._model_name: str = model
        self._selection_confidence_threshold = selection_confidence_threshold
        self._model_factory = model_factory

    def run(self) -> None:
        """Build one function-call result per prompt."""
        prompt_items = load_prompt_items(self.input_path)
        function_definitions = load_function_definitions(self.functions_path)
        function_by_name = {
            function_definition.name: function_definition
            for function_definition in function_definitions
        }

        model = self._build_model()
        tokenizer_vocab = TokenizerVocab.from_model(model)
        selector = FunctionSelector(
            model,
            function_definitions,
            confidence_threshold=self._selection_confidence_threshold,
        )
        decoder = ConstrainedDecoder(
            model,
            tokenizer_vocab,
            function_definitions,
        )

        out: list[FunctionResult] = []
        for item in prompt_items:
            try:
                selected_name = selector.select(item.prompt)
                function_definition = function_by_name.get(selected_name)
                if function_definition is None:
                    raise ValueError(
                        f"Selected unknown function name: {selected_name}"
                    )
                parameters = decoder.decode_parameters(
                    item.prompt,
                    function_definition,
                )
                result = FunctionResult(
                    prompt=item.prompt,
                    name=selected_name,
                    parameters=parameters,
                )
                out.append(result)
            except (
                FunctionSelectorError,
                ConstrainedDecodingError,
                ValueError,
            ) as exc:
                # Emit error row to preserve 1:1 mapping and aid debugging
                err_msg = str(exc)
                print(f"Error processing prompt: {err_msg}", file=sys.stderr)
                out.append(
                    FunctionResult(
                        prompt=item.prompt,
                        name="__error__",
                        parameters={"error": err_msg},
                    )
                )

        write_function_results(self.output_path, out)

    def _build_model(self) -> Small_LLM_Model:
        if self._model_factory is not None:
            return self._model_factory(self._model_name)
        if self._model_name:
            return Small_LLM_Model(model_name=self._model_name)
        return Small_LLM_Model()
