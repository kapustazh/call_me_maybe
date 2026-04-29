from __future__ import annotations

from src.io_utils import (
    load_function_definitions,
    load_prompt_items,
    write_function_results,
)
from src.function_selector import FunctionSelector
from src.models import FunctionResult


class Pipeline:
    def __init__(
        self,
        functions_path: str,
        input_path: str,
        output_path: str,
        model: str = "",
        selection_confidence_threshold: float = 0.90,
    ) -> None:
        self.functions_path: str = functions_path
        self.input_path: str = input_path
        self.output_path: str = output_path
        self._model_name: str = model
        self._selection_confidence_threshold = selection_confidence_threshold

    def run(self) -> None:
        """
        Build one function-call result per prompt.
        """
        prompt_items = load_prompt_items(self.input_path)
        function_definitions = load_function_definitions(self.functions_path)

        from llm_sdk import Small_LLM_Model  # type: ignore

        model = (
            Small_LLM_Model(model_name=self._model_name)
            if self._model_name
            else Small_LLM_Model()
        )
        selector = FunctionSelector(
            model,
            function_definitions,
            confidence_threshold=self._selection_confidence_threshold,
        )

        out: list[FunctionResult] = []
        for item in prompt_items:
            selection = selector.select(item.prompt)
            result = FunctionResult(
                prompt=item.prompt,
                name=selection,
                parameters={},
            )
            out.append(result)

        write_function_results(self.output_path, out)
