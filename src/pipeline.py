from __future__ import annotations

from src.io_utils import (
    load_function_definitions,
    load_prompt_items,
    write_json,
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
    ) -> None:
        self.functions_path: str = functions_path
        self.input_path: str = input_path
        self.output_path: str = output_path
        self._model_name: str = model

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
            confidence_threshold=0.0,
        )

        out: list[dict[str, object]] = []
        for item in prompt_items:
            selection = selector.select(item.prompt)
            result = FunctionResult(
                prompt=item.prompt,
                name=selection.name,
                parameters={},
            )
            out.append(result.model_dump())

        write_json(self.output_path, out)
