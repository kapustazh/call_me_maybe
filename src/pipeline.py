from __future__ import annotations

from src.io_utils import (
    load_function_definitions,
    load_prompt_items,
    write_json,
)
from src.function_selector import (
    BobThePrompter,
    FunctionSelector,
    FunctionSelectorError,
)


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
        Test harness: build selection prompts + get model guess per prompt.
        """
        prompt_items = load_prompt_items(self.input_path)
        function_definitions = load_function_definitions(self.functions_path)
        bob = BobThePrompter(functions=function_definitions)

        selector: FunctionSelector | None = None
        selector_error: str | None = None
        try:
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
        except Exception as exc:
            selector_error = str(exc)

        out: list[dict[str, object]] = []
        for item in prompt_items:
            row: dict[str, object] = {
                "prompt": item.prompt,
                "selection_prompt": bob.build_selection_prompt(item.prompt),
            }
            if selector is None:
                if selector_error:
                    row["model_error"] = selector_error
            else:
                try:
                    sel = selector.select(item.prompt)
                    row["model_guess"] = sel.name
                    row["model_confidence"] = sel.confidence
                    row["function_scores"] = sel.scores
                except FunctionSelectorError as exc:
                    row["model_error"] = str(exc)
            out.append(row)

        write_json(self.output_path, out)
