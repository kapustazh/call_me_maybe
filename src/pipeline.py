from __future__ import annotations

import json
import sys
import time
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore

from src.constrained_decoder import (
    ConstrainedDecoder,
    ConstrainedDecodingError,
)
from src.function_selector import (
    FunctionSelector,
    FunctionSelectorError,
)
from src.io_utils import (
    load_function_definitions,
    load_prompt_items,
    write_function_results,
)
from src.models import FunctionDefinition, FunctionResult
from src.tokenizer_vocab import TokenizerVocab

_WRIER_DELAY = 0.03

_DEBUG_FILE = f"{__file__}_{time.time()}.txt"


def _write_char_by_char(text: str, *, delay: float = _WRIER_DELAY) -> None:
    """Write text char-by-char to stdout for visual feedback."""
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    with open(_DEBUG_FILE, "a") as f:
        f.write(text + "\n")


def _format_params(parameters: dict[str, Any]) -> str:
    """Compact JSON repr of parameters for display."""
    return json.dumps(parameters, ensure_ascii=False)


class Pipeline:
    def __init__(
        self,
        functions_path: str,
        input_path: str,
        output_path: str,
        model_name: str = "",
        selection_confidence_threshold: float | None = None,
        selection_peak_softmax_target: float | None = None,
    ) -> None:
        self.functions_path: str = functions_path
        self.input_path: str = input_path
        self.output_path: str = output_path
        self._model_name: str = model_name
        self._selection_confidence_threshold = selection_confidence_threshold
        self._selection_peak_softmax_target = selection_peak_softmax_target

    @staticmethod
    def _deduplicate_definitions(
        definitions: list[FunctionDefinition],
    ) -> list[FunctionDefinition]:
        """Keep first occurrence of each function name."""
        seen: set[str] = set()
        unique: list[FunctionDefinition] = []
        for fd in definitions:
            if fd.name not in seen:
                seen.add(fd.name)
                unique.append(fd)
        return unique

    def run(self) -> None:
        """Build function-call results and skip invalid prompts."""
        prompt_items = load_prompt_items(self.input_path)
        function_definitions = self._deduplicate_definitions(
            load_function_definitions(self.functions_path)
        )
        function_by_name = {fd.name: fd for fd in function_definitions}

        model = (
            Small_LLM_Model(self._model_name)
            if self._model_name.strip()
            else Small_LLM_Model()
        )
        tokenizer_vocab = TokenizerVocab.from_model(model)
        selector_kwargs: dict[str, float] = {}
        if self._selection_confidence_threshold is not None:
            selector_kwargs["confidence_threshold"] = (
                self._selection_confidence_threshold
            )
        if self._selection_peak_softmax_target is not None:
            selector_kwargs["peak_softmax_target"] = (
                self._selection_peak_softmax_target
            )
        selector = FunctionSelector(
            model,
            function_definitions,
            **selector_kwargs,
        )
        decoder = ConstrainedDecoder(
            model,
            tokenizer_vocab,
            function_definitions,
        )

        total = len(prompt_items)
        out: list[FunctionResult] = []
        skipped_count: int = 0
        for idx, item in enumerate(prompt_items, 1):
            _write_char_by_char(f"[{idx}/{total}] Processing: {item.prompt}\n")
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
                _write_char_by_char(
                    f"  \u2192 {selected_name}"  # arrow symbol
                    f"({_format_params(result.parameters)})\n",
                )
            except (
                FunctionSelectorError,
                ConstrainedDecodingError,
                ValueError,
            ) as exc:
                print(
                    f"  \u2717 Skipped: {exc}", file=sys.stderr
                )  # cross symbol
                skipped_count += 1
                continue

        write_function_results(self.output_path, out)
        print(
            f"\nDone. {len(out)}/{total} results written"
            f" to {self.output_path}",
        )
        if skipped_count:
            print(
                (
                    f"Skipped {skipped_count} prompt(s) due to "
                    "routing/decoding errors"
                ),
                file=sys.stderr,
            )
