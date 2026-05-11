from __future__ import annotations

import json
import sys

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
from src.render import PipelineUIRenderer
from src.tokenizer_vocab import TokenizerVocab


class Pipeline:
    """End-to-end batch runner: load JSON, route prompts, decode parameters.

    Loads prompt tests and function definitions, runs :class:`FunctionSelector`
    and :class:`ConstrainedDecoder` per prompt, writes successful
    :class:`FunctionResult` rows. Uses :class:`PipelineUIRenderer` when TTY
    available; otherwise prints to stdout/stderr.

    Attributes:
        functions_path: Path to function definitions JSON.
        input_path: Path to prompt tests JSON.
        output_path: Path for results JSON output.
    """

    def __init__(
        self,
        functions_path: str,
        input_path: str,
        output_path: str,
        model_name: str,
    ) -> None:
        """Create end-to-end generation pipeline.

        Args:
            functions_path: Path to function definitions JSON.
            input_path: Path to prompt tests JSON.
            output_path: Path to write results JSON.
            model_name: Hugging Face model id (empty string uses SDK default).
        """
        self.functions_path: str = functions_path
        self.input_path: str = input_path
        self.output_path: str = output_path
        self._model_name: str = model_name.strip()

    @staticmethod
    def _deduplicate_definitions(
        definitions: list[FunctionDefinition],
    ) -> list[FunctionDefinition]:
        """Keep first occurrence of each function name.

        Later duplicates with the same ``name`` are dropped so probability mass
        is not split across identical tools.

        Args:
            definitions: Parsed function definitions (possibly with repeats).

        Returns:
            Ordered list with unique ``name`` values.
        """
        seen: set[str] = set()
        unique: list[FunctionDefinition] = []
        for fd in definitions:
            if fd.name not in seen:
                seen.add(fd.name)
                unique.append(fd)
        return unique

    def run(self) -> None:
        """Run selection + constrained decoding over all prompts.

        Writes only successful results to output file.
        Prompts that fail selection or decoding are skipped (stderr).
        """
        prompt_items = load_prompt_items(self.input_path)
        function_definitions = self._deduplicate_definitions(
            load_function_definitions(self.functions_path),
        )
        function_by_name = {fd.name: fd for fd in function_definitions}

        model = (
            Small_LLM_Model(self._model_name)
            if self._model_name
            else Small_LLM_Model()
        )
        tokenizer_vocab = TokenizerVocab.from_model(model)
        selector = FunctionSelector(
            model,
            function_definitions,
        )
        decoder = ConstrainedDecoder(
            model,
            tokenizer_vocab,
            function_definitions,
        )

        total = len(prompt_items)

        def _execute(renderer: PipelineUIRenderer | None) -> None:
            """Process all prompts; stream logs and write results.

            Args:
                renderer: Interactive UI logger, or ``None`` for plain print.
            """
            out: list[FunctionResult] = []
            skipped_count: int = 0

            for idx, item in enumerate(prompt_items, 1):
                header = f"[{idx}/{total}] Processing: {item.prompt}"
                if renderer is not None:
                    renderer.log_info_stream(header + "\n")
                else:
                    print(header)
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
                    params_json = json.dumps(
                        result.parameters,
                        ensure_ascii=False,
                    )
                    ok_line = f"  \u2192 {selected_name}({params_json})"
                    if renderer is not None:
                        renderer.log_ok_stream(ok_line + "\n")
                    else:
                        print(ok_line)
                except (
                    FunctionSelectorError,
                    ConstrainedDecodingError,
                    ValueError,
                ) as exc:
                    err_line = f"  \u2717 Skipped: {exc}"
                    if renderer is not None:
                        renderer.log_err_stream(err_line + "\n")
                    else:
                        print(err_line, file=sys.stderr)
                    skipped_count += 1
                    continue

            if not out:
                print(
                    "Warning: no successful results — output file "
                    "not written.",
                    file=sys.stderr,
                )
            else:
                write_function_results(self.output_path, out)

            done_msg = (
                f"\nDone. {len(out)}/{total} results written"
                f" to {self.output_path}\n"
            )
            skip_msg = (
                f"Skipped {skipped_count} prompt(s) due to "
                "routing/decoding errors\n"
            )
            if renderer is not None:
                renderer.log_info_stream(done_msg)
                if skipped_count:
                    renderer.log_err_stream(skip_msg)
                renderer.log_info_stream("\n[q] or [Esc] when done to quit\n")
                renderer.wait_until_quit()
            else:
                print(done_msg, end="")
                if skipped_count:
                    print(skip_msg, end="", file=sys.stderr)

        try:
            PipelineUIRenderer.run_interactive(_execute)
        except Exception as exc:
            print(
                f"TUI unavailable, falling back to plain output: {exc}",
                file=sys.stderr,
            )
            _execute(None)
