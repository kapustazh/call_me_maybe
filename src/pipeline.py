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
from src.render import LogColorPair, PipelineUIRenderer
from src.tokenizer_vocab import TokenizerVocab

# ANSI SGR (only when target stream is a TTY).
_SGR_CYAN = "\033[36m"
_SGR_GREEN = "\033[32m"
_SGR_RED = "\033[31m"
_SGR_RESET = "\033[0m"


class PipelineNoResultsError(RuntimeError):
    """Raised when every prompt failed selection or constrained decoding.

    No rows are written to the output path in this case.
    """


def _pipeline_no_results_in_chain(
    exc: BaseException,
) -> PipelineNoResultsError | None:
    """Return ``PipelineNoResultsError`` if it appears in ``exc``'s chain."""
    from textual.worker import WorkerFailed

    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        cid = id(cur)
        if cid in seen:
            continue
        seen.add(cid)
        if isinstance(cur, PipelineNoResultsError):
            return cur
        if isinstance(cur, WorkerFailed):
            stack.append(cur.error)
            continue
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)
    return None


def _stream_ok_answer_line(
    renderer: PipelineUIRenderer | None,
    ok_line: str,
) -> None:
    """Append success line char-by-char.

    TUI path uses ``log_token_visual``.
    """
    text = ok_line + "\n"
    if renderer is not None:
        for ch in text:
            renderer.log_token_visual(ch, pair=LogColorPair.OK)
        return
    if sys.stdout.isatty():
        sys.stdout.write(_SGR_GREEN)
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
        sys.stdout.write(_SGR_RESET)
    else:
        print(text, end="")


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

        Raises:
            PipelineNoResultsError: If no prompt produced a successful result
                and the run used plain output (no TUI renderer). The interactive
                path logs a summary and waits for quit instead of raising.
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
                    renderer.log_stream(
                        header + "\n",
                        pair=LogColorPair.INFO,
                    )
                else:
                    if sys.stdout.isatty():
                        print(
                            f"{_SGR_CYAN}{header}{_SGR_RESET}",
                        )
                    else:
                        print(header)
                try:
                    selected_name = selector.select(item.prompt)
                    function_definition = function_by_name.get(
                        selected_name,
                    )
                    if function_definition is None:
                        raise ValueError(
                            "Selected unknown function name: "
                            f"{selected_name}"
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
                    _stream_ok_answer_line(renderer, ok_line)
                except (
                    FunctionSelectorError,
                    ConstrainedDecodingError,
                    ValueError,
                ) as exc:
                    err_line = f"  \u2717 Skipped: {exc}"
                    if renderer is not None:
                        renderer.log_stream(
                            err_line + "\n",
                            pair=LogColorPair.ERR,
                        )
                    else:
                        if sys.stderr.isatty():
                            print(
                                f"{_SGR_RED}{err_line}{_SGR_RESET}",
                                file=sys.stderr,
                            )
                        else:
                            print(err_line, file=sys.stderr)
                    skipped_count += 1
                    continue

            if not out:
                msg = (
                    f"0/{total} OK; {skipped_count} skipped; "
                    f"output not written — nothing written to {self.output_path}."
                )
                if renderer is not None:
                    renderer.log_stream(msg + "\n", pair=LogColorPair.ERR)
                    renderer.log_stream(
                        "\n[q] or [Esc] when done to quit\n",
                        pair=LogColorPair.PLAIN,
                    )
                    renderer.wait_until_quit()
                    return
                else:
                    if sys.stderr.isatty():
                        print(
                            f"{_SGR_RED}{msg}{_SGR_RESET}",
                            file=sys.stderr,
                        )
                    else:
                        print(msg, file=sys.stderr)
                    raise PipelineNoResultsError(msg)

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
                renderer.log_stream(done_msg, pair=LogColorPair.PLAIN)
                if skipped_count:
                    renderer.log_stream(skip_msg, pair=LogColorPair.ERR)
                renderer.log_stream(
                    "\n[q] or [Esc] when done to quit\n",
                    pair=LogColorPair.PLAIN,
                )
                renderer.wait_until_quit()
            else:
                print(done_msg, end="")
                if skipped_count:
                    print(skip_msg, end="", file=sys.stderr)

        try:
            PipelineUIRenderer.run_interactive(_execute)
        except Exception as exc:
            buried = _pipeline_no_results_in_chain(exc)
            if buried is not None:
                raise buried from exc
            print(
                f"TUI unavailable, falling back to plain output: {exc}",
                file=sys.stderr,
            )
            _execute(None)
