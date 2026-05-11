from __future__ import annotations

import json
from typing import Any

from src.models import FunctionDefinition


class Prefix:
    """Utilities for longest-prefix alignment across function name strings.

    Used when building selection prompts so tokenizer continuations share a
    stable anchor (e.g. ``fn`` for ``fn_*`` names).

    Methods:
        longest_common_prefix: LC-prefix via sorted first/last scan.
    """

    @staticmethod
    def longest_common_prefix(strs: list[str]) -> str:
        """Compute longest common prefix across strings.

        Args:
            strs: Input strings.

        Returns:
            Longest common prefix. Empty string if "strs" empty.
        """
        if not strs:
            return ""
        sorted_strs = sorted(strs)
        first = sorted_strs[0]
        last = sorted_strs[-1]
        i = 0
        while i < len(first) and i < len(last) and first[i] == last[i]:
            i += 1

        return first[:i]


class BobThePrompter:
    """Build selection/decode prompts for function calling tasks.

    Centralizes prompt formatting so selector and decoder stay aligned.
    """

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Create prompter for a fixed function list.

        Args:
            functions: Function definitions exposed to the model.
        """
        self._functions = functions

    def function_name_prefix(self) -> str:
        """Return shared prefix where the selection prompt stops before branch.

        Tokenization-friendly for ``fn_*`` tools: aligns continuation tokens
        after ``fn`` when all names share that prefix shape.

        Returns:
            Longest shared prefix string used in selection prompt tail, or
            empty string when undetermined.
        """
        names = [fn.name for fn in self._functions]
        if not names:
            return ""
        if all(name.startswith("fn") for name in names):
            return "fn"
        if len(names) <= 1:
            return ""
        return Prefix.longest_common_prefix(names)

    def build_selection_prompt(self, user_prompt: str) -> str:
        """Build prompt asking model to select function name.

        Args:
            user_prompt: Raw user request.

        Returns:
            Full selection prompt string.
        """
        prefix = self.function_name_prefix()
        fn_lines = "\n".join(
            f"- {fn.name}: {fn.description}" for fn in self._functions
        )
        return (
            "Select the correct function name for the request.\n\n"
            f"Available functions:\n{fn_lines}\n\n"
            f'Request: "{user_prompt}"\n\n'
            f"The correct function is: {prefix}"
        )

    def build_decode_prompt(
        self,
        user_prompt: str,
        chosen_fn: FunctionDefinition,
    ) -> str:
        """Build prompt for argument extraction for chosen function.

        Args:
            user_prompt: Raw user request.
            chosen_fn: Selected function schema.

        Returns:
            Prompt string used to condition constrained decoding.
        """
        example_params: dict[str, Any] = {}
        for param_name, param_def in chosen_fn.parameters.items():
            if param_def.type in ("number", "integer"):
                example_params[param_name] = 10.0
            elif param_def.type == "boolean":
                example_params[param_name] = False
            elif param_def.type == "object":
                example_params[param_name] = {}
            else:
                example_params[param_name] = param_name

        example = json.dumps(
            {"name": chosen_fn.name, "parameters": example_params},
            ensure_ascii=False,
        )
        return (
            "You are parameters extraction assistant from text. "
            "Extract only literal values from the request. "
            f"Description: {chosen_fn.description}\n"
            f"Example: {example}\n\n"
            f"Task: {user_prompt}\n"
        )
