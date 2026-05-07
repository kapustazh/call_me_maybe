from __future__ import annotations

import json
from typing import Any

from src.models import FunctionDefinition


class Prefix:
    """Longest common prefix of function names"""

    @staticmethod
    def longest_common_prefix(strs: list[str]) -> str:
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
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self._functions = functions

    def function_name_prefix(self) -> str:
        """Shared prefix used in selection prompt.

        Keep it tokenization-friendly for Qwen function names: all current
        tools start with ``fn_...`` but continuation tokens are aligned after
        ``fn`` (e.g. ``_add``, ``_get``), not after ``fn_``.
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
