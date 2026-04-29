from __future__ import annotations

from src.models import FunctionDefinition


class Prefix:
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
        names = [fn.name for fn in self._functions]
        return Prefix.longest_common_prefix(names)

    def build_selection_prompt(self, user_prompt: str) -> str:
        prefix = self.function_name_prefix()
        fn_lines = "\n".join(
            f"- {fn.name}: {fn.description}" for fn in self._functions
        )
        return (
            "You are a function calling router.\n"
            "Pick best function for user request.\n\n"
            f"User request:\n{user_prompt}\n\n"
            "Available functions:\n"
            f"{fn_lines}\n\n"
            "Return only function name.\n"
            f"{prefix}"
        )
