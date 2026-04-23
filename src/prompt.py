from __future__ import annotations

from src.models import FunctionDefinition


class Prefix:
    @staticmethod
    def longest_common(strings: list[str]) -> str:
        """
        Return longest common prefix shared by all strings.

        Empty input -> "".
        """
        if not strings:
            return ""
        pref = strings[0]
        for s in strings[1:]:
            i = 0
            n = min(len(pref), len(s))
            while i < n and pref[i] == s[i]:
                i += 1
            pref = pref[:i]
            if pref == "":
                break
        return pref


class BobThePrompter:
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self._functions = functions

    def function_name_prefix(self) -> str:
        names = [fn.name for fn in self._functions]
        return Prefix.longest_common(names)

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
