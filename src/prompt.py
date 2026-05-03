from __future__ import annotations

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
        """Shared text prefix of all function names (character-level LCP)."""
        return Prefix.longest_common_prefix(
            [fn.name for fn in self._functions],
        )

    def build_selection_prompt(self, user_prompt: str) -> str:
        fn_lines = "\n".join(
            f"- {fn.name}: {fn.description}" for fn in self._functions
        )
        _ = self.function_name_prefix()
        return (
            "Select the correct function name for the request.\n\n"
            f"Available functions:\n{fn_lines}\n\n"
            f'Request: "{user_prompt}"\n\n'
            f"The correct function is: fn_"
        )

    def build_parameter_prompt(
        self,
        user_prompt: str,
        function_definition: FunctionDefinition,
        parameter_name: str,
    ) -> str:
        parameter_spec = function_definition.parameters[parameter_name]
        return (
            "You are strict JSON value extractor for function calling.\n"
            "Return only JSON literal value for requested parameter.\n\n"
            f"User prompt:\n{user_prompt}\n\n"
            f"Function name: {function_definition.name}\n"
            f"Function description: {function_definition.description}\n"
            f"Parameter: {parameter_name}\n"
            f"Expected JSON type: {parameter_spec.type}\n\n"
            "JSON literal:"
        )
