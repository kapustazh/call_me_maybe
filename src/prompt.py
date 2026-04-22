from __future__ import annotations

from src.models import FunctionDefinition


class BobThePrompter:
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self._functions = functions
