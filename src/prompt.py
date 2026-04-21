"""Prompt builder for the LLM decision call.

The prompt only guides the model; correctness is enforced downstream by
the constrained decoder. No "prompt and pray".
"""

from __future__ import annotations

from src.models import FunctionDefinition


def build_prompt(
    fn_defs: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Return an instruction string listing functions + the user query."""
    ...


def _format_function(fn: FunctionDefinition) -> str:
    """Render a single function definition as a compact text block."""
    ...


def _format_parameters(fn: FunctionDefinition) -> str:
    """Render a function's parameter list (name: type) for the prompt."""
    ...
