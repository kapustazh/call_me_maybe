"""Constrained decoding: JSON + schema state machine + masking loop.

Guarantees 100% valid, schema-compliant JSON output. Works token by
token using next-token logits from :class:`llm_sdk.Small_LLM_Model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition, FunctionResult
from src.vocab import Vocab


class FsmState(Enum):
    """High-level states of the output JSON schema FSM."""

    START = auto()
    KEY_PROMPT = auto()
    VAL_PROMPT = auto()
    KEY_NAME = auto()
    VAL_NAME = auto()
    KEY_PARAMS = auto()
    VAL_PARAMS = auto()
    END = auto()


@dataclass
class JsonSchemaFSM:
    """Deterministic state machine enforcing the output JSON shape.

    Tracks current state, chosen function (after VAL_NAME), parameter
    cursor and the raw output buffer generated so far.
    """

    fn_defs: list[FunctionDefinition]
    user_prompt: str
    state: FsmState = FsmState.START
    chosen_fn: FunctionDefinition | None = None
    param_cursor: int = 0
    buffer: str = ""

    def is_prefix_valid(self, piece: str) -> bool:
        """Return True if appending *piece* keeps buffer as valid prefix."""
        ...

    def advance(self, piece: str) -> None:
        """Commit *piece* to the buffer and update internal state."""
        ...

    def is_complete(self) -> bool:
        """Return True when buffer is a complete, schema-compliant JSON."""
        ...

    def forced_next_char(self) -> str | None:
        """Return the single legal next char if any (fast path, skip LLM)."""
        ...

    def _enter_state(self, new_state: FsmState) -> None:
        """Transition to *new_state* and reset any per-state counters."""
        ...

    def _expected_key(self) -> str | None:
        """Return the JSON key expected next, or None if inside a value."""
        ...


@dataclass
class ConstrainedDecoder:
    """Drive the LLM generation loop with token-level schema masking."""

    model: Small_LLM_Model
    vocab: Vocab
    max_tokens: int = 256

    def generate_call(
        self,
        fn_defs: list[FunctionDefinition],
        user_prompt: str,
    ) -> FunctionResult:
        """Produce a validated FunctionResult for a single user prompt."""
        ...

    def _mask_logits(
        self,
        logits: list[float],
        fsm: JsonSchemaFSM,
    ) -> list[float]:
        """Set logits of ids invalid under the current FSM state to -inf."""
        ...

    def _pick_next_id(self, masked_logits: list[float]) -> int:
        """Return argmax over masked logits (ties broken by lowest id)."""
        ...

    def _coerce_parameters(
        self,
        raw: dict[str, Any],
        fn: FunctionDefinition,
    ) -> dict[str, Any]:
        """Cast raw parsed JSON values to types declared in *fn*."""
        ...
