from __future__ import annotations
from pydantic import BaseModel, ConfigDict, StringConstraints, Field
from typing import Any, Literal, Annotated

NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class PromptItem(BaseModel):
    """Single prompt record from input test file.

    Attributes:
        prompt: Natural-language request string. Non-empty after trimming.
    """

    prompt: NonEmptyStr


class FunctionParameter(BaseModel):
    """Parameter schema entry for a function definition.

    Attributes:
        type: JSON type name expected for this parameter.
    """

    type: Literal["number", "integer", "boolean", "string", "object"]


class FunctionDefinition(BaseModel):
    """Callable function schema used for routing and constrained decoding.

    Attributes:
        name: Function name exposed to the model (e.g. 'fn_add_numbers').
        description: Short natural-language description of purpose.
        parameters: Mapping of parameter name to its expected JSON type.
        returns: Return type metadata (not used for decoding, kept for parity).
    """

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    description: str
    parameters: dict[str, FunctionParameter] = Field(min_length=1)
    returns: FunctionParameter


class FunctionResult(BaseModel):
    """Output record for one successfully processed prompt.

    Attributes:
        prompt: Original prompt text.
        name: Selected function name.
        parameters: Extracted arguments matching function schema.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
