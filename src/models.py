from __future__ import annotations
from pydantic import BaseModel, ConfigDict, StringConstraints, Field
from typing import Any, Literal, Annotated

NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class PromptItem(BaseModel):
    prompt: NonEmptyStr


class FunctionParameter(BaseModel):
    type: Literal["number", "integer", "boolean", "string", "object"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr
    description: str
    parameters: dict[str, FunctionParameter] = Field(min_length=1)
    returns: FunctionParameter


class FunctionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
