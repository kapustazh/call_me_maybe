from __future__ import annotations
from pydantic import BaseModel, StringConstraints, Field, RootModel
from typing import Any, Literal, Annotated

NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class PromptItem(BaseModel):
    prompt: NonEmptyStr


class FunctionParameter(BaseModel):
    type: Literal["number", "integer", "boolean", "string", "object"]


class FunctionDefinition(BaseModel):
    name: NonEmptyStr
    description: str
    parameters: dict[str, FunctionParameter] = Field(min_length=1)
    returns: FunctionParameter


class FunctionResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class FunctionDefinitionList(RootModel[list[FunctionDefinition]]):
    pass


class PromptItemList(RootModel[list[PromptItem]]):
    pass
