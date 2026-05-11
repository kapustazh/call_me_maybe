from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDefinition, FunctionResult, PromptItem


class JsonFileError(ValueError):
    """Raised when a JSON file cannot be read or parsed.

    Attributes:
        args: Standard ``Exception`` tuple; message describes missing path,
            OS error, or JSON syntax location.
    """


class JsonValidationError(ValueError):
    """Raised when parsed JSON fails Pydantic schema validation.

    Attributes:
        args: Message includes file path and validation error details.
    """


def load_json_file(path: str | Path) -> Any:
    """Load and decode one JSON file.

    Args:
        path: Filesystem path to a ``.json`` file.

    Returns:
        Decoded Python object (typically ``dict`` or ``list``).

    Raises:
        JsonFileError: If the path is missing, not a file, empty, unreadable,
            or contains invalid JSON.
    """
    json_path = Path(path)

    if not json_path.exists():
        raise JsonFileError(f"JSON file not found: {json_path}")

    if not json_path.is_file():
        raise JsonFileError(f"Path is not a file: {json_path}")

    try:
        raw = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JsonFileError(f"Cannot read file '{json_path}': {exc}") from exc

    if raw.strip() == "":
        raise JsonFileError(f"File is empty: {json_path}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = (
            f"Invalid JSON in '{json_path}' at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        )
        raise JsonFileError(msg) from exc


def load_function_definitions(path: str | Path) -> list[FunctionDefinition]:
    """Load function definitions JSON and validate against schema.

    Args:
        path: Path to JSON array of function definition objects.

    Returns:
        Validated list of :class:`FunctionDefinition`.

    Raises:
        JsonFileError: If the file cannot be read or parsed.
        JsonValidationError: If structure does not match the schema.
    """
    data = load_json_file(path)
    try:
        return TypeAdapter(list[FunctionDefinition]).validate_python(data)
    except ValidationError as exc:
        raise JsonValidationError(
            f"Schema validation failed for function definitions file '{path}':"
            f" {exc}"
        ) from exc


def load_prompt_items(path: str | Path) -> list[PromptItem]:
    """Load prompt tests JSON and validate against schema.

    Args:
        path: Path to JSON array of prompt objects.

    Returns:
        Validated list of :class:`PromptItem`.

    Raises:
        JsonFileError: If the file cannot be read or parsed.
        JsonValidationError: If structure does not match the schema.
    """
    data = load_json_file(path)
    try:
        return TypeAdapter(list[PromptItem]).validate_python(data)
    except ValidationError as exc:
        raise JsonValidationError(
            f"Schema validation failed for prompt tests file '{path}': {exc}"
        ) from exc


def write_function_results(
    path: str | Path,
    results: list[FunctionResult],
) -> None:
    """Serialize results to UTF-8 JSON with indentation.

    Creates parent directories if needed. Each row is the dumped
    :class:`FunctionResult` (prompt, name, parameters).

    Args:
        path: Output file path.
        results: Successful pipeline rows to write.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_records = [result.model_dump() for result in results]
    out_path.write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
