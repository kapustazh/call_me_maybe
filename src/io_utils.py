from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from pydantic import TypeAdapter, ValidationError

from src.models import FunctionDefinition, FunctionResult, PromptItem


class JsonFileError(ValueError):
    """Raised when a JSON file cannot be read or parsed."""


class JsonValidationError(ValueError):
    """Raised when parsed JSON does not match the expected schema."""


def load_json_file(path: str | Path) -> Any:
    """Load a JSON file and return decoded Python data with clear errors."""
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


def validate_json_data(
    data: Any,
    adapter: TypeAdapter[Any],
    context: str,
) -> Any:
    """Validate decoded JSON using a TypeAdapter and return validated data."""
    try:
        return adapter.validate_python(data)
    except ValidationError as exc:
        raise JsonValidationError(
            f"Schema validation failed for {context}: {exc}"
        ) from exc


def validate_json_file(path: str | Path) -> bool:
    """Return True if file exists and contains syntactically valid JSON."""
    _ = load_json_file(path)
    return True


def load_function_definitions(path: str | Path) -> list[FunctionDefinition]:
    """Load and validate function definition list."""
    data = load_json_file(path)
    adapter = TypeAdapter(list[FunctionDefinition])
    return cast(
        list[FunctionDefinition],
        validate_json_data(
            data,
            adapter,
            f"function definitions file '{path}'",
        ),
    )


def load_prompt_items(path: str | Path) -> list[PromptItem]:
    """Load and validate prompt item list."""
    data = load_json_file(path)
    adapter = TypeAdapter(list[PromptItem])
    return cast(
        list[PromptItem],
        validate_json_data(data, adapter, f"prompt tests file '{path}'"),
    )


def write_text(path: str | Path, text: str) -> None:
    """Write text to file, creating parent dirs if needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def write_json(path: str | Path, data: object) -> None:
    """Write data as formatted JSON to file."""
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def write_function_results(
    path: str | Path,
    results: list[FunctionResult],
) -> None:
    """Write final result array with only prompt, name, parameters keys."""
    write_json(path, [result.model_dump() for result in results])
