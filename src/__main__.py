import argparse
from argparse import ArgumentParser, Namespace

from src.io_utils import JsonFileError, JsonValidationError
from src.pipeline import Pipeline


def main() -> None:
    parser: ArgumentParser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="Translates prompts into function calls",
    )
    _ = parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the <name>.json with function definitions",
    )
    _ = parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the <name>.json for the function calling tests",
    )
    _ = parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the <name>.json for the output",
    )
    args: Namespace = parser.parse_args()
    try:
        pipeline = Pipeline(
            args.functions_definition,
            args.input,
            args.output,
        )
        pipeline.run()
    except (
        JsonFileError,
        JsonValidationError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
