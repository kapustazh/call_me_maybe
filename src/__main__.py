"""
Call me maybe. Not just maybe, please, call me I miss you so much.
           .               ,.
          T."-._..---.._,-"/|
          l|"-.  _.v._   (" |
          [l /.'_ \\; _~"-.`-t
          Y " _(o} _{o)._ ^.|
          j  T  ,-<v>-.  T  ]
          \\  l (//-^-\\ ) !  !
           \\.\\.  "~"  ./  /c-..,__
             ^r- .._ .- .-"  `- .  ~"--..
              > \\.                      \\
              ]   ^.                     \\
              3  .  ">            .       Y
 ,.__.--._   _j   \\ ~   .         ;       |
(    ~"-._~"^.\\   ^.    ^._      I     .  l
 "-._ ___ ~"-,_7    .Z-._   7"   Y      ;  \\        _
    /"   "~-(r r  _/_--._~-/    /      /,.--^-._   / Y
    "-._    '"~~~>-._~]>--^---./____,.^~        ^.^  !
        ~--._    '   Y---.                        \\./
             ~~--._  l_   )                        \
                   ~-._~~~---._,____..---           \
                       ~----"~       \
                                      \
"""

import argparse
from argparse import ArgumentParser, Namespace

from src import init_runtime_dirs
from src.io_utils import JsonFileError, JsonValidationError
from src.pipeline import Pipeline, PipelineNoResultsError


def main() -> None:
    """Run CLI entrypoint for translating prompts into function calls.

    Reads function schema and prompt tests JSON files.
    Runs selection + constrained decoding. Writes results to output JSON file.

    Raises:
        SystemExit: If input files missing/invalid, schema validation fails,
            or no prompt produced a successful function call.
    """
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
    _ = parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name of the model to use (default: Qwen/Qwen3-0.6B)",
    )
    args: Namespace = parser.parse_args()
    try:
        init_runtime_dirs()
        pipeline = Pipeline(
            args.functions_definition,
            args.input,
            args.output,
            args.model_name,
        )
        pipeline.run()
    except (
        JsonFileError,
        JsonValidationError,
        ValueError,
        PipelineNoResultsError,
    ) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
