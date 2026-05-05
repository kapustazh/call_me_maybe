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
    _ = parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Selection confidence threshold "
            "(default: adaptive min(3/N, 0.9))"
        ),
    )
    _ = parser.add_argument(
        "--selection-peak-target",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Anneal softmax temperature until the top candidate reaches "
            "mass P (default: 0.9; use P>=1.0 to disable annealing)"
        ),
    )
    args: Namespace = parser.parse_args()
    try:
        pipeline = Pipeline(
            args.functions_definition,
            args.input,
            args.output,
            selection_confidence_threshold=args.threshold,
            selection_peak_softmax_target=args.selection_peak_target,
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
