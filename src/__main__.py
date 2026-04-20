"""
Your implementation must use constrained decoding to guarantee 100% valid JSON
output, ensuring near-perfect reliability even
with a small 0.5B parameter model

"""

from argparse import Namespace
from src.pipeline import Pipeline
import argparse

# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
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
        default="data/output/function_calls.json",
        help="Path to the <name>.json for the output",
    )
    args: Namespace = parser.parse_args()
    try:
        pipeline = Pipeline(args.functions_definition, args.input, args.output)
        pipeline.run()
    except Exception as exc:
        print(f"{exc=}")


if __name__ == "__main__":
    # logger.info(msg="Start")
    main()
    # logger.info(msg="Finished succesfully")
