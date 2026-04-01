"""
Your implementation must use constrained decoding to guarantee 100% valid JSON
output, ensuring near-perfect reliability even
with a small 0.5B parameter model

"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="Translates prompts into function calls",
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the <name>.json with function definitions",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the <name>.json for the function calling tests",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the <name>.json for the output",
    )
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
