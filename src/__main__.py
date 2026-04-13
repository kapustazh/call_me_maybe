"""
Your implementation must use constrained decoding to guarantee 100% valid JSON
output, ensuring near-perfect reliability even
with a small 0.5B parameter model

"""

from llm_sdk import Small_LLM_Model
from argparse import Namespace
from src.manager import build_inference
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
    _ = args
    with open(args.input, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    llm_sdk = Small_LLM_Model()
    build_inference(prompt_text, llm_sdk, max_new_tokens=15)


if __name__ == "__main__":
    # logger.info(msg="Start")
    main()
    # logger.info(msg="Finished succesfully")
