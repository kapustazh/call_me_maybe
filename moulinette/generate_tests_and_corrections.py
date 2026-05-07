import json
from typing import Any, List

from pydantic import BaseModel

from moulinette.functions_definition import exercises

class Correction(BaseModel):
    prompt: str
    fn_name: str
    fn_args: dict
    expected_output: Any


def generate_function_calling_corrections(
    exercises: dict[str, list[dict[str, Any]]],
) -> List[Correction]:
    corrections = []
    for fn_to_call, exercises in exercises.items():
        for exercise in exercises:
            correction = Correction(
                prompt=exercise["prompt"],
                fn_name=fn_to_call.__name__,
                fn_args=exercise["fn_args"],
                expected_output=fn_to_call(**exercise["fn_args"])
            )
            corrections.append(correction)
    return corrections

def save_function_calling_corrections(
    output_json_path: str,
) -> None:
    corrections = generate_function_calling_corrections(exercises)
    corrections_list = [correction.model_dump() for correction in corrections]

    with open(output_json_path, "w") as f:
        json.dump(corrections_list, f, indent=2)

def save_function_calling_tests(
    output_json_path: str,
) -> None:
    corrections = generate_function_calling_corrections(exercises)
    corrections_list = [correction.model_dump(exclude={"expected_output", "fn_args", "fn_name"}) for correction in corrections]

    with open(output_json_path, "w") as f:
        json.dump(corrections_list, f, indent=2)