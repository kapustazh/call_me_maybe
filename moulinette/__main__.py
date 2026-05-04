import os
import fire 
import shutil
from pathlib import Path

from moulinette.extract_functions_infos import generate_function_calling_definition
from moulinette.generate_tests_and_corrections import save_function_calling_corrections, save_function_calling_tests
from moulinette.functions_definition import exercises
import json

class Moulinette:
	def __init__(self):
		self.base_dir = Path("data")
		self.dir_exercise_input = self.base_dir / "exercise_input"
		self.dir_exercise_output = self.base_dir / "exercise_output"
		self.dir_exercise_correction = self.base_dir / "exercise_correction"

		self.functions_definition_path = self.dir_exercise_input / "functions_definition.json"
		self.function_calling_tests_path = self.dir_exercise_input / "function_calling_tests.json"
		self.function_calling_corrections_path = self.dir_exercise_correction / "function_calling_corrections.json"

	def prepare_exercises(self):
		shutil.rmtree(self.dir_exercise_input, ignore_errors=True)
		shutil.rmtree(self.dir_exercise_output, ignore_errors=True)
		shutil.rmtree(self.dir_exercise_correction, ignore_errors=True)

		self.dir_exercise_input.mkdir(parents=True, exist_ok=True)
		self.dir_exercise_output.mkdir(parents=True, exist_ok=True)
		self.dir_exercise_correction.mkdir(parents=True, exist_ok=True)

		generate_function_calling_definition(
			import_module_path="moulinette/functions_definition.py",
			output_json_path=self.functions_definition_path,
		)
		save_function_calling_corrections(
			output_json_path=self.function_calling_corrections_path,
		)
		save_function_calling_tests(
			output_json_path=self.function_calling_tests_path,
		)

	def evaluate_student_answers(self, student_answer_path: str = None):
		if student_answer_path is None:
			student_answer_path = self.dir_exercise_output / "function_calls.json"
		if isinstance(student_answer_path, str):
			student_answer_path = Path(student_answer_path)
		student_answers = json.load(open(student_answer_path))

		corrections = json.load(open(self.function_calling_corrections_path))

		fn_name_to_function = {
			fn_to_call.__name__: fn_to_call for fn_to_call in exercises.keys()
		}

		total_score = 0
		for student_answer, correction in zip(student_answers, corrections):
			print(f"{'-'*100}")
			print(f"Student answer: {student_answer}")
			print(f"Correction: {correction}")

			if student_answer["prompt"] != correction["prompt"]:
				print(f"Prompt mismatch: {student_answer['prompt']} != {correction['prompt']}")
				print(f"INVALID EXERCISE: wrong prompt")
				print(f"{'-'*100}")
				continue

			fn_args = student_answer["args"]
			fn_name = student_answer["fn_name"]
			try:
				fn = fn_name_to_function[fn_name]
			except Exception as e:
				print(f"Error: {e}")
				print(f"INVALID EXERCISE: wrong function name")
				print(f"{'-'*100}")
				continue

			try:
				student_output = fn(**fn_args)
			except Exception as e:
				print(f"Error: {e}")
				print(f"INVALID EXERCISE: wrong function arguments")
				print(f"{'-'*100}")
				continue

			if student_output != correction["expected_output"]:
				print(f"Output mismatch: {student_output} != {correction['expected_output']}")
				print(f"INVALID EXERCISE: wrong output")
				print(f"{'-'*100}")
				continue

			total_score += 1
			print(f"VALID EXERCISE")
			print(f"{'-'*100}")
		print(f"Total score: {total_score}/{len(corrections)}")


if __name__ == "__main__":
	fire.Fire(Moulinette)