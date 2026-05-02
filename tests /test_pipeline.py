import json
from pathlib import Path

from src.pipeline import Pipeline


class FakePipelineModel:
    def __init__(self, tokenizer_path: Path, vocab_path: Path) -> None:
        self._tokenizer_path = tokenizer_path
        self._vocab_path = vocab_path

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(token_id) for token_id in input_ids)
        if "JSON literal:" not in text:
            return [0.0] * 256
        marker = "JSON literal:"
        marker_pos = text.rfind(marker)
        generated = "" if marker_pos < 0 else text[marker_pos + len(marker) :]
        if "Function name: fn_add_numbers" in text:
            if "Parameter: a" in text:
                target = "2"
            elif "Parameter: b" in text:
                target = "3"
            else:
                target = "0"
        else:
            return [0.0] * 256
        next_char = (
            target[len(generated)] if len(generated) < len(target) else " "
        )
        logits = [-1000.0] * 256
        logits[ord(next_char)] = 1000.0
        return logits

    def get_path_to_tokenizer_file(self) -> str:
        return str(self._tokenizer_path)

    def get_path_to_vocab_file(self) -> str:
        return str(self._vocab_path)


class GoldenPipelineModel:
    def __init__(
        self,
        tokenizer_path: Path,
        vocab_path: Path,
        expected: list[dict[str, object]],
    ) -> None:
        self._tokenizer_path = tokenizer_path
        self._vocab_path = vocab_path
        self._expected_by_prompt = {
            str(item["prompt"]): item for item in expected
        }

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(token_id) for token_id in input_ids)
        if "JSON literal:" in text:
            target = self._parameter_target(text)
            generated = self._generated_parameter_suffix(text)
        else:
            target = self._selection_target(text)
            generated = self._generated_selection_suffix(text, target)
        next_char = (
            target[len(generated)] if len(generated) < len(target) else " "
        )
        logits = [-1000.0] * 256
        logits[ord(next_char)] = 1000.0
        return logits

    def get_path_to_tokenizer_file(self) -> str:
        return str(self._tokenizer_path)

    def get_path_to_vocab_file(self) -> str:
        return str(self._vocab_path)

    def _selection_target(self, text: str) -> str:
        prompt = _between(text, "User request:\n", "\n\nAvailable functions:")
        item = self._expected_by_prompt[prompt]
        return str(item["name"]).removeprefix("fn_")

    def _parameter_target(self, text: str) -> str:
        prompt = _between(text, "User prompt:\n", "\n\nFunction name:")
        parameter_name = _between(text, "Parameter: ", "\nExpected JSON type:")
        parameters = self._expected_by_prompt[prompt]["parameters"]
        if not isinstance(parameters, dict):
            raise TypeError("Expected parameter mapping")
        return json.dumps(parameters[parameter_name])

    @staticmethod
    def _generated_parameter_suffix(text: str) -> str:
        return text.rsplit("JSON literal:", 1)[1]

    @staticmethod
    def _generated_selection_suffix(text: str, target: str) -> str:
        suffix = text.rsplit("Return only function name.\nfn_", 1)[1]
        return target[: len(suffix)] if target.startswith(suffix) else ""


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_pipeline_skips_invalid_prompts(tmp_path: Path) -> None:
    functions_path = tmp_path / "functions.json"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_path = tmp_path / "vocab.json"

    functions_path.write_text(
        json.dumps(
            [
                {
                    "name": "fn_add_numbers",
                    "description": (
                        "Add two numbers together and return their sum."
                    ),
                    "parameters": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "returns": {"type": "number"},
                },
                {
                    "name": "fn_greet",
                    "description": (
                        "Generate a greeting message for a person by name."
                    ),
                    "parameters": {"name": {"type": "string"}},
                    "returns": {"type": "string"},
                },
            ]
        ),
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(
            [
                {"prompt": "What is the sum of 2 and 3?"},
                {"prompt": "This prompt matches no function at all"},
            ]
        ),
        encoding="utf-8",
    )

    token_map = {chr(code): code for code in range(32, 127)}
    tokenizer_path.write_text(
        json.dumps({"model": {"vocab": token_map}}),
        encoding="utf-8",
    )
    vocab_path.write_text(json.dumps(token_map), encoding="utf-8")

    pipeline = Pipeline(
        str(functions_path),
        str(input_path),
        str(output_path),
        model_factory=lambda _: FakePipelineModel(tokenizer_path, vocab_path),
    )
    pipeline.run()

    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert out == [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_pipeline_matches_function_calls_golden_file(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    functions_path = repo_root / "data/input/functions_definition.json"
    input_path = repo_root / "data/input/function_calling_tests.json"
    expected_path = repo_root / "data/output/function_calls.json"
    output_path = tmp_path / "output.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_path = tmp_path / "vocab.json"

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    token_map = {chr(code): code for code in range(32, 127)}
    tokenizer_path.write_text(
        json.dumps({"model": {"vocab": token_map}}),
        encoding="utf-8",
    )
    vocab_path.write_text(json.dumps(token_map), encoding="utf-8")

    pipeline = Pipeline(
        str(functions_path),
        str(input_path),
        str(output_path),
        model_factory=lambda _: GoldenPipelineModel(
            tokenizer_path,
            vocab_path,
            expected,
        ),
    )
    pipeline.run()

    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert out == expected
