import json
import re
from pathlib import Path

import pytest

from src.pipeline import Pipeline
from src.prompt import Prefix


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
            logits = [-100.0] * 256
            if 'Request: "' in text:
                user_part = text.split('Request: "', 1)[1].split('"', 1)[0]
                if "sum" in user_part.lower():
                    logits[ord("a")] = 10.0
                    logits[ord("g")] = 1.0
            return logits
        marker = "JSON literal:"
        marker_pos = text.rfind(marker)
        end_marker = marker_pos + len(marker)
        generated = "" if marker_pos < 0 else text[end_marker:]
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
        self._name_prefix = Prefix.longest_common_prefix(
            [str(item["name"]) for item in expected],
        )

    def encode(self, text: str) -> list[list[int]]:
        return [[ord(ch) for ch in text]]

    def decode(self, ids: list[int] | object) -> str:
        if isinstance(ids, list):
            return "".join(chr(token_id) for token_id in ids)
        raise TypeError("Expected list[int]")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        text = "".join(chr(token_id) for token_id in input_ids)
        if "You are parameters extraction assistant from text" in text:
            return self._assistant_decode_logits(text)
        if "JSON literal:" in text:
            target = self._parameter_target(text)
            generated = self._generated_parameter_suffix(text)
        else:
            target = self._selection_target(text)
            generated = self._generated_selection_suffix(text)
        next_char = (
            target[len(generated)] if len(generated) < len(target) else " "
        )
        logits = [-1000.0] * 256
        logits[ord(next_char)] = 1000.0
        return logits

    def _task_user_prompt(self, text: str) -> str:
        marker = "Task: "
        pos = text.find(marker)
        if pos < 0:
            raise ValueError("decode prompt missing Task line")
        rest = text[pos + len(marker):]
        return rest.split("\n", 1)[0].strip()

    def _assistant_decode_logits(self, text: str) -> list[float]:
        user = self._task_user_prompt(text)
        if user != getattr(self, "_golden_assistant_user", None):
            self._golden_assistant_user = user
            self._golden_bool_wave = 0

        row = self._expected_by_prompt[user]
        params = row["parameters"]
        if not isinstance(params, dict):
            raise TypeError("Expected parameter mapping")

        def emit(ch: str) -> list[float]:
            out = [-1000.0] * 256
            out[ord(ch)] = 1000.0
            return out

        for pname, val in params.items():
            if not isinstance(val, str):
                continue
            m = re.search(rf'"{re.escape(pname)}": "([^"]*)$', text)
            if not m:
                continue
            partial = m.group(1)
            if len(partial) < len(val):
                return emit(val[len(partial)])

        for pname, val in params.items():
            if isinstance(val, bool) or isinstance(val, str):
                continue
            if isinstance(val, (int, float)):
                target = json.dumps(val, separators=(",", ":"))
                m = re.search(
                    rf'"{re.escape(pname)}":\s*([0-9.eE+-]*)$',
                    text,
                )
                if not m:
                    continue
                partial = m.group(1)
                if len(partial) < len(target):
                    return emit(target[len(partial)])

        for pname in params:
            val = params[pname]
            if not isinstance(val, bool):
                continue
            needle = f'"{pname}": '
            if needle not in text:
                continue
            suf = text.rsplit(needle, 1)[-1]
            if suf == "":
                wave = getattr(self, "_golden_bool_wave", 0)
                self._golden_bool_wave = wave + 1
                return emit("t" if wave % 2 == 0 else "f")
            for word in ("true", "false"):
                if word.startswith(suf) and suf != word:
                    return emit(word[len(suf)])

        return [-1000.0] * 256

    def get_path_to_tokenizer_file(self) -> str:
        return str(self._tokenizer_path)

    def get_path_to_vocab_file(self) -> str:
        return str(self._vocab_path)

    def _selection_target(self, text: str) -> str:
        prompt = _between(text, 'Request: "', '"\n\nThe correct function is:')
        item = self._expected_by_prompt[prompt]
        full = str(item["name"])
        skip = len(self._name_prefix)
        return full[skip:]

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

    def _generated_selection_suffix(self, text: str) -> str:
        marker = "The correct function is: "
        if marker not in text:
            return ""
        tail = text.split(marker, 1)[1]
        if not tail.startswith(self._name_prefix):
            return ""
        skip = len(self._name_prefix)
        return tail[skip:]


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_pipeline_drops_invalid_prompt_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def _fake_model(*_a: object, **_k: object) -> FakePipelineModel:
        return FakePipelineModel(tokenizer_path, vocab_path)

    monkeypatch.setattr("src.pipeline.Small_LLM_Model", _fake_model)
    pipeline = Pipeline(
        str(functions_path),
        str(input_path),
        str(output_path),
        "fake-model",
        # selection_confidence_threshold=0.80,
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


def test_pipeline_matches_results_golden_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    functions_path = repo_root / "data/input/functions_definition.json"
    input_path = repo_root / "data/input/function_calling_tests.json"
    expected_path = repo_root / "data/output/function_calling_results.json"
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

    def _fake_model(*_a: object, **_k: object) -> GoldenPipelineModel:
        return GoldenPipelineModel(
            tokenizer_path,
            vocab_path,
            expected,
        )

    monkeypatch.setattr("src.pipeline.Small_LLM_Model", _fake_model)
    pipeline = Pipeline(
        str(functions_path),
        str(input_path),
        str(output_path),
        "fake-model",
    )
    pipeline.run()

    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert out == expected
