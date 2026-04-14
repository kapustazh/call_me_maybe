---
name: call-me-maybe-implementation
overview: Implement a function-calling pipeline that reads prompts + function definitions, uses an LLM to select a function and extract typed arguments, and uses constrained decoding to guarantee 100% valid, schema-compliant JSON output.
todos:
  - id: parse-and-validate-inputs
    content: Create pydantic models for function definitions + prompt items; implement robust JSON loaders with clear errors.
    status: pending
  - id: token-id-to-piece-map
    content: Load tokenizer/vocab mapping via `Small_LLM_Model.get_path_to_tokenizer_file()` (fallback to vocab) so constraints can be applied per token.
    status: pending
  - id: constrained-decoder-core
    content: Implement a JSON+schema state machine and a token-masking generation loop that guarantees valid, schema-compliant JSON for one result object.
    status: pending
  - id: pipeline-loop-and-output
    content: Update `Pipeline.run()` to iterate prompts, generate one constrained-decoded result per prompt, validate types, and write a JSON array to `--output`.
    status: pending
  - id: cli-and-docs
    content: Align CLI defaults with subject, improve error messages, and rewrite README to satisfy mandatory sections (incl. constrained decoding explanation).
    status: pending
---

## What the project must do

- Read function definitions JSON and prompt tests JSON (defaults under `data/input/`).
- For each prompt, output **one object** with **exactly** keys: `prompt`, `name`, `parameters`.
- Write a single JSON array file (spec default: `data/output/function_calling_results.json`).
- **Guarantee 100% valid JSON** and schema compliance via **constrained decoding** (not “prompt and pray”).

## Repo reality check (current state)

- Entry point exists: `src/__main__.py` parses `--functions_definition`, `--input`, `--output` and runs `Pipeline`.
- `src/pipeline.py` currently just reads the whole input file as raw text and greedily generates 10 tokens. It doesn’t parse the input JSON array, doesn’t use function definitions, and doesn’t implement constrained decoding.
- The SDK you have is `llm_sdk.Small_LLM_Model` in `llm_sdk/llm_sdk/__init__.py` and exposes:
  - `encode(text) -> torch.Tensor`
  - `get_logits_from_input_ids(input_ids: list[int]) -> list[float]`
  - `decode(ids) -> str`
  - `get_path_to_tokenizer_file()` / `get_path_to_vocab_file()`

## High-level architecture

- **Parse inputs** (robustly):
  - `functions_definition.json` → list of validated `FunctionDefinition` models.
  - `function_calling_tests.json` → list of `PromptItem { prompt: str }`.
- **For each prompt**:
  - Construct a *decision prompt* containing the user prompt + available functions (names + descriptions + parameter types).
  - Run **constrained decoding** that can only emit JSON matching your required output object schema.
  - Parse the generated JSON (it must parse every time), validate types against the chosen function definition, and collect results.
- **Write output** as JSON array to requested `--output` path.

## Constrained decoding approach (practical, implementable)

Implement a token-level “masking” loop:

- At each generation step:
  - Ask the model for next-token logits with `get_logits_from_input_ids(context_ids)`.
  - Compute a boolean mask of which token IDs are valid next tokens **given the partial output so far**.
  - Set invalid tokens’ logits to negative infinity and pick argmax (or sample) from remaining.

To make the mask computable, use a **streaming JSON + schema state machine** (not a full JSON-schema engine):

- Hardcode the top-level object shape:
  - Must generate: `{ "prompt": <string>, "name": <one_of_function_names>, "parameters": <object_for_that_function> }`
- Drive generation with a deterministic state machine that knows:
  - Which structural characters are expected next (`{`, `}`, `[`, `]`, `:`, `,`, quotes, whitespace).
  - When it is inside a JSON string vs number.
  - Which key is expected next.
  - For `name`: only allow strings that match one of the function names exactly.
  - For `parameters`: only allow the required keys for that function, with value types restricted by definition (`string`, `number`, `boolean`).

Token-to-text mapping for constraints:

- Use `Small_LLM_Model.get_path_to_tokenizer_file()` (preferred) or `get_path_to_vocab_file()` to build a mapping `token_id -> token_text_piece`.
- A candidate token is valid if appending its text piece to the current generated text keeps the state machine in a non-error state (still a valid prefix).

Stopping condition:

- Stop only when the state machine reaches “completed JSON object” and the last token closes all structures.

## Files to add/modify

- Modify `src/pipeline.py`:
  - Parse both input JSONs.
  - Loop prompts and call a new constrained-decoding generator.
  - Validate and write output JSON array.
- Modify `src/__main__.py`:
  - Align default `--output` with subject expectation (results file name) and improve error messages (no raw `print(f"{e=}")`).
- Add new modules under `src/` (suggested):
  - `src/models.py`: pydantic models: `FunctionDefinition`, `ParameterSpec`, `PromptItem`, `FunctionCallResult`.
  - `src/io_utils.py`: safe JSON read/write helpers (nice errors for missing/invalid JSON).
  - `src/tokenizer_vocab.py`: load tokenizer/vocab files into `id -> piece` mapping.
  - `src/constrained_decoder.py`: the state machine + masking loop.
  - `src/prompting.py`: build the LLM instruction prompt (function list formatting, etc.).
- Update `README.md` to meet the subject’s required sections (first italicized line, algorithm explanation, design decisions, performance analysis, challenges, testing strategy, usage examples, and resources + how AI was used).

## Validation & testing checklist (what to run locally)

- `make install` then `make lint` (flake8 + mypy).
- Run default command and ensure it writes JSON array:
  - `make run` (or `uv run python -m src`).
- Add a small local test script (not submitted) that:
  - Corrupts input JSON to confirm graceful error handling.
  - Uses edge prompts (empty, quotes, unicode, large numbers) to confirm JSON escaping and type correctness.

## Key gotchas to explicitly handle

- Input files may be missing or invalid JSON: must not crash.
- Strings in parameters must be JSON-escaped correctly (quotes, backslashes, unicode).
- Don’t hardcode function names/parameters: always derive from `functions_definition.json`.
- Don’t output extra keys, comments, or prose.
- Keep runtime under ~5 minutes for the test set.

