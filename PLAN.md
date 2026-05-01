---
name: call-me-maybe-implementation
overview: Implement a function-calling pipeline that reads prompts + function definitions, uses an LLM to select a function and extract typed arguments, and uses constrained decoding to guarantee 100% valid, schema-compliant JSON output.
todos:
  - id: parse-and-validate-inputs
    content: Create pydantic models for function definitions + prompt items; implement robust JSON loaders with clear errors.
    status: done
  - id: builtin-vocab-token-map
    content: Build token ID helpers from the model's built-in vocab/tokenizer files and SDK decode behavior so constraints stay aligned with the model.
    status: done
  - id: function-selection
    content: Select functions with one LLM forward pass by scoring each function's distinguishing token at the prompt boundary and applying a confidence threshold.
    status: done
  - id: constrained-decoder-core
    content: Implement hybrid constrained decoding: force-inject JSON structure and use type-specific masked generation only for parameter values.
    status: done
  - id: pipeline-loop-and-output
    content: Update `Pipeline.run()` to iterate prompts, generate one constrained-decoded result per prompt, validate types, and write a JSON array to `--output`.
    status: done
  - id: cli-and-docs
    content: Align CLI defaults with subject, improve error messages, and rewrite README to satisfy mandatory sections (incl. constrained decoding explanation).
    status: done
---

## What the project must do

- Read function definitions JSON and prompt tests JSON (defaults under `data/input/`).
- For each prompt, output **one object** with **exactly** keys: `prompt`, `name`, `parameters`.
- Write a single JSON array file (spec default: `data/output/function_calling_results.json`).
- **Guarantee 100% valid JSON** and schema compliance via **constrained decoding** (not “prompt and pray”).

## Repo reality check (current state)

- Entry point exists: `src/__main__.py` parses `--functions_definition`, `--input`, `--output` and runs `Pipeline`.
- `src/pipeline.py` now loads validated prompt/function JSON, loops over prompts, selects a function, and writes result objects, but still emits empty `parameters`.
- `src/function_selector.py` already contains an LLM-driven selector, but it currently scores complete function-name suffixes with multiple forward passes. It should be tightened to the single-boundary-logit design below.
- `src/constrained_decoder.py` only has early token helper code. It still needs the hybrid value decoder.
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
  - Select the function with a single LLM forward pass over boundary logits.
  - Build the result object with deterministic JSON structure: `prompt`, selected `name`, and generated `parameters`.
  - Decode only parameter values with type-specific constrained generation.
  - Validate types against the selected function definition and collect results.
- **Write output** as JSON array to requested `--output` path.

## Function selection

Use LLM-driven function selection with one forward pass:

- Build a router prompt listing every function name, description, and parameter schema.
- End the prompt exactly at the position where the distinguishing function token should appear.
- Read logits with `get_logits_from_input_ids(context_ids)`.
- For each candidate function, score the token that distinguishes it from the common prefix.
- Apply softmax over candidate scores to get a confidence distribution.
- Select the highest-scoring function only if confidence is at least `0.90`; otherwise raise a clear selection error.

Why this shape:

- Avoids O(N) full-name scoring passes.
- Keeps routing fast and deterministic for the small fixed function list.
- Reuses the model where it is useful: semantic intent classification.

## Constrained decoding approach

Use a hybrid decoder that guarantees valid JSON by construction:

- **Structure**: force-inject braces, keys, quotes, colons, commas, and parameter key order. The LLM never gets to decide JSON syntax.
- **Top-level object**: construct with Python data and `json.dumps`, using exactly `prompt`, `name`, and `parameters`.
- **Parameters object**: derive required keys and value types from the selected `FunctionDefinition`.
- **Values only**: use constrained LLM decoding only for the value slots.

Type-specific value strategy:

- **Strings**:
  - Generate string content with quote-aware masking.
  - Allow normal text tokens, escaped characters, and a closing quote only when the value is complete.
  - Map generated strings to parameters by schema order for multi-parameter functions.
- **Numbers**:
  - Use logit masking over numeric token IDs (`0-9`, `.`, `-`) and parse the result.
  - Keep only prefixes that can still become a valid JSON number.
- **Booleans**:
  - Score `true` and `false` autoregressively across all tokens, because either word may be multi-token.
  - Pick the higher normalized log-probability.

Stopping condition:

- Stop parameter value generation as soon as the value parser reaches a complete valid value for the expected type.
- Never stop based on free-form model prose.

## Vocab strategy

Use the model's built-in vocabulary directly:

- Load vocab assets from `Small_LLM_Model.get_path_to_vocab_file()` and reuse the existing `Vocab` helper where possible.
- Use `Small_LLM_Model.decode([token_id])` to derive each token's text piece for masking checks.
- Cache structural token IDs, numeric token IDs, quote IDs, `true`/`false` token sequences, and frequent punctuation at decoder init.
- Use `Small_LLM_Model.encode()` for prompts and fixed literals so tokenization stays identical to the model.
- Treat candidate token as valid only when its decoded text keeps the current value parser in a valid prefix state.

## Design decisions

1. **Force-injecting structure**: guarantees 100% JSON validity. Spontaneous LLM JSON is unreliable and can emit prose, missing braces, extra keys, or invalid escapes.
2. **Single forward pass selection**: O(1) function scoring at the decision boundary instead of O(N) candidate generation passes.
3. **Masked value decoding**: all parameter values come from model logits under type-specific masks.
4. **Schema-order string mapping**: maps generated string values to multi-parameter function arguments by parameter order.
5. **Precomputed token IDs**: caches structural, boolean, and numeric tokens once at init for speed and simpler masks.
6. **Built-in vocab reuse**: avoids tokenizer drift and keeps constraints aligned with actual model tokenization.

## Files to add/modify

- Modify `src/pipeline.py`:
  - Parse both input JSONs.
  - Loop prompts and call a new constrained-decoding generator.
  - Validate and write output JSON array.
- Modify `src/__main__.py`:
  - Align default `--output` with subject expectation (results file name) and improve error messages (no raw `print(f"{e=}")`).
- Add/update modules under `src/`:
  - `src/models.py`: pydantic models: `FunctionDefinition`, `ParameterSpec`, `PromptItem`, `FunctionCallResult`.
  - `src/io_utils.py`: safe JSON read/write helpers (nice errors for missing/invalid JSON).
  - `src/vocab.py`: built-in vocab loader and token ID lookup helpers.
  - `src/function_selector.py`: single-forward-pass boundary-logit function selector.
  - `src/constrained_decoder.py`: force-injected JSON skeleton plus type-specific value generation.
  - `src/prompt.py`: build selection and value-generation prompts.
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
