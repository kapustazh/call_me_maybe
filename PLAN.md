---
name: call-me-maybe-implementation
overview: Implement function-calling pipeline: read prompts + function definitions, use LLM to select function + extract typed args, use constrained decoding to guarantee 100% valid, schema-compliant JSON output.
todos:
  - id: parse-and-validate-inputs
    content: Create pydantic models for function definitions + prompt items; implement robust JSON loaders with clear errors.
    status: done
  - id: builtin-vocab-token-map
    content: Build token ID helpers from model built-in vocab/tokenizer files + SDK decode behavior so constraints stay aligned with model.
    status: done
  - id: function-selection
    content: Select functions with 1 LLM forward pass: score each function's distinguishing token at prompt boundary + apply confidence threshold.
    status: done
  - id: constrained-decoder-core
    content: Implement hybrid constrained decoding: force-inject JSON structure + use type-specific masked generation only for parameter values.
    status: done
  - id: pipeline-loop-and-output
    content: Update `Pipeline.run()` to iterate prompts, generate 1 constrained-decoded result per prompt, validate types, write JSON array to `--output`.
    status: done
  - id: cli-and-docs
    content: Align CLI defaults with subject, improve error messages, rewrite README to satisfy mandatory sections (incl. constrained decoding explanation).
    status: done
---

## What the project must do

- Read function definitions JSON + prompt tests JSON (defaults under `data/input/`).
- For each prompt, output **one object** with **exactly** keys: `prompt`, `name`, `parameters`.
- Write single JSON array file (spec default: `data/output/function_calling_results.json`).
- **Guarantee 100% valid JSON** + schema compliance via **constrained decoding** (not “prompt and pray”).

## Repo reality check (current state)

- Entry point exists: `src/__main__.py` parses `--functions_definition`, `--input`, `--output` and runs `Pipeline`.
- `src/pipeline.py` loads validated prompt/function JSON, loops prompts, selects function, writes result objects, but still emits empty `parameters`.
- `src/function_selector.py` has LLM-driven selector, but scores complete function-name suffixes with multiple forward passes. Tighten to single-boundary-logit design below.
- `src/constrained_decoder.py` only has early token helper code. It still needs the hybrid value decoder.
- The SDK you have is `llm_sdk.Small_LLM_Model` in `llm_sdk/llm_sdk/__init__.py` and exposes:
  - `encode(text) -> torch.Tensor`
  - `get_logits_from_input_ids(input_ids: list[int]) -> list[float]`
  - `decode(ids) -> str`
  - `get_path_to_tokenizer_file()` / `get_path_to_vocab_file()`

## High-level architecture

- **Parse inputs**:
  - `functions_definition.json` → list of validated `FunctionDefinition` models.
  - `function_calling_tests.json` → list of `PromptItem { prompt: str }`.
- **For each prompt**:
  - Construct *decision prompt*: user prompt + available functions (names + descriptions + parameter types).
  - Select function with 1 LLM forward pass over boundary logits.
  - Build the result object with deterministic JSON structure: `prompt`, selected `name`, and generated `parameters`.
  - Decode only parameter values with type-specific constrained generation.
  - Validate types against the selected function definition and collect results.
- **Write output**: JSON array to requested `--output` path.

## Function selection

Use LLM-driven function selection with 1 forward pass:

- Build router prompt listing every function name, description, parameter schema.
- End the prompt exactly at the position where the distinguishing function token should appear.
- Read logits with `get_logits_from_input_ids(context_ids)`.
- For each candidate function, score token that distinguishes it from common prefix.
- Apply softmax over candidate scores to get a confidence distribution.
- Select highest-scoring function only if confidence ≥ `0.90`; else raise clear selection error.

Why this shape:

- Avoid O(N) full-name scoring passes.
- Keep routing fast + deterministic for small fixed function list.
- Use model where useful: semantic intent classification.

## Constrained decoding approach

Use hybrid decoder that guarantees valid JSON by construction:

- **Structure**: force-inject braces, keys, quotes, colons, commas, parameter key order. LLM never decides JSON syntax.
- **Top-level object**: construct with Python data + `json.dumps`, using exactly `prompt`, `name`, `parameters`.
- **Parameters object**: derive required keys and value types from the selected `FunctionDefinition`.
- **Values only**: constrained LLM decoding only for value slots.

Type-specific value strategy:

- **Strings**:
  - Generate string content with quote-aware masking.
  - Allow normal text tokens, escaped characters, closing quote only when value is complete.
  - Map generated strings to parameters by schema order for multi-parameter functions.
- **Numbers**:
  - Logit mask over numeric token IDs (`0-9`, `.`, `-`) and parse result.
  - Keep only prefixes that can still become valid JSON number.
- **Booleans**:
  - Score `true` and `false` autoregressively across all tokens (either may be multi-token).
  - Pick higher normalized log-probability.

Stopping condition:

- Stop value generation as soon as parser reaches complete valid value for expected type.
- Never stop based on free-form model prose.

## Vocab strategy

Use model built-in vocabulary directly:

- Load vocab assets from `Small_LLM_Model.get_path_to_vocab_file()` and reuse existing `Vocab` helper where possible.
- Use `Small_LLM_Model.decode([token_id])` to derive each token's text piece for masking checks.
- Cache structural token IDs, numeric token IDs, quote IDs, `true`/`false` token sequences, frequent punctuation at decoder init.
- Use `Small_LLM_Model.encode()` for prompts and fixed literals so tokenization stays identical to the model.
- Treat candidate token as valid only when decoded text keeps current value parser in valid prefix state.

## Design decisions

1. **Force-injecting structure**: guarantee 100% JSON validity. Spontaneous LLM JSON unreliable: can emit prose, missing braces, extra keys, invalid escapes.
2. **Single forward pass selection**: O(1) function scoring at decision boundary vs O(N) candidate generation passes.
3. **Masked value decoding**: all parameter values come from model logits under type-specific masks.
4. **Schema-order string mapping**: map generated string values to multi-parameter function args by parameter order.
5. **Precomputed token IDs**: cache structural, boolean, numeric tokens once at init for speed + simpler masks.
6. **Built-in vocab reuse**: avoid tokenizer drift; keep constraints aligned with model tokenization.

## Files to add/modify

- Modify `src/pipeline.py`:
  - Parse both input JSONs.
  - Loop prompts + call constrained-decoding generator.
  - Validate + write output JSON array.
- Modify `src/__main__.py`:
  - Align default `--output` with subject expectation (results file name) + improve error messages (no raw `print(f"{e=}")`).
- Add/update modules under `src/`:
  - `src/models.py`: pydantic models: `FunctionDefinition`, `ParameterSpec`, `PromptItem`, `FunctionCallResult`.
  - `src/io_utils.py`: safe JSON read/write helpers (nice errors for missing/invalid JSON).
  - `src/vocab.py`: built-in vocab loader and token ID lookup helpers.
  - `src/function_selector.py`: single-forward-pass boundary-logit function selector.
  - `src/constrained_decoder.py`: force-injected JSON skeleton + type-specific value generation.
  - `src/prompt.py`: build selection and value-generation prompts.
- Update `README.md` to meet the subject’s required sections (first italicized line, algorithm explanation, design decisions, performance analysis, challenges, testing strategy, usage examples, and resources + how AI was used).

## Validation & testing checklist (what to run locally)

- `make install` then `make lint` (flake8 + mypy).
- Run default command; ensure it writes JSON array:
  - `make run` (or `uv run python -m src`).
- Add small local test script (not submitted):
  - Corrupt input JSON; confirm graceful error handling.
  - Use edge prompts (empty, quotes, unicode, large numbers); confirm JSON escaping + type correctness.

## Key gotchas to explicitly handle

- Input files may be missing/invalid JSON: must not crash.
- Strings in parameters must be JSON-escaped correctly (quotes, backslashes, unicode).
- Don’t hardcode function names/parameters: derive from `functions_definition.json`.
- Don’t output extra keys, comments, prose.
- Keep runtime under ~5 minutes for test set.
