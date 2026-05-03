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

- Entry point works: `src/__main__.py` parses `--functions_definition`, `--input`, `--output` and runs `Pipeline`.
- `src/pipeline.py` loads validated JSON, routes with `FunctionSelector`, decodes typed parameters with `ConstrainedDecoder`, and writes a JSON array.
- Per-prompt failures are handled gracefully: error logged to stderr, prompt skipped from output.
- Selector is model-only (no lexical heuristics), with boundary-token probabilities and continuation tie-break scoring in one log-probability space.
- `TokenizerVocab.from_model()` reads tokenizer JSON first, then falls back to vocab JSON.

## High-level architecture (implemented)

- **Load + validate**:
  - `function_definitions.json` → `FunctionDefinition` list.
  - `function_calling_tests.json` → prompt list.
- **Per prompt**:
  - Build selection prompt in `BobThePrompter`.
  - Select best function via `FunctionSelector.select()`.
  - Decode each required parameter literal via `ConstrainedDecoder.decode_parameters()`.
  - Parse and type-check literal with validator for declared schema type.
- **Write output**:
  - Append successful `{prompt, name, parameters}` rows.
  - Serialize one JSON array to `--output`.

## Function selection (implemented)

- Prompt ends at common function-name prefix.
- First candidate token scored from one boundary model call.
- If multiple candidates share first token, continuation tokens are rescored with autoregressive log-probability.
- Candidate score is normalized by suffix length, then softmaxed.
- Threshold gate uses model-only probability distribution (`DEFAULT_SELECTION_CONFIDENCE = 0.90`).

## Constrained decoding (implemented)

- Decoder constrains **parameter values** by expected JSON type:
  - `string`, `number`, `integer`, `boolean`, `object`.
- Token pieces are filtered by:
  - static allowed-piece checks,
  - prefix-validity checks (`is_valid_prefix`),
  - completion checks (`is_complete`).
- Decoding stops on first complete valid literal, then parsed by type validator.
- Final output object shape is enforced by pipeline writer (`prompt`, `name`, `parameters`).

## Verification checklist

- `uv run python -m pytest "tests /" -q`
- `uv run flake8 src tests`
- `uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
- `uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json`

## Known remaining risks

- Skipped prompts reduce effective accuracy; track skip count in evaluation runs.
- Full `mypy src` still depends on cleanup in files outside immediate routing/output path.
- `tests ` directory has trailing space in name; be explicit in commands until renamed.
