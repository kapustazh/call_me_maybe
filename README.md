*This project has been created as part of the 42 curriculum by mnestere.*

# call_me_maybe

Small-model **function calling**: read prompts and a function schema (JSON),
pick a function with the LM under a confidence gate, extract typed parameters
with constrained decoding plus light heuristics where helpful, write one JSON
array of results:

```json
{
  "prompt": "...",
  "name": "fn_example",
  "parameters": { "...": "..." }
}
```

Rows that fail selection or decoding are skipped (message on stderr); the output
file lists only successful calls.

## Description

Goal is reliable structured output from a ~0.6B causal LM (`Qwen/Qwen3-0.6B` by
default via `llm_sdk`), not free-form JSON from the model alone. The pipeline
loads inputs with **Pydantic**, runs **logit-masked** literal generation per
parameter type, and prints **per-prompt progress** on stdout while it runs.

## Instructions

**Prerequisites:** Python ≥ 3.10, [uv](https://github.com/astral-sh/uv), enough
disk/RAM for the HF checkpoint (first run downloads weights).

```bash
make install    # uv sync, Python 3.11 in this repo
make run        # default paths under data/input → data/output
make test
make lint       # flake8 + mypy (non-strict)
# optional: make lint-strict
```

**CLI** (same flags as `uv run -m src …`):

| Flag | Default |
|------|---------|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

Use a **matching** definitions file for your prompt set (e.g. extended tests
under `data_test/` need `data_test/input/functions_definition.json`), or
selection sees the wrong tool list.

```bash
make run ARGS='--functions_definition data_test/input/functions_definition.json \
  --input data_test/input/function_calling_tests.json \
  --output data_test/output/function_calling_results.json'
```

**Programmatic model:** `Pipeline(..., model_name="HF/model-id")` passes the
id to `Small_LLM_Model`; empty string keeps the SDK default. The CLI does not
expose `--model` yet.

## Algorithm (short)

1. Validate JSON inputs; **deduplicate** function definitions by `name` (first
   wins) so softmax over candidates is not split by duplicates.
2. **Selection prompt** lists tools; suffix is the **longest common prefix** of
   all function names (character LCP), so prompt ending and token suffix for
   scoring stay aligned. Softmax over per-candidate boundary logits with a
   **minimum probability** threshold (`DEFAULT_SELECTION_CONFIDENCE` in
   `function_selector.py`); continuation logits break first-token ties.
3. For the chosen function, each parameter: **heuristic extraction** when
   pattern matches, else **masked autoregressive** decode of one JSON literal,
   then `json.loads` into Python types per schema.

## Design choices

- **Constrained decoding** for literals → syntactically valid JSON fragments.
- **Schema-driven loop** over `parameters` — no per-function hardcoded keys in
  the decoder core.
- **Tokenizer JSON first**, flat vocab fallback (`TokenizerVocab`) so masks
  match the loaded model.
- **Heuristics** only as fast path / fallback for obvious patterns (numbers,
  quoted strings, etc.); routing stays LLM-based.

## Performance and reliability

- One forward per selection step at the boundary; extra forwards only for
  tied first tokens (shared BPE prefix).
- Candidate token **pools per JSON type** built once at decoder init.
- Bounded `max_new_tokens` per literal decode.

## Testing

- `tests /` — tokenizer/vocab, selector, decoder, pipeline (including golden
  output with a fake model where applicable).

## Resources

- Logit: https://en.wikipedia.org/wiki/Logit
- Byte-Pair Encoding: https://en.wikipedia.org/wiki/Byte-pair_encoding
- N-gram: https://en.wikipedia.org/wiki/N-gram
- UTF-8: https://en.wikipedia.org/wiki/UTF-8
- Andrej Karpathy lectures:
  https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3719s
- ML roadmap (RU): https://nareshka.ru/ml-roadmap?module=inference-optimization

## AI usage

AI assistant was used for everything (besides this part of the README and "Resources")
