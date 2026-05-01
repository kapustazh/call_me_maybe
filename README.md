*Constrained function-calling pipeline that converts natural language prompts
into schema-valid JSON results.*

# call_me_maybe

`call_me_maybe` reads prompt tests and function definitions, selects best
function per prompt, constrained-decodes parameter values, and writes final
results with exact shape:

```json
{
  "prompt": "...",
  "name": "fn_name",
  "parameters": {}
}
```

## Algorithm

1. Load and validate input JSON files with pydantic.
2. Build selection prompt from available function definitions.
3. Run boundary-logit function selection with confidence threshold.
4. For selected function, decode each parameter with type-aware token masks.
5. Validate decoded values against JSON type expectations.
6. Emit one result object per prompt and write one JSON array file.

## Design decisions

- **Constrained decoding for values:** avoids malformed JSON literals.
- **Schema-driven parameter loop:** no hardcoded function names or keys.
- **Tokenizer-file first, vocab fallback:** stays aligned to model tokenization.
- **Single pipeline path:** clean separation of IO, selection, and decoding.

## Performance notes

- Selection uses one boundary-logit pass, with continuation rescoring only on
  first-token collisions.
- Decoder precomputes candidate token pools per JSON type.
- `max_new_tokens` bounds each parameter decode step for safety.

## Challenges

- Handling partial JSON string escapes and unicode escape prefixes.
- Number-prefix validation where many prefixes are syntactically valid.
- Keeping strict typing (`mypy --strict`) while integrating dynamic SDK model.

## Testing strategy

- Unit tests for tokenizer/vocab loading and fallback behavior.
- Unit tests for selector confidence handling.
- Unit tests for constrained parameter decoding into typed Python values.

Run checks:

```bash
make lint-strict
make test
```

## Usage

Default paths:

```bash
make run
```

Custom paths:

```bash
make run ARGS="--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output data/output/function_calling_results.json"
```

## Resources

- Logit: https://en.wikipedia.org/wiki/Logit
- Byte-Pair Encoding: https://en.wikipedia.org/wiki/Byte-pair_encoding
- N-gram: https://en.wikipedia.org/wiki/N-gram
- UTF-8: https://en.wikipedia.org/wiki/UTF-8
- Andrej Karpathy lectures:
  https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3719s
- ML roadmap (RU): https://nareshka.ru/ml-roadmap?module=inference-optimization

## AI usage

AI assistant used for implementation scaffolding, type-safety cleanup, and unit
test drafting. All code and behavior reviewed and adjusted in-repo.
