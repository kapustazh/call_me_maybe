# Refactor plan — call_me_maybe

Current snapshot after routing/output stabilization.

## Subject-critical rules (still guiding)

- **IV.3.1**: function selection must stay LLM-driven.
- **V.4**: output rows must use exactly `prompt`, `name`, `parameters`.
- **V.5**: optimize for high selection+argument accuracy and valid JSON.

## What is fixed

- Lexical heuristic path removed from selector and tests.
- Selector scoring uses model-only probability space:
  - boundary token probability,
  - continuation log-prob tie-break for shared first token,
  - softmax + confidence gate.
- Output policy is explicit:
  - per-prompt routing/decoding failures are logged,
  - invalid prompts are skipped,
  - output JSON keeps only valid function-call rows.
- Filename contract aligned to `data/output/function_calling_results.json`:
  - CLI default,
  - README usage,
  - pipeline golden fixture test path.

## Current behavior contract

- `Pipeline.run()`:
  - loads/validates input JSON,
  - routes + decodes per prompt,
  - skips invalid prompts with stderr message,
  - writes one JSON array to output path.
- `__main__.py` handles fatal IO/validation startup errors with non-zero exit.
- Per-prompt model mistakes do not crash run.

## Remaining improvement targets

1. Evaluate skip count impact on accuracy metrics (`subject` asks ~90%+).
2. Keep `README` and planning docs synchronized after any behavior change.
3. Resolve remaining full-repo mypy issues outside routing/output path.
4. Optionally rename `tests ` directory (trailing space) for tooling clarity.

## Quick verification commands

- `uv run python -m pytest "tests /" -q`
- `uv run flake8 src tests`
- `uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
- `uv run python -m src --output data/output/function_calling_results.json`
