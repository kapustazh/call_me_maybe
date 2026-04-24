# Milestones — `call_me_maybe`

Derived from [PLAN.md](PLAN.md). Check boxes as you go. Order matters inside each milestone unless noted.

---

## M1 — Input / output contract (mostly in place; verify + align)

- [ ] Confirm `FunctionResult`-shaped model matches spec: only keys `prompt`, `name`, `parameters` (or extend model + rename field if spec differs).
- [ ] Ensure `io_utils.py` can read and write the **final** result array; no extra keys in written JSON.
- [ ] Set default `--output` in `__main__.py` to `data/output/function_calling_results.json` (or subject’s exact path from brief).
- [ ] End-to-end dry run: valid inputs → one JSON object per test prompt in output file.

---

## M2 — Token id ↔ text (enables masking)

- [ ] Add module (e.g. `src/tokenizer_vocab.py`) that loads `Small_LLM_Model.get_path_to_tokenizer_file()`; fallback to `get_path_to_vocab_file()` if needed.
- [ ] Expose a stable API: e.g. `id_to_text(token_id: int) -> str` and/or batch map for all vocab indices used in masking.
- [ ] Unit-smoke: a few known ids decode to expected pieces; handle edge ids (BOS/UNK if present).
- [ ] Document any tokenizer quirks (BPE merges, leading space, partial UTF-8) that affect “append piece” validity.

---

## M3 — Constrained decoder: JSON shell (structure only)

- [ ] Define generation context: which prefix the model sees (system + user + functions text + `BobThePrompter` or final instruction prompt).
- [ ] State machine: track partial output string, parse position, brace depth, `inside_string`, escape after `\`.
- [ ] Enforce top-level object only: keys `prompt`, `name`, `parameters` in fixed order (or allowed order you choose, consistently).
- [ ] Whitelist structural chars when machine expects `:` `,` `{` `}` string delimiters, etc.
- [ ] Stopping: generation stops when machine reaches “closed root object, valid JSON”.

---

## M4 — Constrained decoder: `name` and `parameters` schema

- [ ] `name` value: only token sequences that concatenate to an **exact** function name from loaded definitions.
- [ ] `prompt` value: JSON string; allow only valid string-prefix tokens; handle `\"`, `\\`, unicode escapes as needed.
- [ ] `parameters` object: keys from chosen function’s `parameters` only; no extras.
- [ ] Type masks per value: `string` / `number` / `integer` / `boolean` / `object` (match your `FunctionParameter` literals).
- [ ] `object` value: if you only support empty `{}` for now, document it; else recursive structure.

---

## M5 — Masking loop (wire to model)

- [ ] In `constrained_decoder.py` (or split): for each step, `get_logits_from_input_ids(context_ids)`.
- [ ] For each `token_id`, get piece string; if append keeps state machine valid, keep logit; else set `-inf`.
- [ ] Argmax (or sample) among allowed tokens; append chosen id; extend context; repeat until stop.
- [ ] `max_new_tokens` safety cap; on failure, clear error (don’t write invalid JSON).

---

## M6 — Pipeline: replace ad-hoc path with spec output

- [ ] In `Pipeline.run()`: for each `PromptItem`, run constrained generator once; build `FunctionResult` (or list of dicts matching schema).
- [ ] **Remove or gate** the current “test harness” (`model_guess` / `selection_prompt` dump) for final submission, or keep behind a dev-only flag.
- [ ] After generation: `json.loads` + validate with pydantic (or your adapter) for each result.
- [ ] `write_json` full array to `--output` path.

---

## M7 — Quality, errors, and performance

- [ ] Input missing / bad JSON: raise `JsonFileError` / `JsonValidationError` with good messages; CLI prints message, not raw `f"{e=}"`.
- [ ] Parameter fill: ensure numbers/strings are correctly escaped and unambiguous (quotes in user text).
- [ ] No hardcoded function names: always from `functions_definition.json`.
- [ ] Measure runtime on `function_calling_tests.json`; ensure under ~5 minutes (PLAN).
- [ ] `make install` and `make lint` clean (flake8 + mypy if in Makefile).

---

## M8 — Documentation (subject checklist)

- [ ] README: first italicized line, algorithm, design decisions, performance, challenges, testing strategy, usage examples.
- [ ] Resources + how you used AI (if required by subject).
- [ ] `PLAN.md` frontmatter: set todo statuses to match reality, or point readers to this file for execution detail.

---

## M9 — Optional local tests (not necessarily submitted)

- [ ] Corrupt a copy of input JSON; confirm graceful error.
- [ ] Edge prompts: empty string, lots of quotes, unicode, very large number strings.
- [ ] Golden-file diff: run once, store expected `function_calling_results.json` for regression (optional).

---

## Quick “next move” (if you do only one thing)

- [ ] **M2 + M3 in parallel:** token map + JSON-shell state machine without LLM, then (M5) connect logits.

---

## Legend

- **Spec output file name:** PLAN says `data/output/function_calling_results.json`; your CLI may still use `function_calls.json` — pick one and align M1.
- **Bob / FunctionSelector:** useful for exploration; final product is constrained-decoded `prompt` + `name` + `parameters`, not “guess only name” at end.
