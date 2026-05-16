# colors
BLUE   := \033[0;34m
YELLOW := \033[0;33m
RESET  := \033[0m

# env vars
UV_CACHE_DIR := $(CURDIR)/.uv-cache
TMPDIR       := $(CURDIR)/.tmp
PYTHON_VER   := 3.11
SRC_DIR      := src
RUN          := uv run -m $(SRC_DIR)

export UV_CACHE_DIR
export TMPDIR

CLEAN_DIRS := .mypy_cache .pytest_cache .uv-cache .tmp .hf .venv

help:
	@printf "$(BLUE)Commands:$(RESET)\n"
	@printf "$(YELLOW)  install      - install dependencies and environment$(RESET)\n"
	@printf "$(YELLOW)  run          - run main module, default dataset (ARGS=\"...\")$(RESET)\n"
	@printf "$(YELLOW)  run-extended - run extended prompt set (24 prompts)$(RESET)\n"
	@printf "$(YELLOW)  run-nested   - run nested object parameter set$(RESET)\n"
	@printf "$(YELLOW)  test         - run unit tests in tests/$(RESET)\n"
	@printf "$(YELLOW)  lint         - run linter (flake8 + mypy)$(RESET)\n"
	@printf "$(YELLOW)  lint-strict  - run mypy in strict mode$(RESET)\n"
	@printf "$(YELLOW)  debug        - run main module under pdb$(RESET)\n"
	@printf "$(YELLOW)  clean        - delete cache$(RESET)\n"

init-dirs:
	@mkdir -p "$(UV_CACHE_DIR)" "$(TMPDIR)"

install: init-dirs
	uv sync --python $(PYTHON_VER)

run: init-dirs
	$(RUN) $(ARGS)

run-extended: init-dirs
	$(RUN) --functions_definition data_test/input/functions_definition.json \
		--input data_test/input/function_calling_tests.json \
		--output data_test/output/function_calling_results.json

run-nested: init-dirs
	$(RUN) --functions_definition data_test_nested/input/functions_definition_nested_object.json \
		--input data_test_nested/input/function_calling_tests.json \
		--output data_test_nested/output/function_calling_results.json

test: install
	PYTHONPATH=. uv run pytest -q

debug: init-dirs
	uv run -m pdb -m $(SRC_DIR) $(ARGS)

clean:
	@echo "$(BLUE)Cleaning cache...$(RESET)"
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf $(CLEAN_DIRS)

lint: install
	@uv run flake8 --jobs 1 $(SRC_DIR)
	@uv run mypy $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict: install
	@uv run flake8 --jobs 1 $(SRC_DIR)
	@uv run mypy $(SRC_DIR) --strict

.PHONY: help init-dirs install run run-extended run-nested test debug clean lint lint-strict
