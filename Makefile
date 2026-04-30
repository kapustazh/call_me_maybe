# colors
BLUE   := \033[0;34m
YELLOW := \033[0;33m
RESET  := \033[0m

# env vars
UV_CACHE_DIR := $(CURDIR)/.uv-cache
TMPDIR       := $(CURDIR)/.tmp
PYTHON_VER   := 3.11
SRC_DIR      := src

export UV_CACHE_DIR
export TMPDIR

CLEAN_DIRS := .mypy_cache .pytest_cache .uv-cache .tmp .hf .venv

help:
	@printf "$(BLUE)Commands:$(RESET)\n"
	@printf "$(YELLOW)  install      - install dependencies and envirnonment$(RESET)\n"
	@printf "$(YELLOW)  run          - run man module (with ARGS=\"...\")$(RESET)\n"
	@printf "$(YELLOW)  lint         - run linter (flake8 + mypy)$(RESET)\n"
	@printf "$(YELLOW)  test         - run tests in tests/$(RESET)\n"
	@printf "$(YELLOW)  clean        - delete cache$(RESET)\n"

init-dirs:
	@mkdir -p "$(UV_CACHE_DIR)" "$(TMPDIR)"

install: init-dirs
	uv sync --python $(PYTHON_VER)

run: init-dirs
	uv run -m $(SRC_DIR) $(ARGS)

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

test: 
	@PYTHONPATH=. uv run pytest -q

.PHONY: help init-dirs install run debug clean lint lint-strict test