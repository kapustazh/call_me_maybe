UV_CACHE_DIR := $(CURDIR)/.uv-cache
TMPDIR := $(CURDIR)/.tmp
export UV_CACHE_DIR
export TMPDIR

init-dirs:
	mkdir -p "$(UV_CACHE_DIR)" "$(TMPDIR)"

install: init-dirs
	uv sync --python 3.11

run: init-dirs
	uv run -m src $(ARGS)

debug: init-dirs
	uv run -m pdb -m src $(ARGS)

clean:
	find . -type f -name '*.py[co]' -delete
	rm -rf .mypy_cache .pytest_cache .uv-cache .tmp .hf
	find . -type d -name __pycache__ -exec rm -rf {} +

lint: install
	uv run flake8 --jobs 1 src
	uv run mypy src\
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--exclude '(^\.venv/)'

lint-strict: install
	uv run flake8 --jobs 1 src
	uv run mypy src --strict --exclude '(^\.venv/)'

.PHONY: init-dirs install run debug clean lint lint-strict