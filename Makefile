install:
	uv sync --python 3.11

run:
	uv run main.py $(ARGS)

debug:
	uv run -m pdb main.py $(ARGS)

clean:
	find . -type f -name '*.py[co]' -delete
	rm -rf .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

lint:
	uv run flake8 *.py
	uv run mypy *.py\
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--exclude '(^\.venv/)'

lint-strict:
	uv run flake8 *.py
	uv run mypy *.py --strict --exclude '(^\.venv/)'

.PHONY: install run debug clean lint lint-strict