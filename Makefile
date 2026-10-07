run:
	uv run src index

run-back:
	uv run fastapi dev

run-front:
	npx live-server src/frontend

install:
	uv sync

clean:
	@echo "\n"
	@uv run -m pyclean . --debris
	@rm -rf .mypy_cache
	@echo "\e[42m\e[30m Directory is cleaned \e[0m"

lint:
	-uv run python3 -m flake8 . --exclude=.venv
	-uv run python3 -m mypy . --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs
