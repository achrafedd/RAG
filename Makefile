run:
	uv run -m src index
	uv run -m src search_dataset \
		--dataset_path data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
		--k 5 \
		--save_directory data/output/search_datase
	uv run -m src evaluate \
		data/output/search_dataset/dataset_docs_public.json \
		data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json

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
	@echo -e "\e[42m\e[30m Directory is cleaned \e[0m"

lint:
	-uv run python3 -m flake8 . --exclude=.venv,data
	-uv run python3 -m mypy . --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs
