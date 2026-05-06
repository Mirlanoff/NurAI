.PHONY: install install-ml install-eval install-agent lint typecheck test quality eval run docker-up docker-dev docker-down

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

install-ml:
	.venv/bin/pip install -e ".[ml]"

install-eval:
	.venv/bin/pip install -e ".[eval]"

install-agent:
	.venv/bin/pip install -e ".[agent]"

lint:
	.venv/bin/ruff check .

typecheck:
	.venv/bin/mypy src

test:
	.venv/bin/pytest

quality: lint typecheck test

eval:
	.venv/bin/python -m nurai.evaluation.runner \
		--corpus eval/corpus.jsonl \
		--dataset eval/golden.jsonl \
		--thresholds eval/thresholds.json \
		--report eval/report.json

run:
	.venv/bin/uvicorn nurai.main:app --reload

docker-up:
	docker compose up --build

docker-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

docker-down:
	docker compose down --remove-orphans
