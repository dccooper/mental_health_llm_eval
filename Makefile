.PHONY: install install-dev test lint format clean docs build publish

# Default target
all: install test lint

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,ui,openai,anthropic,huggingface]"

# Testing
test:
	pytest tests/ --cov=mental_health_llm_eval --cov-report=term-missing

test-watch:
	pytest-watch -- tests/ --cov=mental_health_llm_eval

# Code quality
lint:
	flake8 src/ tests/
	mypy src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf **/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf docs/_build/

# Building and publishing
build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Development environment
setup-dev: clean install-dev
	pre-commit install

# Run the UI
ui:
	streamlit run app/streamlit_app.py

# Run the CLI
cli:
	mh-llm-eval --help 