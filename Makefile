.PHONY: help install test lint format clean run-pipeline run-app generate-data

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	pip install -e ".[dev]"

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing

lint: ## Run linting
	ruff check src/ tests/ run_pipeline.py

format: ## Format code
	ruff format src/ tests/ run_pipeline.py

clean: ## Clean generated files
	rm -rf artifacts/* data/processed/* .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

run-pipeline: ## Run the full pipeline
	python run_pipeline.py

run-app: ## Run Streamlit app
	streamlit run app/streamlit_app.py

generate-data: ## Generate sample data
	python generate_sample_data.py

generate-pdf: ## Generate project summary PDF
	python gen_pdf.py

all: clean install test lint format run-pipeline ## Full pipeline run

.DEFAULT_GOAL := help