# ====== Config ======
ENV_NAME ?= qAlice-Py313
PY       ?= python
PIP      ?= python -m pip

# ====== Help (default) ======
help: ## Show available commands
	@echo "Make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ====== Env & deps ======
env: ## Create the base mamba env (Python 3.13, core libs)
	mamba env create -f environment.yml

extras-mamba: ## Layer optional quantum deps via mamba env update
	mamba env update -n $(ENV_NAME) -f environment.optional.yml

extras-pip: ## Layer optional quantum deps via pip requirements
	$(PIP) install -r requirements-optional.txt

install: ## Editable install of the package (gives `qAlice` CLI)
	$(PIP) install -e .
	# For quantum extras: make extras-pip

dev: ## One-shot: pre-commit hooks + quick smoke test
	pre-commit install
	pytest -q

freeze: ## Export pip freeze to requirements.lock.txt (for debugging)
	$(PY) -m pip freeze > requirements.lock.txt

# ====== Lint / Format ======
lint: ## Run static checks (ruff/black/isort)
	ruff check .
	black --check .
	isort --check-only .

format: ## Auto-fix style issues
	ruff check . --fix
	black .
	isort .

# ====== Tests ======
test: ## Run fast tests (quiet)
	pytest -q

testv: ## Run tests verbosely
	pytest -vv

test-cov: ## Run tests with coverage (requires pytest-cov)
	pytest --cov=qalice_core --cov-report=term-missing

# ====== Build / Package ======
build-deps: ## Install build backend
	$(PIP) install build

package: build-deps ## Build sdist+wheel into dist/
	$(PY) -m build

# ====== Run / CLI ======
run: ## Show CLI help
	lattice --help

hello: ## Demo CLI command
	qalice hello --name "qAlice"

# ====== Clean ======
clean: ## Remove caches, build, coverage artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf build dist *.egg-info
	rm -f .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

# ====== CI parity ======
ci: ## Mimic GitHub Actions locally (install + lint + tests)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(MAKE) lint
	$(MAKE) test
