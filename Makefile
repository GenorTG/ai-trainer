# AI Trainer — Makefile
# Convenience commands for development, testing, linting.

.PHONY: help install install-dev install-test install-lint install-tools
.PHONY: test test-unit test-api test-frontend test-cov test-fast test-watch
.PHONY: lint lint-fix lint-strict format type-check security docstring-check complexity-check
.PHONY: clean clean-cache clean-models clean-venv clean-all
.PHONY: serve serve-trainer serve-inference serve-both
.PHONY: train benchmark compare
.PHONY: deploy deploy-test deploy-status deploy-logs deploy-restart
.PHONY: docker-build docker-run docker-stop docker-clean
.PHONY: pre-commit-install pre-commit-run pre-commit-update
.PHONY: git-init git-status git-commit git-push git-tag git-release
.PHONY: docs docs-build docs-serve
.PHONY: version version-bump-patch version-bump-minor version-bump-major
.PHONY: all-checks ci

# =============================================================================
# HELP
# =============================================================================
help:                   ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# INSTALL
# =============================================================================
install:               ## Install project in dev mode
	pip install -e ./trainer
	pip install -e ./server

install-test:          ## Install test dependencies
	pip install pytest pytest-asyncio pytest-mock pytest-cov pytest-xdist

install-lint:          ## Install lint dependencies
	pip install --break-system-packages ruff mypy bandit pydocstyle vulture

install-tools:         ## Install all dev tools
	pip install pre-commit safety pip-audit

install-dev: install install-test install-lint install-tools  ## Install everything

# =============================================================================
# TEST
# =============================================================================
test:                  ## Run all tests (573)
	python tests/run_all.py

test-unit:                  ## Run unit tests only (416)
	python tests/run_all.py --suite unit

test-api:                   ## Run API tests only (187)
	python tests/run_all.py --suite api

test-frontend:              ## Run frontend tests only (105)
	python tests/run_all.py --suite frontend

test-cov:                   ## Run all tests with coverage report
	python tests/run_all.py --coverage

test-fast:                  ## Run fast tests only (skip slow markers)
	python -m pytest tests/ -m "not slow" -q

test-watch:                 ## Run tests on file change (requires pytest-watch)
	ptw tests/

# =============================================================================
# LINT
# =============================================================================
lint:                  ## Run ruff linter
	ruff check trainer/ server/

lint-fix:               ## Run ruff linter with auto-fix
	ruff check --fix trainer/ server/

lint-strict:            ## Run ruff with all rules
	ruff check --select ALL --ignore D100,D101,D102,D103,D104 trainer/ server/

format:                 ## Format code with ruff
	ruff format trainer/ server/

type-check:             ## Run mypy type checker
	mypy trainer/src server/inference_server --ignore-missing-imports --no-strict-optional

security:               ## Run bandit security linter
	bandit -r trainer/ server/ -ll

docstring-check:        ## Check docstring style
	pydocstyle trainer/ server/

complexity-check:       ## Find dead code
	vulture trainer/src server/inference_server

# =============================================================================
# CLEAN
# =============================================================================
clean:                 ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null

clean-cache: clean

clean-models:                 ## Remove downloaded model files (NOT committed)
	rm -rf models/*.gguf
	rm -rf models/*.safetensors
	rm -rf trainer/output_*

clean-venv:                 ## Remove virtual environments
	rm -rf .venv
	rm -rf trainer/.venv
	rm -rf server/.venv

clean-all: clean clean-models clean-venv   ## Clean EVERYTHING

# =============================================================================
# SERVE
# =============================================================================
serve:                 ## Start trainer WebUI (port 7860)
	fts webui

serve-trainer: serve

serve-inference:       ## Start inference server (with v21 GGUF)
	inference-server serve --model models/v21/chris-ai-gemma4-e4b-v21.Q4_K_M.gguf

serve-both:            ## Start both services
	@echo "Starting both services..."
	@make -j 2 serve-trainer serve-inference

# =============================================================================
# TRAIN / BENCHMARK / COMPARE
# =============================================================================
train:                 ## Run training (v21 with knowledge preservation)
	python trainer/train_v21.py

benchmark:             ## Run benchmarks on v21
	fts benchmark models/v21/chris-ai-gemma4-e4b-v21.Q4_K_M.gguf --suite all

compare:               ## Compare v20 vs v21
	fts compare --models \
  v20=models/v20/chris-ai-gemma4-e4b-v20.Q4_K_M.gguf \
  v21=models/v21/chris-ai-gemma4-e4b-v21.Q4_K_M.gguf \
  --suite trainer/comparison_suite.json

# =============================================================================
# DEPLOY (genorbox1 production)
# =============================================================================
deploy:                ## Deploy to genorbox1
	./tools/deploy.sh

deploy-test:               ## Deploy test files only
	rsync -avz --exclude='__pycache__' --exclude='coverage_html' \
		tests/ genorbox1:/home/genorbox1/llama-server/tests/

deploy-status:             ## Check llama-server status on genorbox1
	ssh genorbox1 "systemctl status chris-ai.service 2>&1 | head -10"

deploy-logs:               ## Tail llama-server logs on genorbox1
	ssh genorbox1 "journalctl -u chris-ai.service -f"

deploy-restart:            ## Restart llama-server on genorbox1
	ssh genorbox1 "sudo systemctl restart chris-ai.service"

# =============================================================================
# DOCKER
# =============================================================================
docker-build:          ## Build inference server Docker image
	cd server && docker build -t ai-trainer/server:latest .

docker-run:            ## Run inference server in Docker
	docker run --rm -p 8888:8888 --gpus all ai-trainer/server:latest

docker-stop:           ## Stop all running containers
	docker stop $(docker ps -q)

docker-clean:          ## Remove all Docker artifacts
	docker system prune -af

# =============================================================================
# PRE-COMMIT
# =============================================================================
pre-commit-install:    ## Install pre-commit hooks
	pre-commit install

pre-commit-run:        ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update:     ## Update pre-commit hook versions
	pre-commit autoupdate

# =============================================================================
# GIT
# =============================================================================
git-init:              ## Initialize git repo
	git init -b main
	git config user.email "amy@smart-samurai.pl"
	git config user.name "Amy"
	git config commit.gpgsign false

git-status:            ## Show git status
	git status --short

git-commit:            ## Commit with conventional message (use m=)
	git commit

git-push:              ## Push to remote
	git push origin main

git-tag:               ## Tag current version (use v=X.Y.Z)
	git tag -a v$(VERSION) -m "Release v$(VERSION)"

git-release:           ## Tag + push release
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)

# =============================================================================
# DOCS
# =============================================================================
docs:                  ## Show docs index
	@echo "Documentation index:"
	@ls -la docs/

docs-build:            ## Build documentation (placeholder)
	@echo "(docs build not yet implemented)"

docs-serve:            ## Serve docs locally (placeholder)
	@echo "(docs serve not yet implemented)"

# =============================================================================
# VERSION
# =============================================================================
version:               ## Show current version
	@grep version trainer/pyproject.toml | head -1
	@grep version server/pyproject.toml | head -1

version-bump-patch:    ## Bump patch version (1.0.0 → 1.0.1)
	@./tools/bump.py patch

version-bump-minor:    ## Bump minor version (1.0.0 → 1.1.0)
	@./tools/bump.py minor

version-bump-major:    ## Bump major version (1.0.0 → 2.0.0)
	@./tools/bump.py major

# =============================================================================
# CI / ALL CHECKS
# =============================================================================
all-checks: lint type-check security test-cov  ## Run all checks

ci:                    ## Same as all-checks but exits on first failure
	make lint && make type-check && make security && make test-cov

# Run by default when no target
.DEFAULT_GOAL := help