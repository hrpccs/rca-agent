.PHONY: help install dev test test-contracts lint format up down logs ps import eval serve frontend clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install python deps via uv
	uv sync

dev: install ## Install deps and create .env from example if missing
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example — edit it (esp. RCA_DEEPSEEK_API_KEY)"; fi

test: ## Run the full test suite (excludes live)
	uv run pytest -q -m "not live"

test-live: ## Run tests including live DeepSeek/DB tests
	uv run pytest -q

test-contracts: ## Run contract conformance gate
	uv run pytest -q tests/contracts

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

# ---------- infrastructure ----------
up: ## Start infra (clickhouse, mysql, otel-collector, tempo, grafana)
	docker compose up -d clickhouse mysql otel-collector tempo grafana
	@echo "Infra up. Grafana: http://localhost:3000 (admin/admin)"

down: ## Stop infra
	docker compose down

logs: ## Tail infra logs
	docker compose logs -f --tail=100

ps: ## Show container status
	docker compose ps

# ---------- data ----------
import: ## Import a case (usage: make import CASE=t001)
	uv run python -m rca_agent.cli import-case $(CASE)

# ---------- run / serve ----------
serve: ## Start the RCA server
	uv run uvicorn rca_agent.server.app:app --host $${RCA_SERVER_HOST:-0.0.0.0} --port $${RCA_SERVER_PORT:-8000} --reload

run: ## Run RCA on a case (usage: make run CASE=t001)
	uv run python -m rca_agent.cli run $(CASE)

eval: ## Evaluate on cases (usage: make eval CASES=t001,t002)
	uv run python -m rca_agent.cli eval --cases $(CASES)

frontend: ## Start the frontend dev server
	cd frontend && npm install && npm run dev

clean: ## Remove caches and run artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache runs/*.json
