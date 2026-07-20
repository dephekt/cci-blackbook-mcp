# Local development helpers. See README.md for the full quickstart.
.PHONY: help up down logs ingest ingest-force smoke test lint

help:
	@echo "make up      - build & start the MCP locally (http://127.0.0.1:8000/mcp)"
	@echo "make ingest       - incrementally refresh the index from data/source/*.pdf"
	@echo "make ingest-force - rebuild every source (stop the MCP first for schema upgrades)"
	@echo "make smoke   - synthetic Voyage connectivity check (needs VOYAGE_API_KEY)"
	@echo "make test    - run the offline unit tests"
	@echo "make lint    - ruff check"
	@echo "make logs    - follow container logs"
	@echo "make down    - stop and remove the container"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f | cat

ingest:
	docker compose exec cci-blackbook cci-blackbook-ingest

ingest-force:
	docker compose run --rm cci-blackbook cci-blackbook-ingest --force

smoke:
	docker compose exec cci-blackbook cci-blackbook-ingest --smoke

test:
	uv run --project . python -m unittest discover -s tests -p 'test_*.py'

lint:
	uvx ruff check app tests
