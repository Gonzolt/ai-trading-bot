.PHONY: test lint up down api ui

test:
	pytest

lint:
	ruff check src tests

up:
	docker compose up -d --build

down:
	docker compose down

api:
	uvicorn trading_bot.dashboard.api.main:app --reload

ui:
	cd dashboard-ui && npm run dev
