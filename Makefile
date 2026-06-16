.PHONY: setup ingest serve eval test clean

setup:
	@echo "==> Installing dependencies..."
	pip install -r requirements.txt
	@echo "==> Creating directories..."
	mkdir -p data/chroma data/corpus logs
	@echo "==> Copying .env if needed..."
	@[ -f .env ] || cp .env.example .env && echo "Created .env — add your ANTHROPIC_API_KEY"
	@echo "==> Scraping and ingesting Airflow docs..."
	python scripts/ingest.py
	@echo "✓ Setup complete. Run 'make serve' to start the API."

ingest:
	python scripts/ingest.py

serve:
	uvicorn src.api.app:app --reload --port 8000

eval:
	python eval/run_eval.py --output eval/results_latest.json

test:
	python -m pytest tests/ -v

clean:
	rm -rf data/chroma data/corpus logs/__pycache__ src/__pycache__
	find . -name "*.pyc" -delete

# Docker
docker-up:
	docker compose up --build

docker-down:
	docker compose down
