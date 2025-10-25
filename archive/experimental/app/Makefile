.PHONY: dev test build run-docker gcloud-build deploy lint clean

PY=python3
PROJECT_ID=periodicdent42
REGION=us-central1
IMAGE_NAME=ard-backend

dev:
	@echo "Starting development server..."
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

test:
	@echo "Running tests..."
	$(PY) -m pytest -v --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	$(PY) -m pytest --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Linting code..."
	$(PY) -m ruff check src/
	$(PY) -m black --check src/

format:
	@echo "Formatting code..."
	$(PY) -m black src/
	$(PY) -m ruff --fix src/

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):latest .

run-docker:
	@echo "Running Docker container..."
	docker run -p 8080:8080 \
		-e PROJECT_ID=$(PROJECT_ID) \
		-e LOCATION=$(REGION) \
		--env-file .env \
		$(IMAGE_NAME):latest

gcloud-build:
	@echo "Building image with Cloud Build..."
	gcloud builds submit --tag gcr.io/$(PROJECT_ID)/$(IMAGE_NAME) .

deploy:
	@echo "Deploying to Cloud Run..."
	bash ../infra/scripts/deploy_cloudrun.sh

logs:
	@echo "Tailing Cloud Run logs..."
	gcloud run services logs tail $(IMAGE_NAME) --region=$(REGION)

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

setup-local:
	@echo "Setting up local environment..."
	$(PY) -m pip install -r requirements.txt
	cp .env.example .env
	@echo "Edit .env with your configuration"

