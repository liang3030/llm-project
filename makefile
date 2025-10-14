.PHONY: install install-dev test lint format type-check clean run setup-hf test-models

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=src/my_llm_project --cov-report=html --cov-report=term

# Run only HuggingFace-specific tests
test-hf:
	pytest tests/unit/test_huggingface*.py tests/integration/test_huggingface*.py -v

# Lint code
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Type check
type-check:
	mypy src/

# Clean cache and build files
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/

# Run the application
run:
	uvicorn src.my_llm_project.api:create_app --factory --host 0.0.0.0 --port 8000


# Run in development mode
dev:
	DEBUG=true uvicorn src.my_llm_project.api:create_app --factory --reload --host 0.0.0.0 --port 8000

# Test API endpoints
test-api:
	@echo "Testing health endpoint..."
	curl -s http://localhost:8000/llm/health | python -m json.tool
	@echo "\nTesting models list..."
	curl -s http://localhost:8000/llm/models | python -m json.tool
	@echo "\nTesting simple generation..."
	curl -s -X POST http://localhost:8000/llm/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello! How are you?", "model_name": "microsoft/DialoGPT-small"}' | python -m json.tool

# Quick model test
quick-test:
	curl -s -X POST http://localhost:8000/llm/test?model_name=microsoft/DialoGPT-small | python -m json.tool

# Install specific HuggingFace dependencies
install-hf:
	pip install langchain-huggingface transformers torch huggingface-hub

# Show model recommendations
recommendations:
	curl -s http://localhost:8000/llm/models/recommendations?task=conversation | python -m json.tool