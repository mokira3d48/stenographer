venv:
	python3 -m venv env

install:
	pip install -r requirements.txt; \
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
	python3 -c "import nltk; nltk.download('punkt')"

dev:
	pip install -e .

test:
	pytest tests

run:
	python3 -m stenographer
