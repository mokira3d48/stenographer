venv:
	python3 -m venv env

install:
	pip install -r requirements.txt; \
	python3 -c "import nltk; nltk.download('punkt')"

dev:
	pip install -e .

test:
	pytest tests

run:
	python3 -m stenographer
