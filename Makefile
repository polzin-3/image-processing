style:
	pre-commit run -a

test:
	pytest tests

install:
	python -m pip install --upgrade pip setuptools wheel
	pip install --upgrade .[test,dev]