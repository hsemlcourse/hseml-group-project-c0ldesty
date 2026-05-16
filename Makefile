lint:
	flake8 src tests --max-line-length=120

test:
	pytest -q
