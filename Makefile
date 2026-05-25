lint:
	flake8 src tests --max-line-length=120

test:
	pytest -q

api:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

streamlit:
	python -m streamlit run src/streamlit_app.py
