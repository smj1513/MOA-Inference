python -m uv sync --extra dev
python -m uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
