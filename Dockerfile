FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock .python-version ./
COPY app ./app

RUN uv sync --frozen --no-dev

RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
