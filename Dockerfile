FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY evaluation/uzbek ./evaluation/uzbek
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
CMD ["uvicorn", "faq.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
