FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system nurai \
    && adduser --system --ingroup nurai nurai

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[agent]"

USER nurai

EXPOSE 8000

CMD ["uvicorn", "nurai.main:app", "--host", "0.0.0.0", "--port", "8000"]
