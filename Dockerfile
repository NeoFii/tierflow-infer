# syntax=docker/dockerfile:1
# Inference-service: GPU image with torch/transformers

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev pkg-config && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
COPY src/ src/
COPY scripts/ scripts/

RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv lock && uv sync --frozen --no-dev

FROM python:3.12-slim AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/logs && chown -R appuser:appuser /app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appuser scripts/ /app/scripts/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8004

CMD ["uvicorn", "inference_service.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
