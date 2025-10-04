# Phase 1: Der "Builder"
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y git --no-install-recommends

RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./

RUN uv pip compile pyproject.toml --output-file requirements.txt

RUN uv pip sync --system --no-cache requirements.txt

# ---

FROM python:3.11-slim


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN adduser --system --group nonroot
WORKDIR /home/nonroot/app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --chown=nonroot:nonroot app/ ./app/

USER nonroot

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
