# Finale, GPU-fähige Dockerfile basierend auf python:slim, läuft als root
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3.11-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip compile pyproject.toml --output-file requirements.txt
RUN uv pip sync --system --no-cache requirements.txt

COPY . .

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
