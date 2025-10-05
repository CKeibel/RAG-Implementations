FROM python:3.11-slim

RUN pip install uv

RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN uv pip compile pyproject.toml --output-file requirements.txt
RUN uv pip sync --system --no-cache requirements.txt

COPY . .

CMD ["celery", "-A", "app.worker.celery_app", "worker", "--loglevel=info"]
