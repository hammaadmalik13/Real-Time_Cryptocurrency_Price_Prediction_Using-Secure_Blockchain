FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY data ./data
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "crypto_lstm.api:app", "--host", "0.0.0.0", "--port", "8000"]

