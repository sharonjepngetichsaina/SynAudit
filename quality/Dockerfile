# Build from the project root:  docker build -f quality/Dockerfile -t synaudit-quality .
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY quality/ quality/
# column types and ranges come from the SDV metadata written by generate_synthetic.py
COPY data/synthetic/*-metadata.json data/synthetic/

ENV PYTHONUNBUFFERED=1 \
    KAFKA_BOOTSTRAP=kafka:9092

# DATASET (adult | credit-default) is set per service in docker compose
CMD ["python", "-m", "quality.consumer"]
