# Built from the repository root (see docker/docker-compose.yml)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY storage/ /app/storage/
COPY sketches/ /app/sketches/
ENTRYPOINT ["python", "-m", "sketches.sketch_service"]
